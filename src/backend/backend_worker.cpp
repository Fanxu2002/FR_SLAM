#include "fr_slam/frontend/lo_frontend.hpp"
#include "fr_slam/frontend/ground_segmenter.hpp"
#include "fr_slam/frontend/ground_input_bridge.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <cstdint>
#include <cstddef>
#include <exception>
#include <utility>
#include <vector>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>
#include <pcl/search/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <Eigen/Eigenvalues>

#include <sophus/so3.hpp>

#include <rclcpp/rclcpp.hpp>
namespace
{
    const rclcpp::Logger kTimingLogger =
        rclcpp::get_logger("scan2local_map.timing");

    double ElapsedMilliseconds(
        const std::chrono::steady_clock::time_point &start,
        const std::chrono::steady_clock::time_point &end)
    {
        return std::chrono::duration<double, std::milli>(
                   end - start)
            .count();
    }
} // namespace

void RegistrationScan2LocalMap::StartBackendWorker()
{
    bool expected = false;

    if (!backend_running_.compare_exchange_strong(
            expected,
            true))
    {
        return;
    }

    backend_thread_ =
        std::thread(
            &RegistrationScan2LocalMap::BackendLoop,
            this);
}

void RegistrationScan2LocalMap::StopBackendWorker()
{
    if (!backend_running_.exchange(false))
    {
        if (backend_thread_.joinable())
        {
            backend_thread_.join();
        }

        return;
    }

    {
        std::lock_guard<std::mutex> lock(
            backend_queue_mutex_);

        backend_queue_.clear();
    }

    backend_condition_.notify_all();

    if (backend_thread_.joinable())
    {
        backend_thread_.join();
    }
}

bool RegistrationScan2LocalMap::BuildFinishedSubmapSnapshot(
    std::size_t submap_id,
    BackendSubmapSnapshot &snapshot) const
{
    for (const Submap &submap :
         submap_manager_.GetAllSubmaps())
    {
        if (submap.id != submap_id)
        {
            continue;
        }

        if (!submap.finished ||
            !submap.has_frozen_cloud ||
            !submap.has_origin_pose ||
            !submap.cloud_S ||
            submap.cloud_S->empty() ||
            !submap.T_WS.matrix().allFinite())
        {
            return false;
        }

        snapshot = BackendSubmapSnapshot();
        snapshot.id = submap.id;
        snapshot.T_WS = submap.T_WS;
        snapshot.keyframe_ids = submap.keyframe_ids;
        snapshot.cloud_S = submap.cloud_S;

        return true;
    }

    return false;
}

bool RegistrationScan2LocalMap::EnqueueBackendKeyframe(
    const Keyframe &keyframe,
    std::size_t current_submap_id)
{
    if (!backend_running_.load() ||
        !keyframe.cloud ||
        keyframe.cloud->empty() ||
        !keyframe.T_WL.matrix().allFinite())
    {
        return false;
    }

    BackendKeyframeJob job;
    job.keyframe = keyframe;
    job.current_submap_id = current_submap_id;

    if (submap_manager_.LastAddStartedNewSubmap())
    {
        BackendSubmapSnapshot finished_snapshot;

        if (BuildFinishedSubmapSnapshot(
                submap_manager_.LastFinishedSubmapId(),
                finished_snapshot))
        {
            job.has_finished_submap = true;
            job.finished_submap =
                std::move(finished_snapshot);
        }
    }

    std::size_t queue_size = 0;

    {
        std::lock_guard<std::mutex> lock(
            backend_queue_mutex_);

        if (!backend_running_.load())
        {
            return false;
        }

        backend_queue_.push_back(
            std::move(job));

        queue_size =
            backend_queue_.size();
    }

    backend_condition_.notify_one();

    if (queue_size >
        backend_backlog_warning_threshold_)
    {
        RCLCPP_WARN(
            kTimingLogger,
            "FR_BACKEND backlog growing"
            " | queued_keyframes=%zu"
            " | warning_threshold=%zu",
            queue_size,
            backend_backlog_warning_threshold_);
    }

    return true;
}

void RegistrationScan2LocalMap::StoreBackendFinishedSubmap(
    const BackendSubmapSnapshot &snapshot)
{
    if (snapshot.id ==
            std::numeric_limits<std::size_t>::max() ||
        !snapshot.cloud_S ||
        snapshot.cloud_S->empty() ||
        !snapshot.T_WS.matrix().allFinite())
    {
        return;
    }

    for (BackendSubmapSnapshot &stored :
         backend_finished_submaps_)
    {
        if (stored.id == snapshot.id)
        {
            stored = snapshot;
            return;
        }
    }

    backend_finished_submaps_.push_back(
        snapshot);
}

void RegistrationScan2LocalMap::RefreshBackendOutputSnapshot()
{
    std::lock_guard<std::mutex> lock(
        backend_output_mutex_);

    backend_pose_graph_snapshot_ =
        pose_graph_;

    backend_raw_map_snapshot_ =
        incremental_global_map_.GetRawMap();

    backend_optimized_map_snapshot_ =
        incremental_global_map_.GetOptimizedMap();

    backend_refined_map_snapshot_ =
        incremental_global_map_.GetRefinedMap();

    backend_refinement_historical_target_snapshot_ =
        refinement_historical_target_debug_;

    backend_refinement_current_before_snapshot_ =
        refinement_current_before_debug_;

    backend_refinement_current_after_snapshot_ =
        refinement_current_after_debug_;

    backend_global_map_revision_snapshot_ =
        global_map_revision_;

    backend_refined_map_revision_snapshot_ =
        refined_map_revision_;

    backend_refinement_debug_revision_snapshot_ =
        refinement_debug_revision_;

    backend_T_map_odom_snapshot_ =
        T_map_odom_;

    backend_has_map_odom_correction_snapshot_ =
        has_map_odom_correction_;

    backend_map_odom_revision_snapshot_ =
        map_odom_revision_;
}

void RegistrationScan2LocalMap::ProcessBackendJob(
    const BackendKeyframeJob &job)
{
    const std::chrono::steady_clock::time_point
        backend_start =
            std::chrono::steady_clock::now();

    double pose_graph_ms = 0.0;
    double global_map_ms = 0.0;
    double loop_ms = 0.0;

    if (job.has_finished_submap)
    {
        StoreBackendFinishedSubmap(
            job.finished_submap);
    }

    if (FindBackendKeyframeById(
            job.keyframe.id) == nullptr)
    {
        backend_keyframes_.push_back(
            job.keyframe);
    }

    const std::chrono::steady_clock::time_point
        pose_graph_start =
            std::chrono::steady_clock::now();

    const bool pose_graph_ok =
        AddKeyframeToPoseGraph(
            job.keyframe);

    pose_graph_ms =
        ElapsedMilliseconds(
            pose_graph_start,
            std::chrono::steady_clock::now());

    if (!pose_graph_ok)
    {
        std::cerr
            << "Async backend PoseGraph insert failed"
            << " | keyframe=" << job.keyframe.id
            << std::endl;

        RefreshBackendOutputSnapshot();
        return;
    }

    const std::chrono::steady_clock::time_point
        global_map_start =
            std::chrono::steady_clock::now();

    const bool global_map_ok =
        UpdateIncrementalGlobalMaps(
            "NEW_KEYFRAME",
            false);

    global_map_ms =
        ElapsedMilliseconds(
            global_map_start,
            std::chrono::steady_clock::now());

    if (!global_map_ok)
    {
        std::cerr
            << "Async backend global map update failed"
            << " | keyframe=" << job.keyframe.id
            << std::endl;
    }

    if (loop_detector_.GetConfig().enabled)
    {
        const std::chrono::steady_clock::time_point
            loop_start =
                std::chrono::steady_clock::now();

        DetectAndVerifyLoopFromKeyframe(
            job.keyframe,
            job.current_submap_id);

        loop_ms =
            ElapsedMilliseconds(
                loop_start,
                std::chrono::steady_clock::now());
    }

    RefreshBackendOutputSnapshot();

    std::size_t remaining_queue = 0;

    {
        std::lock_guard<std::mutex> lock(
            backend_queue_mutex_);

        remaining_queue =
            backend_queue_.size();
    }

    const double total_ms =
        ElapsedMilliseconds(
            backend_start,
            std::chrono::steady_clock::now());

}

void RegistrationScan2LocalMap::BackendLoop()
{
    while (true)
    {
        BackendKeyframeJob job;

        {
            std::unique_lock<std::mutex> lock(
                backend_queue_mutex_);

            backend_condition_.wait(
                lock,
                [this]()
                {
                    return !backend_running_.load() ||
                           !backend_queue_.empty();
                });

            if (!backend_running_.load() &&
                backend_queue_.empty())
            {
                break;
            }

            if (backend_queue_.empty())
            {
                continue;
            }

            job =
                std::move(
                    backend_queue_.front());

            backend_queue_.pop_front();
        }

        try
        {
            ProcessBackendJob(
                job);
        }
        catch (const std::exception &exception)
        {
            std::cerr
                << "Async backend exception"
                << " | keyframe=" << job.keyframe.id
                << " | what=" << exception.what()
                << std::endl;
        }
        catch (...)
        {
            std::cerr
                << "Async backend unknown exception"
                << " | keyframe=" << job.keyframe.id
                << std::endl;
        }
    }
}

// ============================================================================
// AddFrame()
//
// Process ONE LiDAR scan.
//
// Parameters:
//
// cloud_lidar:
//     Current processed LiDAR cloud.
//
//     Coordinate frame:
//         current LiDAR frame.
//
//     It is expected to have already gone through the outer pipeline such as:
//
//         deskew
//         filtering
//         voxelization
//
// timestamp:
//     Current LiDAR scan start timestamp in seconds.
//
//     This is stored in KeyframeManager when the frame becomes a keyframe.
//
// T_WL:
//     Output parameter.
//
//     On success:
//         receives the current accepted LiDAR -> World pose.
//
//     On rejection:
//         this function returns false and does NOT commit the rejected pose.
//
// registration_result:
//     Output registration diagnostics:
//
//         success
//         converged
//         correspondences
//         rmse
//         T_target_source
//         ...
//
// imu_relative_rotation:
//     Optional relative LiDAR rotation prediction derived from IMU.
//
//     In the current frontend design, IMU contributes ONLY to the rotational
//     part of the registration initial guess.
//
// Returns:
//
//     true:
//         current frame is accepted by the frontend.
//
//     false:
//         current frame is rejected / cannot be processed.
//
// ============================================================================
