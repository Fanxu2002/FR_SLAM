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

// ============================================================================
// Async Post-PGO refinement worker.
//
// V1 infrastructure only:
//     - lifecycle
//     - immutable snapshot jobs
//     - latest-only pending slot
//
// Heavy refinement computation runs on the dedicated worker.
// ============================================================================
void RegistrationScan2LocalMap::StartPostPgoRefinementWorker()
{
    if (loop_runtime_config_.post_pgo_refinement_mode != "async")
    {
        return;
    }

    bool expected = false;

    if (!refinement_worker_running_.compare_exchange_strong(
            expected,
            true))
    {
        return;
    }

    refinement_worker_thread_ =
        std::thread(
            &RegistrationScan2LocalMap::PostPgoRefinementLoop,
            this);

    std::cout
        << "BACKEND_REFINE_ASYNC_WORKER_START_V1"
        << " | mode=async"
        << std::endl;
}

void RegistrationScan2LocalMap::StopPostPgoRefinementWorker()
{
    if (!refinement_worker_running_.exchange(false))
    {
        if (refinement_worker_thread_.joinable())
        {
            refinement_worker_thread_.join();
        }

        return;
    }

    {
        std::lock_guard<std::mutex> lock(
            refinement_worker_mutex_);

        pending_refinement_job_.reset();
        completed_refinement_result_.reset();
        refinement_result_ready_.store(false);
    }

    refinement_worker_condition_.notify_all();

    if (refinement_worker_thread_.joinable())
    {
        refinement_worker_thread_.join();
    }

    std::cout
        << "BACKEND_REFINE_ASYNC_WORKER_STOP_V1"
        << std::endl;
}

bool RegistrationScan2LocalMap::SubmitPostPgoRefinementJob()
{
    if (loop_runtime_config_.post_pgo_refinement_mode != "async" ||
        !refinement_worker_running_.load() ||
        global_map_revision_ == 0 ||
        backend_keyframes_.empty() ||
        pose_graph_.NodeCount() == 0)
    {
        return false;
    }

    auto job =
        std::make_unique<PostPgoRefinementJob>();

    job->pgo_epoch =
        pose_graph_optimization_epoch_;

    job->source_global_revision =
        global_map_revision_;

    job->loop_edges =
        pose_graph_.LoopEdgeCount();

    // IMPORTANT:
    // This method is called by the single backend thread.  Therefore these
    // copies are made while backend_keyframes_ / pose_graph_ are not being
    // mutated by another backend operation.
    //
    // Keyframe clouds are immutable shared buffers, so copying Keyframe history
    // does not duplicate all point data.
    job->keyframes =
        backend_keyframes_;

    job->pose_graph =
        pose_graph_;

    bool replaced_pending = false;

    {
        std::lock_guard<std::mutex> lock(
            refinement_worker_mutex_);

        replaced_pending =
            static_cast<bool>(
                pending_refinement_job_);

        if (replaced_pending)
        {
            ++refinement_jobs_replaced_;
        }

        pending_refinement_job_ =
            std::move(job);

        ++refinement_jobs_submitted_;
    }

    refinement_worker_condition_.notify_one();

    std::cout
        << "BACKEND_REFINE_ASYNC_SUBMIT_V1"
        << " | pgo_epoch=" << pose_graph_optimization_epoch_
        << " | global_revision=" << global_map_revision_
        << " | loop_edges=" << pose_graph_.LoopEdgeCount()
        << " | keyframes=" << backend_keyframes_.size()
        << " | replaced_pending="
        << (replaced_pending ? 1 : 0)
        << " | submitted_total="
        << refinement_jobs_submitted_
        << " | replaced_total="
        << refinement_jobs_replaced_
        << std::endl;

    return true;
}

bool RegistrationScan2LocalMap::TryCommitCompletedPostPgoRefinementResult()
{
    std::unique_ptr<PostPgoRefinementResult>
        result;

    {
        std::lock_guard<std::mutex> lock(
            refinement_worker_mutex_);

        if (!completed_refinement_result_)
        {
            refinement_result_ready_.store(false);
            return false;
        }

        result =
            std::move(
                completed_refinement_result_);

        // This completed slot has now been consumed by the backend thread.
        refinement_result_ready_.store(false);
    }

    if (!result ||
        !result->success)
    {
        std::cout
            << "BACKEND_REFINE_ASYNC_DROP_V2"
            << " | reason=COMPUTE_FAILED"
            << std::endl;

        return false;
    }

    // ------------------------------------------------------------
    // Authoritative stale-result guard.
    //
    // global_map_revision_ is NOT used here because ordinary Keyframe
    // growth changes it.  Only a newer accepted main PGO invalidates
    // the immutable refinement snapshot.
    // ------------------------------------------------------------
    if (result->pgo_epoch !=
        pose_graph_optimization_epoch_)
    {
        std::cout
            << "BACKEND_REFINE_ASYNC_DROP_V2"
            << " | reason=STALE_PGO_EPOCH"
            << " | result_epoch="
            << result->pgo_epoch
            << " | live_epoch="
            << pose_graph_optimization_epoch_
            << std::endl;

        return false;
    }

    const std::size_t source_count =
        result->source_keyframe_ids.size();

    if (source_count == 0 ||
        result->refined_poses.size() != source_count ||
        result->adjusted.size() != source_count ||
        backend_keyframes_.size() < source_count)
    {
        std::cout
            << "BACKEND_REFINE_ASYNC_DROP_V2"
            << " | reason=SIZE_MISMATCH"
            << " | source_count="
            << source_count
            << " | live_keyframes="
            << backend_keyframes_.size()
            << std::endl;

        return false;
    }

    // Backend history is append-only.  Verify that the worker snapshot is
    // still exactly the prefix of the live backend history.
    for (std::size_t i = 0;
         i < source_count;
         ++i)
    {
        if (backend_keyframes_[i].id !=
            result->source_keyframe_ids[i])
        {
            std::cout
                << "BACKEND_REFINE_ASYNC_DROP_V2"
                << " | reason=KEYFRAME_PREFIX_MISMATCH"
                << " | index=" << i
                << " | result_id="
                << result->source_keyframe_ids[i]
                << " | live_id="
                << backend_keyframes_[i].id
                << std::endl;

            return false;
        }
    }

    // ------------------------------------------------------------
    // Build a COMPLETE pose array for the CURRENT live backend.
    //
    // Snapshot KFs may receive async refined poses.
    // KFs appended while the worker was running use their current main
    // PoseGraph estimates and remain unadjusted.
    // ------------------------------------------------------------
    std::vector<
        Eigen::Isometry3d,
        Eigen::aligned_allocator<Eigen::Isometry3d>>
        commit_poses(
            backend_keyframes_.size(),
            Eigen::Isometry3d::Identity());

    std::vector<bool>
        commit_adjusted(
            backend_keyframes_.size(),
            false);

    for (std::size_t i = 0;
         i < backend_keyframes_.size();
         ++i)
    {
        const PoseGraphNode *node =
            pose_graph_.GetNode(
                backend_keyframes_[i].id);

        if (node == nullptr ||
            !node->T_WK.matrix().allFinite())
        {
            std::cout
                << "BACKEND_REFINE_ASYNC_DROP_V2"
                << " | reason=LIVE_GRAPH_POSE_INVALID"
                << " | keyframe="
                << backend_keyframes_[i].id
                << std::endl;

            return false;
        }

        commit_poses[i] =
            node->T_WK;
    }

    for (std::size_t i = 0;
         i < source_count;
         ++i)
    {
        if (!result->adjusted[i])
        {
            continue;
        }

        if (!result->refined_poses[i]
                 .matrix()
                 .allFinite())
        {
            std::cout
                << "BACKEND_REFINE_ASYNC_DROP_V2"
                << " | reason=REFINED_POSE_NONFINITE"
                << " | keyframe="
                << backend_keyframes_[i].id
                << std::endl;

            return false;
        }

        commit_poses[i] =
            result->refined_poses[i];

        commit_adjusted[i] =
            true;
    }

    IncrementalGlobalMap::UpdateStats
        refined_stats;

    const bool update_ok =
        incremental_global_map_.UpdateRefinedOverrides(
            backend_keyframes_,
            commit_poses,
            commit_adjusted,
            refined_stats);

    const pcl::PointCloud<LIDAR_POINT>::ConstPtr
        refined_map =
            incremental_global_map_.GetRefinedMap();

    if (!update_ok ||
        !refined_map ||
        refined_map->empty())
    {
        std::cout
            << "BACKEND_REFINE_ASYNC_DROP_V2"
            << " | reason=REFINED_MAP_UPDATE_FAILED"
            << std::endl;

        return false;
    }

    // The committed refined map is now based on the CURRENT optimized map
    // plus the still-valid refinement overrides, so its public map revision is
    // the current global revision, not the old source revision.
    refined_map_revision_ =
        global_map_revision_;

    if (result->historical_debug)
    {
        refinement_historical_target_debug_ =
            result->historical_debug;
    }

    if (result->before_debug)
    {
        refinement_current_before_debug_ =
            result->before_debug;
    }

    if (result->after_debug)
    {
        refinement_current_after_debug_ =
            result->after_debug;
    }

    if (result->historical_debug ||
        result->before_debug ||
        result->after_debug)
    {
        refinement_debug_revision_ =
            global_map_revision_;
    }

    std::size_t adjusted_count = 0;

    for (const bool adjusted :
         commit_adjusted)
    {
        if (adjusted)
        {
            ++adjusted_count;
        }
    }

    std::cout
        << "BACKEND_REFINE_ASYNC_COMMIT_V2"
        << " | pgo_epoch="
        << result->pgo_epoch
        << " | source_global_revision="
        << result->source_global_revision
        << " | live_global_revision="
        << global_map_revision_
        << " | source_keyframes="
        << source_count
        << " | live_keyframes="
        << backend_keyframes_.size()
        << " | adjusted="
        << adjusted_count
        << " | refined_points="
        << refined_map->size()
        << std::endl;

    return true;
}

void RegistrationScan2LocalMap::PostPgoRefinementLoop()
{
    while (true)
    {
        std::unique_ptr<PostPgoRefinementJob>
            job;

        {
            std::unique_lock<std::mutex> lock(
                refinement_worker_mutex_);

            refinement_worker_condition_.wait(
                lock,
                [this]()
                {
                    return
                        !refinement_worker_running_.load() ||
                        static_cast<bool>(
                            pending_refinement_job_);
                });

            if (!refinement_worker_running_.load() &&
                !pending_refinement_job_)
            {
                break;
            }

            if (!pending_refinement_job_)
            {
                continue;
            }

            job =
                std::move(
                    pending_refinement_job_);
        }

        if (!job)
        {
            continue;
        }

        // ------------------------------------------------------------
        // Asynchronous Post-PGO refinement compute.
        //
        // Heavy work:
        //   historical target
        //   voxel
        //   PrepareTarget
        //   Align
        //   temporary local PoseGraph
        //   local PGO
        //
        // All live map mutation remains on the main backend thread.
        // ------------------------------------------------------------
        auto result =
            std::make_unique<PostPgoRefinementResult>();

        const std::chrono::steady_clock::time_point
            async_compute_start =
                std::chrono::steady_clock::now();

        bool compute_ok = false;

        try
        {
            compute_ok =
                ComputePostPgoRefinementAsync(
                    *job,
                    *result);
        }
        catch (const std::exception &exception)
        {
            std::cerr
                << "BACKEND_REFINE_ASYNC_COMPUTE_EXCEPTION_V3"
                << " | what=" << exception.what()
                << std::endl;

            compute_ok = false;
        }
        catch (...)
        {
            std::cerr
                << "BACKEND_REFINE_ASYNC_COMPUTE_EXCEPTION_V3"
                << " | what=unknown"
                << std::endl;

            compute_ok = false;
        }

        result->success =
            compute_ok &&
            result->success;

        std::size_t async_adjusted_count = 0;

        for (const bool adjusted :
             result->adjusted)
        {
            if (adjusted)
            {
                ++async_adjusted_count;
            }
        }

        const bool async_result_success =
            result->success;

        const double async_compute_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() -
                async_compute_start)
                .count();

        {
            std::lock_guard<std::mutex> lock(
                refinement_worker_mutex_);

            // Latest completed result wins.  The backend commit performs the
            // authoritative PGO-epoch stale check.
            completed_refinement_result_ =
                std::move(result);

            refinement_result_ready_.store(true);
        }

        // Wake the MAIN backend worker.  Commit must stay single-writer on
        // backend state; the refinement worker never commits the map itself.
        backend_condition_.notify_one();

        std::cout
            << "BACKEND_REFINE_ASYNC_JOB_V3"
            << " | pgo_epoch=" << job->pgo_epoch
            << " | global_revision=" << job->source_global_revision
            << " | loop_edges=" << job->loop_edges
            << " | keyframes=" << job->keyframes.size()
            << " | success=" << (async_result_success ? 1 : 0)
            << " | adjusted=" << async_adjusted_count
            << " | compute_ms=" << async_compute_ms
            << " | status=compute_result_ready"
            << std::endl;
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

    // Commit completed async refinement only on the single backend thread.
    // This keeps incremental_global_map_, revisions and backend history
    // single-writer.
    TryCommitCompletedPostPgoRefinementResult();

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
                           !backend_queue_.empty() ||
                           refinement_result_ready_.load();
                });

            if (!backend_running_.load() &&
                backend_queue_.empty() &&
                !refinement_result_ready_.load())
            {
                break;
            }

            // A completed refinement result is committed only by this backend
            // thread.  Do it before consuming the next Keyframe job.
            if (refinement_result_ready_.load())
            {
                lock.unlock();

                TryCommitCompletedPostPgoRefinementResult();
                RefreshBackendOutputSnapshot();

                continue;
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
