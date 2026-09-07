#include "fr_slam/frontend/fr_lidar_frontend.hpp"
#include "fr_slam/frontend/fr_ground_segmenter.hpp"
#include "fr_slam/frontend/fr_ground_input_bridge.hpp"

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

    struct RefinementTimingDiagnostics
    {
        std::chrono::steady_clock::time_point start =
            std::chrono::steady_clock::now();

        std::size_t keyframes = 0;
        std::size_t loop_anchors = 0;
        std::size_t groups_considered = 0;
        std::size_t groups_prepared = 0;
        std::size_t groups_optimized = 0;
        std::size_t groups_accepted = 0;
        std::size_t geometry_candidates = 0;
        std::size_t geometry_primary_selected = 0;
        std::size_t geometry_calls = 0;
        std::size_t geometry_fallback_calls = 0;
        std::size_t geometry_anchors = 0;

        double current_select_ms = 0.0;
        double historical_select_ms = 0.0;
        double historical_build_ms = 0.0;
        double historical_voxel_ms = 0.0;
        double prepare_target_ms = 0.0;
        double local_graph_build_ms = 0.0;
        double geometry_align_ms = 0.0;
        double local_pgo_ms = 0.0;
        double debug_clouds_ms = 0.0;
        double refined_map_update_ms = 0.0;
    };

    class RefinementTimingReporter
    {
    public:
        explicit RefinementTimingReporter(
            RefinementTimingDiagnostics &diagnostics)
            : diagnostics_(diagnostics)
        {
        }

        ~RefinementTimingReporter()
        {
            const double total_ms =
                ElapsedMilliseconds(
                    diagnostics_.start,
                    std::chrono::steady_clock::now());

            RCLCPP_INFO(
                kTimingLogger,
                "FR_TIMING REFINEMENT"
                " | total=%.3f ms"
                " | keyframes=%zu"
                " | loop_anchors=%zu"
                " | groups=%zu/%zu/%zu/%zu"
                " | current_select=%.3f"
                " | historical_select=%.3f"
                " | historical_build=%.3f"
                " | historical_voxel=%.3f"
                " | prepare_target=%.3f"
                " | local_graph=%.3f"
                " | geometry_align=%.3f"
                " | geometry_candidates=%zu"
                " | geometry_primary_selected=%zu"
                " | geometry_calls=%zu"
                " | geometry_fallback_calls=%zu"
                " | geometry_anchors=%zu"
                " | local_pgo=%.3f"
                " | debug_clouds=%.3f"
                " | refined_map_update=%.3f",
                total_ms,
                diagnostics_.keyframes,
                diagnostics_.loop_anchors,
                diagnostics_.groups_considered,
                diagnostics_.groups_prepared,
                diagnostics_.groups_optimized,
                diagnostics_.groups_accepted,
                diagnostics_.current_select_ms,
                diagnostics_.historical_select_ms,
                diagnostics_.historical_build_ms,
                diagnostics_.historical_voxel_ms,
                diagnostics_.prepare_target_ms,
                diagnostics_.local_graph_build_ms,
                diagnostics_.geometry_align_ms,
                diagnostics_.geometry_candidates,
                diagnostics_.geometry_primary_selected,
                diagnostics_.geometry_calls,
                diagnostics_.geometry_fallback_calls,
                diagnostics_.geometry_anchors,
                diagnostics_.local_pgo_ms,
                diagnostics_.debug_clouds_ms,
                diagnostics_.refined_map_update_ms);
        }

    private:
        RefinementTimingDiagnostics &diagnostics_;
    };

    double RelativeRotationDeg(
        const Eigen::Isometry3d &T_A,
        const Eigen::Isometry3d &T_B)
    {
        if (!T_A.matrix().allFinite() ||
            !T_B.matrix().allFinite())
        {
            return std::numeric_limits<double>::quiet_NaN();
        }

        const Eigen::Matrix3d R_AB =
            T_A.rotation().transpose() *
            T_B.rotation();

        Eigen::Quaterniond q_AB(R_AB);

        if (!q_AB.coeffs().allFinite() ||
            q_AB.norm() < 1.0e-12)
        {
            return std::numeric_limits<double>::quiet_NaN();
        }

        q_AB.normalize();

        // q and -q represent the same rotation.
        const double w =
            std::clamp(
                std::abs(q_AB.w()),
                0.0,
                1.0);

        return 2.0 *
               std::acos(w) *
               180.0 /
               M_PI;
    }
} // namespace

bool RegistrationScan2LocalMap::RebuildPostPgoRefinedMap()
{
    RefinementTimingDiagnostics refinement_timing;
    RefinementTimingReporter refinement_timing_reporter(
        refinement_timing);

    const std::vector<Keyframe> &keyframes =
        backend_keyframes_;

    refinement_timing.keyframes =
        keyframes.size();

    if (keyframes.empty() ||
        pose_graph_.NodeCount() == 0 ||
        global_map_revision_ == 0)
    {
        return false;
    }

    refined_map_revision_ = 0;

    // Publish-side snapshots may still reference the previous debug clouds.
    // Replace the handles instead of clearing those shared buffers in place.
    refinement_historical_target_debug_ =
        pcl::make_shared<pcl::PointCloud<LIDAR_POINT>>();

    refinement_current_before_debug_ =
        pcl::make_shared<pcl::PointCloud<LIDAR_POINT>>();

    refinement_current_after_debug_ =
        pcl::make_shared<pcl::PointCloud<LIDAR_POINT>>();

    refinement_debug_revision_ = 0;

    refined_keyframe_poses_.clear();
    refined_keyframe_pose_was_adjusted_.clear();

    refined_keyframe_poses_.resize(
        keyframes.size(),
        Eigen::Isometry3d::Identity());

    refined_keyframe_pose_was_adjusted_.resize(
        keyframes.size(),
        false);

    // Immutable G2O snapshot.  V3 never feeds its own refined poses back into
    // neighbor selection, ICP targets, the live frontend, or the main graph.
    std::vector<
        Eigen::Isometry3d,
        Eigen::aligned_allocator<Eigen::Isometry3d>>
        frozen_graph_poses(
            keyframes.size(),
            Eigen::Isometry3d::Identity());

    std::vector<bool> graph_pose_valid(
        keyframes.size(),
        false);

    for (std::size_t i = 0;
         i < keyframes.size();
         ++i)
    {
        const PoseGraphNode *node =
            pose_graph_.GetNode(
                keyframes[i].id);

        if (node == nullptr ||
            !node->T_WK.matrix().allFinite())
        {
            continue;
        }

        frozen_graph_poses[i] =
            node->T_WK;

        refined_keyframe_poses_[i] =
            node->T_WK;

        graph_pose_valid[i] =
            true;
    }

    struct RefinementLoopAnchor
    {
        std::size_t historical_id = 0;
        std::size_t current_id = 0;
    };

    std::vector<RefinementLoopAnchor>
        loop_anchors;

    for (const PoseGraphEdge &edge :
         pose_graph_.GetEdges())
    {
        if (edge.type != PoseGraphEdgeType::Loop)
        {
            continue;
        }

        RefinementLoopAnchor anchor;

        anchor.historical_id =
            std::min(
                edge.from_id,
                edge.to_id);

        anchor.current_id =
            std::max(
                edge.from_id,
                edge.to_id);

        if (anchor.current_id <=
            anchor.historical_id)
        {
            continue;
        }

        if (anchor.current_id -
                anchor.historical_id <
            refinement_min_historical_keyframe_gap_)
        {
            continue;
        }

        loop_anchors.push_back(anchor);
    }

    if (loop_anchors.empty())
    {
        return false;
    }

    std::sort(
        loop_anchors.begin(),
        loop_anchors.end(),
        [](const RefinementLoopAnchor &lhs,
           const RefinementLoopAnchor &rhs)
        {
            return lhs.current_id <
                   rhs.current_id;
        });

    refinement_timing.loop_anchors =
        loop_anchors.size();

    std::size_t groups_considered = 0;
    std::size_t groups_prepared = 0;
    std::size_t groups_optimized = 0;
    std::size_t groups_accepted = 0;
    std::size_t groups_rejected_overlap = 0;
    std::size_t groups_rejected_quality = 0;
    std::size_t groups_rejected_large_update = 0;
    std::size_t geometry_candidates_total = 0;
    std::size_t geometry_primary_selected_total = 0;
    std::size_t geometry_attempts_total = 0;
    std::size_t geometry_fallback_calls_total = 0;
    std::size_t geometry_anchors_total = 0;
    std::size_t adjusted_keyframes = 0;

    // V4 keeps all local graph states and odometry edges, but limits the
    // expensive point-to-plane observations.  A 16-KF window therefore uses
    // at most 8 primary geometry registrations.  If too few primary anchors
    // pass the existing quality gates, up to 4 additional untested Keyframes
    // are tried before rejecting the refinement group.
    constexpr std::size_t kSparsePrimaryGeometryAnchors = 8;
    constexpr std::size_t kSparseMaxFallbackGeometryCalls = 4;

    std::cout
        << "Post-PGO local-window refinement V5 SPARSE+GATE_DIAG started"
        << " | global_revision=" << global_map_revision_
        << " | keyframes=" << keyframes.size()
        << " | loop_anchors=" << loop_anchors.size()
        << " | local_window=" << refinement_local_window_
        << " | historical_window=" << refinement_historical_keyframe_window_
        << " | historical_radius=" << refinement_historical_radius_ << " m"
        << " | min_geometry_anchors=" << refinement_min_geometry_anchors_
        << " | sparse_primary_anchors=" << kSparsePrimaryGeometryAnchors
        << " | sparse_fallback_max=" << kSparseMaxFallbackGeometryCalls
        << " | odom_info_scale=" << refinement_local_odom_information_scale_
        << " | geometry_info_scale=" << refinement_geometry_information_scale_
        << " | max_window_dt=" << refinement_window_max_translation_update_ << " m"
        << " | max_window_dR=" << refinement_window_max_rotation_update_deg_ << " deg"
        << std::endl;

    for (const RefinementLoopAnchor &anchor :
         loop_anchors)
    {
        std::size_t historical_anchor_index =
            std::numeric_limits<std::size_t>::max();

        std::size_t current_anchor_index =
            std::numeric_limits<std::size_t>::max();

        for (std::size_t i = 0;
             i < keyframes.size();
             ++i)
        {
            if (keyframes[i].id ==
                anchor.historical_id)
            {
                historical_anchor_index = i;
            }

            if (keyframes[i].id ==
                anchor.current_id)
            {
                current_anchor_index = i;
            }
        }

        if (historical_anchor_index ==
                std::numeric_limits<std::size_t>::max() ||
            current_anchor_index ==
                std::numeric_limits<std::size_t>::max() ||
            !graph_pose_valid[historical_anchor_index] ||
            !graph_pose_valid[current_anchor_index])
        {
            continue;
        }

        ++groups_considered;
        refinement_timing.groups_considered =
            groups_considered;

        // --------------------------------------------------------------------
        // Current revisit window: one independent SE(3) state per Keyframe.
        // The window ends at the current loop endpoint because those poses are
        // already available when the online loop is accepted.
        // --------------------------------------------------------------------
        const std::chrono::steady_clock::time_point
            current_select_start =
                std::chrono::steady_clock::now();

        std::vector<std::size_t>
            current_indices;

        current_indices.reserve(
            refinement_local_window_ + 1);

        for (std::size_t i = 0;
             i < keyframes.size();
             ++i)
        {
            if (!graph_pose_valid[i] ||
                !keyframes[i].cloud ||
                keyframes[i].cloud->empty())
            {
                continue;
            }

            if (keyframes[i].id >
                anchor.current_id)
            {
                continue;
            }

            const std::size_t current_gap =
                anchor.current_id -
                keyframes[i].id;

            if (current_gap >
                refinement_local_window_)
            {
                continue;
            }

            current_indices.push_back(i);
        }

        refinement_timing.current_select_ms +=
            ElapsedMilliseconds(
                current_select_start,
                std::chrono::steady_clock::now());

        if (current_indices.size() <
            refinement_min_current_keyframes_)
        {
            ++groups_rejected_quality;
            continue;
        }

        // Keep V3 conservative when multiple accepted loop windows overlap.
        // A future global local-BA pass can merge those windows explicitly.
        bool overlaps_previous_refinement = false;

        for (const std::size_t source_index :
             current_indices)
        {
            if (refined_keyframe_pose_was_adjusted_[source_index])
            {
                overlaps_previous_refinement = true;
                break;
            }
        }

        if (overlaps_previous_refinement)
        {
            ++groups_rejected_overlap;

            std::cout
                << "Post-PGO local-window refinement skipped"
                << " | historical_kf=" << anchor.historical_id
                << " | current_kf=" << anchor.current_id
                << " | reason=OVERLAPPING_REFINEMENT_WINDOW"
                << std::endl;

            continue;
        }

        // --------------------------------------------------------------------
        // Build one frozen HISTORICAL LocalMap tied to the historical endpoint
        // of the accepted loop edge.  Every current Keyframe uses exactly this
        // same target, so the geometry constraints are mutually comparable.
        // --------------------------------------------------------------------
        const std::chrono::steady_clock::time_point
            historical_select_start =
                std::chrono::steady_clock::now();

        std::vector<std::pair<double, std::size_t>>
            historical_candidates;

        historical_candidates.reserve(
            refinement_max_historical_keyframes_ * 2);

        const Eigen::Vector3d historical_anchor_position =
            frozen_graph_poses[historical_anchor_index]
                .translation();

        for (std::size_t i = 0;
             i < keyframes.size();
             ++i)
        {
            if (!graph_pose_valid[i] ||
                !keyframes[i].cloud ||
                keyframes[i].cloud->empty())
            {
                continue;
            }

            if (keyframes[i].id >=
                anchor.current_id)
            {
                continue;
            }

            const std::size_t current_to_historical_gap =
                anchor.current_id -
                keyframes[i].id;

            if (current_to_historical_gap <
                refinement_min_historical_keyframe_gap_)
            {
                continue;
            }

            const std::size_t historical_id_gap =
                keyframes[i].id >= anchor.historical_id
                    ? keyframes[i].id - anchor.historical_id
                    : anchor.historical_id - keyframes[i].id;

            if (historical_id_gap >
                refinement_historical_keyframe_window_)
            {
                continue;
            }

            const double distance =
                (frozen_graph_poses[i].translation() -
                 historical_anchor_position)
                    .norm();

            if (!std::isfinite(distance) ||
                distance >
                    refinement_historical_radius_)
            {
                continue;
            }

            historical_candidates.emplace_back(
                distance,
                i);
        }

        if (historical_candidates.size() <
            refinement_min_historical_keyframes_)
        {
            ++groups_rejected_quality;
            continue;
        }

        std::sort(
            historical_candidates.begin(),
            historical_candidates.end(),
            [](const auto &lhs,
               const auto &rhs)
            {
                return lhs.first < rhs.first;
            });

        if (historical_candidates.size() >
            refinement_max_historical_keyframes_)
        {
            historical_candidates.resize(
                refinement_max_historical_keyframes_);
        }

        refinement_timing.historical_select_ms +=
            ElapsedMilliseconds(
                historical_select_start,
                std::chrono::steady_clock::now());

        const std::chrono::steady_clock::time_point
            historical_build_start =
                std::chrono::steady_clock::now();

        pcl::PointCloud<LIDAR_POINT>::Ptr
            historical_target_accumulated =
                pcl::make_shared<
                    pcl::PointCloud<LIDAR_POINT>>();

        std::size_t estimated_historical_points = 0;

        for (const auto &candidate :
             historical_candidates)
        {
            estimated_historical_points +=
                keyframes[candidate.second].cloud->size();
        }

        historical_target_accumulated->reserve(
            estimated_historical_points);

        for (const auto &candidate :
             historical_candidates)
        {
            const std::size_t historical_index =
                candidate.second;

            pcl::PointCloud<LIDAR_POINT>
                historical_world_cloud;

            const Eigen::Matrix4f T_WK_float =
                frozen_graph_poses[historical_index]
                    .matrix()
                    .cast<float>();

            pcl::transformPointCloud(
                *keyframes[historical_index].cloud,
                historical_world_cloud,
                T_WK_float);

            *historical_target_accumulated +=
                historical_world_cloud;
        }

        refinement_timing.historical_build_ms +=
            ElapsedMilliseconds(
                historical_build_start,
                std::chrono::steady_clock::now());

        if (historical_target_accumulated->empty())
        {
            ++groups_rejected_quality;
            continue;
        }

        pcl::PointCloud<LIDAR_POINT>::Ptr
            historical_target_filtered =
                pcl::make_shared<
                    pcl::PointCloud<LIDAR_POINT>>();

        pcl::VoxelGrid<LIDAR_POINT>
            target_voxel;

        target_voxel.setLeafSize(
            refinement_target_voxel_leaf_size_,
            refinement_target_voxel_leaf_size_,
            refinement_target_voxel_leaf_size_);

        target_voxel.setInputCloud(
            historical_target_accumulated);

        const std::chrono::steady_clock::time_point
            historical_voxel_start =
                std::chrono::steady_clock::now();

        target_voxel.filter(
            *historical_target_filtered);

        refinement_timing.historical_voxel_ms +=
            ElapsedMilliseconds(
                historical_voxel_start,
                std::chrono::steady_clock::now());

        if (historical_target_filtered->size() <
            refinement_min_target_points_)
        {
            ++groups_rejected_quality;
            continue;
        }

        PreparedLidarTarget
            prepared_refinement_target;

        const std::chrono::steady_clock::time_point
            prepare_target_start =
                std::chrono::steady_clock::now();

        const bool prepare_target_ok =
            backend_refinement_registration_.PrepareTarget(
                historical_target_filtered,
                prepared_refinement_target);

        refinement_timing.prepare_target_ms +=
            ElapsedMilliseconds(
                prepare_target_start,
                std::chrono::steady_clock::now());

        if (!prepare_target_ok)
        {
            ++groups_rejected_quality;
            continue;
        }

        ++groups_prepared;
        refinement_timing.groups_prepared =
            groups_prepared;

        const std::chrono::steady_clock::time_point
            local_graph_start =
                std::chrono::steady_clock::now();

        // --------------------------------------------------------------------
        // Temporary LOCAL PoseGraph.
        //
        // local node 0:
        //     fixed map-frame anchor at Identity.
        //
        // local nodes 1..N:
        //     current-window Keyframe poses initialized from frozen G2O.
        //
        // Odometry edges preserve the current trajectory shape.
        // Geometry edges are soft absolute-pose observations produced by
        // Keyframe-to-HistoricalLocalMap point-to-plane registration.
        // --------------------------------------------------------------------
        PoseGraph local_pose_graph;

        if (!local_pose_graph.AddNode(
                0,
                Eigen::Isometry3d::Identity(),
                true))
        {
            ++groups_rejected_quality;
            continue;
        }

        std::vector<std::size_t>
            local_node_ids(
                current_indices.size(),
                0);

        bool local_graph_ok = true;

        for (std::size_t k = 0;
             k < current_indices.size();
             ++k)
        {
            const std::size_t local_node_id =
                k + 1;

            local_node_ids[k] =
                local_node_id;

            const std::size_t source_index =
                current_indices[k];

            if (!local_pose_graph.AddNode(
                    local_node_id,
                    frozen_graph_poses[source_index],
                    false))
            {
                local_graph_ok = false;
                break;
            }
        }

        if (!local_graph_ok)
        {
            ++groups_rejected_quality;
            continue;
        }

        Eigen::Matrix<double, 6, 6>
            odom_information =
                Eigen::Matrix<double, 6, 6>::Identity() *
                refinement_local_odom_information_scale_;

        for (std::size_t k = 1;
             k < current_indices.size();
             ++k)
        {
            const std::size_t previous_index =
                current_indices[k - 1];

            const std::size_t current_index =
                current_indices[k];

            const Eigen::Isometry3d Z_previous_current =
                frozen_graph_poses[previous_index].inverse() *
                frozen_graph_poses[current_index];

            if (!local_pose_graph.AddOdometryEdge(
                    local_node_ids[k - 1],
                    local_node_ids[k],
                    Z_previous_current,
                    odom_information))
            {
                local_graph_ok = false;
                break;
            }
        }

        if (!local_graph_ok)
        {
            ++groups_rejected_quality;
            continue;
        }

        refinement_timing.local_graph_build_ms +=
            ElapsedMilliseconds(
                local_graph_start,
                std::chrono::steady_clock::now());

        std::vector<bool>
            geometry_anchor_valid(
                current_indices.size(),
                false);

        std::vector<bool>
            geometry_was_tested(
                current_indices.size(),
                false);

        std::vector<std::size_t>
            geometry_correspondences(
                current_indices.size(),
                0);

        std::vector<double>
            geometry_rmse(
                current_indices.size(),
                std::numeric_limits<double>::infinity());

        std::vector<double>
            geometry_delta_t(
                current_indices.size(),
                std::numeric_limits<double>::infinity());

        std::vector<double>
            geometry_delta_R_deg(
                current_indices.size(),
                std::numeric_limits<double>::infinity());

        std::size_t geometry_anchor_count = 0;
        std::size_t geometry_primary_selected = 0;
        std::size_t geometry_fallback_calls = 0;

        // V5 diagnostic counters. These counters are observational only;
        // they do not change any geometry acceptance threshold or graph math.
        std::size_t geometry_attempts_group = 0;
        std::size_t geometry_reject_align_failed = 0;
        std::size_t geometry_reject_result_failed = 0;
        std::size_t geometry_reject_not_converged = 0;
        std::size_t geometry_reject_transform_nonfinite = 0;
        std::size_t geometry_reject_rmse_nonfinite = 0;
        std::size_t geometry_reject_corr_low = 0;
        std::size_t geometry_reject_rmse_high = 0;
        std::size_t geometry_reject_correction_nonfinite = 0;
        std::size_t geometry_reject_delta_t_high = 0;
        std::size_t geometry_reject_delta_R_high = 0;
        std::size_t geometry_reject_add_edge_failed = 0;

        geometry_candidates_total +=
            current_indices.size();

        refinement_timing.geometry_candidates +=
            current_indices.size();

        const Eigen::Matrix<double, 6, 6>
            geometry_information =
                Eigen::Matrix<double, 6, 6>::Identity() *
                refinement_geometry_information_scale_;

        // --------------------------------------------------------------------
        // V4 sparse geometry selection.
        //
        // Keep the first and last Keyframes and distribute the remaining
        // anchors approximately uniformly over the local window.  For the
        // current default 16-KF window this selects:
        //
        //     0, 2, 4, 6, 9, 11, 13, 15
        //
        // (exact integer rounding is intentional).  All 16 graph nodes and
        // all 15 odometry edges remain in the optimization.
        // --------------------------------------------------------------------
        const std::size_t primary_target_count =
            std::min(
                kSparsePrimaryGeometryAnchors,
                current_indices.size());

        std::vector<std::size_t>
            primary_geometry_indices;

        primary_geometry_indices.reserve(
            primary_target_count);

        if (primary_target_count == 1)
        {
            primary_geometry_indices.push_back(0);
        }
        else if (primary_target_count > 1)
        {
            const std::size_t last_index =
                current_indices.size() - 1;

            const std::size_t denominator =
                primary_target_count - 1;

            for (std::size_t slot = 0;
                 slot < primary_target_count;
                 ++slot)
            {
                const std::size_t numerator =
                    slot * last_index;

                // Rounded integer interpolation in [0, last_index].
                const std::size_t k =
                    (numerator + denominator / 2) /
                    denominator;

                if (primary_geometry_indices.empty() ||
                    primary_geometry_indices.back() != k)
                {
                    primary_geometry_indices.push_back(k);
                }
            }

            // Defensive guarantee: the current loop endpoint is always tested.
            if (primary_geometry_indices.empty() ||
                primary_geometry_indices.back() != last_index)
            {
                primary_geometry_indices.push_back(
                    last_index);
            }
        }

        geometry_primary_selected =
            primary_geometry_indices.size();

        geometry_primary_selected_total +=
            geometry_primary_selected;

        refinement_timing.geometry_primary_selected +=
            geometry_primary_selected;

        auto log_geometry_attempt =
            [&](const char *outcome,
                const char *reason,
                const std::size_t k,
                const std::size_t source_keyframe_id,
                const bool is_fallback,
                const std::size_t correspondences,
                const double rmse,
                const double correction_translation,
                const double correction_rotation_deg)
        {
            RCLCPP_INFO(
                kTimingLogger,
                "FR_REFINEMENT_GEOMETRY_ATTEMPT"
                " | historical_kf=%zu"
                " | current_kf=%zu"
                " | source_kf=%zu"
                " | local_index=%zu"
                " | sample=%s"
                " | outcome=%s"
                " | reason=%s"
                " | corr=%zu"
                " | rmse=%.6f"
                " | delta_t=%.6f"
                " | delta_R_deg=%.6f",
                anchor.historical_id,
                anchor.current_id,
                source_keyframe_id,
                k,
                is_fallback ? "fallback" : "primary",
                outcome,
                reason,
                correspondences,
                rmse,
                correction_translation,
                correction_rotation_deg);
        };

        auto try_geometry_anchor =
            [&](const std::size_t k,
                const bool is_fallback) -> bool
        {
            if (k >= current_indices.size() ||
                geometry_was_tested[k])
            {
                return true;
            }

            geometry_was_tested[k] = true;

            const std::size_t source_index =
                current_indices[k];

            const Keyframe &source_keyframe =
                keyframes[source_index];

            ++geometry_attempts_group;
            ++geometry_attempts_total;
            ++refinement_timing.geometry_calls;

            if (is_fallback)
            {
                ++geometry_fallback_calls;
                ++geometry_fallback_calls_total;
                ++refinement_timing.geometry_fallback_calls;
            }

            LidarRegistrationResult geometry_result;

            const std::chrono::steady_clock::time_point
                geometry_align_start =
                    std::chrono::steady_clock::now();

            const bool align_success =
                backend_refinement_registration_.Align(
                    source_keyframe.cloud,
                    prepared_refinement_target,
                    frozen_graph_poses[source_index],
                    geometry_result);

            refinement_timing.geometry_align_ms +=
                ElapsedMilliseconds(
                    geometry_align_start,
                    std::chrono::steady_clock::now());

            const double nan_value =
                std::numeric_limits<double>::quiet_NaN();

            if (!align_success)
            {
                ++geometry_reject_align_failed;
                log_geometry_attempt(
                    "REJECT",
                    "ALIGN_RETURN_FALSE",
                    k,
                    source_keyframe.id,
                    is_fallback,
                    0,
                    nan_value,
                    nan_value,
                    nan_value);
                return true;
            }

            if (!geometry_result.success)
            {
                ++geometry_reject_result_failed;
                log_geometry_attempt(
                    "REJECT",
                    "RESULT_SUCCESS_FALSE",
                    k,
                    source_keyframe.id,
                    is_fallback,
                    geometry_result.correspondences,
                    geometry_result.rmse,
                    nan_value,
                    nan_value);
                return true;
            }

            if (!geometry_result.converged)
            {
                ++geometry_reject_not_converged;
                log_geometry_attempt(
                    "REJECT",
                    "NOT_CONVERGED",
                    k,
                    source_keyframe.id,
                    is_fallback,
                    geometry_result.correspondences,
                    geometry_result.rmse,
                    nan_value,
                    nan_value);
                return true;
            }

            if (!geometry_result.T_target_source.matrix().allFinite())
            {
                ++geometry_reject_transform_nonfinite;
                log_geometry_attempt(
                    "REJECT",
                    "TRANSFORM_NONFINITE",
                    k,
                    source_keyframe.id,
                    is_fallback,
                    geometry_result.correspondences,
                    geometry_result.rmse,
                    nan_value,
                    nan_value);
                return true;
            }

            if (!std::isfinite(geometry_result.rmse))
            {
                ++geometry_reject_rmse_nonfinite;
                log_geometry_attempt(
                    "REJECT",
                    "RMSE_NONFINITE",
                    k,
                    source_keyframe.id,
                    is_fallback,
                    geometry_result.correspondences,
                    geometry_result.rmse,
                    nan_value,
                    nan_value);
                return true;
            }

            geometry_correspondences[k] =
                geometry_result.correspondences;

            geometry_rmse[k] =
                geometry_result.rmse;

            const Eigen::Isometry3d T_initial_geometry =
                frozen_graph_poses[source_index].inverse() *
                geometry_result.T_target_source;

            const double correction_translation =
                T_initial_geometry.translation().norm();

            const double correction_rotation_deg =
                RelativeRotationDeg(
                    frozen_graph_poses[source_index],
                    geometry_result.T_target_source);

            geometry_delta_t[k] =
                correction_translation;

            geometry_delta_R_deg[k] =
                correction_rotation_deg;

            // ------------------------------------------------------------
            // Refinement pose-delta diagnostics.
            //
            // T_local_delta answers:
            //     how much does this Keyframe need to move relative to its
            //     current post-PGO pose, expressed in the Keyframe-local
            //     perturbation convention used by the existing gate?
            //
            // T_world_delta answers:
            //     is there a coherent rigid correction in the map/world
            //     frame shared by several current-window Keyframes?
            //
            // If many good ICP results show nearly the same world-frame
            // yaw / translation, the current absolute 0.40 m / 3 deg gate is
            // rejecting a coherent correction rather than random outliers.
            // ------------------------------------------------------------
            const Eigen::Isometry3d T_local_delta =
                T_initial_geometry;

            const Eigen::Isometry3d T_world_delta =
                geometry_result.T_target_source *
                frozen_graph_poses[source_index].inverse();

            const Eigen::Vector3d local_rpy_zyx =
                T_local_delta.rotation().eulerAngles(2, 1, 0);

            const Eigen::Vector3d world_rpy_zyx =
                T_world_delta.rotation().eulerAngles(2, 1, 0);

            constexpr double kRadToDeg =
                180.0 / M_PI;

            const double local_yaw_deg =
                local_rpy_zyx[0] * kRadToDeg;
            const double local_pitch_deg =
                local_rpy_zyx[1] * kRadToDeg;
            const double local_roll_deg =
                local_rpy_zyx[2] * kRadToDeg;

            const double world_yaw_deg =
                world_rpy_zyx[0] * kRadToDeg;
            const double world_pitch_deg =
                world_rpy_zyx[1] * kRadToDeg;
            const double world_roll_deg =
                world_rpy_zyx[2] * kRadToDeg;

            RCLCPP_INFO(
                kTimingLogger,
                "FR_REFINEMENT_DELTA_COMPONENTS"
                " | historical_kf=%zu"
                " | current_kf=%zu"
                " | source_kf=%zu"
                " | local_index=%zu"
                " | sample=%s"
                " | corr=%zu"
                " | rmse=%.6f"
                " | local_dt=[%.6f %.6f %.6f]"
                " | local_rpy_deg=[%.6f %.6f %.6f]"
                " | world_dt=[%.6f %.6f %.6f]"
                " | world_rpy_deg=[%.6f %.6f %.6f]"
                " | delta_t_norm=%.6f"
                " | delta_R_deg=%.6f",
                anchor.historical_id,
                anchor.current_id,
                source_keyframe.id,
                k,
                is_fallback ? "fallback" : "primary",
                geometry_result.correspondences,
                geometry_result.rmse,
                T_local_delta.translation().x(),
                T_local_delta.translation().y(),
                T_local_delta.translation().z(),
                local_roll_deg,
                local_pitch_deg,
                local_yaw_deg,
                T_world_delta.translation().x(),
                T_world_delta.translation().y(),
                T_world_delta.translation().z(),
                world_roll_deg,
                world_pitch_deg,
                world_yaw_deg,
                correction_translation,
                correction_rotation_deg);

            if (geometry_result.correspondences <
                refinement_geometry_min_correspondences_)
            {
                ++geometry_reject_corr_low;
                log_geometry_attempt(
                    "REJECT",
                    "CORRESPONDENCES_LOW",
                    k,
                    source_keyframe.id,
                    is_fallback,
                    geometry_result.correspondences,
                    geometry_result.rmse,
                    correction_translation,
                    correction_rotation_deg);
                return true;
            }

            if (geometry_result.rmse >
                refinement_geometry_max_rmse_)
            {
                ++geometry_reject_rmse_high;
                log_geometry_attempt(
                    "REJECT",
                    "RMSE_HIGH",
                    k,
                    source_keyframe.id,
                    is_fallback,
                    geometry_result.correspondences,
                    geometry_result.rmse,
                    correction_translation,
                    correction_rotation_deg);
                return true;
            }

            if (!std::isfinite(correction_translation) ||
                !std::isfinite(correction_rotation_deg))
            {
                ++geometry_reject_correction_nonfinite;
                log_geometry_attempt(
                    "REJECT",
                    "CORRECTION_NONFINITE",
                    k,
                    source_keyframe.id,
                    is_fallback,
                    geometry_result.correspondences,
                    geometry_result.rmse,
                    correction_translation,
                    correction_rotation_deg);
                return true;
            }

            if (correction_translation >
                refinement_geometry_max_translation_correction_)
            {
                ++geometry_reject_delta_t_high;
                log_geometry_attempt(
                    "REJECT",
                    "DELTA_T_HIGH",
                    k,
                    source_keyframe.id,
                    is_fallback,
                    geometry_result.correspondences,
                    geometry_result.rmse,
                    correction_translation,
                    correction_rotation_deg);
                return true;
            }

            if (correction_rotation_deg >
                refinement_geometry_max_rotation_correction_deg_)
            {
                ++geometry_reject_delta_R_high;
                log_geometry_attempt(
                    "REJECT",
                    "DELTA_R_HIGH",
                    k,
                    source_keyframe.id,
                    is_fallback,
                    geometry_result.correspondences,
                    geometry_result.rmse,
                    correction_translation,
                    correction_rotation_deg);
                return true;
            }

            // Fixed node 0 is Identity in map coordinates. Therefore the
            // measurement 0 -> k is simply the absolute map pose returned by
            // the point-to-plane registration.
            if (!local_pose_graph.AddLoopEdge(
                    0,
                    local_node_ids[k],
                    geometry_result.T_target_source,
                    geometry_information))
            {
                ++geometry_reject_add_edge_failed;
                log_geometry_attempt(
                    "REJECT",
                    "ADD_LOOP_EDGE_FAILED",
                    k,
                    source_keyframe.id,
                    is_fallback,
                    geometry_result.correspondences,
                    geometry_result.rmse,
                    correction_translation,
                    correction_rotation_deg);
                return false;
            }

            geometry_anchor_valid[k] = true;

            ++geometry_anchor_count;
            ++geometry_anchors_total;

            refinement_timing.geometry_anchors =
                geometry_anchors_total;

            log_geometry_attempt(
                "ACCEPT",
                "OK",
                k,
                source_keyframe.id,
                is_fallback,
                geometry_result.correspondences,
                geometry_result.rmse,
                correction_translation,
                correction_rotation_deg);

            return true;
        };

        for (const std::size_t k :
             primary_geometry_indices)
        {
            if (!try_geometry_anchor(
                    k,
                    false))
            {
                local_graph_ok = false;
                break;
            }
        }

        // If the sparse primary set happens to fall on weak geometry, test a
        // few additional Keyframes.  The fallback is intentionally bounded so
        // V4 cannot silently return to V3's all-frame Full Align cost.
        if (local_graph_ok &&
            geometry_anchor_count <
                refinement_min_geometry_anchors_)
        {
            for (std::size_t k = 0;
                 k < current_indices.size() &&
                 geometry_anchor_count <
                     refinement_min_geometry_anchors_ &&
                 geometry_fallback_calls <
                     kSparseMaxFallbackGeometryCalls;
                 ++k)
            {
                if (geometry_was_tested[k])
                {
                    continue;
                }

                if (!try_geometry_anchor(
                        k,
                        true))
                {
                    local_graph_ok = false;
                    break;
                }
            }
        }

        RCLCPP_INFO(
            kTimingLogger,
            "FR_REFINEMENT_GEOMETRY_SUMMARY"
            " | historical_kf=%zu"
            " | current_kf=%zu"
            " | candidates=%zu"
            " | primary_selected=%zu"
            " | fallback_calls=%zu"
            " | attempts=%zu"
            " | accepted=%zu"
            " | reject_align_failed=%zu"
            " | reject_result_failed=%zu"
            " | reject_not_converged=%zu"
            " | reject_transform_nonfinite=%zu"
            " | reject_rmse_nonfinite=%zu"
            " | reject_corr_low=%zu"
            " | reject_rmse_high=%zu"
            " | reject_correction_nonfinite=%zu"
            " | reject_delta_t_high=%zu"
            " | reject_delta_R_high=%zu"
            " | reject_add_edge_failed=%zu"
            " | min_corr=%zu"
            " | max_rmse=%.6f"
            " | max_delta_t=%.6f"
            " | max_delta_R_deg=%.6f",
            anchor.historical_id,
            anchor.current_id,
            current_indices.size(),
            geometry_primary_selected,
            geometry_fallback_calls,
            geometry_attempts_group,
            geometry_anchor_count,
            geometry_reject_align_failed,
            geometry_reject_result_failed,
            geometry_reject_not_converged,
            geometry_reject_transform_nonfinite,
            geometry_reject_rmse_nonfinite,
            geometry_reject_corr_low,
            geometry_reject_rmse_high,
            geometry_reject_correction_nonfinite,
            geometry_reject_delta_t_high,
            geometry_reject_delta_R_high,
            geometry_reject_add_edge_failed,
            refinement_geometry_min_correspondences_,
            refinement_geometry_max_rmse_,
            refinement_geometry_max_translation_correction_,
            refinement_geometry_max_rotation_correction_deg_);

        std::cout
            << "Post-PGO sparse geometry sampling"
            << " | historical_kf=" << anchor.historical_id
            << " | current_kf=" << anchor.current_id
            << " | candidates=" << current_indices.size()
            << " | primary_selected=" << geometry_primary_selected
            << " | fallback_calls=" << geometry_fallback_calls
            << " | total_calls="
            << std::count(
                   geometry_was_tested.begin(),
                   geometry_was_tested.end(),
                   true)
            << " | accepted_anchors=" << geometry_anchor_count
            << std::endl;

        if (!local_graph_ok ||
            geometry_anchor_count <
                refinement_min_geometry_anchors_)
        {
            ++groups_rejected_quality;

            std::cout
                << "Post-PGO local-window refinement rejected"
                << " | historical_kf=" << anchor.historical_id
                << " | current_kf=" << anchor.current_id
                << " | current_kfs=" << current_indices.size()
                << " | geometry_anchors=" << geometry_anchor_count
                << " | required=" << refinement_min_geometry_anchors_
                << " | reason=INSUFFICIENT_GEOMETRY_ANCHORS"
                << std::endl;

            continue;
        }

        PoseGraphOptimizationResult
            local_optimization_result;

        const std::chrono::steady_clock::time_point
            local_pgo_start =
                std::chrono::steady_clock::now();

        const bool local_pgo_ok =
            pose_graph_optimizer_.Optimize(
                local_pose_graph,
                local_optimization_result);

        refinement_timing.local_pgo_ms +=
            ElapsedMilliseconds(
                local_pgo_start,
                std::chrono::steady_clock::now());

        if (!local_pgo_ok ||
            !local_optimization_result.success)
        {
            ++groups_rejected_quality;
            continue;
        }

        ++groups_optimized;
        refinement_timing.groups_optimized =
            groups_optimized;

        std::vector<
            Eigen::Isometry3d,
            Eigen::aligned_allocator<Eigen::Isometry3d>>
            optimized_local_poses(
                current_indices.size(),
                Eigen::Isometry3d::Identity());

        double max_pose_delta_t = 0.0;
        double max_pose_delta_R_deg = 0.0;
        double sum_pose_delta_t = 0.0;
        double sum_pose_delta_R_deg = 0.0;

        bool optimized_window_valid = true;

        for (std::size_t k = 0;
             k < current_indices.size();
             ++k)
        {
            const PoseGraphNode *local_node =
                local_pose_graph.GetNode(
                    local_node_ids[k]);

            if (local_node == nullptr ||
                !local_node->T_WK.matrix().allFinite())
            {
                optimized_window_valid = false;
                break;
            }

            const std::size_t source_index =
                current_indices[k];

            optimized_local_poses[k] =
                local_node->T_WK;

            const Eigen::Isometry3d T_initial_optimized =
                frozen_graph_poses[source_index].inverse() *
                optimized_local_poses[k];

            const double delta_t =
                T_initial_optimized.translation().norm();

            const double delta_R_deg =
                RelativeRotationDeg(
                    frozen_graph_poses[source_index],
                    optimized_local_poses[k]);

            if (!std::isfinite(delta_t) ||
                !std::isfinite(delta_R_deg))
            {
                optimized_window_valid = false;
                break;
            }

            max_pose_delta_t =
                std::max(
                    max_pose_delta_t,
                    delta_t);

            max_pose_delta_R_deg =
                std::max(
                    max_pose_delta_R_deg,
                    delta_R_deg);

            sum_pose_delta_t +=
                delta_t;

            sum_pose_delta_R_deg +=
                delta_R_deg;
        }

        double max_relative_odom_translation_change = 0.0;
        double max_relative_odom_rotation_change_deg = 0.0;

        if (optimized_window_valid)
        {
            for (std::size_t k = 1;
                 k < current_indices.size();
                 ++k)
            {
                const std::size_t previous_index =
                    current_indices[k - 1];

                const std::size_t current_index =
                    current_indices[k];

                const Eigen::Isometry3d Z_before =
                    frozen_graph_poses[previous_index].inverse() *
                    frozen_graph_poses[current_index];

                const Eigen::Isometry3d Z_after =
                    optimized_local_poses[k - 1].inverse() *
                    optimized_local_poses[k];

                const Eigen::Isometry3d Z_error =
                    Z_before.inverse() *
                    Z_after;

                max_relative_odom_translation_change =
                    std::max(
                        max_relative_odom_translation_change,
                        Z_error.translation().norm());

                max_relative_odom_rotation_change_deg =
                    std::max(
                        max_relative_odom_rotation_change_deg,
                        RelativeRotationDeg(
                            Z_before,
                            Z_after));
            }
        }

        const bool refinement_window_translation_ok =
            optimized_window_valid &&
            max_pose_delta_t <=
                refinement_window_max_translation_update_;

        const bool refinement_window_rotation_ok =
            optimized_window_valid &&
            max_pose_delta_R_deg <=
                refinement_window_max_rotation_update_deg_;

        const bool refinement_window_relative_translation_ok =
            optimized_window_valid &&
            max_relative_odom_translation_change <=
                refinement_window_max_relative_odom_translation_change_;

        const bool refinement_window_relative_rotation_ok =
            optimized_window_valid &&
            max_relative_odom_rotation_change_deg <=
                refinement_window_max_relative_odom_rotation_change_deg_;

        // Coherent refinement gate:
        //   1) allow a reasonably large absolute rigid correction after the
        //      global loop optimization;
        //   2) reject solutions that distort the local odometry shape.
        const bool refinement_window_gate_accepted =
            optimized_window_valid &&
            refinement_window_translation_ok &&
            refinement_window_rotation_ok &&
            refinement_window_relative_translation_ok &&
            refinement_window_relative_rotation_ok;

        RCLCPP_INFO(
            kTimingLogger,
            "FR_REFINEMENT_WINDOW_GATE | historical_kf=%zu | current_kf=%zu "
            "| geometry_anchors=%zu | chi2_before=%.9f | chi2_after=%.9f "
            "| max_delta_t=%.6f | max_delta_t_limit=%.6f "
            "| max_delta_R_deg=%.6f | max_delta_R_limit_deg=%.6f "
            "| max_rel_odom_dt=%.6f | max_rel_odom_dt_limit=%.6f "
            "| max_rel_odom_dR_deg=%.6f | max_rel_odom_dR_limit_deg=%.6f "
            "| optimized_window_valid=%s | translation_ok=%s | rotation_ok=%s "
            "| relative_translation_ok=%s | relative_rotation_ok=%s "
            "| decision=%s",
            anchor.historical_id,
            anchor.current_id,
            geometry_anchor_count,
            local_optimization_result.chi2_before,
            local_optimization_result.chi2_after,
            max_pose_delta_t,
            refinement_window_max_translation_update_,
            max_pose_delta_R_deg,
            refinement_window_max_rotation_update_deg_,
            max_relative_odom_translation_change,
            refinement_window_max_relative_odom_translation_change_,
            max_relative_odom_rotation_change_deg,
            refinement_window_max_relative_odom_rotation_change_deg_,
            optimized_window_valid ? "true" : "false",
            refinement_window_translation_ok ? "true" : "false",
            refinement_window_rotation_ok ? "true" : "false",
            refinement_window_relative_translation_ok ? "true" : "false",
            refinement_window_relative_rotation_ok ? "true" : "false",
            refinement_window_gate_accepted ? "ACCEPT" : "REJECT");

        if (!refinement_window_gate_accepted)
        {
            ++groups_rejected_large_update;

            const char *rejection_reason =
                (!refinement_window_relative_translation_ok ||
                 !refinement_window_relative_rotation_ok)
                    ? "RELATIVE_ODOM_SHAPE_CHANGE"
                    : "LARGE_LOCAL_WINDOW_UPDATE";

            std::cout
                << "Post-PGO local-window refinement rejected"
                << " | historical_kf=" << anchor.historical_id
                << " | current_kf=" << anchor.current_id
                << " | geometry_anchors=" << geometry_anchor_count
                << " | chi2_before=" << local_optimization_result.chi2_before
                << " | chi2_after=" << local_optimization_result.chi2_after
                << " | max_delta_t=" << max_pose_delta_t << " m"
                << " | max_delta_R=" << max_pose_delta_R_deg << " deg"
                << " | max_odom_rel_dt="
                << max_relative_odom_translation_change << " m"
                << " | max_odom_rel_dR="
                << max_relative_odom_rotation_change_deg << " deg"
                << " | reason=" << rejection_reason
                << std::endl;

            continue;
        }

        const double mean_pose_delta_t =
            sum_pose_delta_t /
            static_cast<double>(
                current_indices.size());

        const double mean_pose_delta_R_deg =
            sum_pose_delta_R_deg /
            static_cast<double>(
                current_indices.size());

        for (std::size_t k = 0;
             k < current_indices.size();
             ++k)
        {
            const std::size_t source_index =
                current_indices[k];

            refined_keyframe_poses_[source_index] =
                optimized_local_poses[k];

            refined_keyframe_pose_was_adjusted_[source_index] =
                true;

            ++adjusted_keyframes;

            const Eigen::Isometry3d T_initial_optimized =
                frozen_graph_poses[source_index].inverse() *
                optimized_local_poses[k];

            const double final_delta_t =
                T_initial_optimized.translation().norm();

            const double final_delta_R_deg =
                RelativeRotationDeg(
                    frozen_graph_poses[source_index],
                    optimized_local_poses[k]);

            std::cout
                << "Post-PGO local pose refined"
                << " | keyframe=" << keyframes[source_index].id
                << " | geometry_tested="
                << (geometry_was_tested[k] ? "true" : "false")
                << " | geometry_anchor="
                << (geometry_anchor_valid[k] ? "true" : "false")
                << " | geometry_corr=" << geometry_correspondences[k]
                << " | geometry_rmse=" << geometry_rmse[k]
                << " | geometry_delta_t=" << geometry_delta_t[k]
                << " | geometry_delta_R=" << geometry_delta_R_deg[k]
                << " | final_delta_t=" << final_delta_t
                << " | final_delta_R=" << final_delta_R_deg
                << std::endl;
        }

        // --------------------------------------------------------------------
        // Save one COHERENT debug triplet for this accepted refinement group.
        // If several non-overlapping loop groups are accepted in one pass, the
        // latest accepted group replaces the previous debug triplet.  This is
        // intentional: the three RViz topics must always describe exactly the
        // same historical/current window rather than a confusing mixture.
        // --------------------------------------------------------------------
        const std::chrono::steady_clock::time_point
            debug_clouds_start =
                std::chrono::steady_clock::now();

        pcl::PointCloud<LIDAR_POINT>::Ptr current_before_debug =
            pcl::make_shared<pcl::PointCloud<LIDAR_POINT>>();

        pcl::PointCloud<LIDAR_POINT>::Ptr current_after_debug =
            pcl::make_shared<pcl::PointCloud<LIDAR_POINT>>();

        std::size_t estimated_current_debug_points = 0;

        for (const std::size_t source_index :
             current_indices)
        {
            if (keyframes[source_index].cloud)
            {
                estimated_current_debug_points +=
                    keyframes[source_index].cloud->size();
            }
        }

        current_before_debug->reserve(
            estimated_current_debug_points);

        current_after_debug->reserve(
            estimated_current_debug_points);

        for (std::size_t k = 0;
             k < current_indices.size();
             ++k)
        {
            const std::size_t source_index =
                current_indices[k];

            if (!keyframes[source_index].cloud ||
                keyframes[source_index].cloud->empty())
            {
                continue;
            }

            pcl::PointCloud<LIDAR_POINT> before_world_cloud;
            pcl::PointCloud<LIDAR_POINT> after_world_cloud;

            pcl::transformPointCloud(
                *keyframes[source_index].cloud,
                before_world_cloud,
                frozen_graph_poses[source_index]
                    .matrix()
                    .cast<float>());

            pcl::transformPointCloud(
                *keyframes[source_index].cloud,
                after_world_cloud,
                optimized_local_poses[k]
                    .matrix()
                    .cast<float>());

            *current_before_debug +=
                before_world_cloud;

            *current_after_debug +=
                after_world_cloud;
        }

        refinement_historical_target_debug_ =
            pcl::make_shared<pcl::PointCloud<LIDAR_POINT>>(
                *historical_target_filtered);

        refinement_current_before_debug_ =
            current_before_debug;

        refinement_current_after_debug_ =
            current_after_debug;

        refinement_debug_revision_ =
            global_map_revision_;

        std::cout
            << "Post-PGO refinement debug snapshots updated"
            << " | revision=" << refinement_debug_revision_
            << " | historical_kf=" << anchor.historical_id
            << " | current_kf=" << anchor.current_id
            << " | historical_points="
            << refinement_historical_target_debug_->size()
            << " | current_before_points="
            << refinement_current_before_debug_->size()
            << " | current_after_points="
            << refinement_current_after_debug_->size()
            << std::endl;

        refinement_timing.debug_clouds_ms +=
            ElapsedMilliseconds(
                debug_clouds_start,
                std::chrono::steady_clock::now());

        ++groups_accepted;
        refinement_timing.groups_accepted =
            groups_accepted;

        std::cout
            << "Post-PGO local-window refined"
            << " | historical_kf=" << anchor.historical_id
            << " | current_kf=" << anchor.current_id
            << " | current_kfs=" << current_indices.size()
            << " | historical_kfs=" << historical_candidates.size()
            << " | target_points=" << historical_target_filtered->size()
            << " | geometry_candidates=" << current_indices.size()
            << " | geometry_primary_selected=" << geometry_primary_selected
            << " | geometry_fallback_calls=" << geometry_fallback_calls
            << " | geometry_anchors=" << geometry_anchor_count
            << " | odom_edges=" << (current_indices.size() - 1)
            << " | chi2_before=" << local_optimization_result.chi2_before
            << " | chi2_after=" << local_optimization_result.chi2_after
            << " | mean_delta_t=" << mean_pose_delta_t << " m"
            << " | max_delta_t=" << max_pose_delta_t << " m"
            << " | mean_delta_R=" << mean_pose_delta_R_deg << " deg"
            << " | max_delta_R=" << max_pose_delta_R_deg << " deg"
            << " | max_odom_rel_dt="
            << max_relative_odom_translation_change << " m"
            << " | max_odom_rel_dR="
            << max_relative_odom_rotation_change_deg << " deg"
            << std::endl;
    }

    // ------------------------------------------------------------------------
    // Incrementally update /refined_map.
    //
    // The optimized block map is the base.  Only backend blocks containing
    // accepted locally-refined Keyframes become refined override blocks.  If
    // no local window was accepted, the refined map simply mirrors optimized
    // blocks without replaying every Keyframe cloud.
    // ------------------------------------------------------------------------
    std::size_t used_keyframes = 0;

    for (std::size_t i = 0;
         i < keyframes.size();
         ++i)
    {
        if (graph_pose_valid[i] &&
            keyframes[i].cloud &&
            !keyframes[i].cloud->empty() &&
            refined_keyframe_poses_[i].matrix().allFinite())
        {
            ++used_keyframes;
        }
    }

    if (used_keyframes == 0)
    {
        return false;
    }

    IncrementalGlobalMap::UpdateStats
        refined_stats;

    const std::chrono::steady_clock::time_point
        refined_map_update_start =
            std::chrono::steady_clock::now();

    const bool refined_update_ok =
        incremental_global_map_.UpdateRefinedOverrides(
            keyframes,
            refined_keyframe_poses_,
            refined_keyframe_pose_was_adjusted_,
            refined_stats);

    const pcl::PointCloud<LIDAR_POINT>::ConstPtr refined_map =
        incremental_global_map_.GetRefinedMap();

    refinement_timing.refined_map_update_ms +=
        ElapsedMilliseconds(
            refined_map_update_start,
            std::chrono::steady_clock::now());

    if (!refined_update_ok)
    {
        return false;
    }

    if (!refined_map ||
        refined_map->empty())
    {
        return false;
    }

    refined_map_revision_ =
        global_map_revision_;

    std::cout
        << "Post-PGO local-window refined map updated incrementally"
        << " | revision=" << refined_map_revision_
        << " | groups_considered=" << groups_considered
        << " | groups_prepared=" << groups_prepared
        << " | groups_optimized=" << groups_optimized
        << " | groups_accepted=" << groups_accepted
        << " | groups_rejected_overlap=" << groups_rejected_overlap
        << " | groups_rejected_quality=" << groups_rejected_quality
        << " | groups_rejected_large_update="
        << groups_rejected_large_update
        << " | geometry_candidates=" << geometry_candidates_total
        << " | geometry_primary_selected=" << geometry_primary_selected_total
        << " | geometry_attempts=" << geometry_attempts_total
        << " | geometry_fallback_calls=" << geometry_fallback_calls_total
        << " | geometry_anchors=" << geometry_anchors_total
        << " | adjusted_keyframes=" << adjusted_keyframes
        << " | used_keyframes=" << used_keyframes
        << " | refined_dirty_keyframes="
        << refined_stats.dirty_keyframes
        << " | refined_dirty_blocks="
        << refined_stats.dirty_blocks
        << " | refined_rebuilt_blocks="
        << refined_stats.rebuilt_blocks
        << " | refined_reused_blocks="
        << refined_stats.reused_blocks
        << " | total_blocks="
        << refined_stats.total_blocks
        << " | refined_points="
        << refined_map->size()
        << std::endl;

    return true;
}

// ============================================================================
// UpdateMapOdomCorrection()
//
// The frontend raw pose and the backend optimized pose refer to the SAME
// physical Keyframe but live in two different coordinate frames after loop
// correction:
//
//     raw frontend:       T_odom_K  == Keyframe::T_WL
//     corrected backend:  T_map_K   == PoseGraphNode::T_WK
//
// Therefore:
//
//     T_map_K = T_map_odom * T_odom_K
//
// and:
//
//     T_map_odom = T_map_K * T_odom_K^-1
//
// Between backend optimizations this correction is held constant.  New raw
// frontend scans remain continuous, while their corrected pose is obtained by
// left-multiplying this bridge.
// ============================================================================
