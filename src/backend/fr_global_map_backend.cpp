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
} // namespace

bool RegistrationScan2LocalMap::UpdateIncrementalGlobalMaps(
    const char *reason,
    bool clear_refined_overrides)
{
    const std::chrono::steady_clock::time_point
        map_update_start =
            std::chrono::steady_clock::now();

    const std::vector<Keyframe> &keyframes =
        backend_keyframes_;

    if (keyframes.empty() ||
        pose_graph_.NodeCount() == 0)
    {
        return false;
    }

    std::vector<
        Eigen::Isometry3d,
        Eigen::aligned_allocator<Eigen::Isometry3d>>
        optimized_poses(
            keyframes.size(),
            Eigen::Isometry3d::Identity());

    std::vector<bool>
        optimized_pose_valid(
            keyframes.size(),
            false);

    std::size_t missing_graph_nodes = 0;

    const std::chrono::steady_clock::time_point
        pose_snapshot_start =
            std::chrono::steady_clock::now();

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
            ++missing_graph_nodes;
            continue;
        }

        optimized_poses[i] =
            node->T_WK;

        optimized_pose_valid[i] =
            true;
    }

    const double pose_snapshot_ms =
        ElapsedMilliseconds(
            pose_snapshot_start,
            std::chrono::steady_clock::now());

    IncrementalGlobalMap::UpdateStats raw_stats;
    IncrementalGlobalMap::UpdateStats optimized_stats;

    const std::chrono::steady_clock::time_point
        raw_update_start =
            std::chrono::steady_clock::now();

    const bool raw_ok =
        incremental_global_map_.UpdateRaw(
            keyframes,
            raw_stats);

    const double raw_update_ms =
        ElapsedMilliseconds(
            raw_update_start,
            std::chrono::steady_clock::now());

    const std::chrono::steady_clock::time_point
        optimized_update_start =
            std::chrono::steady_clock::now();

    const bool optimized_ok =
        incremental_global_map_.UpdateOptimized(
            keyframes,
            optimized_poses,
            optimized_pose_valid,
            clear_refined_overrides,
            optimized_stats);

    const double optimized_update_ms =
        ElapsedMilliseconds(
            optimized_update_start,
            std::chrono::steady_clock::now());

    const pcl::PointCloud<LIDAR_POINT>::ConstPtr raw_map =
        incremental_global_map_.GetRawMap();

    const pcl::PointCloud<LIDAR_POINT>::ConstPtr optimized_map =
        incremental_global_map_.GetOptimizedMap();

    if (!raw_ok ||
        !optimized_ok ||
        !raw_map ||
        !optimized_map ||
        raw_map->empty() ||
        optimized_map->empty())
    {
        return false;
    }

    ++global_map_revision_;

    // During ordinary incremental growth an already-accepted local refinement
    // remains valid and IncrementalGlobalMap preserves its override blocks.
    // After a new main PoseGraph optimization, however, old overrides are
    // deliberately cleared and RebuildPostPgoRefinedMap() will create a new
    // refinement result for this graph revision.
    if (clear_refined_overrides)
    {
        refined_map_revision_ = 0;
    }
    else
    {
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr refined_map =
            incremental_global_map_.GetRefinedMap();

        if (refined_map &&
            !refined_map->empty())
        {
            refined_map_revision_ =
                global_map_revision_;
        }
    }

    std::cout
        << "Incremental global map update"
        << " | reason=" << (reason != nullptr ? reason : "UNKNOWN")
        << " | revision=" << global_map_revision_
        << " | keyframes=" << keyframes.size()
        << " | missing_graph_nodes=" << missing_graph_nodes
        << " | keyframes_per_block="
        << incremental_global_map_.KeyframesPerBlock()
        << " | total_blocks="
        << incremental_global_map_.BlockCount()
        << " | raw_dirty_keyframes=" << raw_stats.dirty_keyframes
        << " | raw_dirty_blocks=" << raw_stats.dirty_blocks
        << " | raw_rebuilt_blocks=" << raw_stats.rebuilt_blocks
        << " | raw_reused_blocks=" << raw_stats.reused_blocks
        << " | raw_points=" << raw_map->size()
        << " | opt_dirty_keyframes=" << optimized_stats.dirty_keyframes
        << " | opt_dirty_blocks=" << optimized_stats.dirty_blocks
        << " | opt_rebuilt_blocks=" << optimized_stats.rebuilt_blocks
        << " | opt_reused_blocks=" << optimized_stats.reused_blocks
        << " | optimized_points=" << optimized_map->size()
        << " | block_voxel_leaf="
        << incremental_global_map_.VoxelLeafSize() << " m"
        << std::endl;

    const double map_update_total_ms =
        ElapsedMilliseconds(
            map_update_start,
            std::chrono::steady_clock::now());

    RCLCPP_INFO(
        kTimingLogger,
        "FR_TIMING GLOBAL_MAP"
        " | reason=%s"
        " | total=%.3f ms"
        " | pose_snapshot=%.3f"
        " | raw_update=%.3f"
        " | optimized_update=%.3f"
        " | keyframes=%zu"
        " | raw_dirty_blocks=%zu"
        " | optimized_dirty_blocks=%zu"
        " | raw_points=%zu"
        " | optimized_points=%zu",
        reason != nullptr ? reason : "UNKNOWN",
        map_update_total_ms,
        pose_snapshot_ms,
        raw_update_ms,
        optimized_update_ms,
        keyframes.size(),
        raw_stats.dirty_blocks,
        optimized_stats.dirty_blocks,
        raw_map->size(),
        optimized_map->size());

    return true;
}

// ============================================================================
// RebuildGlobalMapSnapshots()
//
// Compatibility wrapper retained for the existing post-G2O call site.  The
// implementation is now incremental dirty-block maintenance, not a full map
// replay + global VoxelGrid.
// ============================================================================
bool RegistrationScan2LocalMap::RebuildGlobalMapSnapshots()
{
    return UpdateIncrementalGlobalMaps(
        "POSE_GRAPH",
        true);
}

// ============================================================================
// RebuildPostPgoRefinedMap()
//
// Post-PGO refinement V4: sparse-geometry local-window multi-pose optimization.
//
// Every current-window Keyframe still gets its own SE(3) variable and every
// consecutive odometry edge is retained.  The expensive point-to-plane
// registration is evaluated only on a uniformly distributed subset of
// Keyframes.  Those accepted geometry anchors constrain the whole local graph,
// and the odometry chain propagates the correction to unregistered Keyframes.
// The temporary graph solution is used only for
// /refined_map and never overwrites the main PoseGraph.
//
// Why V4 exists:
//     V1: each Keyframe independently registered to a LocalMap -> too free.
//     V2: one shared rigid correction for the whole window -> too rigid.
//     V3: local pose graph works well, but performs Full Align for every KF.
//     V4: keep the V3 graph, but sparsify only the expensive geometry anchors.
//
// V4 uses the same small temporary Keyframe PoseGraph:
//
//     fixed map anchor (Identity)
//          |       |       |       geometry edges from point-to-plane ICP
//          v       v       v
//        KF_i --- KF_i+1 --- KF_i+2 --- ...
//             odom       odom
//
// The geometry edges allow different Keyframes to receive different small
// corrections.  The stronger consecutive odometry edges prevent frame-to-frame
// jumps and preserve local trajectory continuity.
//
// Important: this is a practical Local-Pose-Graph refinement, not yet direct
// point-residual LiDAR bundle adjustment.  Point-to-plane ICP is first reduced
// to one soft SE(3) observation per accepted Keyframe.
//
// Safety rules:
//   1. Only accepted main-graph Loop edges define refinement regions.
//   2. Historical geometry is frozen from the G2O pose snapshot.
//   3. The main PoseGraph and frontend T_WL are never overwritten.
//   4. Each geometry observation must pass correspondence/RMSE/small-delta gates.
//   5. The whole local-window result must remain inside a final small-update gate.
//   6. Overlapping accepted refinement windows are skipped conservatively.
// ============================================================================

bool RegistrationScan2LocalMap::UpdateMapOdomCorrection(
    std::size_t anchor_keyframe_id)
{
    const Keyframe *anchor_keyframe =
        FindBackendKeyframeById(
            anchor_keyframe_id);

    const PoseGraphNode *anchor_node =
        pose_graph_.GetNode(
            anchor_keyframe_id);

    if (anchor_keyframe == nullptr ||
        anchor_node == nullptr ||
        !anchor_keyframe->T_WL.matrix().allFinite() ||
        !anchor_node->T_WK.matrix().allFinite())
    {
        return false;
    }

    const Eigen::Isometry3d T_map_odom_new =
        anchor_node->T_WK *
        anchor_keyframe->T_WL.inverse();

    if (!T_map_odom_new.matrix().allFinite())
    {
        return false;
    }

    const Eigen::AngleAxisd correction_rotation(
        T_map_odom_new.rotation());

    const double correction_rotation_deg =
        std::abs(
            correction_rotation.angle()) *
        180.0 /
        3.14159265358979323846;

    // Sanity check: applying the bridge to the raw anchor pose must recover the
    // optimized graph pose to numerical precision.
    const Eigen::Isometry3d T_map_K_check =
        T_map_odom_new *
        anchor_keyframe->T_WL;

    const Eigen::Isometry3d T_check_error =
        anchor_node->T_WK.inverse() *
        T_map_K_check;

    const double check_translation_error =
        T_check_error.translation().norm();

    const Eigen::AngleAxisd check_rotation(
        T_check_error.rotation());

    const double check_rotation_error_deg =
        std::abs(
            check_rotation.angle()) *
        180.0 /
        3.14159265358979323846;

    if (!std::isfinite(check_translation_error) ||
        !std::isfinite(check_rotation_error_deg) ||
        check_translation_error > 1.0e-6 ||
        check_rotation_error_deg > 1.0e-6)
    {
        std::cerr
            << "Map->odom correction anchor consistency check failed"
            << " | anchor_kf=" << anchor_keyframe_id
            << " | translation_error=" << check_translation_error << " m"
            << " | rotation_error=" << check_rotation_error_deg << " deg"
            << std::endl;
        return false;
    }

    T_map_odom_ =
        T_map_odom_new;

    has_map_odom_correction_ =
        true;

    map_odom_anchor_keyframe_id_ =
        anchor_keyframe_id;

    ++map_odom_revision_;

    std::cout
        << "Map->odom correction updated"
        << " | revision=" << map_odom_revision_
        << " | anchor_kf=" << map_odom_anchor_keyframe_id_
        << " | translation=["
        << T_map_odom_.translation().x() << " "
        << T_map_odom_.translation().y() << " "
        << T_map_odom_.translation().z() << "]"
        << " | translation_norm="
        << T_map_odom_.translation().norm() << " m"
        << " | rotation="
        << correction_rotation_deg << " deg"
        << " | anchor_check_translation="
        << check_translation_error << " m"
        << " | anchor_check_rotation="
        << check_rotation_error_deg << " deg"
        << std::endl;

    return true;
}

// Return the latest ACCEPTED LiDAR -> World pose.
//
// Rejected scans never modify this value.
