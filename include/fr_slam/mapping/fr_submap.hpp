#pragma once

#include "fr_slam/mapping/fr_local_map.hpp"
#include "fr_slam/common/fr_point_types.hpp"

#include <cstddef>
#include <vector>

#include <Eigen/Geometry>

#include <pcl/point_cloud.h>

// ============================================================================
// One Submap.
//
// Frontend use:
//     local_map is currently built in WORLD coordinates and is used as part of
//     the Scan-to-Submap tracking target.
//
// Backend use:
//     T_WS defines this Submap's own coordinate-frame origin in World.
//     Later, one Submap will correspond to one PoseGraph node.
//
// Transform convention:
//
//     p_W = T_WS * p_S
//
// W = World frame
// S = this Submap frame
// ============================================================================
struct Submap
{
    Submap(
        std::size_t submap_id,
        const LocalMapConfig &local_map_config)
        : id(submap_id),
          local_map(local_map_config)
    {
    }

    // Unique Submap ID.
    std::size_t id = 0;

    // ------------------------------------------------------------------------
    // Initial / optimized Submap pose in World.
    //
    // For now this is initialized from the FIRST keyframe inserted into this
    // Submap:
    //
    //     T_WS = T_WL_first_keyframe
    //
    // Later this becomes the PoseGraph state X_i.
    // ------------------------------------------------------------------------
    Eigen::Isometry3d T_WS =
        Eigen::Isometry3d::Identity();

    // True after T_WS has been initialized exactly once.
    bool has_origin_pose = false;

    // Keyframes contained in this Submap.
    std::vector<std::size_t> keyframe_ids;

    // ------------------------------------------------------------------------
    // Existing FRONTEND map builder.
    //
    // IMPORTANT:
    //     local_map remains in WORLD coordinates:
    //
    //         p_W = T_WL * p_L
    //
    // We deliberately do NOT change this because the live Scan-to-Submap
    // frontend already uses this representation.
    // ------------------------------------------------------------------------
    LocalMap local_map;

    // ------------------------------------------------------------------------
    // Frozen BACKEND cloud expressed in this Submap's own S frame.
    //
    // It is created exactly once when this Submap becomes finished:
    //
    //     p_S = T_WS^{-1} * p_W
    //
    // After that, cloud_S never follows later PoseGraph corrections.
    // Instead, an optimized Submap pose can move the entire rigid Submap:
    //
    //     p_W_optimized = T_WS_optimized * p_S
    //
    // Future users:
    //     - Scan Context
    //     - Loop Verification
    //     - Global map reconstruction after g2o
    // ------------------------------------------------------------------------
    pcl::PointCloud<LIDAR_POINT>::Ptr cloud_S =
        pcl::make_shared<
            pcl::PointCloud<LIDAR_POINT>>();

    // True only after cloud_S has been successfully frozen.
    bool has_frozen_cloud = false;

    // Finished Submaps no longer accept normal new keyframes.
    bool finished = false;
};