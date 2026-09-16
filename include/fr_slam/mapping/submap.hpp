#pragma once

#include "fr_slam/mapping/local_map.hpp"
#include "fr_slam/common/point_types.hpp"

#include <cstddef>
#include <limits>
#include <vector>

#include <Eigen/Geometry>

#include <pcl/point_cloud.h>

// ============================================================================
// FR-SLAM Fixed-frame Overlapping Submap.
//
// Core invariant:
//
//     A point inserted into cloud_S is NEVER transformed again.
//
// Frontend continuous pose:
//
//     T_O_L
//
// Fixed Submap creation anchor:
//
//     T_O_S_creation
//
// Keyframe pose inside this Submap:
//
//     T_S_L = T_O_S_creation^-1 * T_O_L
//
// Inserted point:
//
//     p_S = T_S_L * p_L
//
// cloud_O is only a derived frontend tracking view:
//
//     p_O = T_O_S_creation * p_S
//
// T_O_S_creation never follows later odometry / PGO corrections.
// ============================================================================
enum class SubmapState
{
    Growing,
    Primary,
    Finished
};

struct Submap
{
    Submap(
        std::size_t submap_id,
        const LocalMapConfig &)
        : id(submap_id)
    {
    }

    std::size_t id = 0;

    SubmapState state =
        SubmapState::Growing;

    // ------------------------------------------------------------------------
    // Fixed frontend anchor.
    //
    // Initialized exactly once when the Submap is created.
    // Never modified afterwards by frontend or backend.
    // ------------------------------------------------------------------------
    Eigen::Isometry3d T_O_S_creation =
        Eigen::Isometry3d::Identity();

    // ------------------------------------------------------------------------
    // Compatibility pose used by the existing backend interface.
    //
    // At creation this is identical to T_O_S_creation.
    //
    // IMPORTANT:
    // Frontend tracking / insertion MUST NOT use this field anymore.
    // T_O_S_creation is the immutable frontend anchor.
    // ------------------------------------------------------------------------
    Eigen::Isometry3d T_WS =
        Eigen::Isometry3d::Identity();

    bool has_origin_pose = false;

    std::vector<std::size_t>
        keyframe_ids;

    // ------------------------------------------------------------------------
    // Source of truth: immutable historical geometry in Submap frame.
    //
    // Existing points are NEVER rebuilt from Keyframes.
    // New Keyframes may append points while state is Primary / Growing.
    // ------------------------------------------------------------------------
    pcl::PointCloud<LIDAR_POINT>::Ptr cloud_S =
        pcl::make_shared<
            pcl::PointCloud<LIDAR_POINT>>();

    // ------------------------------------------------------------------------
    // Derived frontend tracking view in continuous odom coordinates.
    //
    // For every point:
    //
    //     p_O = T_O_S_creation * p_S
    //
    // Existing points are also append-only because T_O_S_creation is fixed.
    //
    // This preserves the existing LO/LIO registration API:
    //
    //     target = odom/world-like frontend frame
    //     result = T_O_L
    //
    // while internal Submap geometry remains fixed in S.
    // ------------------------------------------------------------------------
    pcl::PointCloud<LIDAR_POINT>::Ptr cloud_O =
        pcl::make_shared<
            pcl::PointCloud<LIDAR_POINT>>();

    // Becomes true when the Submap is FINISHED.
    // cloud_S already exists before this; "frozen" means no more appends.
    bool has_frozen_cloud = false;

    bool finished = false;

    double start_timestamp =
        std::numeric_limits<double>::quiet_NaN();

    double end_timestamp =
        std::numeric_limits<double>::quiet_NaN();
};
