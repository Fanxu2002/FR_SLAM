#pragma once

#include "fr_slam/common/fr_point_types.hpp"

#include <cstddef>
#include <limits>
#include <mutex>
#include <utility>
#include <vector>

#include <pcl/point_cloud.h>

// ============================================================================
// Hierarchical 3m loop-retrieval RViz debug bridge.
//
// This header is intentionally independent from RegistrationScan2LocalMap's
// public API.  The asynchronous loop backend writes the exact 3m retrieval
// windows that were used for Region Scan Context, while the ROS node reads a
// cheap immutable snapshot and publishes it.
//
// Frame convention for both clouds:
//     raw frontend odom frame (T_WL / T_odom_L keyframe poses)
//
// Therefore RViz shows exactly where the frontend believes the current and
// historical 3m regions are before loop correction.
// ============================================================================
namespace fr_slam_debug
{

struct LoopRetrievalDebugSnapshot
{
    pcl::PointCloud<LIDAR_POINT>::ConstPtr current_window_odom;
    pcl::PointCloud<LIDAR_POINT>::ConstPtr historical_window_odom;

    std::vector<std::size_t> current_keyframe_ids;
    std::vector<std::size_t> historical_keyframe_ids;

    std::size_t revision = 0;

    std::size_t current_keyframe_id =
        std::numeric_limits<std::size_t>::max();

    std::size_t historical_anchor_keyframe_id =
        std::numeric_limits<std::size_t>::max();

    std::size_t historical_submap_id =
        std::numeric_limits<std::size_t>::max();

    // -1 = BACKWARD_3M, +1 = FORWARD_3M, 0 = no historical region selected.
    int historical_direction = 0;

    // 1-based rank after Submap aggregation.  Zero means unavailable.
    std::size_t submap_rank = 0;

    double current_arc_length_m = 0.0;
    double historical_arc_length_m = 0.0;

    double region_sc_distance =
        std::numeric_limits<double>::infinity();

    double region_sc_similarity = 0.0;
};

inline std::mutex &LoopRetrievalDebugMutex()
{
    static std::mutex mutex;
    return mutex;
}

inline LoopRetrievalDebugSnapshot &MutableLoopRetrievalDebugSnapshot()
{
    static LoopRetrievalDebugSnapshot snapshot;
    return snapshot;
}

inline void UpdateLoopRetrievalDebugSnapshot(
    LoopRetrievalDebugSnapshot snapshot)
{
    std::lock_guard<std::mutex> lock(
        LoopRetrievalDebugMutex());

    LoopRetrievalDebugSnapshot &stored =
        MutableLoopRetrievalDebugSnapshot();

    snapshot.revision =
        stored.revision + 1;

    stored =
        std::move(snapshot);
}

inline LoopRetrievalDebugSnapshot GetLoopRetrievalDebugSnapshot()
{
    std::lock_guard<std::mutex> lock(
        LoopRetrievalDebugMutex());

    return MutableLoopRetrievalDebugSnapshot();
}

} // namespace fr_slam_debug
