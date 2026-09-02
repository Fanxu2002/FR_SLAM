#pragma once

#include <cstddef>
#include <deque>
#include <limits>
#include <vector>

#include <pcl/point_cloud.h>

#include "fr_slam/mapping/fr_keyframe.hpp"
#include "fr_slam/common/fr_point_types.hpp"
#include "fr_slam/mapping/fr_submap.hpp"

struct SubmapManagerConfig
{
    // A complete Submap contains this many keyframes, INCLUDING overlap KFs.
    std::size_t max_keyframes_per_submap = 15;

    // Recent keyframes copied into the next Active Submap.
    std::size_t overlap_keyframes = 5;

    // ------------------------------------------------------------------------
    // Submap V1.1 transition rule.
    //
    // Immediately after a Submap switch:
    //
    //     Previous = mature finished Submap
    //     Active   = only overlap keyframes
    //
    // Do NOT immediately use the small Active Submap alone.
    //
    // While:
    //
    //     ActiveKeyframeCount < transition_until_active_keyframes
    //
    // tracking target becomes:
    //
    //     Previous + Active
    //
    // After the Active Submap reaches this size:
    //
    //     tracking target = Active only
    //
    // Default:
    //
    //     overlap = 5
    //     transition_until = 10
    //
    // so Active grows:
    //
    //     5 -> 6 -> 7 -> 8 -> 9 : Previous + Active
    //     10 -> ...              : Active only
    // ------------------------------------------------------------------------
    std::size_t transition_until_active_keyframes = 10;
};

// ============================================================================
// SubmapManager
//
// Keyframe
//    |
//    v
// Active Submap
//    |
//    +-- full --> finish as Previous
//                  |
//                  v
//             create new Active
//                  |
//                  v
//             seed overlap KFs
//
// Submap V1.1 additionally provides a smooth TRACKING target:
//
//     no Previous:
//         Active
//
//     just switched / Active still small:
//         Previous + Active
//
//     Active mature:
//         Active
// ============================================================================
class SubmapManager
{
public:
    SubmapManager(
        const SubmapManagerConfig &config,
        const LocalMapConfig &local_map_config);

    bool AddKeyframe(
        const Keyframe &keyframe);

    const Submap *ActiveSubmap() const;

    const Submap *PreviousSubmap() const;

    pcl::PointCloud<LIDAR_POINT>::ConstPtr
    GetActiveMap() const;

    pcl::PointCloud<LIDAR_POINT>::ConstPtr
    GetPreviousMap() const;

    // Current target used by normal registration / recovery.
    //
    // ACTIVE mode:
    //     return Active map.
    //
    // TRANSITION mode:
    //     return voxelized Previous + Active map.
    pcl::PointCloud<LIDAR_POINT>::ConstPtr
    GetTrackingMap() const;

    bool IsTransitionActive() const;

    std::size_t TrackingPointCount() const;

    std::size_t SubmapCount() const;

    std::size_t ActiveSubmapId() const;

    std::size_t PreviousSubmapId() const;

    std::size_t ActiveKeyframeCount() const;

    std::size_t ActivePointCount() const;

    bool LastAddStartedNewSubmap() const;

    std::size_t LastFinishedSubmapId() const;

    const std::vector<Submap> &GetAllSubmaps() const;

    void Clear();

private:
    static constexpr std::size_t kInvalidIndex =
        std::numeric_limits<std::size_t>::max();

    bool CreateFirstSubmap();

    bool AddKeyframeToSubmap(
        Submap &submap,
        const Keyframe &keyframe);

    void PushRecentOverlapKeyframe(
        const Keyframe &keyframe);

    // Freeze the current WORLD-frame LocalMap into a Submap-local backend
    // cloud before the Submap is marked finished.
    //
    //     p_S = T_WS^{-1} * p_W
    //
    // This does not modify the frontend LocalMap.
    bool FreezeSubmapLocalCloud(
        Submap &submap);

    bool FinishActiveAndStartNext();

    // Rebuild only the temporary transition target.
    //
    // Previous and Active themselves remain independent Submaps.
    bool RebuildTransitionMap();

private:
    SubmapManagerConfig config_;

    LocalMapConfig local_map_config_;

    std::vector<Submap> submaps_;

    std::size_t active_index_ = kInvalidIndex;

    std::size_t previous_index_ = kInvalidIndex;

    std::size_t next_submap_id_ = 0;

    // Keyframe::cloud is a shared_ptr, so this does not deep-copy point data.
    std::deque<Keyframe> recent_overlap_keyframes_;

    // Temporary Previous + Active target used only during Submap transition.
    //
    // It is in WORLD coordinates, exactly like each Submap LocalMap.
    pcl::PointCloud<LIDAR_POINT>::Ptr transition_map_ =
        pcl::make_shared<
            pcl::PointCloud<LIDAR_POINT>>();

    bool last_add_started_new_submap_ = false;

    std::size_t last_finished_submap_id_ = kInvalidIndex;
};