#pragma once

#include <cstddef>
#include <limits>
#include <vector>

#include <pcl/point_cloud.h>

#include "fr_slam/mapping/keyframe.hpp"
#include "fr_slam/common/point_types.hpp"
#include "fr_slam/mapping/submap.hpp"

struct SubmapManagerConfig
{
    // Final Submap size.
    //
    // S0 = KF0  ... KF29
    // S1 = KF15 ... KF44
    // S2 = KF30 ... KF59
    std::size_t max_keyframes_per_submap = 30;

    // Number of overlapping Keyframes.
    //
    // Growing begins BEFORE insertion when:
    //
    //     primary_count >= max_keyframes_per_submap - overlap_keyframes
    //
    // With 30 / 15:
    //
    //     KF15 arrives
    //       -> create S1
    //       -> insert KF15 into S0 and S1
    std::size_t overlap_keyframes = 15;

    // Retained only for source compatibility with the old V1.1 API.
    // The Final Fixed-frame Submap architecture does NOT use a
    // Previous+Active transition target.
    std::size_t transition_until_active_keyframes = 15;
};

// ============================================================================
// FR-SLAM Final Fixed-frame Overlapping Submap Manager.
//
// Lifecycle:
//
//     PRIMARY
//       old geometry immutable
//       new Keyframes may append
//
//     GROWING
//       old geometry immutable
//       new Keyframes may append
//       not used for tracking
//
//     FINISHED
//       geometry fully immutable
//       no new Keyframes
//
// Tracking target:
//
//     PRIMARY.cloud_O only
//
// Internal source of truth:
//
//     PRIMARY.cloud_S
//
// No Previous+Active target merge.
// No reconstruction from historical Keyframe poses.
// ============================================================================
class SubmapManager
{
public:
    SubmapManager(
        const SubmapManagerConfig &config,
        const LocalMapConfig &local_map_config);

    bool AddKeyframe(
        const Keyframe &keyframe);

    // Legacy name retained:
    // ActiveSubmap() == PRIMARY Submap.
    const Submap *ActiveSubmap() const;

    // Latest FINISHED Submap.
    const Submap *PreviousSubmap() const;

    const Submap *GrowingSubmap() const;

    // Frontend odom-frame views.
    pcl::PointCloud<LIDAR_POINT>::ConstPtr
    GetActiveMap() const;

    pcl::PointCloud<LIDAR_POINT>::ConstPtr
    GetPreviousMap() const;

    // Registration target = PRIMARY only.
    pcl::PointCloud<LIDAR_POINT>::ConstPtr
    GetTrackingMap() const;

    // Final architecture has no Previous+Primary transition target.
    bool IsTransitionActive() const;

    std::size_t TrackingPointCount() const;

    std::size_t SubmapCount() const;

    std::size_t ActiveSubmapId() const;

    std::size_t PreviousSubmapId() const;

    std::size_t GrowingSubmapId() const;

    std::size_t ActiveKeyframeCount() const;

    std::size_t ActivePointCount() const;

    bool LastAddStartedNewSubmap() const;

    std::size_t LastFinishedSubmapId() const;

    const std::vector<Submap> &
    GetAllSubmaps() const;

    void Clear();

private:
    static constexpr std::size_t kInvalidIndex =
        std::numeric_limits<std::size_t>::max();

    bool CreateFirstPrimary(
        const Keyframe &keyframe);

    bool CreateGrowing(
        const Keyframe &keyframe);

    bool AddKeyframeToSubmap(
        Submap &submap,
        const Keyframe &keyframe);

    bool FinishPrimaryAndPromoteGrowing();

private:
    SubmapManagerConfig config_;

    // Only voxel_leaf_size is reused.
    // The old LocalMap rolling-window behavior is NOT used.
    LocalMapConfig local_map_config_;

    std::vector<Submap> submaps_;

    // PRIMARY
    std::size_t active_index_ =
        kInvalidIndex;

    // GROWING
    std::size_t growing_index_ =
        kInvalidIndex;

    // Latest FINISHED Submap.
    std::size_t previous_index_ =
        kInvalidIndex;

    std::size_t next_submap_id_ = 0;

    bool last_add_started_new_submap_ = false;

    std::size_t last_finished_submap_id_ =
        kInvalidIndex;
};
