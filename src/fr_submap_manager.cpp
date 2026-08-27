#include "fr_slam/fr_submap_manager.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>

#include <pcl/filters/voxel_grid.h>

SubmapManager::SubmapManager(
    const SubmapManagerConfig &config,
    const LocalMapConfig &local_map_config)
    : config_(config),
      local_map_config_(local_map_config)
{
    // ------------------------------------------------------------------------
    // Sanitize Submap size.
    // ------------------------------------------------------------------------
    if (config_.max_keyframes_per_submap == 0)
    {
        config_.max_keyframes_per_submap = 1;
    }

    // Overlap must be strictly smaller than a complete Submap.
    if (config_.overlap_keyframes >=
        config_.max_keyframes_per_submap)
    {
        config_.overlap_keyframes =
            config_.max_keyframes_per_submap > 1
                ? config_.max_keyframes_per_submap - 1
                : 0;
    }

    // ------------------------------------------------------------------------
    // Sanitize transition threshold.
    //
    // We want:
    //
    //     overlap < transition_until <= max_keyframes_per_submap
    //
    // whenever the Submap has enough capacity for a real transition phase.
    // ------------------------------------------------------------------------
    if (config_.max_keyframes_per_submap <= 1)
    {
        config_.transition_until_active_keyframes =
            config_.max_keyframes_per_submap;
    }
    else
    {
        const std::size_t minimum_transition_size =
            std::min(
                config_.max_keyframes_per_submap,
                config_.overlap_keyframes + 1);

        if (config_.transition_until_active_keyframes <
            minimum_transition_size)
        {
            config_.transition_until_active_keyframes =
                minimum_transition_size;
        }

        if (config_.transition_until_active_keyframes >
            config_.max_keyframes_per_submap)
        {
            config_.transition_until_active_keyframes =
                config_.max_keyframes_per_submap;
        }
    }

    // Keep the same safety default as LocalMap.
    if (!(local_map_config_.voxel_leaf_size > 0.0f))
    {
        local_map_config_.voxel_leaf_size =
            0.30f;
    }

    // ------------------------------------------------------------------------
    // IMPORTANT:
    //
    // Old LocalMap was often configured:
    //
    //     max_frames = 10
    //
    // A Submap must not silently pop its own old keyframes before it reaches
    // max_keyframes_per_submap.
    // ------------------------------------------------------------------------
    local_map_config_.max_frames =
        std::max(
            local_map_config_.max_frames,
            config_.max_keyframes_per_submap);
}

bool SubmapManager::AddKeyframe(
    const Keyframe &keyframe)
{
    last_add_started_new_submap_ = false;
    last_finished_submap_id_ = kInvalidIndex;

    if (!keyframe.cloud ||
        keyframe.cloud->empty() ||
        !std::isfinite(keyframe.timestamp) ||
        !keyframe.T_WL.matrix().allFinite())
    {
        return false;
    }

    if (active_index_ == kInvalidIndex)
    {
        if (!CreateFirstSubmap())
        {
            return false;
        }
    }

    if (active_index_ >= submaps_.size())
    {
        return false;
    }

    Submap &active =
        submaps_[active_index_];

    if (active.finished)
    {
        return false;
    }

    // Avoid accidental duplicate insertion.
    if (!active.keyframe_ids.empty() &&
        active.keyframe_ids.back() == keyframe.id)
    {
        return true;
    }

    if (!AddKeyframeToSubmap(
            active,
            keyframe))
    {
        return false;
    }

    PushRecentOverlapKeyframe(
        keyframe);

    // ------------------------------------------------------------------------
    // Active full -> finish it and create a new Active seeded with overlap KFs.
    // ------------------------------------------------------------------------
    if (active.keyframe_ids.size() >=
        config_.max_keyframes_per_submap)
    {
        if (!FinishActiveAndStartNext())
        {
            return false;
        }
    }

    // ------------------------------------------------------------------------
    // Submap V1.1:
    //
    // Rebuild the temporary Previous + Active transition target whenever the
    // Active Submap changes.
    // ------------------------------------------------------------------------
    return RebuildTransitionMap();
}

const Submap *SubmapManager::ActiveSubmap() const
{
    if (active_index_ == kInvalidIndex ||
        active_index_ >= submaps_.size())
    {
        return nullptr;
    }

    return &submaps_[active_index_];
}

const Submap *SubmapManager::PreviousSubmap() const
{
    if (previous_index_ == kInvalidIndex ||
        previous_index_ >= submaps_.size())
    {
        return nullptr;
    }

    return &submaps_[previous_index_];
}

pcl::PointCloud<LIDAR_POINT>::ConstPtr
SubmapManager::GetActiveMap() const
{
    const Submap *active =
        ActiveSubmap();

    if (active == nullptr)
    {
        return nullptr;
    }

    return active->local_map.GetMap();
}

pcl::PointCloud<LIDAR_POINT>::ConstPtr
SubmapManager::GetPreviousMap() const
{
    const Submap *previous =
        PreviousSubmap();

    if (previous == nullptr)
    {
        return nullptr;
    }

    return previous->local_map.GetMap();
}

pcl::PointCloud<LIDAR_POINT>::ConstPtr
SubmapManager::GetTrackingMap() const
{
    if (IsTransitionActive())
    {
        if (transition_map_ &&
            !transition_map_->empty())
        {
            return transition_map_;
        }
    }

    return GetActiveMap();
}

bool SubmapManager::IsTransitionActive() const
{
    const Submap *active =
        ActiveSubmap();

    const Submap *previous =
        PreviousSubmap();

    if (active == nullptr ||
        previous == nullptr)
    {
        return false;
    }

    return active->keyframe_ids.size() <
           config_.transition_until_active_keyframes;
}

std::size_t SubmapManager::TrackingPointCount() const
{
    const pcl::PointCloud<LIDAR_POINT>::ConstPtr tracking_map =
        GetTrackingMap();

    return tracking_map
               ? tracking_map->size()
               : 0;
}

std::size_t SubmapManager::SubmapCount() const
{
    return submaps_.size();
}

std::size_t SubmapManager::ActiveSubmapId() const
{
    const Submap *active =
        ActiveSubmap();

    return active != nullptr
               ? active->id
               : kInvalidIndex;
}

std::size_t SubmapManager::PreviousSubmapId() const
{
    const Submap *previous =
        PreviousSubmap();

    return previous != nullptr
               ? previous->id
               : kInvalidIndex;
}

std::size_t SubmapManager::ActiveKeyframeCount() const
{
    const Submap *active =
        ActiveSubmap();

    return active != nullptr
               ? active->keyframe_ids.size()
               : 0;
}

std::size_t SubmapManager::ActivePointCount() const
{
    const Submap *active =
        ActiveSubmap();

    return active != nullptr
               ? active->local_map.PointCount()
               : 0;
}

bool SubmapManager::LastAddStartedNewSubmap() const
{
    return last_add_started_new_submap_;
}

std::size_t SubmapManager::LastFinishedSubmapId() const
{
    return last_finished_submap_id_;
}

const std::vector<Submap> &
SubmapManager::GetAllSubmaps() const
{
    return submaps_;
}

void SubmapManager::Clear()
{
    submaps_.clear();
    recent_overlap_keyframes_.clear();

    active_index_ = kInvalidIndex;
    previous_index_ = kInvalidIndex;
    next_submap_id_ = 0;

    transition_map_ =
        pcl::make_shared<
            pcl::PointCloud<LIDAR_POINT>>();

    last_add_started_new_submap_ = false;
    last_finished_submap_id_ = kInvalidIndex;
}

bool SubmapManager::CreateFirstSubmap()
{
    if (!submaps_.empty())
    {
        return false;
    }

    submaps_.emplace_back(
        next_submap_id_,
        local_map_config_);

    ++next_submap_id_;

    active_index_ = 0;
    previous_index_ = kInvalidIndex;

    return true;
}

bool SubmapManager::AddKeyframeToSubmap(
    Submap &submap,
    const Keyframe &keyframe)
{
    if (submap.finished ||
        !keyframe.cloud ||
        keyframe.cloud->empty() ||
        !keyframe.T_WL.matrix().allFinite())
    {
        return false;
    }

    // Existing LocalMap does:
    //
    //     p_W = T_WL * p_L
    //     merge
    //     voxel
    //
    // LocalMap remains the INTERNAL map builder of each Submap.
    if (!submap.local_map.AddFrame(
            keyframe.cloud,
            keyframe.T_WL))
    {
        return false;
    }

    // ------------------------------------------------------------------------
    // Backend Submap origin pose.
    //
    // Initialize exactly ONCE, using the first keyframe that is successfully
    // inserted into this Submap:
    //
    //     T_WS = T_WL(first keyframe in this Submap)
    //
    // IMPORTANT:
    // This is deliberately placed inside AddKeyframeToSubmap(), not only in
    // AddKeyframe(), because a newly created Submap is initially seeded with
    // overlap keyframes through this same helper function.
    //
    // Example:
    //
    //     Submap0 = KF0  ... KF14
    //     Submap1 = KF10 ... KF24
    //
    // Then:
    //
    //     T_WS0 = T_WL(KF0)
    //     T_WS1 = T_WL(KF10)
    //
    // which gives each Submap a stable local origin for the future PoseGraph.
    // ------------------------------------------------------------------------
    if (!submap.has_origin_pose)
    {
        submap.T_WS =
            keyframe.T_WL;

        submap.has_origin_pose =
            true;
    }

    submap.keyframe_ids.push_back(
        keyframe.id);

    return true;
}

void SubmapManager::PushRecentOverlapKeyframe(
    const Keyframe &keyframe)
{
    if (config_.overlap_keyframes == 0)
    {
        recent_overlap_keyframes_.clear();
        return;
    }

    recent_overlap_keyframes_.push_back(
        keyframe);

    while (recent_overlap_keyframes_.size() >
           config_.overlap_keyframes)
    {
        recent_overlap_keyframes_.pop_front();
    }
}

bool SubmapManager::FreezeSubmapLocalCloud(
    Submap &submap)
{
    // A finished backend cloud is immutable. Repeated calls are harmless.
    if (submap.has_frozen_cloud)
    {
        return submap.cloud_S &&
               !submap.cloud_S->empty();
    }

    if (!submap.has_origin_pose ||
        !submap.T_WS.matrix().allFinite())
    {
        return false;
    }

    const pcl::PointCloud<LIDAR_POINT>::ConstPtr cloud_W =
        submap.local_map.GetMap();

    if (!cloud_W ||
        cloud_W->empty())
    {
        return false;
    }

    const Eigen::Isometry3d T_SW =
        submap.T_WS.inverse();

    if (!T_SW.matrix().allFinite())
    {
        return false;
    }

    pcl::PointCloud<LIDAR_POINT>::Ptr cloud_S =
        pcl::make_shared<
            pcl::PointCloud<LIDAR_POINT>>();

    cloud_S->reserve(
        cloud_W->size());

    // ------------------------------------------------------------------------
    // WORLD -> SUBMAP
    //
    // LocalMap currently stores:
    //
    //     p_W
    //
    // Backend storage must be independent from the current global drift:
    //
    //     p_S = T_WS^{-1} * p_W
    //
    // Copy the complete custom point first so intensity / ring / time_offset
    // are preserved, then overwrite XYZ only.
    // ------------------------------------------------------------------------
    for (const LIDAR_POINT &point_W :
         cloud_W->points)
    {
        if (!std::isfinite(point_W.x) ||
            !std::isfinite(point_W.y) ||
            !std::isfinite(point_W.z))
        {
            continue;
        }

        const Eigen::Vector3d p_W(
            static_cast<double>(point_W.x),
            static_cast<double>(point_W.y),
            static_cast<double>(point_W.z));

        const Eigen::Vector3d p_S =
            T_SW * p_W;

        if (!p_S.allFinite())
        {
            continue;
        }

        LIDAR_POINT point_S =
            point_W;

        point_S.x =
            static_cast<float>(p_S.x());

        point_S.y =
            static_cast<float>(p_S.y());

        point_S.z =
            static_cast<float>(p_S.z());

        cloud_S->push_back(
            point_S);
    }

    if (cloud_S->empty())
    {
        return false;
    }

    cloud_S->width =
        static_cast<std::uint32_t>(
            cloud_S->size());

    cloud_S->height = 1;
    cloud_S->is_dense = true;

    submap.cloud_S =
        cloud_S;

    submap.has_frozen_cloud =
        true;

    std::cout
        << "Submap backend cloud frozen"
        << " | submap=" << submap.id
        << " | frame=S"
        << " | world_points=" << cloud_W->size()
        << " | local_points=" << cloud_S->size()
        << std::endl;

    return true;
}


bool SubmapManager::FinishActiveAndStartNext()
{
    if (active_index_ == kInvalidIndex ||
        active_index_ >= submaps_.size())
    {
        return false;
    }

    Submap &finished_submap =
        submaps_[active_index_];

    if (finished_submap.finished)
    {
        return false;
    }

    // ------------------------------------------------------------------------
    // Freeze a backend-local snapshot BEFORE changing Submap lifecycle state.
    //
    // If this fails, keep the Submap Active. This avoids ending up with a
    // "finished" Submap that has no cloud_S for Scan Context / loop closure.
    // ------------------------------------------------------------------------
    if (!FreezeSubmapLocalCloud(
            finished_submap))
    {
        return false;
    }

    finished_submap.finished = true;

    previous_index_ =
        active_index_;

    last_finished_submap_id_ =
        finished_submap.id;

    // Create the next Active Submap.
    submaps_.emplace_back(
        next_submap_id_,
        local_map_config_);

    ++next_submap_id_;

    active_index_ =
        submaps_.size() - 1;

    last_add_started_new_submap_ = true;

    // ------------------------------------------------------------------------
    // Seed overlap:
    //
    //     Previous: KF0 ... KF14
    //     Active  :        KF10 ... KF14
    //
    // These duplicated overlap keyframes are intentional.
    //
    // When Previous + Active are merged for transition tracking, a VoxelGrid
    // is applied again so duplicated overlap geometry does not simply double
    // the target density.
    // ------------------------------------------------------------------------
    Submap &new_active =
        submaps_[active_index_];

    for (const Keyframe &overlap_keyframe :
         recent_overlap_keyframes_)
    {
        if (!AddKeyframeToSubmap(
                new_active,
                overlap_keyframe))
        {
            return false;
        }
    }

    return true;
}

bool SubmapManager::RebuildTransitionMap()
{
    // No transition phase:
    //
    // GetTrackingMap() will directly return the Active Submap map.
    if (!IsTransitionActive())
    {
        transition_map_->clear();

        transition_map_->width = 0;
        transition_map_->height = 1;
        transition_map_->is_dense = false;

        return true;
    }

    const pcl::PointCloud<LIDAR_POINT>::ConstPtr previous_map =
        GetPreviousMap();

    const pcl::PointCloud<LIDAR_POINT>::ConstPtr active_map =
        GetActiveMap();

    if (!previous_map ||
        previous_map->empty() ||
        !active_map ||
        active_map->empty())
    {
        return false;
    }

    // ------------------------------------------------------------------------
    // Merge two WORLD-frame maps.
    //
    // We intentionally merge map clouds rather than raw Keyframes here:
    //
    //     Previous LocalMap
    //          +
    //     Active LocalMap
    //
    // Then voxel once more to collapse duplicated overlap geometry.
    // ------------------------------------------------------------------------
    pcl::PointCloud<LIDAR_POINT>::Ptr merged =
        pcl::make_shared<
            pcl::PointCloud<LIDAR_POINT>>();

    merged->reserve(
        previous_map->size() +
        active_map->size());

    *merged +=
        *previous_map;

    *merged +=
        *active_map;

    if (merged->empty())
    {
        return false;
    }

    pcl::VoxelGrid<LIDAR_POINT> voxel;

    voxel.setInputCloud(
        merged);

    voxel.setDownsampleAllData(
        false);

    voxel.setLeafSize(
        local_map_config_.voxel_leaf_size,
        local_map_config_.voxel_leaf_size,
        local_map_config_.voxel_leaf_size);

    pcl::PointCloud<LIDAR_POINT>::Ptr filtered =
        pcl::make_shared<
            pcl::PointCloud<LIDAR_POINT>>();

    voxel.filter(
        *filtered);

    if (!filtered ||
        filtered->empty())
    {
        return false;
    }

    transition_map_ =
        filtered;

    return true;
}