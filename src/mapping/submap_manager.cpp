#include <iomanip>
#include <iostream>
#include "fr_slam/mapping/submap_manager.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

#include <pcl/filters/voxel_grid.h>

// ============================================================================
// Constructor
// ============================================================================
SubmapManager::SubmapManager(
    const SubmapManagerConfig &config,
    const LocalMapConfig &local_map_config)
    : config_(config),
      local_map_config_(local_map_config)
{
    if (config_.max_keyframes_per_submap < 2)
    {
        config_.max_keyframes_per_submap = 2;
    }

    // Final architecture always uses real overlap.
    if (config_.overlap_keyframes == 0)
    {
        config_.overlap_keyframes = 1;
    }

    if (config_.overlap_keyframes >=
        config_.max_keyframes_per_submap)
    {
        config_.overlap_keyframes =
            config_.max_keyframes_per_submap - 1;
    }

    if (!(local_map_config_.voxel_leaf_size > 0.0f))
    {
        local_map_config_.voxel_leaf_size =
            0.30f;
    }
}

// ============================================================================
// Add one accepted Keyframe.
//
// IMPORTANT ORDER:
//
//     lifecycle preparation
//          ↓
//     insert current KF
//          ↓
//     finish/promote if required
//
// Therefore KF15 creates S1 BEFORE KF15 insertion.
// ============================================================================
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

    // ------------------------------------------------------------------------
    // First Keyframe:
    //
    //     create S0 PRIMARY
    //     anchor = current T_O_L
    // ------------------------------------------------------------------------
    if (active_index_ == kInvalidIndex)
    {
        if (!CreateFirstPrimary(
                keyframe))
        {
            return false;
        }
    }

    if (active_index_ >= submaps_.size())
    {
        return false;
    }

    // ------------------------------------------------------------------------
    // Create GROWING before insertion.
    //
    // N = 30
    // overlap = 15
    //
    // growing_start_count = 15
    //
    // After KF14:
    //     S0 count = 15
    //
    // KF15 arrives:
    //     create S1 anchored at KF15
    //     then insert KF15 into S0 and S1
    // ------------------------------------------------------------------------
    const std::size_t growing_start_count =
        config_.max_keyframes_per_submap -
        config_.overlap_keyframes;

    if (growing_index_ == kInvalidIndex &&
        submaps_[active_index_].keyframe_ids.size() >=
            growing_start_count)
    {
        if (!CreateGrowing(
                keyframe))
        {
            return false;
        }
    }

    // ------------------------------------------------------------------------
    // Match-first / commit-second is enforced by the caller.
    //
    // We are now in the commit stage.
    // ------------------------------------------------------------------------
    if (!AddKeyframeToSubmap(
            submaps_[active_index_],
            keyframe))
    {
        return false;
    }

    if (growing_index_ != kInvalidIndex)
    {
        if (growing_index_ >= submaps_.size())
        {
            return false;
        }

        if (!AddKeyframeToSubmap(
                submaps_[growing_index_],
                keyframe))
        {
            return false;
        }
    }

    // ------------------------------------------------------------------------
    // PRIMARY full:
    //
    //     old PRIMARY -> FINISHED
    //     GROWING     -> PRIMARY
    //
    // With 30/15:
    //
    //     after KF29:
    //         S0 = 30
    //         S1 = 15
    //
    //     promote S1 immediately.
    //
    // Next KF30 arrives:
    //     S1 count already 15
    //     -> create S2 before KF30 insertion.
    // ------------------------------------------------------------------------
    if (submaps_[active_index_].keyframe_ids.size() >=
        config_.max_keyframes_per_submap)
    {
        if (!FinishPrimaryAndPromoteGrowing())
        {
            return false;
        }
    }

    return true;
}

// ============================================================================
// Accessors
// ============================================================================
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

const Submap *SubmapManager::GrowingSubmap() const
{
    if (growing_index_ == kInvalidIndex ||
        growing_index_ >= submaps_.size())
    {
        return nullptr;
    }

    return &submaps_[growing_index_];
}

pcl::PointCloud<LIDAR_POINT>::ConstPtr
SubmapManager::GetActiveMap() const
{
    const Submap *primary =
        ActiveSubmap();

    if (primary == nullptr)
    {
        return nullptr;
    }

    return primary->cloud_O;
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

    return previous->cloud_O;
}

pcl::PointCloud<LIDAR_POINT>::ConstPtr
SubmapManager::GetTrackingMap() const
{
    // FINAL rule:
    //
    // Tracking uses PRIMARY only.
    //
    // Never concatenate PRIMARY + GROWING.
    // Never concatenate PREVIOUS + PRIMARY.
    return GetActiveMap();
}

bool SubmapManager::IsTransitionActive() const
{
    return false;
}

std::size_t SubmapManager::TrackingPointCount() const
{
    const pcl::PointCloud<LIDAR_POINT>::ConstPtr cloud =
        GetTrackingMap();

    return cloud
               ? cloud->size()
               : 0;
}

std::size_t SubmapManager::SubmapCount() const
{
    return submaps_.size();
}

std::size_t SubmapManager::ActiveSubmapId() const
{
    const Submap *primary =
        ActiveSubmap();

    return primary != nullptr
               ? primary->id
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

std::size_t SubmapManager::GrowingSubmapId() const
{
    const Submap *growing =
        GrowingSubmap();

    return growing != nullptr
               ? growing->id
               : kInvalidIndex;
}

std::size_t SubmapManager::ActiveKeyframeCount() const
{
    const Submap *primary =
        ActiveSubmap();

    return primary != nullptr
               ? primary->keyframe_ids.size()
               : 0;
}

std::size_t SubmapManager::ActivePointCount() const
{
    const Submap *primary =
        ActiveSubmap();

    return primary != nullptr &&
                   primary->cloud_O
               ? primary->cloud_O->size()
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

    active_index_ = kInvalidIndex;
    growing_index_ = kInvalidIndex;
    previous_index_ = kInvalidIndex;

    next_submap_id_ = 0;

    last_add_started_new_submap_ = false;
    last_finished_submap_id_ = kInvalidIndex;
}

// ============================================================================
// Create initial PRIMARY.
//
// Anchor is frozen at creation:
//
//     T_O_S_creation = T_O_L(current)
// ============================================================================
bool SubmapManager::CreateFirstPrimary(
    const Keyframe &keyframe)
{
    if (!submaps_.empty() ||
        !keyframe.T_WL.matrix().allFinite())
    {
        return false;
    }

    submaps_.emplace_back(
        next_submap_id_,
        local_map_config_);

    ++next_submap_id_;

    active_index_ =
        submaps_.size() - 1;

    Submap &primary =
        submaps_[active_index_];

    primary.state =
        SubmapState::Primary;

    primary.T_O_S_creation =
        keyframe.T_WL;

    // Compatibility copy for existing backend code.
    primary.T_WS =
        primary.T_O_S_creation;

    primary.has_origin_pose = true;

    primary.start_timestamp =
        keyframe.timestamp;

    primary.end_timestamp =
        keyframe.timestamp;

    return true;
}

// ============================================================================
// Create GROWING before the current overlap Keyframe is inserted.
//
// Example:
//
//     after KF14:
//         S0 count = 15
//
//     KF15 arrives:
//         CreateGrowing(KF15)
//         anchor S1 = T_O_L(KF15)
//         then KF15 is inserted into both S0 / S1.
// ============================================================================
bool SubmapManager::CreateGrowing(
    const Keyframe &keyframe)
{
    if (growing_index_ != kInvalidIndex ||
        active_index_ == kInvalidIndex ||
        active_index_ >= submaps_.size() ||
        !keyframe.T_WL.matrix().allFinite())
    {
        return false;
    }

    submaps_.emplace_back(
        next_submap_id_,
        local_map_config_);

    ++next_submap_id_;

    growing_index_ =
        submaps_.size() - 1;

    Submap &growing =
        submaps_[growing_index_];

    growing.state =
        SubmapState::Growing;

    growing.T_O_S_creation =
        keyframe.T_WL;

    // Compatibility copy for existing backend code.
    growing.T_WS =
        growing.T_O_S_creation;

    growing.has_origin_pose = true;

    growing.start_timestamp =
        keyframe.timestamp;

    growing.end_timestamp =
        keyframe.timestamp;

    last_add_started_new_submap_ =
        true;

    std::cout
        << "[SUBMAP_AUDIT] CREATE_GROWING"
        << " timestamp=" << keyframe.timestamp
        << " growing_id=" << growing.id
        << " primary_id=" << submaps_[active_index_].id
        << " primary_count="
        << submaps_[active_index_].keyframe_ids.size()
        << " anchor_z="
        << growing.T_O_S_creation.translation().z()
        << std::endl;

    return true;
}

// ============================================================================
// Insert one Keyframe.
//
// CRITICAL INVARIANT:
//
//     old cloud_S points are NEVER transformed / rebuilt.
//
// Current Keyframe:
//
//     T_S_L = T_O_S_creation^-1 * T_O_L
//
// Point:
//
//     p_S = T_S_L * p_L
//
// We voxel-filter ONLY the incoming Keyframe before append.
// We never voxel-filter the complete accumulated Submap again, because doing
// so would move historical voxel centroids and violate immutable geometry.
// ============================================================================
bool SubmapManager::AddKeyframeToSubmap(
    Submap &submap,
    const Keyframe &keyframe)
{
    if (submap.state == SubmapState::Finished ||
        submap.finished ||
        !submap.has_origin_pose ||
        !submap.T_O_S_creation.matrix().allFinite() ||
        !keyframe.cloud ||
        keyframe.cloud->empty() ||
        !keyframe.T_WL.matrix().allFinite())
    {
        return false;
    }

    // Deterministic duplicate protection.
    if (!submap.keyframe_ids.empty() &&
        submap.keyframe_ids.back() ==
            keyframe.id)
    {
        return true;
    }

    const Eigen::Isometry3d T_S_O =
        submap.T_O_S_creation.inverse();

    const Eigen::Isometry3d T_S_L =
        T_S_O *
        keyframe.T_WL;

    if (!T_S_L.matrix().allFinite())
    {
        return false;
    }

    pcl::PointCloud<LIDAR_POINT>::Ptr frame_S =
        pcl::make_shared<
            pcl::PointCloud<LIDAR_POINT>>();

    frame_S->reserve(
        keyframe.cloud->size());

    for (const LIDAR_POINT &point_L :
         keyframe.cloud->points)
    {
        if (!std::isfinite(point_L.x) ||
            !std::isfinite(point_L.y) ||
            !std::isfinite(point_L.z))
        {
            continue;
        }

        const Eigen::Vector3d p_L(
            static_cast<double>(point_L.x),
            static_cast<double>(point_L.y),
            static_cast<double>(point_L.z));

        const Eigen::Vector3d p_S =
            T_S_L * p_L;

        if (!p_S.allFinite())
        {
            continue;
        }

        LIDAR_POINT point_S =
            point_L;

        point_S.x =
            static_cast<float>(p_S.x());

        point_S.y =
            static_cast<float>(p_S.y());

        point_S.z =
            static_cast<float>(p_S.z());

        frame_S->push_back(
            point_S);
    }

    if (frame_S->empty())
    {
        return false;
    }

    frame_S->width =
        static_cast<std::uint32_t>(
            frame_S->size());

    frame_S->height = 1;
    frame_S->is_dense = false;

    // ------------------------------------------------------------------------
    // Downsample only THIS newly inserted Keyframe.
    //
    // Historical Submap points remain untouched forever.
    // ------------------------------------------------------------------------
    pcl::PointCloud<LIDAR_POINT>::Ptr filtered_S =
        pcl::make_shared<
            pcl::PointCloud<LIDAR_POINT>>();

    pcl::VoxelGrid<LIDAR_POINT> voxel;

    voxel.setInputCloud(
        frame_S);

    voxel.setDownsampleAllData(
        false);

    voxel.setLeafSize(
        local_map_config_.voxel_leaf_size,
        local_map_config_.voxel_leaf_size,
        local_map_config_.voxel_leaf_size);

    voxel.filter(
        *filtered_S);

    if (!filtered_S ||
        filtered_S->empty())
    {
        return false;
    }

    // ========================================================
    // FR_LONG_SPAN_SAME_COUNT_V1
    //
    // Keep ALL Keyframes in Submap lifecycle, but only every
    // second local Keyframe contributes geometry.
    //
    // 60 lifecycle KFs -> approximately 30 geometry KFs.
    // ========================================================
    const std::size_t local_keyframe_index =
        submap.keyframe_ids.size();

    const bool keep_geometry =
        (local_keyframe_index % 2U) == 0U;

    // ------------------------------------------------------------------------
    // FR_MAP_KF_INSERT_V1
    //
    // Diagnostic only.
    //
    // Record the EXACT Keyframe pose that is about to be baked into this
    // fixed-frame Submap geometry. Old cloud_S points are never rebuilt, so
    // this is the pose-error inheritance event we want to analyze offline.
    // ------------------------------------------------------------------------
    std::cout
        << std::setprecision(17)
        << "FR_MAP_KF_INSERT_V1"
        << " | submap_id=" << submap.id
        << " | submap_state="
        << static_cast<int>(submap.state)
        << " | kf_id=" << keyframe.id
        << " | kf_t=" << keyframe.timestamp

        << " | anchor_xyz=["
        << submap.T_O_S_creation.translation().x() << " "
        << submap.T_O_S_creation.translation().y() << " "
        << submap.T_O_S_creation.translation().z() << "]"

        << " | kf_xyz=["
        << keyframe.T_WL.translation().x() << " "
        << keyframe.T_WL.translation().y() << " "
        << keyframe.T_WL.translation().z() << "]"

        << " | tsl_xyz=["
        << T_S_L.translation().x() << " "
        << T_S_L.translation().y() << " "
        << T_S_L.translation().z() << "]"

        << " | raw_points="
        << keyframe.cloud->size()

        << " | filtered_points="
        << filtered_S->size()

        << " | local_index="
        << local_keyframe_index

        << " | geometry_kept="
        << (keep_geometry ? 1 : 0)

        << " | submap_points_before="
        << submap.cloud_S->size()

        << std::endl;

    if (keep_geometry)
    {
        // ------------------------------------------------------------------------
        // Append source-of-truth S-frame geometry.
        // ------------------------------------------------------------------------
        submap.cloud_S->reserve(
            submap.cloud_S->size() +
            filtered_S->size());

        submap.cloud_O->reserve(
            submap.cloud_O->size() +
            filtered_S->size());

        for (const LIDAR_POINT &point_S :
             filtered_S->points)
        {
            submap.cloud_S->push_back(
                point_S);

            const Eigen::Vector3d p_S(
                static_cast<double>(point_S.x),
                static_cast<double>(point_S.y),
                static_cast<double>(point_S.z));

            const Eigen::Vector3d p_O =
                submap.T_O_S_creation *
                p_S;

            if (!p_O.allFinite())
            {
                return false;
            }

            LIDAR_POINT point_O =
                point_S;

            point_O.x =
                static_cast<float>(p_O.x());

            point_O.y =
                static_cast<float>(p_O.y());

            point_O.z =
                static_cast<float>(p_O.z());

            submap.cloud_O->push_back(
                point_O);
        }

        submap.cloud_S->width =
            static_cast<std::uint32_t>(
                submap.cloud_S->size());

        submap.cloud_S->height = 1;
        submap.cloud_S->is_dense = false;

        submap.cloud_O->width =
            static_cast<std::uint32_t>(
                submap.cloud_O->size());

        submap.cloud_O->height = 1;
        submap.cloud_O->is_dense = false;
    }

    submap.keyframe_ids.push_back(
        keyframe.id);

    submap.end_timestamp =
        keyframe.timestamp;

    return true;
}

// ============================================================================
// PRIMARY -> FINISHED
// GROWING -> PRIMARY
//
// No point cloud rebuild occurs here.
//
// cloud_S was already born in the correct Submap frame and therefore simply
// becomes fully immutable.
// ============================================================================
bool SubmapManager::FinishPrimaryAndPromoteGrowing()
{
    if (active_index_ == kInvalidIndex ||
        active_index_ >= submaps_.size() ||
        growing_index_ == kInvalidIndex ||
        growing_index_ >= submaps_.size())
    {
        return false;
    }

    const std::size_t finished_index =
        active_index_;

    Submap &finished =
        submaps_[finished_index];

    if (finished.state !=
            SubmapState::Primary ||
        finished.finished)
    {
        return false;
    }

    finished.state =
        SubmapState::Finished;

    finished.finished = true;

    finished.has_frozen_cloud = true;

    previous_index_ =
        finished_index;

    last_finished_submap_id_ =
        finished.id;

    active_index_ =
        growing_index_;

    growing_index_ =
        kInvalidIndex;

    Submap &new_primary =
        submaps_[active_index_];

    if (new_primary.finished ||
        new_primary.state !=
            SubmapState::Growing)
    {
        return false;
    }

    new_primary.state =
        SubmapState::Primary;

    last_add_started_new_submap_ =
        true;

    std::cout
        << "[SUBMAP_AUDIT] PROMOTE"
        << " finished_id=" << finished.id
        << " finished_count=" << finished.keyframe_ids.size()
        << " new_primary_id=" << new_primary.id
        << " new_primary_count=" << new_primary.keyframe_ids.size()
        << " finished_anchor_z="
        << finished.T_O_S_creation.translation().z()
        << " new_primary_anchor_z="
        << new_primary.T_O_S_creation.translation().z()
        << std::endl;

    return true;
}
