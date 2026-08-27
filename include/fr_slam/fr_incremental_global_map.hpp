#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>

#include "fr_slam/fr_keyframe.hpp"
#include "fr_slam/fr_point_types.hpp"

// ============================================================================
// IncrementalGlobalMap
//
// Backend-only global map cache.
//
// Design goal:
//     Keyframe cloud + Keyframe pose remain the source of truth.
//     The global point cloud is only a derived visualization / export product.
//
// To avoid transforming and voxelizing EVERY historical Keyframe after every
// update, Keyframes are grouped into small backend-only blocks:
//
//     block_id = keyframe_id / keyframes_per_block
//
// These blocks are deliberately NOT the frontend Submap objects.  The frontend
// Submap lifecycle remains dedicated to Scan-to-LocalMap tracking and loop
// geometry.  This keeps mapping decoupled from Active/Finished Submap state and
// follows the same high-level idea as keyframe-centric LiDAR-SAM backends.
//
// Layers:
//     raw       : immutable frontend Keyframe::T_WL
//     optimized : current PoseGraph T_WK
//     refined   : optimized blocks + small override blocks for accepted local
//                 refinement windows
//
// A changed pose marks only its Keyframe block dirty.  Dirty blocks are rebuilt
// from their local Keyframe clouds and voxelized independently.  The published
// global map is then assembled by concatenating already-voxelized blocks.
// ============================================================================
class IncrementalGlobalMap
{
public:
    struct UpdateStats
    {
        std::size_t keyframes_seen = 0;
        std::size_t dirty_keyframes = 0;
        std::size_t dirty_blocks = 0;
        std::size_t rebuilt_blocks = 0;
        std::size_t reused_blocks = 0;
        std::size_t total_blocks = 0;
        std::size_t map_points = 0;
    };

    explicit IncrementalGlobalMap(
        std::size_t keyframes_per_block = 10,
        float voxel_leaf_size = 0.30f,
        double pose_translation_dirty_threshold = 0.01,
        double pose_rotation_dirty_threshold_deg = 0.05)
        : keyframes_per_block_(
              std::max<std::size_t>(1, keyframes_per_block)),
          voxel_leaf_size_(voxel_leaf_size),
          pose_translation_dirty_threshold_(
              std::max(0.0, pose_translation_dirty_threshold)),
          pose_rotation_dirty_threshold_deg_(
              std::max(0.0, pose_rotation_dirty_threshold_deg))
    {
    }

    void Clear()
    {
        blocks_.clear();

        raw_pose_cache_.clear();
        raw_pose_valid_.clear();

        optimized_pose_cache_.clear();
        optimized_pose_valid_.clear();

        refined_pose_override_cache_.clear();
        refined_pose_override_valid_.clear();

        raw_map_->clear();
        optimized_map_->clear();
        refined_map_->clear();
    }

    std::size_t KeyframesPerBlock() const
    {
        return keyframes_per_block_;
    }

    float VoxelLeafSize() const
    {
        return voxel_leaf_size_;
    }

    std::size_t BlockCount() const
    {
        return blocks_.size();
    }

    pcl::PointCloud<LIDAR_POINT>::ConstPtr GetRawMap() const
    {
        return raw_map_;
    }

    pcl::PointCloud<LIDAR_POINT>::ConstPtr GetOptimizedMap() const
    {
        return optimized_map_;
    }

    pcl::PointCloud<LIDAR_POINT>::ConstPtr GetRefinedMap() const
    {
        return refined_map_;
    }

    // ------------------------------------------------------------------------
    // UpdateRaw()
    //
    // Raw frontend Keyframe poses are intended to be immutable.  Therefore in
    // normal operation only the block receiving a NEW Keyframe becomes dirty.
    // If an old raw pose ever changes unexpectedly, the same dirty detection
    // still rebuilds that block correctly.
    // ------------------------------------------------------------------------
    bool UpdateRaw(
        const std::vector<Keyframe> &keyframes,
        UpdateStats &stats)
    {
        std::vector<
            Eigen::Isometry3d,
            Eigen::aligned_allocator<Eigen::Isometry3d>> poses(
                keyframes.size(),
                Eigen::Isometry3d::Identity());

        std::vector<bool> valid(
            keyframes.size(),
            false);

        for (std::size_t i = 0;
             i < keyframes.size();
             ++i)
        {
            if (!keyframes[i].cloud ||
                keyframes[i].cloud->empty() ||
                !keyframes[i].T_WL.matrix().allFinite())
            {
                continue;
            }

            poses[i] = keyframes[i].T_WL;
            valid[i] = true;
        }

        return UpdateLayer(
            keyframes,
            poses,
            valid,
            Layer::RAW,
            raw_pose_cache_,
            raw_pose_valid_,
            stats);
    }

    // ------------------------------------------------------------------------
    // UpdateOptimized()
    //
    // When clear_refined_overrides is true (after a new main PoseGraph
    // optimization), all old refinement overrides are invalid because they
    // were computed from a previous graph snapshot.
    //
    // When false (ordinary new Keyframe), accepted refinement overrides remain
    // valid.  If the new Keyframe lands in an overridden block, that block is
    // rebuilt so the new Keyframe is included with its optimized pose.
    // ------------------------------------------------------------------------
    bool UpdateOptimized(
        const std::vector<Keyframe> &keyframes,
        const std::vector<
            Eigen::Isometry3d,
            Eigen::aligned_allocator<Eigen::Isometry3d>> &poses,
        const std::vector<bool> &valid,
        bool clear_refined_overrides,
        UpdateStats &stats)
    {
        if (clear_refined_overrides)
        {
            ClearRefinedOverrides();
        }

        std::unordered_set<std::size_t> dirty_blocks;

        if (!DetectDirtyBlocks(
                keyframes,
                poses,
                valid,
                optimized_pose_cache_,
                optimized_pose_valid_,
                dirty_blocks,
                stats))
        {
            return false;
        }

        if (!RebuildDirtyBlocks(
                keyframes,
                poses,
                valid,
                dirty_blocks,
                Layer::OPTIMIZED))
        {
            return false;
        }

        // Existing refined override blocks are based on the same optimized
        // graph snapshot during normal incremental growth.  If a new Keyframe
        // enters one of those blocks, rebuild only that override block using
        // the cached per-KF refinement overrides where available and optimized
        // poses everywhere else.
        if (!clear_refined_overrides &&
            !dirty_blocks.empty() &&
            HasAnyRefinedOverrides())
        {
            std::unordered_set<std::size_t> affected_refined_blocks;

            for (const std::size_t block_id : dirty_blocks)
            {
                const auto block_it =
                    blocks_.find(block_id);

                if (block_it != blocks_.end() &&
                    block_it->second.has_refined_override)
                {
                    affected_refined_blocks.insert(block_id);
                }
            }

            if (!affected_refined_blocks.empty())
            {
                std::vector<
                    Eigen::Isometry3d,
                    Eigen::aligned_allocator<Eigen::Isometry3d>>
                    refined_poses = poses;

                std::vector<bool>
                    refined_valid = valid;

                ApplyCachedRefinedOverrides(
                    keyframes,
                    refined_poses,
                    refined_valid);

                if (!RebuildDirtyBlocks(
                        keyframes,
                        refined_poses,
                        refined_valid,
                        affected_refined_blocks,
                        Layer::REFINED_OVERRIDE))
                {
                    return false;
                }
            }
        }

        BuildPublishedMap(
            Layer::OPTIMIZED);

        BuildPublishedMap(
            Layer::REFINED_OVERRIDE);

        stats.dirty_blocks = dirty_blocks.size();
        stats.rebuilt_blocks = dirty_blocks.size();
        stats.total_blocks = CountLayerBlocks(Layer::OPTIMIZED);
        stats.reused_blocks =
            stats.total_blocks >= stats.rebuilt_blocks
                ? stats.total_blocks - stats.rebuilt_blocks
                : 0;
        stats.map_points = optimized_map_->size();

        return !optimized_map_->empty();
    }

    // ------------------------------------------------------------------------
    // UpdateRefinedOverrides()
    //
    // The optimized layer is the base map.  Only blocks containing accepted
    // refined Keyframes are rebuilt as overrides.  All other blocks are reused
    // directly from the optimized layer.
    //
    // This means a local refinement affecting 15 Keyframes typically rebuilds
    // only 1-3 blocks rather than the entire global map.
    // ------------------------------------------------------------------------
    bool UpdateRefinedOverrides(
        const std::vector<Keyframe> &keyframes,
        const std::vector<
            Eigen::Isometry3d,
            Eigen::aligned_allocator<Eigen::Isometry3d>> &final_poses,
        const std::vector<bool> &adjusted,
        UpdateStats &stats)
    {
        ClearRefinedOverrides();

        stats = UpdateStats();
        stats.keyframes_seen = keyframes.size();

        if (keyframes.size() != final_poses.size() ||
            keyframes.size() != adjusted.size())
        {
            return false;
        }

        EnsurePoseCacheSize(
            keyframes,
            refined_pose_override_cache_,
            refined_pose_override_valid_);

        std::unordered_set<std::size_t> dirty_blocks;

        for (std::size_t i = 0;
             i < keyframes.size();
             ++i)
        {
            if (!adjusted[i] ||
                !keyframes[i].cloud ||
                keyframes[i].cloud->empty() ||
                !final_poses[i].matrix().allFinite())
            {
                continue;
            }

            const std::size_t keyframe_id =
                keyframes[i].id;

            if (keyframe_id >= refined_pose_override_cache_.size())
            {
                continue;
            }

            refined_pose_override_cache_[keyframe_id] =
                final_poses[i];

            refined_pose_override_valid_[keyframe_id] =
                true;

            dirty_blocks.insert(
                BlockId(keyframe_id));

            ++stats.dirty_keyframes;
        }

        // No accepted refinement: /refined_map should be exactly the current
        // optimized block map.  This is the expected fallback behavior.
        if (dirty_blocks.empty())
        {
            BuildPublishedMap(
                Layer::REFINED_OVERRIDE);

            stats.total_blocks = CountLayerBlocks(Layer::OPTIMIZED);
            stats.reused_blocks = stats.total_blocks;
            stats.map_points = refined_map_->size();

            return !refined_map_->empty();
        }

        std::vector<
            Eigen::Isometry3d,
            Eigen::aligned_allocator<Eigen::Isometry3d>>
            refined_poses = final_poses;

        std::vector<bool>
            refined_valid(
                keyframes.size(),
                false);

        // final_poses already contains optimized poses for unadjusted KFs in
        // the caller.  Validate all available poses so each rebuilt override
        // block remains complete.
        for (std::size_t i = 0;
             i < keyframes.size();
             ++i)
        {
            refined_valid[i] =
                keyframes[i].cloud &&
                !keyframes[i].cloud->empty() &&
                final_poses[i].matrix().allFinite();
        }

        if (!RebuildDirtyBlocks(
                keyframes,
                refined_poses,
                refined_valid,
                dirty_blocks,
                Layer::REFINED_OVERRIDE))
        {
            return false;
        }

        BuildPublishedMap(
            Layer::REFINED_OVERRIDE);

        stats.dirty_blocks = dirty_blocks.size();
        stats.rebuilt_blocks = dirty_blocks.size();
        stats.total_blocks = CountLayerBlocks(Layer::OPTIMIZED);
        stats.reused_blocks =
            stats.total_blocks >= stats.rebuilt_blocks
                ? stats.total_blocks - stats.rebuilt_blocks
                : 0;
        stats.map_points = refined_map_->size();

        return !refined_map_->empty();
    }

private:
    enum class Layer
    {
        RAW,
        OPTIMIZED,
        REFINED_OVERRIDE
    };

    struct Block
    {
        pcl::PointCloud<LIDAR_POINT>::Ptr raw =
            pcl::make_shared<pcl::PointCloud<LIDAR_POINT>>();

        pcl::PointCloud<LIDAR_POINT>::Ptr optimized =
            pcl::make_shared<pcl::PointCloud<LIDAR_POINT>>();

        pcl::PointCloud<LIDAR_POINT>::Ptr refined_override =
            pcl::make_shared<pcl::PointCloud<LIDAR_POINT>>();

        bool has_raw = false;
        bool has_optimized = false;
        bool has_refined_override = false;
    };

    std::size_t BlockId(
        std::size_t keyframe_id) const
    {
        return keyframe_id /
               keyframes_per_block_;
    }

    static double RotationDifferenceDeg(
        const Eigen::Isometry3d &T_A,
        const Eigen::Isometry3d &T_B)
    {
        if (!T_A.matrix().allFinite() ||
            !T_B.matrix().allFinite())
        {
            return std::numeric_limits<double>::infinity();
        }

        const Eigen::Matrix3d R_AB =
            T_A.rotation().transpose() *
            T_B.rotation();

        Eigen::Quaterniond q_AB(R_AB);

        if (!q_AB.coeffs().allFinite() ||
            q_AB.norm() < 1.0e-12)
        {
            return std::numeric_limits<double>::infinity();
        }

        q_AB.normalize();

        const double w =
            std::clamp(
                std::abs(q_AB.w()),
                0.0,
                1.0);

        return 2.0 *
               std::acos(w) *
               180.0 /
               3.14159265358979323846;
    }

    bool PoseChanged(
        const Eigen::Isometry3d &old_pose,
        const Eigen::Isometry3d &new_pose) const
    {
        if (!old_pose.matrix().allFinite() ||
            !new_pose.matrix().allFinite())
        {
            return true;
        }

        const double translation_change =
            (old_pose.translation() -
             new_pose.translation())
                .norm();

        const double rotation_change_deg =
            RotationDifferenceDeg(
                old_pose,
                new_pose);

        return translation_change >
                   pose_translation_dirty_threshold_ ||
               rotation_change_deg >
                   pose_rotation_dirty_threshold_deg_;
    }

    void EnsurePoseCacheSize(
        const std::vector<Keyframe> &keyframes,
        std::vector<
            Eigen::Isometry3d,
            Eigen::aligned_allocator<Eigen::Isometry3d>> &pose_cache,
        std::vector<bool> &valid_cache) const
    {
        std::size_t required_size = 0;

        for (const Keyframe &keyframe : keyframes)
        {
            required_size =
                std::max(
                    required_size,
                    keyframe.id + 1);
        }

        if (pose_cache.size() < required_size)
        {
            pose_cache.resize(
                required_size,
                Eigen::Isometry3d::Identity());
        }

        if (valid_cache.size() < required_size)
        {
            valid_cache.resize(
                required_size,
                false);
        }
    }

    bool DetectDirtyBlocks(
        const std::vector<Keyframe> &keyframes,
        const std::vector<
            Eigen::Isometry3d,
            Eigen::aligned_allocator<Eigen::Isometry3d>> &poses,
        const std::vector<bool> &valid,
        std::vector<
            Eigen::Isometry3d,
            Eigen::aligned_allocator<Eigen::Isometry3d>> &pose_cache,
        std::vector<bool> &valid_cache,
        std::unordered_set<std::size_t> &dirty_blocks,
        UpdateStats &stats)
    {
        stats = UpdateStats();
        stats.keyframes_seen = keyframes.size();

        if (keyframes.size() != poses.size() ||
            keyframes.size() != valid.size())
        {
            return false;
        }

        EnsurePoseCacheSize(
            keyframes,
            pose_cache,
            valid_cache);

        for (std::size_t i = 0;
             i < keyframes.size();
             ++i)
        {
            if (!valid[i] ||
                !keyframes[i].cloud ||
                keyframes[i].cloud->empty() ||
                !poses[i].matrix().allFinite())
            {
                continue;
            }

            const std::size_t keyframe_id =
                keyframes[i].id;

            bool dirty = false;

            if (keyframe_id >= valid_cache.size() ||
                !valid_cache[keyframe_id])
            {
                dirty = true;
            }
            else if (PoseChanged(
                         pose_cache[keyframe_id],
                         poses[i]))
            {
                dirty = true;
            }

            if (!dirty)
            {
                continue;
            }

            pose_cache[keyframe_id] =
                poses[i];

            valid_cache[keyframe_id] =
                true;

            dirty_blocks.insert(
                BlockId(keyframe_id));

            ++stats.dirty_keyframes;
        }

        return true;
    }

    bool UpdateLayer(
        const std::vector<Keyframe> &keyframes,
        const std::vector<
            Eigen::Isometry3d,
            Eigen::aligned_allocator<Eigen::Isometry3d>> &poses,
        const std::vector<bool> &valid,
        Layer layer,
        std::vector<
            Eigen::Isometry3d,
            Eigen::aligned_allocator<Eigen::Isometry3d>> &pose_cache,
        std::vector<bool> &valid_cache,
        UpdateStats &stats)
    {
        std::unordered_set<std::size_t> dirty_blocks;

        if (!DetectDirtyBlocks(
                keyframes,
                poses,
                valid,
                pose_cache,
                valid_cache,
                dirty_blocks,
                stats))
        {
            return false;
        }

        if (!RebuildDirtyBlocks(
                keyframes,
                poses,
                valid,
                dirty_blocks,
                layer))
        {
            return false;
        }

        BuildPublishedMap(layer);

        stats.dirty_blocks = dirty_blocks.size();
        stats.rebuilt_blocks = dirty_blocks.size();
        stats.total_blocks = CountLayerBlocks(layer);
        stats.reused_blocks =
            stats.total_blocks >= stats.rebuilt_blocks
                ? stats.total_blocks - stats.rebuilt_blocks
                : 0;

        if (layer == Layer::RAW)
        {
            stats.map_points = raw_map_->size();
        }
        else if (layer == Layer::OPTIMIZED)
        {
            stats.map_points = optimized_map_->size();
        }
        else
        {
            stats.map_points = refined_map_->size();
        }

        return true;
    }

    bool RebuildDirtyBlocks(
        const std::vector<Keyframe> &keyframes,
        const std::vector<
            Eigen::Isometry3d,
            Eigen::aligned_allocator<Eigen::Isometry3d>> &poses,
        const std::vector<bool> &valid,
        const std::unordered_set<std::size_t> &dirty_blocks,
        Layer layer)
    {
        if (dirty_blocks.empty())
        {
            return true;
        }

        if (keyframes.size() != poses.size() ||
            keyframes.size() != valid.size())
        {
            return false;
        }

        std::unordered_map<
            std::size_t,
            pcl::PointCloud<LIDAR_POINT>::Ptr>
            accumulators;

        for (const std::size_t block_id : dirty_blocks)
        {
            accumulators[block_id] =
                pcl::make_shared<pcl::PointCloud<LIDAR_POINT>>();
        }

        for (std::size_t i = 0;
             i < keyframes.size();
             ++i)
        {
            if (!valid[i] ||
                !keyframes[i].cloud ||
                keyframes[i].cloud->empty() ||
                !poses[i].matrix().allFinite())
            {
                continue;
            }

            const std::size_t block_id =
                BlockId(keyframes[i].id);

            const auto accumulator_it =
                accumulators.find(block_id);

            if (accumulator_it ==
                accumulators.end())
            {
                continue;
            }

            pcl::PointCloud<LIDAR_POINT>
                transformed_cloud;

            const Eigen::Matrix4f T_float =
                poses[i].matrix().cast<float>();

            pcl::transformPointCloud(
                *keyframes[i].cloud,
                transformed_cloud,
                T_float);

            *(accumulator_it->second) +=
                transformed_cloud;
        }

        for (const auto &entry : accumulators)
        {
            const std::size_t block_id =
                entry.first;

            const pcl::PointCloud<LIDAR_POINT>::Ptr &accumulated =
                entry.second;

            Block &block =
                blocks_[block_id];

            pcl::PointCloud<LIDAR_POINT>::Ptr filtered =
                pcl::make_shared<pcl::PointCloud<LIDAR_POINT>>();

            if (accumulated &&
                !accumulated->empty())
            {
                pcl::VoxelGrid<LIDAR_POINT> voxel;

                voxel.setLeafSize(
                    voxel_leaf_size_,
                    voxel_leaf_size_,
                    voxel_leaf_size_);

                voxel.setInputCloud(
                    accumulated);

                voxel.filter(
                    *filtered);
            }

            if (layer == Layer::RAW)
            {
                block.raw = filtered;
                block.has_raw = !filtered->empty();
            }
            else if (layer == Layer::OPTIMIZED)
            {
                block.optimized = filtered;
                block.has_optimized = !filtered->empty();
            }
            else
            {
                block.refined_override = filtered;
                block.has_refined_override = !filtered->empty();
            }
        }

        return true;
    }

    void BuildPublishedMap(
        Layer layer)
    {
        std::vector<std::size_t> block_ids;
        block_ids.reserve(blocks_.size());

        for (const auto &entry : blocks_)
        {
            block_ids.push_back(entry.first);
        }

        std::sort(
            block_ids.begin(),
            block_ids.end());

        pcl::PointCloud<LIDAR_POINT>::Ptr output =
            pcl::make_shared<pcl::PointCloud<LIDAR_POINT>>();

        std::size_t reserve_points = 0;

        for (const std::size_t block_id : block_ids)
        {
            const auto block_it =
                blocks_.find(block_id);

            if (block_it == blocks_.end())
            {
                continue;
            }

            const Block &block =
                block_it->second;

            if (layer == Layer::RAW)
            {
                if (block.has_raw && block.raw)
                {
                    reserve_points += block.raw->size();
                }
            }
            else if (layer == Layer::OPTIMIZED)
            {
                if (block.has_optimized && block.optimized)
                {
                    reserve_points += block.optimized->size();
                }
            }
            else
            {
                if (block.has_refined_override &&
                    block.refined_override)
                {
                    reserve_points +=
                        block.refined_override->size();
                }
                else if (block.has_optimized &&
                         block.optimized)
                {
                    reserve_points +=
                        block.optimized->size();
                }
            }
        }

        output->reserve(reserve_points);

        for (const std::size_t block_id : block_ids)
        {
            const auto block_it =
                blocks_.find(block_id);

            if (block_it == blocks_.end())
            {
                continue;
            }

            const Block &block =
                block_it->second;

            if (layer == Layer::RAW)
            {
                if (block.has_raw && block.raw)
                {
                    *output += *block.raw;
                }
            }
            else if (layer == Layer::OPTIMIZED)
            {
                if (block.has_optimized && block.optimized)
                {
                    *output += *block.optimized;
                }
            }
            else
            {
                if (block.has_refined_override &&
                    block.refined_override)
                {
                    *output +=
                        *block.refined_override;
                }
                else if (block.has_optimized &&
                         block.optimized)
                {
                    *output +=
                        *block.optimized;
                }
            }
        }

        output->width =
            static_cast<std::uint32_t>(output->size());

        output->height = 1;
        output->is_dense = false;

        if (layer == Layer::RAW)
        {
            raw_map_ = output;
        }
        else if (layer == Layer::OPTIMIZED)
        {
            optimized_map_ = output;
        }
        else
        {
            refined_map_ = output;
        }
    }

    std::size_t CountLayerBlocks(
        Layer layer) const
    {
        std::size_t count = 0;

        for (const auto &entry : blocks_)
        {
            const Block &block =
                entry.second;

            if (layer == Layer::RAW &&
                block.has_raw)
            {
                ++count;
            }
            else if (layer == Layer::OPTIMIZED &&
                     block.has_optimized)
            {
                ++count;
            }
            else if (layer == Layer::REFINED_OVERRIDE &&
                     (block.has_refined_override ||
                      block.has_optimized))
            {
                ++count;
            }
        }

        return count;
    }

    bool HasAnyRefinedOverrides() const
    {
        for (const bool valid :
             refined_pose_override_valid_)
        {
            if (valid)
            {
                return true;
            }
        }

        return false;
    }

    void ClearRefinedOverrides()
    {
        refined_pose_override_cache_.clear();
        refined_pose_override_valid_.clear();

        for (auto &entry : blocks_)
        {
            entry.second.refined_override->clear();
            entry.second.has_refined_override = false;
        }

        BuildPublishedMap(
            Layer::REFINED_OVERRIDE);
    }

    void ApplyCachedRefinedOverrides(
        const std::vector<Keyframe> &keyframes,
        std::vector<
            Eigen::Isometry3d,
            Eigen::aligned_allocator<Eigen::Isometry3d>> &poses,
        std::vector<bool> &valid) const
    {
        if (keyframes.size() != poses.size() ||
            keyframes.size() != valid.size())
        {
            return;
        }

        for (std::size_t i = 0;
             i < keyframes.size();
             ++i)
        {
            const std::size_t keyframe_id =
                keyframes[i].id;

            if (keyframe_id >=
                    refined_pose_override_valid_.size() ||
                !refined_pose_override_valid_[keyframe_id] ||
                keyframe_id >=
                    refined_pose_override_cache_.size())
            {
                continue;
            }

            poses[i] =
                refined_pose_override_cache_[keyframe_id];

            valid[i] =
                poses[i].matrix().allFinite();
        }
    }

private:
    std::size_t keyframes_per_block_ = 10;
    float voxel_leaf_size_ = 0.30f;

    double pose_translation_dirty_threshold_ = 0.01;
    double pose_rotation_dirty_threshold_deg_ = 0.05;

    std::unordered_map<std::size_t, Block>
        blocks_;

    std::vector<
        Eigen::Isometry3d,
        Eigen::aligned_allocator<Eigen::Isometry3d>>
        raw_pose_cache_;

    std::vector<bool>
        raw_pose_valid_;

    std::vector<
        Eigen::Isometry3d,
        Eigen::aligned_allocator<Eigen::Isometry3d>>
        optimized_pose_cache_;

    std::vector<bool>
        optimized_pose_valid_;

    std::vector<
        Eigen::Isometry3d,
        Eigen::aligned_allocator<Eigen::Isometry3d>>
        refined_pose_override_cache_;

    std::vector<bool>
        refined_pose_override_valid_;

    pcl::PointCloud<LIDAR_POINT>::Ptr raw_map_ =
        pcl::make_shared<pcl::PointCloud<LIDAR_POINT>>();

    pcl::PointCloud<LIDAR_POINT>::Ptr optimized_map_ =
        pcl::make_shared<pcl::PointCloud<LIDAR_POINT>>();

    pcl::PointCloud<LIDAR_POINT>::Ptr refined_map_ =
        pcl::make_shared<pcl::PointCloud<LIDAR_POINT>>();
};
