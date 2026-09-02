#include "fr_slam/mapping/fr_local_map.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

#include <pcl/filters/voxel_grid.h>

LocalMap::LocalMap(
    const LocalMapConfig &config)
    : config_(config),
      local_map_(
          pcl::make_shared<
              pcl::PointCloud<LIDAR_POINT>>())
{
    if (config_.max_frames == 0)
    {
        config_.max_frames = 1;
    }

    if (!(config_.voxel_leaf_size > 0.0f))
    {
        config_.voxel_leaf_size = 0.30f;
    }
}

bool LocalMap::AddFrame(
    const pcl::PointCloud<LIDAR_POINT>::ConstPtr &cloud_lidar,
    const Eigen::Isometry3d &T_WL)
{
    if (!cloud_lidar ||
        cloud_lidar->empty() ||
        !T_WL.matrix().allFinite())
    {
        return false;
    }

    pcl::PointCloud<LIDAR_POINT>::Ptr world_cloud =
        TransformToWorld(
            cloud_lidar,
            T_WL);

    if (!world_cloud ||
        world_cloud->empty())
    {
        return false;
    }

    world_frames_.push_back(
        world_cloud);

    while (world_frames_.size() >
           config_.max_frames)
    {
        world_frames_.pop_front();
    }

    return RebuildLocalMap();
}

pcl::PointCloud<LIDAR_POINT>::ConstPtr
LocalMap::GetMap() const
{
    return local_map_;
}

std::size_t LocalMap::FrameCount() const
{
    return world_frames_.size();
}

std::size_t LocalMap::PointCount() const
{
    if (!local_map_)
    {
        return 0;
    }

    return local_map_->size();
}

void LocalMap::Reset()
{
    world_frames_.clear();

    local_map_ =
        pcl::make_shared<
            pcl::PointCloud<LIDAR_POINT>>();
}

pcl::PointCloud<LIDAR_POINT>::Ptr
LocalMap::TransformToWorld(
    const pcl::PointCloud<LIDAR_POINT>::ConstPtr &cloud_lidar,
    const Eigen::Isometry3d &T_WL) const
{
    pcl::PointCloud<LIDAR_POINT>::Ptr world_cloud =
        pcl::make_shared<
            pcl::PointCloud<LIDAR_POINT>>();

    world_cloud->reserve(
        cloud_lidar->size());

    for (const LIDAR_POINT &point_lidar :
         cloud_lidar->points)
    {
        const Eigen::Vector3d p_lidar(
            static_cast<double>(point_lidar.x),
            static_cast<double>(point_lidar.y),
            static_cast<double>(point_lidar.z));

        const Eigen::Vector3d p_world =
            T_WL * p_lidar;

        if (!p_world.allFinite())
        {
            continue;
        }

        LIDAR_POINT point_world =
            point_lidar;

        point_world.x =
            static_cast<float>(p_world.x());

        point_world.y =
            static_cast<float>(p_world.y());

        point_world.z =
            static_cast<float>(p_world.z());

        // ring/time_offset/intensity are preserved.  They are not used
        // by point-to-plane registration, but keeping them avoids losing
        // information in the unified point type.
        world_cloud->push_back(
            point_world);
    }

    world_cloud->width =
        static_cast<std::uint32_t>(
            world_cloud->size());

    world_cloud->height = 1;
    world_cloud->is_dense = false;

    return world_cloud;
}

bool LocalMap::RebuildLocalMap()
{
    if (world_frames_.empty())
    {
        local_map_->clear();
        return false;
    }

    pcl::PointCloud<LIDAR_POINT>::Ptr merged =
        pcl::make_shared<
            pcl::PointCloud<LIDAR_POINT>>();

    std::size_t total_points = 0;

    for (const auto &frame :
         world_frames_)
    {
        if (frame)
        {
            total_points +=
                frame->size();
        }
    }

    merged->reserve(
        total_points);

    for (const auto &frame :
         world_frames_)
    {
        if (!frame ||
            frame->empty())
        {
            continue;
        }

        *merged +=
            *frame;
    }

    if (merged->empty())
    {
        return false;
    }

    pcl::VoxelGrid<LIDAR_POINT> voxel_filter;

    voxel_filter.setInputCloud(
        merged);

    voxel_filter.setLeafSize(
        config_.voxel_leaf_size,
        config_.voxel_leaf_size,
        config_.voxel_leaf_size);

    pcl::PointCloud<LIDAR_POINT>::Ptr filtered =
        pcl::make_shared<
            pcl::PointCloud<LIDAR_POINT>>();

    voxel_filter.filter(
        *filtered);

    if (filtered->empty())
    {
        return false;
    }

    filtered->width =
        static_cast<std::uint32_t>(
            filtered->size());

    filtered->height = 1;
    filtered->is_dense = false;

    local_map_ =
        filtered;

    return true;
}
