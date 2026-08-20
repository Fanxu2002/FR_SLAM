#pragma once

#include <cstddef>
#include <deque>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pcl/point_cloud.h>

#include "fr_slam/fr_point_types.hpp"

struct LocalMapConfig
{
    // Keep the most recent N accepted LiDAR frames in the local map.
    // At 10 Hz, 10 frames are about 1 second of motion.
    std::size_t max_frames = 10;

    // Voxel size used after merging all local-map frames.
    // Keep this close to the registration voxel size for the first version.
    float voxel_leaf_size = 0.30f;
};
/*
         新的一帧 Scan
              │
              │ + 当前位姿 T_WL
              ▼
       TransformToWorld
              │
              ▼
          Scan_k^W
              │
              ▼
        world_frames_
        ┌───────────┐
        │ Scan_1^W  │
        │ Scan_2^W  │
        │ Scan_3^W  │
        │ ...       │
        │ Scan_k^W  │
        └───────────┘
              │
              ▼
            Merge
              │
              ▼
         VoxelGrid
              │
              ▼
          local_map_
              │
              ▼
        给 Registration
          当 Target
*/
class LocalMap
{
public:
    explicit LocalMap(
        const LocalMapConfig &config =
            LocalMapConfig());

    // Add one accepted LiDAR frame to the local map.
    // cloud_lidar is expressed in the LiDAR frame.
    // T_WL maps LiDAR coordinates into world coordinates:
    //     p_W = T_WL * p_L
    bool AddFrame(
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &cloud_lidar,
        const Eigen::Isometry3d &T_WL);

    // Current local map, expressed in world coordinates.
    pcl::PointCloud<LIDAR_POINT>::ConstPtr GetMap() const;

    std::size_t FrameCount() const;

    std::size_t PointCount() const;

    void Reset();

private:
    pcl::PointCloud<LIDAR_POINT>::Ptr TransformToWorld(
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &cloud_lidar,
        const Eigen::Isometry3d &T_WL) const;

    bool RebuildLocalMap();

private:
    LocalMapConfig config_;

    // Each stored frame is already transformed into world coordinates.
    std::deque<pcl::PointCloud<LIDAR_POINT>::Ptr>
        world_frames_;

    pcl::PointCloud<LIDAR_POINT>::Ptr
        local_map_;
};
