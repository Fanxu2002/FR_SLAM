#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>

#include "fr_slam/lidar/fr_lidar_registration_config.hpp"
#include "fr_slam/common/fr_point_types.hpp"

// ============================================================================
// Prepared target frame for point-to-plane registration.
// Compatible with Robust V1 + Degeneracy V2A/V2B.
//
// The whole registration pipeline uses LIDAR_POINT directly:
// - cloud   : original target cloud
// - kdtree  : nearest-neighbor search on the same LIDAR_POINT cloud
// - planes  : precomputed local plane for each target point
//
// KDTree indices directly index cloud and planes because they all share
// exactly the same point order.
// ============================================================================
struct PreparedLidarTarget
{
    pcl::PointCloud<LIDAR_POINT>::ConstPtr cloud;

    std::shared_ptr<
        pcl::KdTreeFLANN<LIDAR_POINT>>
        kdtree;

    std::vector<TargetPlane> planes;

    std::size_t valid_planes = 0;
    std::size_t invalid_planes = 0;

    bool ready = false;
};

class LidarRegistration
{
public:
    explicit LidarRegistration(
        const LidarRegistrationConfig &config =
            LidarRegistrationConfig());

    bool PrepareTarget(
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &target,
        PreparedLidarTarget &prepared_target) const;
    /*
    PreparedLidarTarget
    │
    ├── cloud
    │    └── target 原始点云
    │
    ├── kdtree
    │    └── 在 target 内部做近邻搜索
    │
    ├── planes
    │    └── 每个 target 点对应的局部平面
    │
    ├── valid_planes
    │    └── 成功拟合的平面数量
    │
    ├── invalid_planes
    │    └── 拟合失败/不满足条件的数量
    │
    └── ready
        └── 整个 target 是否已经准备完成
    */

    // Fast path: register source against an already prepared target.
    bool Align(
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &source,
        const PreparedLidarTarget &target,
        const Eigen::Isometry3d &initial_guess,
        LidarRegistrationResult &result) const;

    // Compatibility overload: prepare target first, then align.
    bool Align(
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &source,
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &target,
        const Eigen::Isometry3d &initial_guess,
        LidarRegistrationResult &result) const;

private:
    bool FitLocalPlane(
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &target,
        const std::vector<int> &neighbor_indices,
        Eigen::Vector3d &plane_point,
        Eigen::Vector3d &plane_normal) const;

private:
    LidarRegistrationConfig config_;
};