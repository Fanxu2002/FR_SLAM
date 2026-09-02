#pragma once

#include <cstddef>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pcl/point_cloud.h>

#include "fr_slam/fr_point_types.hpp"

// ============================================================================
// A SLAM keyframe.
//
// Coordinate convention:
//
//     T_WL
//
// means:
//
//     LiDAR -> World
//
// The point cloud stored in a Keyframe stays in the ORIGINAL LiDAR frame.
// It is NOT permanently transformed into the World frame.
//
// Therefore:
//
//     p_W = T_WL * p_L
//
// Keeping cloud + pose separately is important for future pose-graph
// optimization, loop closure, map correction, and submap rebuilding.
// ============================================================================
struct Keyframe
{
    // Unique keyframe ID:
    //
    //     KF0, KF1, KF2, ...
    std::size_t id = 0;

    // LiDAR scan start timestamp, unit: second.
    double timestamp = 0.0;

    // Global LiDAR pose: LiDAR -> World.
    Eigen::Isometry3d T_WL = Eigen::Isometry3d::Identity();

    // Keyframe point cloud in LiDAR coordinates.
    pcl::PointCloud<LIDAR_POINT>::Ptr cloud =
        pcl::make_shared<pcl::PointCloud<LIDAR_POINT>>();

    // ========================================================================
    // Dynamic odometry information from frontend V2B Hessian.
    //
    // Order:
    //     [tx ty tz rx ry rz]
    //
    // Frame:
    //     current Keyframe LiDAR frame
    // ========================================================================
    bool has_odom_information = false;

    Eigen::Matrix<double, 6, 6> odom_information =
        Eigen::Matrix<double, 6, 6>::Identity();
};
