#pragma once

#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pcl/point_cloud.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/bilateral.h>

#include "fr_slam/fr_point_types.hpp"
#include "fr_slam/fr_lidar_frame.hpp"
#include "fr_slam/fr_imu_types.hpp"
#include "fr_slam/fr_lidar_deskewer.hpp"
#include "fr_slam/fr_PreprocessorConfig.hpp"

class PreProcessor
{
private:
    PreprocessorConfig config_;

    // Deskew is part of the LiDAR preprocessing pipeline.
    // The actual deskew mathematics remains inside LidarDeskewer.
    LidarDeskewer deskewer_;

public:
    virtual ~PreProcessor() = default;

    // ============================================================
    // Complete LiDAR preprocessing pipeline:
    //
    // Raw LiDAR
    //   -> Deskew
    //   -> Basic preprocess
    //   -> SOR (if enabled)
    //   -> ROR (if enabled)
    //   -> VoxelGrid (if enabled)
    //
    // imu_poses must cover the complete LiDAR scan time range.
    // ============================================================
    LIDAR_FRAME Process(
        const LIDAR_FRAME &lidar_frame,
        const std::vector<IMU_POSE> &imu_poses,
        bool use_translation = false);

    // ============================================================
    // Deskew only.
    //
    // This wrapper keeps LidarDeskewer inside PreProcessor while
    // preserving the original LidarDeskewer implementation.
    // ============================================================
    bool Deskew(
        const LIDAR_FRAME &lidar_frame,
        const std::vector<IMU_POSE> &imu_poses,
        LIDAR_FRAME &output_frame,
        bool use_translation = false);

    // T_IL:
    // LiDAR frame -> IMU frame
    void SetDeskewExtrinsic(
        const Eigen::Quaterniond &Q_IL,
        const Eigen::Vector3d &P_IL);

    // Existing preprocessing functions are kept so they can still
    // be tested independently.
    LIDAR_FRAME preprocess(
        const LIDAR_FRAME &lidar_frame);

    LIDAR_FRAME VoxelGrid(
        const LIDAR_FRAME &lidar_frame) const;

    LIDAR_FRAME SOR(
        const LIDAR_FRAME &lidar_frame) const;

    LIDAR_FRAME ROR(
        const LIDAR_FRAME &lidar_frame) const;
};