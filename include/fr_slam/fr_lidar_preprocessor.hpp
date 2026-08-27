#pragma once

#include <cstddef>
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


struct PreprocessorTiming
{
    double deskew_ms = 0.0;
    double basic_ms = 0.0;
    double voxel_ms = 0.0;
    double sor_ms = 0.0;
    double ror_ms = 0.0;
    double total_ms = 0.0;

    std::size_t input_points = 0;
    std::size_t after_basic_points = 0;
    std::size_t after_voxel_points = 0;
    std::size_t after_sor_points = 0;
    std::size_t after_ror_points = 0;
};

class PreProcessor
{
private:
    PreprocessorConfig config_;

    // Deskew is part of the LiDAR preprocessing pipeline.
    // The actual deskew mathematics remains inside LidarDeskewer.
    LidarDeskewer deskewer_;

    // Runtime gates used by the real-time node.
    // They allow expensive outlier filters to be disabled without changing
    // the existing PreprocessorConfig file.
    bool process_enable_sor_ = false;
    bool process_enable_ror_ = false;

    PreprocessorTiming last_timing_;

public:
    virtual ~PreProcessor() = default;

    // ============================================================
    // Complete LiDAR preprocessing pipeline:
    //
    // Raw LiDAR
    //   -> Deskew
    //   -> Basic preprocess
    //   -> VoxelGrid (if enabled)
    //   -> SOR (optional, after voxel)
    //   -> ROR (optional, after voxel)
    //
    // imu_poses must cover the complete LiDAR scan time range.
    // ============================================================
    LIDAR_FRAME Process(
        const LIDAR_FRAME &lidar_frame,
        const std::vector<IMU_POSE> &imu_poses,
        bool use_translation = false);

    // Enable/disable the expensive SOR/ROR stages used by Process().
    // The underlying SOR()/ROR() functions still keep their original
    // PreprocessorConfig checks.
    void SetOutlierFiltersEnabled(
        bool enable_sor,
        bool enable_ror);

    const PreprocessorTiming &GetLastTiming() const;

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