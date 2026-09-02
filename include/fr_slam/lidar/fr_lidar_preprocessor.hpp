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

#include "fr_slam/common/fr_point_types.hpp"
#include "fr_slam/common/fr_lidar_frame.hpp"
#include "fr_slam/imu/fr_imu_types.hpp"
#include "fr_slam/lidar/fr_lidar_deskew.hpp"
#include "fr_slam/lidar/fr_preprocessor_config.hpp"


enum class PreprocessorSorMode
{
    ALWAYS,
    OFF,
    ADAPTIVE
};

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

    bool sor_executed = false;
    bool ror_executed = false;
};

class PreProcessor
{
private:
    PreprocessorConfig config_;

    // Deskew is part of the LiDAR preprocessing pipeline.
    // The actual deskew mathematics remains inside LidarDeskewer.
    LidarDeskewer deskewer_;

    // Runtime outlier-filter policy used by the real-time node.
    //
    // ALWAYS:
    //     run SOR on every frame.
    //
    // OFF:
    //     skip SOR entirely.
    //
    // ADAPTIVE:
    //     run SOR only while the post-voxel cloud is not larger than
    //     sor_adaptive_max_points_.  Dense 360-degree scans therefore avoid
    //     the expensive StatisticalOutlierRemoval KNN pass and continue
    //     directly to the much cheaper ROR stage.
    PreprocessorSorMode sor_mode_ =
        PreprocessorSorMode::OFF;

    std::size_t sor_adaptive_max_points_ =
        6000;

    bool process_enable_ror_ = false;

    bool ShouldRunSor(
        std::size_t after_voxel_points) const;

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

    // Replace the complete preprocessing configuration.  Runtime SOR/ROR
    // policy can still be applied afterwards with SetOutlierFilterPolicy().
    void SetConfig(
        const PreprocessorConfig &config);

    // Legacy binary switch kept for compatibility.  enable_sor=true maps to
    // ALWAYS and enable_sor=false maps to OFF.
    void SetOutlierFiltersEnabled(
        bool enable_sor,
        bool enable_ror);

    // Preferred runtime policy for SOR performance experiments.
    void SetOutlierFilterPolicy(
        PreprocessorSorMode sor_mode,
        bool enable_ror,
        std::size_t sor_adaptive_max_points = 6000);

    PreprocessorSorMode GetSorMode() const;

    std::size_t GetSorAdaptiveMaxPoints() const;

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