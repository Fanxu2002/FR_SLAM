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

#include "fr_slam/common/point_types.hpp"
#include "fr_slam/common/lidar_frame.hpp"
#include "fr_slam/imu/imu_types.hpp"

// Reuse the already validated public preprocessing policy/timing types:
//     PreprocessorSorMode
//     PreprocessorTiming
//
// The LO PreProcessor implementation itself is NOT used by LioPreProcessor.
#include "fr_slam/lidar/lidar_preprocessor.hpp"

#include "fr_slam/lidar/lio_lidar_deskew.hpp"
#include "fr_slam/lidar/preprocessor_config.hpp"

// ============================================================================
// LIO-specific LiDAR preprocessing pipeline.
//
// IMPORTANT:
//
// LO:
//     PreProcessor
//         -> LidarDeskewer
//
// LIO:
//     LioPreProcessor
//         -> LioLidarDeskewer
//
// This class intentionally keeps the LIO deskew path independent from the
// validated legacy LO deskew implementation.
// ============================================================================

class LioPreProcessor
{
private:
    PreprocessorConfig config_;

    // LIO-specific deskewer.
    LioLidarDeskewer deskewer_;

    PreprocessorSorMode sor_mode_ =
        PreprocessorSorMode::OFF;

    std::size_t sor_adaptive_max_points_ =
        6000;

    bool process_enable_ror_ =
        false;

    bool ShouldRunSor(
        std::size_t after_voxel_points) const;

    PreprocessorTiming last_timing_;

public:
    virtual ~LioPreProcessor() = default;

    // ============================================================
    // Complete LIO preprocessing pipeline:
    //
    // Raw LiDAR
    //   -> LIO Deskew
    //   -> Basic preprocess
    //   -> VoxelGrid
    //   -> optional SOR
    //   -> optional ROR
    // ============================================================

    LIDAR_FRAME Process(
        const LIDAR_FRAME &lidar_frame,
        const std::vector<IMU_POSE> &imu_poses,
        bool use_translation = false);

    void SetConfig(
        const PreprocessorConfig &config);

    void SetOutlierFiltersEnabled(
        bool enable_sor,
        bool enable_ror);

    void SetOutlierFilterPolicy(
        PreprocessorSorMode sor_mode,
        bool enable_ror,
        std::size_t sor_adaptive_max_points = 6000);

    PreprocessorSorMode GetSorMode() const;

    std::size_t GetSorAdaptiveMaxPoints() const;

    const PreprocessorTiming &GetLastTiming() const;

    // ============================================================
    // LIO deskew only.
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

    LIDAR_FRAME preprocess(
        const LIDAR_FRAME &lidar_frame);

    LIDAR_FRAME VoxelGrid(
        const LIDAR_FRAME &lidar_frame) const;

    LIDAR_FRAME SOR(
        const LIDAR_FRAME &lidar_frame) const;

    LIDAR_FRAME ROR(
        const LIDAR_FRAME &lidar_frame) const;
};
