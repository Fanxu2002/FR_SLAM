#pragma once

#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "fr_slam/common/lidar_frame.hpp"
#include "fr_slam/frontend/ieskf.hpp"
#include "fr_slam/frontend/lio_measurement.hpp"
#include "fr_slam/imu/imu_integrator.hpp"
#include "fr_slam/lidar/lio_preprocessor.hpp"
#include "fr_slam/lidar/lidar_registration.hpp"
#include "fr_slam/lidar/preprocessor_config.hpp"

// ============================================================================
// LIO frontend estimator.
//
// Final responsibility split:
//
//   LioFrontend:
//     IMU propagation
//     -> full-SE(3) deskew
//     -> LiDAR point-to-plane IESKF update
//     -> corrected navigation state / T_WL
//
//   RegistrationScan2LocalMap:
//     owns the tracking map / SubmapManager
//     owns KeyframeManager
//     owns backend / loop / PGO
//
// There is deliberately NO second LocalMap inside LioFrontend.
// The LiDAR update consumes the PreparedLidarTarget owned by the existing
// mapping frontend so tracking geometry has exactly one source of truth.
// ============================================================================

struct LioFrontendConfig
{
    IeskfConfig ieskf;
    LioMeasurementConfig lidar_measurement;
    PreprocessorConfig preprocessor;

    // Keep the same runtime outlier-filter policy used by the existing LO node.
    PreprocessorSorMode preprocessor_sor_mode =
        PreprocessorSorMode::OFF;

    std::size_t preprocessor_sor_adaptive_max_points =
        6000;

    bool preprocessor_enable_ror =
        false;

    double time_epsilon = 1.0e-6;
};

struct LioFrameDiagnostics
{
    bool valid = false;
    bool has_lidar_update = false;

    LioState before_propagation;
    LioState after_propagation;
    LioState after_lidar_update;

    // Selected covariance terms.
    double pred_var_z = 0.0;
    double pred_var_vz = 0.0;
    double pred_var_baz = 0.0;

    double post_var_z = 0.0;
    double post_var_vz = 0.0;
    double post_var_baz = 0.0;

    double pred_cov_z_vz = 0.0;
    double pred_cov_z_baz = 0.0;
    double pred_cov_vz_baz = 0.0;

    double post_cov_z_vz = 0.0;
    double post_cov_z_baz = 0.0;
    double post_cov_vz_baz = 0.0;
};

struct LioFrontendResult
{
    bool success = false;
    bool first_mapping_frame = false;

    LIDAR_FRAME processed_frame;

    LioState state;
    Eigen::Isometry3d T_WL =
        Eigen::Isometry3d::Identity();

    IeskfLidarUpdateResult lidar_update;
};

class LioFrontend
{
public:
    LioFrontend();

    explicit LioFrontend(
        const LioFrontendConfig &config);

    bool SetConfig(
        const LioFrontendConfig &config);

    const LioFrontendConfig &Config() const;

    bool Initialize(
        const LioState &initial_state);

    bool IsInitialized() const;

    const Ieskf &Filter() const;
    Ieskf &Filter();

    // ------------------------------------------------------------------------
    // Process one LiDAR scan.
    //
    // imu_data must cover:
    //
    //     [Filter().State().timestamp, scan_end]
    //
    // prepared_target:
    //     nullptr only for the very first mapping frame, before any map exists.
    //     Otherwise it must point to the tracking target owned by
    //     RegistrationScan2LocalMap.
    //
    // The persistent IESKF state is kept at scan_start. Scan-internal IMU
    // integration to scan_end is used only to build the deskew trajectory.
    // ------------------------------------------------------------------------
    bool ProcessFrame(
        const LIDAR_FRAME &raw_frame,
        const std::vector<IMU_DATA> &imu_data,
        const PreparedLidarTarget *prepared_target,
        LioFrontendResult &result);

private:
    bool PropagateFilterToTime(
        const std::vector<IMU_DATA> &imu_data,
        double target_time);

    bool BuildDeskewTrajectory(
        const std::vector<IMU_DATA> &imu_data,
        double scan_start_time,
        double scan_end_time,
        std::vector<IMU_POSE> &imu_poses);

    bool FindScanTimeRange(
        const LIDAR_FRAME &frame,
        double &scan_start_time,
        double &scan_end_time) const;

    static IMU_STATE ToImuState(
        const LioState &state);

    static Eigen::Isometry3d StateToLidarPose(
        const LioState &state);

private:
    LioFrontendConfig config_;

    Ieskf ieskf_;
    LioMeasurementBuilder measurement_builder_;

    ImuIntegrator imu_integrator_;
    LioPreProcessor preprocessor_;
};
