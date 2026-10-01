#pragma once

#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "fr_slam/common/lidar_frame.hpp"
#include "fr_slam/frontend/ieskf.hpp"
#include "fr_slam/frontend/wall_association.hpp"
#include "fr_slam/frontend/lio_measurement.hpp"
#include "fr_slam/imu/imu_integrator.hpp"
#include "fr_slam/lidar/lio_preprocessor.hpp"
#include "fr_slam/lidar/lidar_registration.hpp"
#include "fr_slam/lidar/preprocessor_config.hpp"
#include "fr_slam/frontend/ground_constraint_config.hpp"
#include "fr_slam/frontend/ground_segmenter.hpp"

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
    GroundConstraintConfig ground;

    // Wall association + temporal filtering + pose injection.
    // Disabled by default; must be explicitly enabled.
    bool wall_constraint_enable = false;

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

    double pred_var_g1 = 0.0;
    double pred_var_g2 = 0.0;
    double post_var_g1 = 0.0;
    double post_var_g2 = 0.0;
};

struct LioSubmapContext
{
    bool valid = false;

    std::size_t primary_submap_id =
        std::numeric_limits<std::size_t>::max();

    Eigen::Isometry3d T_O_S_creation =
        Eigen::Isometry3d::Identity();
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

    fr_slam::WallAssociationResult wall_association;
    LioFrameDiagnostics diagnostics;
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
        const LioSubmapContext &submap_context,
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

    bool ProcessGroundMeasurement(
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &ground_cloud_L,
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &registration_cloud_L,
        const PreparedLidarTarget *prepared_target,
        bool allow_pose_correction);

    bool ProcessWallMeasurement(
        const fr_slam::WallAssociationResult &wall_association,
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &registration_cloud_L,
        const PreparedLidarTarget *prepared_target,
        const LioSubmapContext &submap_context);

    void ResetGroundRuntime();

    static IMU_STATE ToImuState(
        const LioState &state);

    static Eigen::Isometry3d StateToLidarPose(
        const LioState &state);

private:
    LioFrontendConfig config_;

    Ieskf ieskf_;
    LioMeasurementBuilder measurement_builder_;

    // Final V1 Wall Association.
    //
    // Lifetime is tied to the current PRIMARY Submap.
    fr_slam::WallAssociation wall_association_;

    // ========================================================================
    // FR_WALL_WORLD_SHADOW_V4
    //
    // Diagnostic-only trajectory-global Wall association.
    //
    // IMPORTANT:
    //   - receives T_OL (frontend world/odom frame), NOT T_SL;
    //   - is NEVER reset on PRIMARY Submap changes;
    //   - contributes NOTHING to IESKF;
    //   - exists only to verify cross-submap persistent_wall_id continuity.
    // ========================================================================
    fr_slam::WallAssociation wall_world_shadow_association_;

    std::size_t wall_world_shadow_frame_index_ = 0;


    // ========================================================================
    // FR_MULTI_FAMILY_WALL_HEADING_V1
    //
    // Trajectory-global frozen structural heading families.
    // New families are anchored by an already-frozen family.
    // ========================================================================
    std::vector<Eigen::Vector3d>
        wall_heading_family_heading_O_;

    std::vector<Eigen::Vector3d>
        wall_heading_family_bootstrap_sum_O_;

    std::vector<std::size_t>
        wall_heading_family_bootstrap_count_;

    std::vector<bool>
        wall_heading_family_frozen_;

    std::vector<std::size_t>
        wall_heading_family_source_id_;

    std::vector<std::size_t>
        wall_heading_family_last_bootstrap_frame_;

    std::size_t wall_association_submap_id_ =
        std::numeric_limits<std::size_t>::max();

    std::size_t wall_association_frame_index_ = 0;

    // ------------------------------------------------------------------------
    // Wall temporal injection filter.
    // ------------------------------------------------------------------------
    bool wall_temporal_initialized_ = false;

    std::size_t wall_consistent_frames_ = 0;

    std::size_t wall_temporal_last_frame_index_ = 0;

    Eigen::Vector3d wall_previous_delta_t_S_ =
        Eigen::Vector3d::Zero();

    double wall_previous_yaw_rad_ = 0.0;

    Eigen::Vector3d wall_filtered_delta_t_S_ =
        Eigen::Vector3d::Zero();

    double wall_filtered_yaw_rad_ = 0.0;

    // Reuse the Ground segmentation already computed for this frame.
    fr_slam::GroundSegmentationResult
        last_ground_segmentation_result_;

    bool last_ground_segmentation_valid_ = false;

    ImuIntegrator imu_integrator_;
    LioPreProcessor preprocessor_;

    fr_slam::GroundSegmenter ground_segmenter_;

    bool ground_reference_frozen_ = false;

    std::vector<Eigen::Vector3d>
        ground_bootstrap_normals_W_;

    std::vector<double>
        ground_bootstrap_plane_d_W_;

    Eigen::Vector3d ground_reference_normal_W_ =
        Eigen::Vector3d::UnitZ();

    double ground_reference_plane_d_W_ = 0.0;
    // ========================================================================
    // FR-SLAM Continuous Piecewise Frozen Ground
    //
    // Mapping Submap lifecycle and Ground Piece lifecycle are independent.
    // ========================================================================

    // Mapping-submap ID is only a hint for diagnostics/pending reset.
    std::size_t ground_last_mapping_submap_id_ =
        static_cast<std::size_t>(-1);

    // Persistent ACTIVE Ground piece.
    bool ground_active_anchor_valid_ =
        false;

    Eigen::Vector3d ground_active_anchor_W_ =
        Eigen::Vector3d::Zero();

    std::size_t ground_active_piece_id_ =
        0;

    // Physical slope of ACTIVE Ground relative to gravity.
    //
    // This is used ONLY for terrain-transition detection.
    // Height residual is NEVER allowed to trigger a Ground switch.
    bool ground_active_slope_valid_ =
        false;

    double ground_active_slope_deg_ =
        0.0;


    // PENDING Ground candidate.
    bool ground_pending_active_ =
        false;

    std::vector<Eigen::Vector3d>
        ground_pending_normals_W_;

    std::vector<Eigen::Vector3d>
        ground_pending_anchors_W_;

    std::size_t ground_pending_sample_count_ =
        0;

};
