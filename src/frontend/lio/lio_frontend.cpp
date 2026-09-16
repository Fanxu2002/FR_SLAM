#include "fr_slam/frontend/lio_frontend.hpp"
#include <iomanip>
#include <sstream>

#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>

namespace
{

    bool ImuDataIsFinite(
        const IMU_DATA &imu)
    {
        return std::isfinite(imu.timestamp) &&
               imu.gyro.allFinite() &&
               imu.accelerometer.allFinite();
    }

} // namespace

LioFrontend::LioFrontend()
    : ieskf_(config_.ieskf),
      measurement_builder_(config_.lidar_measurement)
{
    preprocessor_.SetConfig(
        config_.preprocessor);

    preprocessor_.SetOutlierFilterPolicy(
        config_.preprocessor_sor_mode,
        config_.preprocessor_enable_ror,
        config_.preprocessor_sor_adaptive_max_points);
}

LioFrontend::LioFrontend(
    const LioFrontendConfig &config)
    : config_(config),
      ieskf_(config.ieskf),
      measurement_builder_(config.lidar_measurement)
{
    preprocessor_.SetConfig(
        config_.preprocessor);

    preprocessor_.SetOutlierFilterPolicy(
        config_.preprocessor_sor_mode,
        config_.preprocessor_enable_ror,
        config_.preprocessor_sor_adaptive_max_points);
}

bool LioFrontend::SetConfig(
    const LioFrontendConfig &config)
{
    if (!std::isfinite(config.time_epsilon) ||
        config.time_epsilon <= 0.0)
    {
        return false;
    }

    if (!ieskf_.SetConfig(config.ieskf))
    {
        return false;
    }

    if (!measurement_builder_.SetConfig(
            config.lidar_measurement))
    {
        return false;
    }

    config_ =
        config;

    preprocessor_.SetConfig(
        config_.preprocessor);

    preprocessor_.SetOutlierFilterPolicy(
        config_.preprocessor_sor_mode,
        config_.preprocessor_enable_ror,
        config_.preprocessor_sor_adaptive_max_points);

    if (ieskf_.IsInitialized())
    {
        const LioState &state =
            ieskf_.State();

        preprocessor_.SetDeskewExtrinsic(
            state.Q_IL,
            state.P_IL);
    }

    return true;
}

const LioFrontendConfig &
LioFrontend::Config() const
{
    return config_;
}

bool LioFrontend::Initialize(
    const LioState &initial_state)
{
    if (!ieskf_.Initialize(
            initial_state))
    {
        return false;
    }

    const LioState &state =
        ieskf_.State();

    preprocessor_.SetDeskewExtrinsic(
        state.Q_IL,
        state.P_IL);

    return true;
}

bool LioFrontend::IsInitialized() const
{
    return ieskf_.IsInitialized();
}

const Ieskf &
LioFrontend::Filter() const
{
    return ieskf_;
}

Ieskf &
LioFrontend::Filter()
{
    return ieskf_;
}

bool LioFrontend::FindScanTimeRange(
    const LIDAR_FRAME &frame,
    double &scan_start_time,
    double &scan_end_time) const
{
    if (!frame.cloud ||
        frame.cloud->empty() ||
        !std::isfinite(frame.scan_start_time) ||
        !frame.has_point_time)
    {
        return false;
    }

    // The deskewer uses frame.scan_start_time as its reference frame.
    // The IESKF state used by the LiDAR measurement MUST therefore also live
    // exactly at frame.scan_start_time.
    scan_start_time =
        frame.scan_start_time;

    double minimum_offset =
        std::numeric_limits<double>::infinity();

    double maximum_offset =
        -std::numeric_limits<double>::infinity();

    for (const LIDAR_POINT &point :
         frame.cloud->points)
    {
        const double offset =
            static_cast<double>(
                point.time_offset);

        if (!std::isfinite(offset))
        {
            continue;
        }

        minimum_offset =
            std::min(
                minimum_offset,
                offset);

        maximum_offset =
            std::max(
                maximum_offset,
                offset);
    }

    if (!std::isfinite(minimum_offset) ||
        !std::isfinite(maximum_offset))
    {
        return false;
    }

    // Current adapters define scan_start_time as the scan reference / earliest
    // point time.  If a future adapter produces materially negative offsets,
    // reject it here instead of silently updating an IESKF state at the wrong
    // timestamp.
    if (minimum_offset <
        -config_.time_epsilon)
    {
        return false;
    }

    scan_end_time =
        frame.scan_start_time +
        std::max(
            0.0,
            maximum_offset);

    return std::isfinite(scan_end_time) &&
           scan_end_time + config_.time_epsilon >=
               scan_start_time;
}

IMU_STATE
LioFrontend::ToImuState(
    const LioState &state)
{
    IMU_STATE output;

    output.timestamp =
        state.timestamp;

    output.Q_WI =
        state.Q_WI;

    output.P_WI =
        state.P_WI;

    output.V_WI =
        state.V_WI;

    output.gyro_bias =
        state.gyro_bias;

    output.accel_bias =
        state.accel_bias;

    output.gravity_W =
        state.gravity_W;

    return output;
}

Eigen::Isometry3d
LioFrontend::StateToLidarPose(
    const LioState &state)
{
    Eigen::Quaterniond Q_WI =
        state.Q_WI;

    Eigen::Quaterniond Q_IL =
        state.Q_IL;

    Q_WI.normalize();
    Q_IL.normalize();

    const Eigen::Matrix3d R_WI =
        Q_WI.toRotationMatrix();

    const Eigen::Matrix3d R_IL =
        Q_IL.toRotationMatrix();

    Eigen::Isometry3d T_WL =
        Eigen::Isometry3d::Identity();

    T_WL.linear() =
        R_WI *
        R_IL;

    T_WL.translation() =
        state.P_WI +
        R_WI *
            state.P_IL;

    return T_WL;
}

bool LioFrontend::PropagateFilterToTime(
    const std::vector<IMU_DATA> &imu_data,
    double target_time)
{
    if (!ieskf_.IsInitialized() ||
        !std::isfinite(target_time))
    {
        return false;
    }

    const double current_time =
        ieskf_.State().timestamp;

    if (!std::isfinite(current_time))
    {
        std::cerr
            << "LIO_PROPAGATE_INVALID_STATE_TIME"
            << " | current_time=" << current_time
            << " | target_time=" << target_time
            << std::endl;

        return false;
    }

    if (target_time <
        current_time -
            config_.time_epsilon)
    {
        std::cerr
            << "LIO_PROPAGATE_BACKWARD_TIME"
            << " | current_time=" << current_time
            << " | target_time=" << target_time
            << std::endl;

        return false;
    }

    if (std::abs(
            target_time -
            current_time) <=
        config_.time_epsilon)
    {
        return true;
    }

    if (imu_data.size() < 2)
    {
        std::cerr
            << "LIO_PROPAGATE_TOO_FEW_IMU"
            << " | size=" << imu_data.size()
            << std::endl;

        return false;
    }

    // ============================================================
    // Extract exactly:
    //
    //     current IESKF time -> current LiDAR scan_start
    // ============================================================
    std::vector<IMU_DATA>
        exact_imu;

    if (!imu_integrator_.Extract(
            imu_data,
            current_time,
            target_time,
            exact_imu))
    {
        std::cerr
            << "LIO_PROPAGATE_EXTRACT_FAILED"
            << " | current_time=" << current_time
            << " | target_time=" << target_time
            << std::endl;

        return false;
    }

    if (exact_imu.size() < 2)
    {
        return false;
    }

    const double max_imu_dt =
        ieskf_.Config().max_imu_dt;

    if (!std::isfinite(max_imu_dt) ||
        max_imu_dt <= 0.0)
    {
        std::cerr
            << "LIO_PROPAGATE_INVALID_MAX_DT"
            << " | max_dt=" << max_imu_dt
            << std::endl;

        return false;
    }

    // ============================================================
    // Short-gap recovery policy.
    //
    // max_imu_dt:
    //     maximum numerical propagation sub-step.
    //
    // max_bridge_dt:
    //     maximum physical IMU dropout that we are willing to bridge.
    //
    // A short dropout is linearly interpolated in measurement space
    // and divided into sub-steps <= max_imu_dt.
    //
    // We DO NOT simply enlarge max_imu_dt.
    // ============================================================
    constexpr double max_bridge_dt =
        0.25;

    // ============================================================
    // Atomic propagation:
    //
    // Never modify the persistent IESKF unless the WHOLE interval
    // reaches target_time successfully.
    // ============================================================
    Ieskf trial_filter =
        ieskf_;

    // ============================================================
    // Propagation Z diagnostics.
    //
    // Diagnostic only. Decompose the exact nominal velocity
    // propagation into:
    //
    //     R_WI * a_m
    //   - R_WI * b_a
    //   + g_W
    //
    // using exactly the same sub-steps as IESKF propagation.
    // ============================================================
    const LioState propagation_start_state =
        trial_filter.State();

    double propagation_diagnostic_dt =
        0.0;

    Eigen::Vector3d propagation_dv_raw_W =
        Eigen::Vector3d::Zero();

    Eigen::Vector3d propagation_dv_bias_W =
        Eigen::Vector3d::Zero();

    Eigen::Vector3d propagation_dv_gravity_W =
        Eigen::Vector3d::Zero();

    Eigen::Vector3d propagation_accel_measurement_dt_I =
        Eigen::Vector3d::Zero();

    for (std::size_t index = 0;
         index + 1 < exact_imu.size();
         ++index)
    {
        const IMU_DATA &imu_begin =
            exact_imu[index];

        const IMU_DATA &imu_end =
            exact_imu[index + 1];

        if (!std::isfinite(imu_begin.timestamp) ||
            !std::isfinite(imu_end.timestamp) ||
            !imu_begin.gyro.allFinite() ||
            !imu_end.gyro.allFinite() ||
            !imu_begin.accelerometer.allFinite() ||
            !imu_end.accelerometer.allFinite())
        {
            std::cerr
                << "LIO_PROPAGATE_INVALID_IMU"
                << " | index=" << index
                << std::endl;

            return false;
        }

        const double interval_dt =
            imu_end.timestamp -
            imu_begin.timestamp;

        if (!std::isfinite(interval_dt) ||
            interval_dt <= 0.0)
        {
            std::cerr
                << "LIO_PROPAGATE_INVALID_DT"
                << " | index=" << index
                << " | dt=" << interval_dt
                << std::endl;

            return false;
        }

        // --------------------------------------------------------
        // Abnormally large data loss.
        //
        // Do not silently invent a long IMU trajectory.
        // --------------------------------------------------------
        if (interval_dt >
            max_bridge_dt)
        {
            std::cerr
                << "LIO_PROPAGATE_GAP_TOO_LARGE"
                << " | index=" << index
                << " | dt=" << interval_dt
                << " | bridge_limit="
                << max_bridge_dt
                << std::endl;

            return false;
        }

        // --------------------------------------------------------
        // Number of numerical propagation steps.
        //
        // Example:
        //
        //     dt = 0.058 s
        //     max_imu_dt = 0.050 s
        //
        // becomes:
        //
        //     2 x 0.029 s
        // --------------------------------------------------------
        const std::size_t substep_count =
            std::max<std::size_t>(
                1,
                static_cast<std::size_t>(
                    std::ceil(
                        interval_dt /
                        max_imu_dt)));

        if (substep_count > 1)
        {
            std::cerr
                << "LIO_PROPAGATE_GAP_BRIDGED"
                << " | index=" << index
                << " | dt=" << interval_dt
                << " | substeps="
                << substep_count
                << " | substep_dt~="
                << interval_dt /
                       static_cast<double>(
                           substep_count)
                << std::endl;
        }

        // --------------------------------------------------------
        // Linear interpolation of IMU measurements across the short
        // missing interval.
        //
        // This does NOT claim that the missing measurements are known.
        // It is only a bounded recovery strategy for a short dropout.
        // --------------------------------------------------------
        for (std::size_t step = 0;
             step < substep_count;
             ++step)
        {
            const double alpha0 =
                static_cast<double>(step) /
                static_cast<double>(
                    substep_count);

            const double alpha1 =
                static_cast<double>(step + 1) /
                static_cast<double>(
                    substep_count);

            IMU_DATA imu0;
            IMU_DATA imu1;

            imu0.timestamp =
                imu_begin.timestamp +
                alpha0 *
                    interval_dt;

            imu1.timestamp =
                imu_begin.timestamp +
                alpha1 *
                    interval_dt;

            imu0.gyro =
                (1.0 - alpha0) *
                    imu_begin.gyro +
                alpha0 *
                    imu_end.gyro;

            imu1.gyro =
                (1.0 - alpha1) *
                    imu_begin.gyro +
                alpha1 *
                    imu_end.gyro;

            imu0.accelerometer =
                (1.0 - alpha0) *
                    imu_begin.accelerometer +
                alpha0 *
                    imu_end.accelerometer;

            imu1.accelerometer =
                (1.0 - alpha1) *
                    imu_begin.accelerometer +
                alpha1 *
                    imu_end.accelerometer;

            const LioState substep_state_before =
                trial_filter.State();

            if (!trial_filter.Propagate(
                    imu0,
                    imu1))
            {
                std::cerr
                    << "LIO_PROPAGATE_SUBSTEP_FAILED"
                    << " | interval=" << index
                    << " | substep=" << step
                    << " | dt="
                    << imu1.timestamp -
                           imu0.timestamp
                    << std::endl;

                // Persistent ieskf_ remains untouched.
                return false;
            }
            const LioState &substep_state_after =
                trial_filter.State();

            const double substep_dt =
                imu1.timestamp -
                imu0.timestamp;

            const Eigen::Matrix3d R_WI_begin =
                substep_state_before.Q_WI
                    .normalized()
                    .toRotationMatrix();

            const Eigen::Matrix3d R_WI_end =
                substep_state_after.Q_WI
                    .normalized()
                    .toRotationMatrix();

            // ----------------------------------------------------
            // Raw accelerometer contribution:
            //
            //     R_WI * a_m
            // ----------------------------------------------------
            const Eigen::Vector3d raw_accel_begin_W =
                R_WI_begin *
                imu0.accelerometer;

            const Eigen::Vector3d raw_accel_end_W =
                R_WI_end *
                imu1.accelerometer;

            const Eigen::Vector3d raw_accel_mid_W =
                0.5 *
                (raw_accel_begin_W +
                 raw_accel_end_W);

            // ----------------------------------------------------
            // Accelerometer bias contribution:
            //
            //     -R_WI * b_a
            // ----------------------------------------------------
            const Eigen::Vector3d bias_accel_begin_W =
                -R_WI_begin *
                substep_state_before.accel_bias;

            const Eigen::Vector3d bias_accel_end_W =
                -R_WI_end *
                substep_state_before.accel_bias;

            const Eigen::Vector3d bias_accel_mid_W =
                0.5 *
                (bias_accel_begin_W +
                 bias_accel_end_W);

            // ----------------------------------------------------
            // Gravity contribution.
            // ----------------------------------------------------
            const Eigen::Vector3d gravity_accel_W =
                substep_state_before.gravity_W;

            propagation_dv_raw_W +=
                raw_accel_mid_W *
                substep_dt;

            propagation_dv_bias_W +=
                bias_accel_mid_W *
                substep_dt;

            propagation_dv_gravity_W +=
                gravity_accel_W *
                substep_dt;

            propagation_accel_measurement_dt_I +=
                0.5 *
                (imu0.accelerometer +
                 imu1.accelerometer) *
                substep_dt;

            propagation_diagnostic_dt +=
                substep_dt;
        }
    }

    const double propagated_time =
        trial_filter.State().timestamp;

    if (!std::isfinite(propagated_time) ||
        std::abs(
            propagated_time -
            target_time) >
            10.0 *
                config_.time_epsilon)
    {
        std::cerr
            << "LIO_PROPAGATE_TARGET_MISMATCH"
            << " | propagated_time="
            << propagated_time
            << " | target_time="
            << target_time
            << std::endl;

        return false;
    }

    if (propagation_diagnostic_dt >
        1.0e-12)
    {
        const LioState &propagation_end_state =
            trial_filter.State();

        const Eigen::Vector3d actual_delta_v_W =
            propagation_end_state.V_WI -
            propagation_start_state.V_WI;

        const Eigen::Vector3d decomposed_delta_v_W =
            propagation_dv_raw_W +
            propagation_dv_bias_W +
            propagation_dv_gravity_W;

        const Eigen::Vector3d decomposition_error_W =
            actual_delta_v_W -
            decomposed_delta_v_W;

        const Eigen::Vector3d mean_raw_accel_W =
            propagation_dv_raw_W /
            propagation_diagnostic_dt;

        const Eigen::Vector3d mean_bias_accel_W =
            propagation_dv_bias_W /
            propagation_diagnostic_dt;

        const Eigen::Vector3d mean_gravity_accel_W =
            propagation_dv_gravity_W /
            propagation_diagnostic_dt;

        const Eigen::Vector3d mean_effective_accel_W =
            decomposed_delta_v_W /
            propagation_diagnostic_dt;

        const Eigen::Vector3d mean_accel_measurement_I =
            propagation_accel_measurement_dt_I /
            propagation_diagnostic_dt;

        std::cerr
            << std::fixed
            << std::setprecision(9)
            << "LIO PROP Z"
            << " | t0="
            << current_time
            << " | t1="
            << target_time
            << " | dt="
            << propagation_diagnostic_dt
            << " | vz0="
            << propagation_start_state.V_WI.z()
            << " | vz1="
            << propagation_end_state.V_WI.z()
            << " | dvz="
            << actual_delta_v_W.z()
            << " | raw_Wz="
            << mean_raw_accel_W.z()
            << " | bias_Wz="
            << mean_bias_accel_W.z()
            << " | gravity_Wz="
            << mean_gravity_accel_W.z()
            << " | effective_Wz="
            << mean_effective_accel_W.z()
            << " | check_dvz="
            << decomposition_error_W.z()
            << " | mean_am_I=["
            << mean_accel_measurement_I.x()
            << " "
            << mean_accel_measurement_I.y()
            << " "
            << mean_accel_measurement_I.z()
            << "]"
            << " | ba=["
            << propagation_start_state.accel_bias.x()
            << " "
            << propagation_start_state.accel_bias.y()
            << " "
            << propagation_start_state.accel_bias.z()
            << "]"
            << " | g=["
            << propagation_start_state.gravity_W.x()
            << " "
            << propagation_start_state.gravity_W.y()
            << " "
            << propagation_start_state.gravity_W.z()
            << "]"
            << std::endl;
    }
    // ============================================================
    // Atomic commit.
    // ============================================================
    ieskf_ =
        trial_filter;

    return true;
}
bool LioFrontend::BuildDeskewTrajectory(
    const std::vector<IMU_DATA> &imu_data,
    double scan_start_time,
    double scan_end_time,
    std::vector<IMU_POSE> &imu_poses)
{
    imu_poses.clear();

    if (!ieskf_.IsInitialized() ||
        !std::isfinite(scan_start_time) ||
        !std::isfinite(scan_end_time) ||
        scan_end_time <
            scan_start_time -
                config_.time_epsilon)
    {
        return false;
    }

    const LioState &filter_state =
        ieskf_.State();

    if (std::abs(
            filter_state.timestamp -
            scan_start_time) >
        10.0 *
            config_.time_epsilon)
    {
        return false;
    }

    if (imu_data.size() < 2)
    {
        return false;
    }

    std::vector<IMU_DATA>
        scan_imu;

    if (!imu_integrator_.Extract(
            imu_data,
            scan_start_time,
            scan_end_time,
            scan_imu))
    {
        return false;
    }

    if (scan_imu.size() < 2)
    {
        return false;
    }

    const IMU_STATE scan_start_state =
        ToImuState(
            filter_state);

    IMU_STATE temporary_scan_end_state;

    if (!imu_integrator_.Integrate(
            scan_imu,
            scan_start_state,
            imu_poses,
            temporary_scan_end_state))
    {
        return false;
    }

    if (imu_poses.size() < 2)
    {
        return false;
    }

    return true;
}

bool LioFrontend::ProcessFrame(
    const LIDAR_FRAME &raw_frame,
    const std::vector<IMU_DATA> &imu_data,
    const PreparedLidarTarget *prepared_target,
    LioFrontendResult &result)
{
    result =
        LioFrontendResult();

    if (!ieskf_.IsInitialized())
    {
        std::cerr << "LIO_REJECT | stage=IESKF_NOT_INITIALIZED" << std::endl;
        return false;
    }

    double scan_start_time =
        0.0;

    double scan_end_time =
        0.0;

    if (!FindScanTimeRange(
            raw_frame,
            scan_start_time,
            scan_end_time))
    {
        std::cerr << "LIO_REJECT | stage=FIND_SCAN_TIME_RANGE" << std::endl;
        return false;
    }

    // ========================================================================
    // LIO_Z_DIAG: state before frame propagation.
    const LioState state_before_propagation =
        ieskf_.State();

    // 1. Advance the persistent IESKF only to the LiDAR reference time.
    // ========================================================================
    if (!PropagateFilterToTime(
            imu_data,
            scan_start_time))
    {
        std::cerr << "LIO_REJECT | stage=PROPAGATE_TO_SCAN_START" << std::endl;
        return false;
    }

    // ========================================================================
    // 2. Build a temporary scan-internal IMU trajectory for deskew.
    //
    // The IESKF itself remains at scan_start.
    // ========================================================================
    std::vector<IMU_POSE>
        imu_poses;

    if (!BuildDeskewTrajectory(
            imu_data,
            scan_start_time,
            scan_end_time,
            imu_poses))
    {
        std::cerr << "LIO_REJECT | stage=BUILD_DESKEW_TRAJECTORY" << std::endl;
        return false;
    }

    // ========================================================================
    // 3. Full SE(3) deskew + filtering.
    // ========================================================================
    const LioState predicted_state =
        ieskf_.State();

    // LIO_Z_DIAG: predicted covariance / state at scan_start.
    result.diagnostics.valid = true;
    result.diagnostics.before_propagation =
        state_before_propagation;
    result.diagnostics.after_propagation =
        predicted_state;

    const Ieskf::StateMatrix &predicted_covariance =
        ieskf_.Covariance();

    constexpr int kPositionZ =
        LioStateIndex::POSITION + 2;

    constexpr int kVelocityZ =
        LioStateIndex::VELOCITY + 2;

    constexpr int kAccelBiasZ =
        LioStateIndex::ACCEL_BIAS + 2;

    constexpr int kGravity1 =
        LioStateIndex::GRAVITY;

    constexpr int kGravity2 =
        LioStateIndex::GRAVITY + 1;

    result.diagnostics.pred_var_z =
        predicted_covariance(kPositionZ, kPositionZ);

    result.diagnostics.pred_var_vz =
        predicted_covariance(kVelocityZ, kVelocityZ);

    result.diagnostics.pred_var_baz =
        predicted_covariance(kAccelBiasZ, kAccelBiasZ);

    result.diagnostics.pred_cov_z_vz =
        predicted_covariance(kPositionZ, kVelocityZ);

    result.diagnostics.pred_cov_z_baz =
        predicted_covariance(kPositionZ, kAccelBiasZ);

    result.diagnostics.pred_cov_vz_baz =
        predicted_covariance(kVelocityZ, kAccelBiasZ);

    result.diagnostics.pred_var_g1 =
        predicted_covariance(kGravity1, kGravity1);

    result.diagnostics.pred_var_g2 =
        predicted_covariance(kGravity2, kGravity2);

    preprocessor_.SetDeskewExtrinsic(
        predicted_state.Q_IL,
        predicted_state.P_IL);

    result.processed_frame =
        preprocessor_.Process(
            raw_frame,
            imu_poses,
            true);

    if (!result.processed_frame.cloud ||
        result.processed_frame.cloud->empty())
    {
        std::cerr << "LIO_REJECT | stage=PREPROCESS" << std::endl;
        return false;
    }

    // ========================================================================
    // 4. First mapping frame.
    //
    // There is no historical tracking target yet, therefore there is no LiDAR
    // correction to perform. The IMU-predicted state at scan_start defines the
    // first LiDAR pose. The mapping owner will commit this cloud/pose and build
    // the first PreparedLidarTarget.
    // ========================================================================
    if (prepared_target == nullptr)
    {
        result.first_mapping_frame =
            true;

        result.state =
            ieskf_.State();

        result.T_WL =
            StateToLidarPose(
                result.state);

        result.success =
            result.T_WL.matrix().allFinite();

        return result.success;
    }

    if (!prepared_target->ready ||
        !prepared_target->cloud ||
        prepared_target->cloud->empty() ||
        !prepared_target->kdtree)
    {
        std::cerr << "LIO_REJECT | stage=INVALID_PREPARED_TARGET" << std::endl;
        return false;
    }

    // ========================================================================
    // 5. IESKF point-to-plane correction against the SAME tracking target that
    // the existing SubmapManager/backend owns.
    // ========================================================================
    if (!ieskf_.IteratedLidarUpdate(
            result.processed_frame.cloud,
            *prepared_target,
            measurement_builder_,
            result.lidar_update))
    {
        std::cerr << "LIO_REJECT | stage=LIDAR_UPDATE" << std::endl;
        return false;
    }

    // ========================================================================
    // 6. Final corrected state / LiDAR pose.
    // ========================================================================
    result.state =
        ieskf_.State();

    // ========================================================================
    // Diagnostic only:
    // Compare IMU-predicted state at scan_start with the posterior state
    // after the complete iterated LiDAR update.
    // ========================================================================
    const LioState &posterior_state =
        result.state;

    // LIO_Z_DIAG: posterior covariance / state after LiDAR update.
    result.diagnostics.has_lidar_update = true;
    result.diagnostics.after_lidar_update =
        posterior_state;

    const Ieskf::StateMatrix &posterior_covariance =
        ieskf_.Covariance();

    result.diagnostics.post_var_z =
        posterior_covariance(kPositionZ, kPositionZ);

    result.diagnostics.post_var_vz =
        posterior_covariance(kVelocityZ, kVelocityZ);

    result.diagnostics.post_var_baz =
        posterior_covariance(kAccelBiasZ, kAccelBiasZ);

    result.diagnostics.post_cov_z_vz =
        posterior_covariance(kPositionZ, kVelocityZ);

    result.diagnostics.post_cov_z_baz =
        posterior_covariance(kPositionZ, kAccelBiasZ);

    result.diagnostics.post_cov_vz_baz =
        posterior_covariance(kVelocityZ, kAccelBiasZ);

    result.diagnostics.post_var_g1 =
        posterior_covariance(kGravity1, kGravity1);

    result.diagnostics.post_var_g2 =
        posterior_covariance(kGravity2, kGravity2);

    const Eigen::Vector3d lidar_delta_position_WI =
        posterior_state.P_WI -
        predicted_state.P_WI;

    const Eigen::Vector3d lidar_delta_velocity_WI =
        posterior_state.V_WI -
        predicted_state.V_WI;

    const Eigen::Vector3d lidar_delta_gyro_bias =
        posterior_state.gyro_bias -
        predicted_state.gyro_bias;

    const Eigen::Vector3d lidar_delta_accel_bias =
        posterior_state.accel_bias -
        predicted_state.accel_bias;

    const Eigen::Vector3d lidar_delta_gravity =
        posterior_state.gravity_W -
        predicted_state.gravity_W;

    const Eigen::Vector3d lidar_delta_extrinsic_position =
        posterior_state.P_IL -
        predicted_state.P_IL;

    constexpr double kLioStateRadToDeg =
        57.29577951308232;

    const double lidar_delta_rotation_deg =
        predicted_state.Q_WI.angularDistance(
            posterior_state.Q_WI) *
        kLioStateRadToDeg;

    const double lidar_delta_extrinsic_rotation_deg =
        predicted_state.Q_IL.angularDistance(
            posterior_state.Q_IL) *
        kLioStateRadToDeg;

    const Eigen::Isometry3d predicted_T_WL =
        StateToLidarPose(
            predicted_state);

    const Eigen::Isometry3d posterior_T_WL =
        StateToLidarPose(
            posterior_state);

    const double predicted_lidar_z =
        predicted_T_WL.translation().z();

    const double posterior_lidar_z =
        posterior_T_WL.translation().z();

    const double lidar_delta_lidar_z =
        posterior_lidar_z -
        predicted_lidar_z;

    // LIO_Z_DIAG: one compact line per successful LiDAR frame.
    const double imu_delta_z =
        predicted_state.P_WI.z() -
        state_before_propagation.P_WI.z();

    const double imu_delta_vz =
        predicted_state.V_WI.z() -
        state_before_propagation.V_WI.z();

    const double lidar_delta_baz =
        posterior_state.accel_bias.z() -
        predicted_state.accel_bias.z();

    std::ostringstream z_diagnostic_log;

    z_diagnostic_log
        << std::fixed
        << std::setprecision(9)
        << "LIO_Z_DIAG"
        << " | t=" << scan_start_time
        << " | imu_dt="
        << predicted_state.timestamp - state_before_propagation.timestamp
        << " | pre_z=" << state_before_propagation.P_WI.z()
        << " | pred_z=" << predicted_state.P_WI.z()
        << " | post_z=" << posterior_state.P_WI.z()
        << " | imu_dz=" << imu_delta_z
        << " | lidar_dz=" << lidar_delta_position_WI.z()
        << " | pre_vz=" << state_before_propagation.V_WI.z()
        << " | pred_vz=" << predicted_state.V_WI.z()
        << " | post_vz=" << posterior_state.V_WI.z()
        << " | imu_dvz=" << imu_delta_vz
        << " | lidar_dvz=" << lidar_delta_velocity_WI.z()
        << " | pre_baz=" << state_before_propagation.accel_bias.z()
        << " | pred_baz=" << predicted_state.accel_bias.z()
        << " | post_baz=" << posterior_state.accel_bias.z()
        << " | lidar_dbaz=" << lidar_delta_baz
        << " | pred_gx=" << predicted_state.gravity_W.x()
        << " | pred_gy=" << predicted_state.gravity_W.y()
        << " | pred_gz=" << predicted_state.gravity_W.z()
        << " | post_gx=" << posterior_state.gravity_W.x()
        << " | post_gy=" << posterior_state.gravity_W.y()
        << " | post_gz=" << posterior_state.gravity_W.z()
        << " | Pz_pred=" << result.diagnostics.pred_var_z
        << " | Pz_post=" << result.diagnostics.post_var_z
        << " | Pvz_pred=" << result.diagnostics.pred_var_vz
        << " | Pvz_post=" << result.diagnostics.post_var_vz
        << " | Pbaz_pred=" << result.diagnostics.pred_var_baz
        << " | Pbaz_post=" << result.diagnostics.post_var_baz
        << " | Pzvz_pred=" << result.diagnostics.pred_cov_z_vz
        << " | Pzvz_post=" << result.diagnostics.post_cov_z_vz
        << " | Pzbaz_pred=" << result.diagnostics.pred_cov_z_baz
        << " | Pzbaz_post=" << result.diagnostics.post_cov_z_baz
        << " | Pvzbaz_pred=" << result.diagnostics.pred_cov_vz_baz
        << " | Pvzbaz_post=" << result.diagnostics.post_cov_vz_baz
        << " | Pg1_pred=" << result.diagnostics.pred_var_g1
        << " | Pg1_post=" << result.diagnostics.post_var_g1
        << " | Pg2_pred=" << result.diagnostics.pred_var_g2
        << " | Pg2_post=" << result.diagnostics.post_var_g2
        << " | info_x=" << result.lidar_update.position_information_x
        << " | info_y=" << result.lidar_update.position_information_y
        << " | info_z=" << result.lidar_update.position_information_z
        << " | info_z_ratio="
        << result.lidar_update.position_information_z_ratio
        << " | corr=" << result.lidar_update.correspondences
        << " | rmse=" << result.lidar_update.final_rmse
        << " | robust_rmse=" << result.lidar_update.final_robust_rmse
        << " | downweighted="
        << result.lidar_update.downweighted_correspondences
        << " | dext_z=" << lidar_delta_extrinsic_position.z()
        << " | dext_rot_deg="
        << lidar_delta_extrinsic_rotation_deg;

    std::cout
        << z_diagnostic_log.str()
        << std::endl;

    std::ostringstream state_update_log;

    state_update_log
        << std::fixed
        << std::setprecision(9)
        << "LIO STATE UPDATE"
        << " | t=" << scan_start_time
        << " | pred_z=" << predicted_state.P_WI.z()
        << " | post_z=" << posterior_state.P_WI.z()
        << " | dz=" << lidar_delta_position_WI.z()
        << " | pred_vz=" << predicted_state.V_WI.z()
        << " | post_vz=" << posterior_state.V_WI.z()
        << " | dvz=" << lidar_delta_velocity_WI.z()
        << " | pred_lidar_z=" << predicted_lidar_z
        << " | post_lidar_z=" << posterior_lidar_z
        << " | dlidar_z=" << lidar_delta_lidar_z
        << " | dP_WI=["
        << lidar_delta_position_WI.x() << " "
        << lidar_delta_position_WI.y() << " "
        << lidar_delta_position_WI.z() << "]"
        << " | dV_WI=["
        << lidar_delta_velocity_WI.x() << " "
        << lidar_delta_velocity_WI.y() << " "
        << lidar_delta_velocity_WI.z() << "]"
        << " | dR_WI_deg=" << lidar_delta_rotation_deg
        << " | ba=["
        << posterior_state.accel_bias.x() << " "
        << posterior_state.accel_bias.y() << " "
        << posterior_state.accel_bias.z() << "]"
        << " | dba=["
        << lidar_delta_accel_bias.x() << " "
        << lidar_delta_accel_bias.y() << " "
        << lidar_delta_accel_bias.z() << "]"
        << " | bg=["
        << posterior_state.gyro_bias.x() << " "
        << posterior_state.gyro_bias.y() << " "
        << posterior_state.gyro_bias.z() << "]"
        << " | dbg=["
        << lidar_delta_gyro_bias.x() << " "
        << lidar_delta_gyro_bias.y() << " "
        << lidar_delta_gyro_bias.z() << "]"
        << " | g=["
        << posterior_state.gravity_W.x() << " "
        << posterior_state.gravity_W.y() << " "
        << posterior_state.gravity_W.z() << "]"
        << " | dg=["
        << lidar_delta_gravity.x() << " "
        << lidar_delta_gravity.y() << " "
        << lidar_delta_gravity.z() << "]"
        << " | P_IL=["
        << posterior_state.P_IL.x() << " "
        << posterior_state.P_IL.y() << " "
        << posterior_state.P_IL.z() << "]"
        << " | dP_IL=["
        << lidar_delta_extrinsic_position.x() << " "
        << lidar_delta_extrinsic_position.y() << " "
        << lidar_delta_extrinsic_position.z() << "]"
        << " | q_IL=["
        << posterior_state.Q_IL.x() << " "
        << posterior_state.Q_IL.y() << " "
        << posterior_state.Q_IL.z() << " "
        << posterior_state.Q_IL.w() << "]"
        << " | dR_IL_deg="
        << lidar_delta_extrinsic_rotation_deg;

    std::cerr
        << state_update_log.str()
        << std::endl;

    result.T_WL =
        StateToLidarPose(
            result.state);

    if (!result.T_WL.matrix().allFinite())
    {
        std::cerr << "LIO_REJECT | stage=FINAL_POSE_NONFINITE" << std::endl;
        return false;
    }

    result.success =
        true;

    return true;
}
