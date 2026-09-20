#include "fr_slam/frontend/lio_frontend.hpp"
#include <array>
#include <Eigen/Eigenvalues>
#include "fr_slam/frontend/ground_input_bridge.hpp"
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
      measurement_builder_(config_.lidar_measurement),
      ground_segmenter_(config_.ground.segmentation)
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
      measurement_builder_(config.lidar_measurement),
      ground_segmenter_(config.ground.segmentation)
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

    ground_segmenter_.SetConfig(
        config_.ground.segmentation);

    ResetGroundRuntime();

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
    ground_segmenter_.Reset();
    ResetGroundRuntime();
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
        0.35;

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

        // ============================================================
        // FR_IMU_GAP_CV_BRIDGE_V1
        //
        // Real IMU interval:
        //
        //   interval_dt <= max_imu_dt
        //       -> normal IMU propagation, unchanged.
        //
        //   interval_dt > max_imu_dt
        //       -> degraded dropout recovery:
        //          preserve gyro propagation,
        //          suppress invented translational acceleration.
        //
        // During the missing interval we impose the propagation model:
        //
        //       dV/dt = 0
        //
        // by synthesizing accelerometer samples satisfying:
        //
        //       R_WI * (a_m - b_a) + g_W = 0
        //
        // No Ground / LiDAR / measurement code is modified here.
        // ============================================================
        const bool use_constant_velocity_gap_bridge =
            interval_dt > max_imu_dt;

        const Eigen::Vector3d gap_velocity_before =
            trial_filter.State().V_WI;


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

            // --------------------------------------------------------
            // FR_IMU_GAP_CV_BRIDGE_V1
            //
            // For a real dropout, the interpolated accelerometer above
            // is NOT treated as measured dynamics.
            //
            // Gyro remains the interpolated gyro.
            //
            // Synthetic accelerometer:
            //
            //     a_m = b_a + R_IW * (-g_W)
            //
            // so nominal world linear acceleration is approximately 0.
            // --------------------------------------------------------
            if (use_constant_velocity_gap_bridge)
            {
                const LioState bridge_state =
                    trial_filter.State();

                if (!bridge_state.Q_WI.coeffs().allFinite() ||
                    bridge_state.Q_WI.norm() <= 1.0e-12 ||
                    !bridge_state.gyro_bias.allFinite() ||
                    !bridge_state.accel_bias.allFinite() ||
                    !bridge_state.gravity_W.allFinite())
                {
                    std::cerr
                        << "LIO_PROPAGATE_CV_BRIDGE_INVALID_STATE"
                        << " | interval=" << index
                        << " | substep=" << step
                        << std::endl;

                    return false;
                }

                const double bridge_substep_dt =
                    imu1.timestamp -
                    imu0.timestamp;

                if (!std::isfinite(bridge_substep_dt) ||
                    bridge_substep_dt <= 0.0)
                {
                    std::cerr
                        << "LIO_PROPAGATE_CV_BRIDGE_INVALID_DT"
                        << " | interval=" << index
                        << " | substep=" << step
                        << " | dt=" << bridge_substep_dt
                        << std::endl;

                    return false;
                }

                const Eigen::Matrix3d R_WI_begin =
                    bridge_state.Q_WI
                        .normalized()
                        .toRotationMatrix();

                const Eigen::Vector3d omega_mid_I =
                    0.5 *
                        (
                            imu0.gyro +
                            imu1.gyro
                        ) -
                    bridge_state.gyro_bias;

                if (!omega_mid_I.allFinite())
                {
                    return false;
                }

                Eigen::Matrix3d R_WI_end =
                    R_WI_begin;

                const double omega_norm =
                    omega_mid_I.norm();

                if (omega_norm > 1.0e-12)
                {
                    const double angle =
                        omega_norm *
                        bridge_substep_dt;

                    R_WI_end =
                        R_WI_begin *
                        Eigen::AngleAxisd(
                            angle,
                            omega_mid_I /
                                omega_norm)
                            .toRotationMatrix();
                }

                const Eigen::Vector3d
                    specific_force_begin_I =
                        R_WI_begin.transpose() *
                        (-bridge_state.gravity_W);

                const Eigen::Vector3d
                    specific_force_end_I =
                        R_WI_end.transpose() *
                        (-bridge_state.gravity_W);

                imu0.accelerometer =
                    bridge_state.accel_bias +
                    specific_force_begin_I;

                imu1.accelerometer =
                    bridge_state.accel_bias +
                    specific_force_end_I;

                if (!imu0.accelerometer.allFinite() ||
                    !imu1.accelerometer.allFinite())
                {
                    return false;
                }
            }


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

        if (use_constant_velocity_gap_bridge)
        {
            const Eigen::Vector3d gap_velocity_after =
                trial_filter.State().V_WI;

            std::cerr
                << "LIO_PROPAGATE_CV_BRIDGE"
                << " | index=" << index
                << " | dt=" << interval_dt
                << " | substeps=" << substep_count
                << " | v_before=["
                << gap_velocity_before.transpose()
                << "]"
                << " | v_after=["
                << gap_velocity_after.transpose()
                << "]"
                << " | dv=["
                << (
                    gap_velocity_after -
                    gap_velocity_before
                ).transpose()
                << "]"
                << std::endl;
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

void LioFrontend::ResetGroundRuntime()
{
    ground_reference_frozen_ = false;

    ground_bootstrap_normals_W_.clear();
    ground_bootstrap_plane_d_W_.clear();

    ground_reference_normal_W_ =
        Eigen::Vector3d::UnitZ();

    ground_reference_plane_d_W_ =
        0.0;
}


bool LioFrontend::ProcessGroundMeasurement(
    const pcl::PointCloud<LIDAR_POINT>::ConstPtr &ground_cloud_L,
    const pcl::PointCloud<LIDAR_POINT>::ConstPtr &registration_cloud_L,
    const PreparedLidarTarget *prepared_target,
    bool allow_pose_correction)
{
    // This cache belongs strictly to the current call/frame.
    last_ground_segmentation_valid_ = false;


    // ========================================================================
    // LIO FINAL GROUND V1
    //
    // LiDAR IESKF owns the baseline state.
    //
    // Ground:
    //   - never enters the LiDAR Hessian,
    //   - never updates velocity / bias / gravity / extrinsic,
    //   - owns only bounded roll/pitch/z correction,
    //   - uses all-scene LiDAR geometry ONLY as a safety evaluator.
    //
    // Accepted corrected T_WL is injected back into the nominal IESKF pose
    // BEFORE the mapping/Submap commit.
    // ========================================================================

    if (!config_.ground.enabled ||
        config_.ground.mode == "off" ||
        config_.ground.mode == "disabled")
    {
        return true;
    }

    if (!ground_cloud_L ||
        ground_cloud_L->empty())
    {
        return true;
    }

    // ------------------------------------------------------------------------
    // A. Dense current-scan Ground segmentation.
    // ------------------------------------------------------------------------
    pcl::PointCloud<pcl::PointXYZ>::Ptr ground_input(
        new pcl::PointCloud<pcl::PointXYZ>);

    ground_input->reserve(
        ground_cloud_L->size());

    for (const LIDAR_POINT &point :
         ground_cloud_L->points)
    {
        if (!std::isfinite(point.x) ||
            !std::isfinite(point.y) ||
            !std::isfinite(point.z))
        {
            continue;
        }

        ground_input->emplace_back(
            point.x,
            point.y,
            point.z);
    }

    if (ground_input->empty())
    {
        return true;
    }

    const fr_slam::GroundSegmentationResult ground_result =
        ground_segmenter_.Segment(
            ground_input);

    last_ground_segmentation_result_ =
        ground_result;

    last_ground_segmentation_valid_ =
        true;

    // ------------------------------------------------------------------------
    // Ground trusted-constraint audit.
    //
    // Diagnostics only.
    // This block MUST NOT modify segmentation, reference or IESKF state.
    // ------------------------------------------------------------------------
    std::cout
        << "LIO_GROUND_GATE"
        << " | success="
        << (ground_result.success ? 1 : 0)
        << " | plane="
        << (ground_result.support_plane_valid ? 1 : 0)
        << " | constraint="
        << (ground_result.support_constraint_valid ? 1 : 0)
        << " | mask="
        << ground_result.support_constraint_rejection_mask
        << " | points="
        << ground_result.support_ground_points
        << " | cells="
        << ground_result.support_ground_cells
        << " | inlier="
        << ground_result.support_plane_inlier_ratio
        << " | rmse="
        << ground_result.support_plane_rmse_m
        << " | tilt="
        << ground_result.support_ground_tilt_deg
        << std::endl;



    if (!ground_result.success ||
        !ground_result.support_plane_valid ||
        !ground_result.support_constraint_valid ||
        !ground_result.support_ground_cloud ||
        ground_result.support_ground_cloud->empty() ||
        !ground_result.support_ground_normal_L.allFinite() ||
        !std::isfinite(
            ground_result.support_ground_plane_d))
    {
        return true;
    }

    Eigen::Vector3d normal_L =
        ground_result.support_ground_normal_L;

    const double normal_L_norm =
        normal_L.norm();

    if (!std::isfinite(normal_L_norm) ||
        normal_L_norm <= 1.0e-12)
    {
        return true;
    }

    normal_L /= normal_L_norm;

    double plane_d_L =
        ground_result.support_ground_plane_d;

    if (normal_L.z() < 0.0)
    {
        normal_L = -normal_L;
        plane_d_L = -plane_d_L;
    }

    // ------------------------------------------------------------------------
    // B. Baseline = state AFTER LiDAR IESKF update.
    // ------------------------------------------------------------------------
    const LioState baseline_state =
        ieskf_.State();

    const Eigen::Isometry3d baseline_T_WL =
        StateToLidarPose(
            baseline_state);

    if (!baseline_T_WL.matrix().allFinite())
    {
        return true;
    }

    Eigen::Vector3d current_normal_W =
        baseline_T_WL.rotation() *
        normal_L;

    const double current_normal_norm =
        current_normal_W.norm();

    if (!current_normal_W.allFinite() ||
        !std::isfinite(current_normal_norm) ||
        current_normal_norm <= 1.0e-12)
    {
        return true;
    }

    current_normal_W /=
        current_normal_norm;

    if (current_normal_W.z() < 0.0)
    {
        current_normal_W =
            -current_normal_W;

        normal_L =
            -normal_L;

        plane_d_L =
            -plane_d_L;
    }

    const Eigen::Vector3d point_on_plane_L =
        -plane_d_L *
        normal_L;

    const Eigen::Vector3d point_on_plane_W =
        baseline_T_WL *
        point_on_plane_L;

    const double plane_d_W =
        -current_normal_W.dot(
            point_on_plane_W);

    if (!std::isfinite(plane_d_W))
    {
        return true;
    }

    // ------------------------------------------------------------------------
    // C. BOOTSTRAP -> FROZEN.
    //
    // Frozen reference is independent of Submap lifecycle.
    // ------------------------------------------------------------------------
    if (!ground_reference_frozen_)
    {
        ground_bootstrap_normals_W_.push_back(
            current_normal_W);

        ground_bootstrap_plane_d_W_.push_back(
            plane_d_W);

        const std::size_t required =
            std::max<std::size_t>(
                1,
                config_.ground.anchor_bootstrap_frames);

        if (ground_bootstrap_normals_W_.size() >
            required)
        {
            ground_bootstrap_normals_W_.erase(
                ground_bootstrap_normals_W_.begin());

            ground_bootstrap_plane_d_W_.erase(
                ground_bootstrap_plane_d_W_.begin());
        }

        if (ground_bootstrap_normals_W_.size() <
            required)
        {
            return true;
        }

        Eigen::Vector3d mean_normal =
            Eigen::Vector3d::Zero();

        double mean_d =
            0.0;

        for (std::size_t i = 0;
             i < required;
             ++i)
        {
            mean_normal +=
                ground_bootstrap_normals_W_[i];

            mean_d +=
                ground_bootstrap_plane_d_W_[i];
        }

        const double mean_normal_norm =
            mean_normal.norm();

        if (!std::isfinite(mean_normal_norm) ||
            mean_normal_norm <= 1.0e-12)
        {
            return true;
        }

        mean_normal /=
            mean_normal_norm;

        mean_d /=
            static_cast<double>(
                required);

        constexpr double kRadToDeg =
            57.29577951308232;

        double maximum_normal_spread_deg =
            0.0;

        double maximum_d_spread =
            0.0;

        for (std::size_t i = 0;
             i < required;
             ++i)
        {
            const double cosine =
                std::clamp(
                    mean_normal.dot(
                        ground_bootstrap_normals_W_[i]),
                    -1.0,
                    1.0);

            maximum_normal_spread_deg =
                std::max(
                    maximum_normal_spread_deg,
                    std::acos(cosine) *
                        kRadToDeg);

            maximum_d_spread =
                std::max(
                    maximum_d_spread,
                    std::abs(
                        ground_bootstrap_plane_d_W_[i] -
                        mean_d));
        }

        if (maximum_normal_spread_deg >
                config_.ground
                    .anchor_maximum_normal_spread_deg ||
            maximum_d_spread >
                config_.ground
                    .anchor_maximum_plane_d_spread_m)
        {
            return true;
        }

        // ================================================================
        // FR_GROUND_FLAT_REFERENCE_V1
        //
        // Flat-floor operating mode:
        //
        // The bootstrap Ground estimate is still used for:
        //   - normal consistency validation
        //   - plane-distance consistency validation
        //   - initial vertical anchor
        //
        // But its small world-frame tilt is NOT frozen permanently.
        //
        // Old bootstrap plane:
        //
        //     mean_normal^T p_W + mean_d = 0
        //
        // At world x=y=0:
        //
        //     z0 = -mean_d / mean_normal.z()
        //
        // Freeze a horizontal plane through exactly the same z-intercept:
        //
        //     UnitZ^T p_W + flat_d = 0
        //
        //     flat_d = mean_d / mean_normal.z()
        //
        // Therefore this change removes ONLY the bootstrap tilt while
        // preserving its initial vertical anchor.
        // ================================================================

        const double bootstrap_normal_z =
            mean_normal.z();

        if (!std::isfinite(bootstrap_normal_z) ||
            bootstrap_normal_z <= 1.0e-6)
        {
            return true;
        }

        const double flat_reference_plane_d_W =
            mean_d /
            bootstrap_normal_z;

        if (!std::isfinite(
                flat_reference_plane_d_W))
        {
            return true;
        }

        const Eigen::Vector3d
            bootstrap_reference_normal_W =
                mean_normal;

        const double
            bootstrap_reference_plane_d_W =
                mean_d;

        ground_reference_normal_W_ =
            Eigen::Vector3d::UnitZ();

        ground_reference_plane_d_W_ =
            flat_reference_plane_d_W;

        std::cout
            << "LIO_GROUND_FLAT_REFERENCE"
            << " | bootstrap_n_W=["
            << bootstrap_reference_normal_W.transpose()
            << "]"
            << " | bootstrap_d_W="
            << bootstrap_reference_plane_d_W
            << " | flat_n_W=["
            << ground_reference_normal_W_.transpose()
            << "]"
            << " | flat_d_W="
            << ground_reference_plane_d_W_
            << " | dz_origin="
            << (
                -ground_reference_plane_d_W_ -
                (
                    -bootstrap_reference_plane_d_W /
                    bootstrap_reference_normal_W.z()
                )
            )
            << std::endl;

        ground_reference_frozen_ =
            true;

        std::cout
            << "LIO_GROUND_REFERENCE_FROZEN"
            << " | piece="
            << ground_active_piece_id_
            << " | n_W=["
            << ground_reference_normal_W_.transpose()
            << "]"
            << " | d_W="
            << ground_reference_plane_d_W_
            << " | samples="
            << required
            << std::endl;

        return true;
    }

    if (!allow_pose_correction)
    {
        return true;
    }

    // ------------------------------------------------------------------------
    // D. Frozen reference sanity.
    // ------------------------------------------------------------------------
    const Eigen::Vector3d reference_normal_W =
        ground_reference_normal_W_.normalized();

    if (!reference_normal_W.allFinite() ||
        reference_normal_W.z() <
            config_.ground.minimum_reference_normal_z)
    {
        return true;
    }

    const double normal_cosine =
        std::clamp(
            current_normal_W.dot(
                reference_normal_W),
            -1.0,
            1.0);

    const double normal_residual_rad =
        std::acos(
            normal_cosine);

    const double normal_residual_deg =
        normal_residual_rad *
        57.29577951308232;

    // ------------------------------------------------------------------------
    // Ground-height residual = median signed distance of CURRENT support
    // points to the FROZEN plane.
    //
    // This avoids contaminating height with current-normal error and long XY
    // travel.
    // ------------------------------------------------------------------------
    const auto median_ground_height_residual =
        [&ground_result,
         &reference_normal_W,
         this](
            const Eigen::Matrix3d &R_WL,
            const Eigen::Vector3d &t_WL,
            double &median_residual_m)
        {
            std::vector<double> residuals;

            residuals.reserve(
                ground_result
                    .support_ground_cloud
                    ->size());

            for (const pcl::PointXYZ &point :
                 ground_result
                     .support_ground_cloud
                     ->points)
            {
                const Eigen::Vector3d p_L(
                    static_cast<double>(point.x),
                    static_cast<double>(point.y),
                    static_cast<double>(point.z));

                if (!p_L.allFinite())
                {
                    continue;
                }

                const Eigen::Vector3d p_W =
                    R_WL * p_L +
                    t_WL;

                const double residual =
                    reference_normal_W.dot(
                        p_W) +
                    ground_reference_plane_d_W_;

                if (std::isfinite(residual))
                {
                    residuals.push_back(
                        residual);
                }
            }

            if (residuals.empty())
            {
                return false;
            }

            const std::size_t middle =
                residuals.size() / 2;

            std::nth_element(
                residuals.begin(),
                residuals.begin() +
                    static_cast<std::ptrdiff_t>(
                        middle),
                residuals.end());

            median_residual_m =
                residuals[middle];

            if ((residuals.size() % 2U) == 0U &&
                middle > 0U)
            {
                const double upper =
                    median_residual_m;

                std::nth_element(
                    residuals.begin(),
                    residuals.begin() +
                        static_cast<std::ptrdiff_t>(
                            middle - 1U),
                    residuals.end());

                median_residual_m =
                    0.5 *
                    (upper +
                     residuals[middle - 1U]);
            }

            return
                std::isfinite(
                    median_residual_m);
        };

    double baseline_height_residual_m =
        0.0;

    if (!median_ground_height_residual(
            baseline_T_WL.rotation(),
            baseline_T_WL.translation(),
            baseline_height_residual_m))
    {
        return true;
    }

    if (!std::isfinite(normal_residual_deg) ||
        !std::isfinite(baseline_height_residual_m) ||
        normal_residual_deg >
            config_.ground.maximum_normal_residual_deg ||
        std::abs(baseline_height_residual_m) >
            config_.ground.maximum_height_residual_m)
    {
        return true;
    }

    // ------------------------------------------------------------------------
    // E. General geometry BASELINE evaluation.
    //
    // General geometry does NOT solve Ground correction.
    // It is used only as a candidate safety evaluator.
    // ------------------------------------------------------------------------
    if (!registration_cloud_L ||
        registration_cloud_L->empty() ||
        prepared_target == nullptr ||
        !prepared_target->ready)
    {
        return true;
    }

    LioMeasurementResult baseline_geometry;

    if (!measurement_builder_.Build(
            baseline_state,
            registration_cloud_L,
            *prepared_target,
            baseline_geometry) ||
        !baseline_geometry.success ||
        !std::isfinite(
            baseline_geometry.rmse))
    {
        return true;
    }

    // ------------------------------------------------------------------------
    // F. Ground-only tilt proposal.
    //
    // We rotate current Ground normal toward Frozen reference with a bounded
    // step, then restore the baseline tangent heading around the NEW normal.
    // Therefore Ground changes tilt but does not own heading.
    // ------------------------------------------------------------------------
    Eigen::Vector3d forward_axis_L(
        config_.ground.forward_axis_x,
        config_.ground.forward_axis_y,
        config_.ground.forward_axis_z);

    const double forward_axis_norm =
        forward_axis_L.norm();

    if (!forward_axis_L.allFinite() ||
        !std::isfinite(forward_axis_norm) ||
        forward_axis_norm <= 1.0e-12)
    {
        return true;
    }

    forward_axis_L /=
        forward_axis_norm;

    const double maximum_tilt_rad =
        config_.ground.maximum_tilt_correction_deg *
        0.017453292519943295;

    const double bounded_tilt_rad =
        std::min(
            normal_residual_rad,
            std::max(
                0.0,
                maximum_tilt_rad));

    Eigen::Vector3d tilt_axis_W =
        current_normal_W.cross(
            reference_normal_W);

    const double tilt_axis_norm =
        tilt_axis_W.norm();

    if (normal_residual_rad > 1.0e-12)
    {
        if (!tilt_axis_W.allFinite() ||
            !std::isfinite(tilt_axis_norm) ||
            tilt_axis_norm <= 1.0e-12)
        {
            return true;
        }

        tilt_axis_W /=
            tilt_axis_norm;
    }

    // Runtime extrinsic, used only to evaluate candidate geometry.
    Eigen::Quaterniond q_IL =
        baseline_state.Q_IL.normalized();

    Eigen::Isometry3d T_IL =
        Eigen::Isometry3d::Identity();

    T_IL.linear() =
        q_IL.toRotationMatrix();

    T_IL.translation() =
        baseline_state.P_IL;

    const std::vector<double> alphas =
    {
        config_.ground.line_search_alpha_1,
        config_.ground.line_search_alpha_2,
        config_.ground.line_search_alpha_3
    };

    const double maximum_allowed_rmse =
        std::max(
            baseline_geometry.rmse *
                config_.ground.maximum_general_rmse_ratio,
            baseline_geometry.rmse +
                config_.ground
                    .maximum_general_rmse_absolute_increase_m);

    const double minimum_allowed_correspondences =
        config_.ground
            .minimum_general_correspondence_ratio *
        static_cast<double>(
            baseline_geometry.correspondences);

    std::string last_reject_reason =
        "NO_VALID_ALPHA";

    for (const double configured_alpha :
         alphas)
    {
        if (!std::isfinite(configured_alpha) ||
            configured_alpha <= 0.0)
        {
            continue;
        }

        const double alpha =
            std::clamp(
                configured_alpha,
                0.0,
                1.0);

        // ------------------------------------------------------------
        // F1. Alpha-scaled bounded tilt.
        // ------------------------------------------------------------
        Eigen::Matrix3d candidate_R_WL =
            baseline_T_WL.rotation();

        if (bounded_tilt_rad > 1.0e-12)
        {
            const double applied_tilt_rad =
                alpha *
                bounded_tilt_rad;

            const Eigen::Matrix3d tilt_rotation_W =
                Eigen::AngleAxisd(
                    applied_tilt_rad,
                    tilt_axis_W)
                    .toRotationMatrix();

            const Eigen::Matrix3d tilted_R_WL =
                tilt_rotation_W *
                baseline_T_WL.rotation();

            Eigen::Vector3d candidate_normal_W =
                tilt_rotation_W *
                current_normal_W;

            const double candidate_normal_norm =
                candidate_normal_W.norm();

            if (!candidate_normal_W.allFinite() ||
                candidate_normal_norm <= 1.0e-12)
            {
                last_reject_reason =
                    "TILT_NORMAL_INVALID";
                continue;
            }

            candidate_normal_W /=
                candidate_normal_norm;

            // Baseline heading projected onto the candidate Ground tangent.
            Eigen::Vector3d baseline_forward_W =
                baseline_T_WL.rotation() *
                forward_axis_L;

            baseline_forward_W -=
                candidate_normal_W *
                candidate_normal_W.dot(
                    baseline_forward_W);

            Eigen::Vector3d tilted_forward_W =
                tilted_R_WL *
                forward_axis_L;

            tilted_forward_W -=
                candidate_normal_W *
                candidate_normal_W.dot(
                    tilted_forward_W);

            const double baseline_heading_norm =
                baseline_forward_W.norm();

            const double tilted_heading_norm =
                tilted_forward_W.norm();

            if (!std::isfinite(
                    baseline_heading_norm) ||
                !std::isfinite(
                    tilted_heading_norm) ||
                baseline_heading_norm <
                    config_.ground
                        .minimum_heading_projection_norm ||
                tilted_heading_norm <
                    config_.ground
                        .minimum_heading_projection_norm)
            {
                last_reject_reason =
                    "HEADING_PROJECTION_INVALID";
                continue;
            }

            baseline_forward_W /=
                baseline_heading_norm;

            tilted_forward_W /=
                tilted_heading_norm;

            const double heading_cosine =
                std::clamp(
                    tilted_forward_W.dot(
                        baseline_forward_W),
                    -1.0,
                    1.0);

            const double heading_sine =
                candidate_normal_W.dot(
                    tilted_forward_W.cross(
                        baseline_forward_W));

            const double heading_correction_rad =
                std::atan2(
                    heading_sine,
                    heading_cosine);

            candidate_R_WL =
                Eigen::AngleAxisd(
                    heading_correction_rad,
                    candidate_normal_W)
                    .toRotationMatrix() *
                tilted_R_WL;
        }

        if (!candidate_R_WL.allFinite())
        {
            last_reject_reason =
                "CANDIDATE_ROTATION_INVALID";
            continue;
        }

        // ------------------------------------------------------------
        // F2. Per-alpha Ground median Z correction.
        //
        // x/y remain EXACTLY baseline.
        // ------------------------------------------------------------
        double height_after_tilt_m =
            0.0;

        if (!median_ground_height_residual(
                candidate_R_WL,
                baseline_T_WL.translation(),
                height_after_tilt_m))
        {
            last_reject_reason =
                "HEIGHT_MEDIAN_FAILED";
            continue;
        }

        const double raw_delta_z_m =
            -height_after_tilt_m /
            reference_normal_W.z();

        const double bounded_delta_z_m =
            std::clamp(
                raw_delta_z_m,
                -config_.ground.maximum_z_correction_m,
                config_.ground.maximum_z_correction_m);

        const double applied_delta_z_m =
            alpha *
            bounded_delta_z_m;

        Eigen::Isometry3d candidate_T_WL =
            baseline_T_WL;

        candidate_T_WL.linear() =
            candidate_R_WL;

        candidate_T_WL.translation().z() +=
            applied_delta_z_m;

        if (!candidate_T_WL.matrix().allFinite())
        {
            last_reject_reason =
                "CANDIDATE_POSE_INVALID";
            continue;
        }

        // ------------------------------------------------------------
        // G. Hard correction safety budget.
        // ------------------------------------------------------------
        const double translation_correction_m =
            (candidate_T_WL.translation() -
             baseline_T_WL.translation())
                .norm();

        const Eigen::AngleAxisd rotation_correction(
            candidate_T_WL.rotation() *
            baseline_T_WL.rotation().transpose());

        const double rotation_correction_deg =
            std::abs(
                rotation_correction.angle()) *
            57.29577951308232;

        if (!std::isfinite(
                translation_correction_m) ||
            !std::isfinite(
                rotation_correction_deg) ||
            translation_correction_m >
                config_.ground
                    .maximum_total_translation_correction_m +
                    1.0e-9 ||
            rotation_correction_deg >
                config_.ground
                    .maximum_total_rotation_correction_deg +
                    1.0e-9)
        {
            last_reject_reason =
                "CORRECTION_SAFETY_GATE";
            continue;
        }

        // Ground residual must not become worse.
        Eigen::Vector3d candidate_normal_W =
            candidate_R_WL *
            normal_L;

        candidate_normal_W.normalize();

        const double candidate_normal_deg =
            std::acos(
                std::clamp(
                    candidate_normal_W.dot(
                        reference_normal_W),
                    -1.0,
                    1.0)) *
            57.29577951308232;

        double candidate_height_residual_m =
            0.0;

        if (!median_ground_height_residual(
                candidate_R_WL,
                candidate_T_WL.translation(),
                candidate_height_residual_m))
        {
            last_reject_reason =
                "GROUND_AFTER_INVALID";
            continue;
        }

        if (candidate_normal_deg >
                normal_residual_deg +
                    config_.ground
                        .ground_sanity_normal_tolerance_deg ||
            std::abs(candidate_height_residual_m) >
                std::abs(
                    baseline_height_residual_m) +
                    config_.ground
                        .ground_sanity_height_tolerance_m)
        {
            last_reject_reason =
                "GROUND_SANITY_GATE";
            continue;
        }

        // ------------------------------------------------------------
        // H. Build a temporary candidate navigation state for GENERAL
        // geometry evaluation. No state is committed yet.
        // ------------------------------------------------------------
        const Eigen::Isometry3d candidate_T_WI =
            candidate_T_WL *
            T_IL.inverse();

        Eigen::Quaterniond candidate_Q_WI(
            candidate_T_WI.rotation());

        if (!candidate_Q_WI.coeffs().allFinite() ||
            candidate_Q_WI.norm() <= 1.0e-12 ||
            !candidate_T_WI.translation().allFinite())
        {
            last_reject_reason =
                "CANDIDATE_STATE_INVALID";
            continue;
        }

        candidate_Q_WI.normalize();

        LioState candidate_state =
            baseline_state;

        candidate_state.Q_WI =
            candidate_Q_WI;

        candidate_state.P_WI =
            candidate_T_WI.translation();

        LioMeasurementResult candidate_geometry;

        if (!measurement_builder_.Build(
                candidate_state,
                registration_cloud_L,
                *prepared_target,
                candidate_geometry) ||
            !candidate_geometry.success ||
            !std::isfinite(
                candidate_geometry.rmse))
        {
            last_reject_reason =
                "GENERAL_GEOMETRY_EVALUATION_FAILED";
            continue;
        }

        if (candidate_geometry.rmse >
            maximum_allowed_rmse)
        {
            last_reject_reason =
                "GENERAL_RMSE_SAFETY_GATE";
            continue;
        }

        if (static_cast<double>(
                candidate_geometry.correspondences) <
            minimum_allowed_correspondences)
        {
            last_reject_reason =
                "GENERAL_CORRESPONDENCE_SAFETY_GATE";
            continue;
        }

        // ------------------------------------------------------------
        // I. ACCEPT.
        //
        // Only Q_WI / P_WI change. Covariance, velocity, bias, gravity and
        // physical extrinsic remain exactly the LiDAR-IESKF result.
        // ------------------------------------------------------------
        // ------------------------------------------------------------
        // I. Ground V2-A state-consistent pseudo-measurement.
        //
        // Direct Ground observations:
        //
        //   - tilt
        //   - height
        //   - Ground-normal velocity
        //
        // The update is applied through the IESKF so both nominal state
        // and covariance are updated consistently.
        // ------------------------------------------------------------
        constexpr double kGroundNormalVelocitySigmaMps =
            0.10;

        const double ground_normal_sigma_rad =
            config_.ground.normal_sigma_deg *
            (3.14159265358979323846 / 180.0);

        const double normal_velocity_before =
            reference_normal_W.dot(
                baseline_state.V_WI);

        if (!ieskf_.GroundStateUpdate(
                candidate_T_WL,
                reference_normal_W,
                ground_normal_sigma_rad,
                config_.ground.height_sigma_m,
                kGroundNormalVelocitySigmaMps))
        {
            last_reject_reason =
                "GROUND_STATE_UPDATE_FAILED";
            continue;
        }

        const LioState ground_updated_state =
            ieskf_.State();

        const double normal_velocity_after =
            reference_normal_W.dot(
                ground_updated_state.V_WI);

        const double normal_velocity_correction =
            normal_velocity_after -
            normal_velocity_before;

        std::cout
            << "LIO_GROUND_APPLIED"
            << " | alpha="
            << alpha
            << " | normal="
            << normal_residual_deg
            << "->"
            << candidate_normal_deg
            << " deg"
            << " | height="
            << baseline_height_residual_m
            << "->"
            << candidate_height_residual_m
            << " m"
            << " | dz_req="
            << raw_delta_z_m
            << " | dz_apply="
            << applied_delta_z_m
            << " | general_rmse="
            << baseline_geometry.rmse
            << "->"
            << candidate_geometry.rmse
            << " | corr="
            << baseline_geometry.correspondences
            << "->"
            << candidate_geometry.correspondences
            << " | vn="
            << normal_velocity_before
            << "->"
            << normal_velocity_after
            << " | dvn="
            << normal_velocity_correction
            << " | vz="
            << baseline_state.V_WI.z()
            << "->"
            << ground_updated_state.V_WI.z()

            << " | z="
            
<< baseline_T_WL.translation().z()
            << "->"
            << candidate_T_WL.translation().z()
            << std::endl;

        return true;
    }

    static std::size_t ground_reject_counter =
        0;

    ++ground_reject_counter;

    if (ground_reject_counter % 10 == 0)
    {
        std::cout
            << "LIO_GROUND_REJECT"
            << " | reason="
            << last_reject_reason
            << " | normal_res_deg="
            << normal_residual_deg
            << " | height_res="
            << baseline_height_residual_m
            << " | baseline_rmse="
            << baseline_geometry.rmse
            << std::endl;
    }

    return true;
}



bool LioFrontend::ProcessWallMeasurement(
    const fr_slam::WallAssociationResult &wall_association,
    const pcl::PointCloud<LIDAR_POINT>::ConstPtr &registration_cloud_L,
    const PreparedLidarTarget *prepared_target,
    const LioSubmapContext &submap_context)
{
    // ========================================================================
    // Final V1 Wall Pose Constraint - Stage A.
    //
    // DIAGNOSTICS / PROPOSAL ONLY.
    //
    // Wall owns ONLY:
    //
    //     1. translation along trusted wall normals
    //     2. heading rotation around the Submap up axis
    //
    // Wall does NOT own:
    //
    //     z / roll / pitch / along-wall translation / velocity
    //
    // No IESKF state is changed in this stage.
    // ========================================================================

    (void)registration_cloud_L;
    (void)prepared_target;

    if (!submap_context.valid ||
        !submap_context.T_O_S_creation.matrix().allFinite() ||
        wall_association.active_static_walls.empty())
    {
        return true;
    }

    const LioState baseline_state =
        ieskf_.State();

    const Eigen::Isometry3d baseline_T_OL =
        StateToLidarPose(
            baseline_state);

    if (!baseline_T_OL.matrix().allFinite())
    {
        return true;
    }

    // ------------------------------------------------------------------------
    // Association lives in the immutable PRIMARY Submap frame S.
    //
    // For Stage A we use odom-frame +Z expressed in S as the heading axis.
    // Ground has already corrected roll / pitch before this function.
    //
    // Later, if needed, this can be replaced by the frozen Ground normal
    // without changing the Wall association representation.
    // ------------------------------------------------------------------------
    
    if (!ground_reference_frozen_ ||
        !ground_reference_normal_W_.allFinite() ||
        ground_reference_normal_W_.norm() <= 1.0e-9)
    {
        return true;
    }

    const Eigen::Vector3d heading_axis_O =
        ground_reference_normal_W_.normalized();

    Eigen::Vector3d up_S =
        submap_context.T_O_S_creation
            .rotation()
            .transpose() *
        heading_axis_O;


    const double up_norm =
        up_S.norm();

    if (!up_S.allFinite() ||
        !std::isfinite(up_norm) ||
        up_norm <= 1.0e-12)
    {
        return true;
    }

    up_S /=
        up_norm;

    // Conservative first-pass gates, selected from the observed ACTIVE_STATIC
    // distribution.
    constexpr double kMinimumQuality =
        0.85;

    constexpr double kMaximumNormalDifferenceDeg =
        1.5;

    constexpr double kMaximumPlaneDistanceDifferenceM =
        0.10;

    
constexpr double kMaximumTranslationCorrectionM =
        0.03;


    
constexpr double kMaximumYawCorrectionDeg =
        0.50;


    Eigen::Matrix3d translation_H =
        Eigen::Matrix3d::Zero();

    Eigen::Vector3d translation_b =
        Eigen::Vector3d::Zero();

    double yaw_weighted_sum =
        0.0;

    double yaw_weight_sum =
        0.0;

    std::size_t used_walls =
        0;

    for (const fr_slam::ActiveWallAssociation &wall :
         wall_association.active_static_walls)
    {
        if (!wall.reference_normal_A.allFinite() ||
            !wall.observed_normal_A.allFinite() ||
            !std::isfinite(wall.reference_d_A) ||
            !std::isfinite(wall.observed_d_A) ||
            !std::isfinite(wall.quality) ||
            !std::isfinite(wall.normal_difference_deg) ||
            !std::isfinite(
                wall.plane_distance_difference_m))
        {
            continue;
        }

        if (wall.quality <
                kMinimumQuality ||
            wall.normal_difference_deg >
                kMaximumNormalDifferenceDeg ||
            std::abs(
                wall.plane_distance_difference_m) >
                kMaximumPlaneDistanceDifferenceM)
        {
            continue;
        }

        // ------------------------------------------------------------
        // Project reference / observed wall normals into the Ground
        // tangent plane.
        // ------------------------------------------------------------
        Eigen::Vector3d reference_normal =
            wall.reference_normal_A -
            up_S *
                up_S.dot(
                    wall.reference_normal_A);

        Eigen::Vector3d observed_normal =
            wall.observed_normal_A -
            up_S *
                up_S.dot(
                    wall.observed_normal_A);

        const double reference_norm =
            reference_normal.norm();

        const double observed_norm =
            observed_normal.norm();

        if (!std::isfinite(reference_norm) ||
            !std::isfinite(observed_norm) ||
            reference_norm <= 1.0e-6 ||
            observed_norm <= 1.0e-6)
        {
            continue;
        }

        reference_normal /=
            reference_norm;

        observed_normal /=
            observed_norm;

        // Association already aligns observation sign with its frozen
        // persistent reference. Guard it again before pose solving.
        if (reference_normal.dot(
                observed_normal) < 0.0)
        {
            observed_normal =
                -observed_normal;
        }

        const double weight =
            std::clamp(
                wall.quality,
                0.0,
                1.0);

        if (weight <= 1.0e-6)
        {
            continue;
        }

        // ------------------------------------------------------------
        // Translation residual.
        //
        // Plane:
        //
        //     n^T p + d = 0
        //
        // Under pose translation correction dt:
        //
        //     d_obs' = d_obs - n^T dt
        //
        // therefore:
        //
        //     n^T dt = d_obs - d_ref
        //
        // so for one wall:
        //
        //     dt = (d_obs - d_ref) n
        //
        // Multiple walls are solved simultaneously below.
        // ------------------------------------------------------------
        const double distance_residual =
            wall.observed_d_A -
            wall.reference_d_A;

        translation_H +=
            weight *
            reference_normal *
            reference_normal.transpose();

        translation_b +=
            weight *
            distance_residual *
            reference_normal;

        // ------------------------------------------------------------
        // Heading residual:
        //
        // signed angle from current observed normal toward frozen
        // reference normal around up_S.
        // ------------------------------------------------------------
        const double yaw_sine =
            up_S.dot(
                observed_normal.cross(
                    reference_normal));

        const double yaw_cosine =
            std::clamp(
                observed_normal.dot(
                    reference_normal),
                -1.0,
                1.0);

        const double yaw_residual =
            std::atan2(
                yaw_sine,
                yaw_cosine);

        if (!std::isfinite(distance_residual) ||
            !std::isfinite(yaw_residual))
        {
            continue;
        }

        yaw_weighted_sum +=
            weight *
            yaw_residual;

        yaw_weight_sum +=
            weight;

        ++used_walls;
    }

    if (used_walls == 0 ||
        yaw_weight_sum <= 1.0e-12)
    {
        return true;
    }

    // ------------------------------------------------------------------------
    // Pseudoinverse translation solve.
    //
    // One wall:
    //     rank = 1
    //     only wall-normal translation is observable.
    //
    // Two nonparallel walls:
    //     rank = 2
    //     both tangent-plane translation axes can become observable.
    //
    // No artificial along-wall correction is introduced.
    // ------------------------------------------------------------------------
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>
        eigen_solver(
            translation_H);

    if (eigen_solver.info() !=
        Eigen::Success)
    {
        return true;
    }

    const Eigen::Vector3d eigenvalues =
        eigen_solver.eigenvalues();

    const Eigen::Matrix3d eigenvectors =
        eigen_solver.eigenvectors();

    const double maximum_eigenvalue =
        eigenvalues.maxCoeff();

    if (!eigenvalues.allFinite() ||
        !eigenvectors.allFinite() ||
        !std::isfinite(maximum_eigenvalue) ||
        maximum_eigenvalue <= 1.0e-12)
    {
        return true;
    }

    Eigen::Vector3d inverse_eigenvalues =
        Eigen::Vector3d::Zero();

    std::size_t observable_rank =
        0;

    const double eigen_threshold =
        std::max(
            1.0e-6,
            maximum_eigenvalue *
                1.0e-3);

    for (int i = 0;
         i < 3;
         ++i)
    {
        if (eigenvalues(i) >
            eigen_threshold)
        {
            inverse_eigenvalues(i) =
                1.0 /
                eigenvalues(i);

            ++observable_rank;
        }
    }

    Eigen::Vector3d raw_delta_t_S =
        eigenvectors *
        inverse_eigenvalues.asDiagonal() *
        eigenvectors.transpose() *
        translation_b;

    if (!raw_delta_t_S.allFinite())
    {
        return true;
    }

    // Numerical protection: Wall translation is tangent to the Ground axis.
    raw_delta_t_S -=
        up_S *
        up_S.dot(
            raw_delta_t_S);

    const double raw_translation_norm =
        raw_delta_t_S.norm();

    Eigen::Vector3d bounded_delta_t_S =
        raw_delta_t_S;

    if (std::isfinite(raw_translation_norm) &&
        raw_translation_norm >
            kMaximumTranslationCorrectionM &&
        raw_translation_norm >
            1.0e-12)
    {
        bounded_delta_t_S *=
            kMaximumTranslationCorrectionM /
            raw_translation_norm;
    }

    const double raw_yaw_rad =
        yaw_weighted_sum /
        yaw_weight_sum;

    const double maximum_yaw_rad =
        kMaximumYawCorrectionDeg *
        0.017453292519943295;

    const double bounded_yaw_rad =
        std::clamp(
            raw_yaw_rad,
            -maximum_yaw_rad,
            maximum_yaw_rad);

    if (!std::isfinite(raw_yaw_rad) ||
        !std::isfinite(bounded_yaw_rad) ||
        !bounded_delta_t_S.allFinite())
    {
        return true;
    }

    std::cout
        << "LIO_WALL_PROPOSAL"
        << " | submap="
        << submap_context.primary_submap_id
        << " | used="
        << used_walls
        << " | rank="
        << observable_rank
        << " | raw_dt_S=["
        << raw_delta_t_S.x()
        << ","
        << raw_delta_t_S.y()
        << ","
        << raw_delta_t_S.z()
        << "]"
        << " | raw_dt_norm="
        << raw_delta_t_S.norm()
        << " | bounded_dt_S=["
        << bounded_delta_t_S.x()
        << ","
        << bounded_delta_t_S.y()
        << ","
        << bounded_delta_t_S.z()
        << "]"
        << " | raw_yaw_deg="
        << raw_yaw_rad *
               57.29577951308232
        << " | bounded_yaw_deg="
        << bounded_yaw_rad *
               57.29577951308232
        << std::endl;

    // ========================================================================
    // Wall temporal injection controller.
    //
    // Goals:
    //   1. ignore tiny frame-to-frame Wall noise;
    //   2. require three consecutive, direction-consistent proposals;
    //   3. smooth the proposal with EMA;
    //   4. after one injection, restart confirmation from zero.
    //
    // Therefore Wall can inject at most once every three valid proposal
    // frames, instead of once per frame.
    // ========================================================================

    constexpr double kWallTranslationDeadbandM =
        0.005;

    constexpr double kWallYawDeadbandDeg =
        0.05;

    constexpr double kWallYawDeadbandRad =
        kWallYawDeadbandDeg *
        0.017453292519943295;

    constexpr double kWallDirectionCosineThreshold =
        0.80;

    constexpr std::size_t kWallRequiredConsistentFrames =
        3;

    constexpr double kWallEmaAlpha =
        0.40;

    const double temporal_translation_norm =
        bounded_delta_t_S.norm();

    const double temporal_abs_yaw_rad =
        std::abs(
            bounded_yaw_rad);

    const bool translation_active =
        std::isfinite(
            temporal_translation_norm) &&
        temporal_translation_norm >=
            kWallTranslationDeadbandM;

    const bool yaw_active =
        std::isfinite(
            temporal_abs_yaw_rad) &&
        temporal_abs_yaw_rad >=
            kWallYawDeadbandRad;

    // ------------------------------------------------------------------------
    // Deadband.
    //
    // If neither translation nor heading contains a meaningful correction,
    // break the temporal chain completely.
    // ------------------------------------------------------------------------
    if (!translation_active &&
        !yaw_active)
    {
        wall_temporal_initialized_ =
            false;

        wall_consistent_frames_ =
            0;

        wall_temporal_last_frame_index_ =
            0;

        wall_previous_delta_t_S_.setZero();

        wall_previous_yaw_rad_ =
            0.0;

        wall_filtered_delta_t_S_.setZero();

        wall_filtered_yaw_rad_ =
            0.0;

        return true;
    }

    const Eigen::Vector3d temporal_input_delta_t_S =
        translation_active
            ? bounded_delta_t_S
            : Eigen::Vector3d::Zero();

    const double temporal_input_yaw_rad =
        yaw_active
            ? bounded_yaw_rad
            : 0.0;

    // ------------------------------------------------------------------------
    // First valid proposal, or a proposal after a frame gap.
    //
    // A temporal sequence must consist of consecutive Wall association
    // frames. Missing Wall observations break confirmation.
    // ------------------------------------------------------------------------
    const bool frame_is_consecutive =
        wall_temporal_initialized_ &&
        wall_temporal_last_frame_index_ > 0 &&
        wall_association_frame_index_ ==
            wall_temporal_last_frame_index_ + 1;

    if (!wall_temporal_initialized_ ||
        !frame_is_consecutive)
    {
        wall_temporal_initialized_ =
            true;

        wall_consistent_frames_ =
            1;

        wall_temporal_last_frame_index_ =
            wall_association_frame_index_;

        wall_previous_delta_t_S_ =
            temporal_input_delta_t_S;

        wall_previous_yaw_rad_ =
            temporal_input_yaw_rad;

        wall_filtered_delta_t_S_ =
            temporal_input_delta_t_S;

        wall_filtered_yaw_rad_ =
            temporal_input_yaw_rad;

        return true;
    }

    // ------------------------------------------------------------------------
    // Translation direction consistency.
    //
    // A newly appearing translation component starts a new sequence.
    // ------------------------------------------------------------------------
    const double previous_translation_norm =
        wall_previous_delta_t_S_.norm();

    const bool previous_translation_active =
        std::isfinite(
            previous_translation_norm) &&
        previous_translation_norm >=
            kWallTranslationDeadbandM;

    bool translation_consistent =
        true;

    if (translation_active)
    {
        if (!previous_translation_active)
        {
            translation_consistent =
                false;
        }
        else
        {
            const double direction_cosine =
                bounded_delta_t_S.dot(
                    wall_previous_delta_t_S_) /
                (temporal_translation_norm *
                 previous_translation_norm);

            translation_consistent =
                std::isfinite(
                    direction_cosine) &&
                direction_cosine >=
                    kWallDirectionCosineThreshold;
        }
    }

    // ------------------------------------------------------------------------
    // Heading sign consistency.
    //
    // A newly appearing yaw component also starts a new sequence.
    // ------------------------------------------------------------------------
    const bool previous_yaw_active =
        std::isfinite(
            wall_previous_yaw_rad_) &&
        std::abs(
            wall_previous_yaw_rad_) >=
            kWallYawDeadbandRad;

    bool yaw_consistent =
        true;

    if (yaw_active)
    {
        if (!previous_yaw_active)
        {
            yaw_consistent =
                false;
        }
        else
        {
            yaw_consistent =
                bounded_yaw_rad *
                    wall_previous_yaw_rad_ >
                0.0;
        }
    }

    // ------------------------------------------------------------------------
    // Direction changed:
    // current proposal becomes frame #1 of a new sequence.
    // ------------------------------------------------------------------------
    if (!translation_consistent ||
        !yaw_consistent)
    {
        wall_consistent_frames_ =
            1;

        wall_temporal_last_frame_index_ =
            wall_association_frame_index_;

        wall_previous_delta_t_S_ =
            temporal_input_delta_t_S;

        wall_previous_yaw_rad_ =
            temporal_input_yaw_rad;

        wall_filtered_delta_t_S_ =
            temporal_input_delta_t_S;

        wall_filtered_yaw_rad_ =
            temporal_input_yaw_rad;

        return true;
    }

    // ------------------------------------------------------------------------
    // Consistent proposal: update EMA.
    //
    // Inactive dimensions are forced to zero instead of carrying stale
    // correction from a previous frame.
    // ------------------------------------------------------------------------
    ++wall_consistent_frames_;

    wall_temporal_last_frame_index_ =
        wall_association_frame_index_;

    if (translation_active)
    {
        wall_filtered_delta_t_S_ =
            kWallEmaAlpha *
                bounded_delta_t_S +
            (1.0 - kWallEmaAlpha) *
                wall_filtered_delta_t_S_;
    }
    else
    {
        wall_filtered_delta_t_S_.setZero();
    }

    if (yaw_active)
    {
        wall_filtered_yaw_rad_ =
            kWallEmaAlpha *
                bounded_yaw_rad +
            (1.0 - kWallEmaAlpha) *
                wall_filtered_yaw_rad_;
    }
    else
    {
        wall_filtered_yaw_rad_ =
            0.0;
    }

    wall_previous_delta_t_S_ =
        temporal_input_delta_t_S;

    wall_previous_yaw_rad_ =
        temporal_input_yaw_rad;

    if (wall_consistent_frames_ <
        kWallRequiredConsistentFrames)
    {
        return true;
    }

    // ------------------------------------------------------------------------
    // Three-frame confirmation completed.
    //
    // Copy the filtered correction to local variables. Temporal state is then
    // cleared immediately. Whether Stage B accepts or rejects this candidate,
    // another injection requires a NEW three-frame confirmation.
    // ------------------------------------------------------------------------
    const Eigen::Vector3d wall_injection_delta_t_S =
        wall_filtered_delta_t_S_;

    const double wall_injection_yaw_rad =
        wall_filtered_yaw_rad_;

    std::cout
        << "LIO_WALL_TEMPORAL_READY"
        << " | submap="
        << submap_context.primary_submap_id
        << " | frame="
        << wall_association_frame_index_
        << " | confirmed="
        << wall_consistent_frames_
        << " | raw_dt_norm="
        << bounded_delta_t_S.norm()
        << " | filtered_dt_norm="
        << wall_injection_delta_t_S.norm()
        << " | raw_yaw_deg="
        << bounded_yaw_rad *
               57.29577951308232
        << " | filtered_yaw_deg="
        << wall_injection_yaw_rad *
               57.29577951308232
        << std::endl;

    wall_temporal_initialized_ =
        false;

    wall_consistent_frames_ =
        0;

    wall_temporal_last_frame_index_ =
        0;

    wall_previous_delta_t_S_.setZero();

    wall_previous_yaw_rad_ =
        0.0;

    wall_filtered_delta_t_S_.setZero();

    wall_filtered_yaw_rad_ =
        0.0;

    // ========================================================================
    // Stage B: guarded Wall pose injection.
    //
    // Wall owns:
    //   - observable wall-normal translation
    //   - heading around the frozen Ground normal
    //
    // Wall does NOT modify velocity / Ground-normal position / tilt.
    // ========================================================================

    if (!registration_cloud_L ||
        registration_cloud_L->empty() ||
        prepared_target == nullptr)
    {
        return true;
    }

    if (wall_injection_delta_t_S.norm() <= 1.0e-12 &&
        std::abs(wall_injection_yaw_rad) <= 1.0e-12)
    {
        return true;
    }

    // ------------------------------------------------------------------------
    // Baseline general LiDAR geometry.
    // ------------------------------------------------------------------------
    LioMeasurementResult baseline_geometry;

    if (!measurement_builder_.Build(
            baseline_state,
            registration_cloud_L,
            *prepared_target,
            baseline_geometry) ||
        !baseline_geometry.success ||
        !std::isfinite(
            baseline_geometry.rmse))
    {
        return true;
    }

    // ------------------------------------------------------------------------
    // Runtime LiDAR -> IMU extrinsic.
    // ------------------------------------------------------------------------
    if (!baseline_state.Q_IL.coeffs().allFinite() ||
        baseline_state.Q_IL.norm() <= 1.0e-12 ||
        !baseline_state.P_IL.allFinite())
    {
        return true;
    }

    Eigen::Quaterniond q_IL =
        baseline_state.Q_IL.normalized();

    Eigen::Isometry3d T_IL =
        Eigen::Isometry3d::Identity();

    T_IL.linear() =
        q_IL.toRotationMatrix();

    T_IL.translation() =
        baseline_state.P_IL;

    // Reuse the already verified Ground general-geometry safety policy.
    const double maximum_allowed_rmse =
        std::max(
            baseline_geometry.rmse *
                config_.ground.maximum_general_rmse_ratio,
            baseline_geometry.rmse +
                config_.ground
                    .maximum_general_rmse_absolute_increase_m);

    const double minimum_allowed_correspondences =
        config_.ground
            .minimum_general_correspondence_ratio *
        static_cast<double>(
            baseline_geometry.correspondences);

    const std::array<double, 3> wall_alphas =
    {
        1.0,
        0.5,
        0.25
    };

    std::string last_wall_reject_reason =
        "NO_VALID_ALPHA";

    for (const double alpha :
         wall_alphas)
    {
        const Eigen::Vector3d applied_delta_t_S =
            alpha *
            wall_injection_delta_t_S;

        const double applied_yaw_rad =
            alpha *
            wall_injection_yaw_rad;

        // ------------------------------------------------------------
        // Wall-specific objective gate.
        //
        // Translation objective:
        //
        // J(dt) - J(0)
        //   = dt^T H dt - 2 dt^T b
        //
        // Heading objective:
        //
        // J(dpsi) - J(0)
        //   = W*dpsi^2 - 2*dpsi*sum(w*r)
        //
        // Negative means improvement.
        // ------------------------------------------------------------
        const double translation_cost_delta =
            applied_delta_t_S.dot(
                translation_H *
                applied_delta_t_S) -
            2.0 *
            applied_delta_t_S.dot(
                translation_b);

        const double yaw_cost_delta =
            yaw_weight_sum *
                applied_yaw_rad *
                applied_yaw_rad -
            2.0 *
                applied_yaw_rad *
                yaw_weighted_sum;

        if (!std::isfinite(
                translation_cost_delta) ||
            !std::isfinite(
                yaw_cost_delta))
        {
            last_wall_reject_reason =
                "WALL_COST_INVALID";
            continue;
        }

        constexpr double kWallCostTolerance =
            1.0e-12;

        const bool translation_not_worse =
            translation_cost_delta <=
            kWallCostTolerance;

        const bool yaw_not_worse =
            yaw_cost_delta <=
            kWallCostTolerance;

        const bool wall_improved =
            translation_cost_delta <
                -kWallCostTolerance ||
            yaw_cost_delta <
                -kWallCostTolerance;

        if (!translation_not_worse ||
            !yaw_not_worse ||
            !wall_improved)
        {
            last_wall_reject_reason =
                "WALL_RESIDUAL_GATE";
            continue;
        }

        // ------------------------------------------------------------
        // Convert S-frame translation correction into odom/world frame.
        //
        // p_O = T_O_S_creation * p_S
        // ------------------------------------------------------------
        Eigen::Vector3d applied_delta_t_O =
            submap_context
                .T_O_S_creation
                .rotation() *
            applied_delta_t_S;

        // Numerical protection:
        // Wall must remain tangent to the frozen Ground plane.
        applied_delta_t_O -=
            heading_axis_O *
            heading_axis_O.dot(
                applied_delta_t_O);

        if (!applied_delta_t_O.allFinite())
        {
            last_wall_reject_reason =
                "WALL_TRANSLATION_INVALID";
            continue;
        }

        Eigen::Isometry3d candidate_T_OL =
            baseline_T_OL;

        candidate_T_OL.translation() +=
            applied_delta_t_O;

        const Eigen::Matrix3d heading_rotation_O =
            Eigen::AngleAxisd(
                applied_yaw_rad,
                heading_axis_O)
                .toRotationMatrix();

        candidate_T_OL.linear() =
            heading_rotation_O *
            baseline_T_OL.rotation();

        if (!candidate_T_OL.matrix().allFinite())
        {
            last_wall_reject_reason =
                "CANDIDATE_POSE_INVALID";
            continue;
        }

        // ------------------------------------------------------------
        // Build temporary candidate IESKF state.
        // ------------------------------------------------------------
        const Eigen::Isometry3d candidate_T_WI =
            candidate_T_OL *
            T_IL.inverse();

        Eigen::Quaterniond candidate_Q_WI(
            candidate_T_WI.rotation());

        if (!candidate_Q_WI.coeffs().allFinite() ||
            candidate_Q_WI.norm() <= 1.0e-12 ||
            !candidate_T_WI.translation().allFinite())
        {
            last_wall_reject_reason =
                "CANDIDATE_STATE_INVALID";
            continue;
        }

        candidate_Q_WI.normalize();

        LioState candidate_state =
            baseline_state;

        candidate_state.Q_WI =
            candidate_Q_WI;

        candidate_state.P_WI =
            candidate_T_WI.translation();

        // Wall NEVER changes velocity.
        candidate_state.V_WI =
            baseline_state.V_WI;

        // ------------------------------------------------------------
        // General LiDAR geometry safety gate.
        // ------------------------------------------------------------
        LioMeasurementResult candidate_geometry;

        if (!measurement_builder_.Build(
                candidate_state,
                registration_cloud_L,
                *prepared_target,
                candidate_geometry) ||
            !candidate_geometry.success ||
            !std::isfinite(
                candidate_geometry.rmse))
        {
            last_wall_reject_reason =
                "GENERAL_GEOMETRY_EVALUATION_FAILED";
            continue;
        }

        if (candidate_geometry.rmse >
            maximum_allowed_rmse)
        {
            last_wall_reject_reason =
                "GENERAL_RMSE_SAFETY_GATE";
            continue;
        }

        if (static_cast<double>(
                candidate_geometry.correspondences) <
            minimum_allowed_correspondences)
        {
            last_wall_reject_reason =
                "GENERAL_CORRESPONDENCE_SAFETY_GATE";
            continue;
        }

        // ------------------------------------------------------------
        // ACCEPT.
        //
        // Preserve the Ground-corrected velocity exactly.
        // ------------------------------------------------------------
        if (!ieskf_.InjectCorrectedLidarPose(
                candidate_T_OL,
                baseline_state.V_WI))
        {
            last_wall_reject_reason =
                "STATE_INJECTION_FAILED";
            continue;
        }

        std::cout
            << "LIO_WALL_APPLIED"
            << " | submap="
            << submap_context.primary_submap_id
            << " | alpha="
            << alpha
            << " | used="
            << used_walls
            << " | rank="
            << observable_rank
            << " | dt_S=["
            << applied_delta_t_S.x()
            << ","
            << applied_delta_t_S.y()
            << ","
            << applied_delta_t_S.z()
            << "]"
            << " | dt_norm="
            << applied_delta_t_S.norm()
            << " | yaw_deg="
            << applied_yaw_rad *
                   57.29577951308232
            << " | wall_dt_cost_delta="
            << translation_cost_delta
            << " | wall_yaw_cost_delta="
            << yaw_cost_delta
            << " | general_rmse="
            << baseline_geometry.rmse
            << "->"
            << candidate_geometry.rmse
            << " | corr="
            << baseline_geometry.correspondences
            << "->"
            << candidate_geometry.correspondences
            << " | velocity_delta="
            << (ieskf_.State().V_WI -
                baseline_state.V_WI).norm()
            << std::endl;

        return true;
    }

    static std::size_t wall_reject_counter =
        0;

    ++wall_reject_counter;

    if (wall_reject_counter % 10 == 0)
    {
        std::cout
            << "LIO_WALL_REJECT"
            << " | submap="
            << submap_context.primary_submap_id
            << " | reason="
            << last_wall_reject_reason
            << " | used="
            << used_walls
            << " | rank="
            << observable_rank
            << " | baseline_rmse="
            << baseline_geometry.rmse
            << std::endl;
    }

    return true;
}


bool LioFrontend::ProcessFrame(
    const LIDAR_FRAME &raw_frame,
    const std::vector<IMU_DATA> &imu_data,
    const PreparedLidarTarget *prepared_target,
    const LioSubmapContext &submap_context,
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
    // ========================================================================
    // 3.1 Dense Ground input.
    //
    // preprocess() publishes the Basic dense cloud BEFORE registration
    // Voxel/SOR/ROR. Ground must use this cloud rather than the sparse
    // registration cloud.
    //
    // The bridge is one-shot. Publish it back immediately because the
    // downstream mapping/Keyframe path also consumes this same dense cloud.
    // ========================================================================
    pcl::PointCloud<LIDAR_POINT>::ConstPtr ground_measurement_cloud =
        fr_slam::ConsumeGroundIcpDenseInput();

    const char *ground_measurement_source =
        "BASIC_BRIDGE";

    if (!ground_measurement_cloud ||
        ground_measurement_cloud->empty())
    {
        ground_measurement_cloud =
            result.processed_frame.cloud;

        ground_measurement_source =
            "REGISTRATION_FALLBACK";
    }

    if (ground_measurement_cloud &&
        !ground_measurement_cloud->empty())
    {
        fr_slam::PublishGroundIcpDenseInput(
            ground_measurement_cloud);
    }

    static std::size_t ground_input_debug_counter = 0;
    ++ground_input_debug_counter;

    if (ground_input_debug_counter % 20 == 0)
    {
        std::cout
            << "LIO_GROUND_INPUT"
            << " | source="
            << ground_measurement_source
            << " | dense="
            << (ground_measurement_cloud
                    ? ground_measurement_cloud->size()
                    : 0)
            << " | sparse="
            << result.processed_frame.cloud->size()
            << std::endl;
    }

    if (prepared_target == nullptr)
    {
        result.first_mapping_frame =
            true;

        

        
ProcessGroundMeasurement(
        ground_measurement_cloud,
        result.processed_frame.cloud,
        prepared_target,
        false);



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
    // ========================================================================
    // FR_JOINT_GROUND_V1
    //
    // Build/refresh the CURRENT-frame Ground segmentation BEFORE the
    // iterated update.  With allow_pose_correction=false this may bootstrap
    // the active Piecewise Frozen reference, but it NEVER modifies the state.
    // ========================================================================
    
    // ========================================================================
    // ========================================================================
    // FR_CONTINUOUS_GROUND_FINAL
    //
    // Mapping Submap changes NEVER invalidate the ACTIVE Ground.
    //
    // A new mapping submap is only a lifecycle hint:
    //   - keep ACTIVE Ground untouched
    //   - clear an unfinished PENDING candidate
    //
    // Ground Piece lifetime is now independent of Mapping Submap lifetime.
    // ========================================================================
    if (config_.ground.enabled &&
        config_.ground.mode == "piecewise_frozen" &&
        submap_context.valid &&
        ground_last_mapping_submap_id_ !=
            submap_context.primary_submap_id)
    {
        const std::size_t previous_mapping_submap_id =
            ground_last_mapping_submap_id_;

        ground_last_mapping_submap_id_ =
            submap_context.primary_submap_id;

        // Pending may be discarded at a mapping boundary.
        // ACTIVE is NEVER invalidated here.
        ground_pending_active_ =
            false;

        ground_pending_normals_W_.clear();
        ground_pending_anchors_W_.clear();

        ground_pending_sample_count_ =
            0;

        std::cout
            << "LIO_GROUND_SUBMAP_HINT"
            << " | previous="
            << previous_mapping_submap_id
            << " | current="
            << ground_last_mapping_submap_id_
            << " | active_piece="
            << ground_active_piece_id_
            << " | active_valid="
            << (ground_reference_frozen_ ? 1 : 0)
            << std::endl;
    }

    if (!ProcessGroundMeasurement(
            ground_measurement_cloud,
            result.processed_frame.cloud,
            prepared_target,
            false))
    {
        std::cerr
            << "LIO_REJECT | stage=GROUND_PREPARE"
            << std::endl;
        return false;
    }

    // ========================================================================
    // FR_JOINT_WALL_V1
    //
    // Build CURRENT-frame Wall association BEFORE the IEKF update.
    // The resulting association + baseline pose are frozen during all
    // inner IEKF iterations.
    // ========================================================================
    fr_slam::WallAssociationResult
        joint_wall_association;

    bool joint_wall_ready =
        false;

    Eigen::Isometry3d joint_wall_baseline_T_SL =
        Eigen::Isometry3d::Identity();

    Eigen::Vector3d joint_wall_up_S =
        Eigen::Vector3d::UnitZ();

    if (config_.wall_constraint_enable &&
        submap_context.valid &&
        submap_context.T_O_S_creation.matrix().allFinite())
    {
        if (wall_association_submap_id_ !=
            submap_context.primary_submap_id)
        {
            wall_association_.Reset();

            wall_association_frame_index_ = 0;

            wall_temporal_initialized_ = false;
            wall_consistent_frames_ = 0;
            wall_temporal_last_frame_index_ = 0;

            wall_previous_delta_t_S_.setZero();
            wall_previous_yaw_rad_ = 0.0;

            wall_filtered_delta_t_S_.setZero();
            wall_filtered_yaw_rad_ = 0.0;

            wall_association_submap_id_ =
                submap_context.primary_submap_id;

            std::cout
                << "LIO_JOINT_WALL_SUBMAP_RESET"
                << " | submap="
                << wall_association_submap_id_
                << std::endl;
        }

        if (ground_measurement_cloud &&
            !ground_measurement_cloud->empty())
        {
            const Eigen::Isometry3d baseline_T_OL_wall =
                StateToLidarPose(
                    ieskf_.State());

            if (baseline_T_OL_wall.matrix().allFinite())
            {
                const fr_slam::MultiPlaneExtractor
                    multi_plane_extractor;

                Eigen::Vector3d up_direction_L =
                    baseline_T_OL_wall
                        .rotation()
                        .transpose() *
                    Eigen::Vector3d::UnitZ();

                if (!up_direction_L.allFinite() ||
                    up_direction_L.norm() < 1.0e-9)
                {
                    up_direction_L =
                        Eigen::Vector3d::UnitZ();
                }

                fr_slam::MultiPlaneExtractionResult
                    multi_plane_result =
                        multi_plane_extractor.Extract(
                            ground_measurement_cloud,
                            up_direction_L);

                fr_slam::RefinePlaneConstraintEligibility(
                    multi_plane_result,
                    last_ground_segmentation_result_,
                    multi_plane_extractor.GetConfig(),
                    baseline_T_OL_wall.rotation(),
                    baseline_T_OL_wall.translation());

                fr_slam::StoreLatestMultiPlaneResult(
                    multi_plane_result);

                joint_wall_baseline_T_SL =
                    submap_context
                        .T_O_S_creation
                        .inverse() *
                    baseline_T_OL_wall;

                if (joint_wall_baseline_T_SL
                        .matrix()
                        .allFinite())
                {
                    ++wall_association_frame_index_;

                    result.wall_association =
                        wall_association_.Update(
                            multi_plane_result,
                            joint_wall_baseline_T_SL,
                            wall_association_frame_index_);

                    joint_wall_association =
                        result.wall_association;

                    Eigen::Vector3d heading_axis_O =
                        Eigen::Vector3d::UnitZ();

                    if (ground_reference_frozen_ &&
                        ground_reference_normal_W_.allFinite() &&
                        ground_reference_normal_W_.norm() >
                            1.0e-9)
                    {
                        heading_axis_O =
                            ground_reference_normal_W_
                                .normalized();
                    }

                    joint_wall_up_S =
                        submap_context
                            .T_O_S_creation
                            .rotation()
                            .transpose() *
                        heading_axis_O;

                    if (joint_wall_up_S.allFinite() &&
                        joint_wall_up_S.norm() >
                            1.0e-9)
                    {
                        joint_wall_up_S.normalize();

                        joint_wall_ready =
                            !joint_wall_association
                                 .active_static_walls
                                 .empty();
                    }

                    if (joint_wall_ready)
                    {
                        std::cout
                            << "LIO_JOINT_WALL_PREPARE"
                            << " | submap="
                            << submap_context.primary_submap_id
                            << " | frame="
                            << wall_association_frame_index_
                            << " | active_static="
                            << joint_wall_association
                                   .active_static_walls
                                   .size()
                            << std::endl;
                    }
                }
            }
        }
    }

    
    // ========================================================================
    // CONTINUOUS PIECEWISE FROZEN GROUND LIFECYCLE
    //
    // ACTIVE:
    //   Persistent and used by Joint IESKF.
    //
    // PENDING:
    //   Candidate only. It NEVER becomes a measurement directly.
    //
    // Switch:
    //   Pending must be temporally stable, then it replaces ACTIVE atomically
    //   using a continuity anchor projected onto the OLD ACTIVE plane.
    // ========================================================================

    bool ground_joint_measurement_allowed =
        true;

    const auto reset_pending_ground =
        [this]()
        {
            ground_pending_active_ =
                false;

            ground_pending_normals_W_.clear();
            ground_pending_anchors_W_.clear();

            ground_pending_sample_count_ =
                0;
        };

    // ------------------------------------------------------------------------
    // Initial bootstrap compatibility.
    //
    // Existing ProcessGroundMeasurement() still creates the FIRST frozen
    // reference after anchor_bootstrap_frames reliable measurements.
    //
    // Convert that initial plane representation into:
    //
    //     n_A^T (x - a_A) = 0
    //
    // exactly once.
    // ------------------------------------------------------------------------
    if (ground_reference_frozen_ &&
        !ground_active_anchor_valid_ &&
        ground_reference_normal_W_.allFinite() &&
        ground_reference_normal_W_.norm() >
            1.0e-12 &&
        std::isfinite(
            ground_reference_plane_d_W_))
    {
        ground_reference_normal_W_.normalize();

        ground_active_anchor_W_ =
            -ground_reference_plane_d_W_ *
            ground_reference_normal_W_;

        if (!ground_active_anchor_W_.allFinite())
        {
            std::cerr
                << "LIO_GROUND_ACTIVE_INIT_INVALID"
                << std::endl;

            return false;
        }

        ground_active_anchor_valid_ =
            true;

        ground_active_piece_id_ =
            0;

        // ------------------------------------------------------------
        // FR_GROUND_PHYSICAL_SLOPE_SWITCH_FINAL
        //
        // Initial ACTIVE physical slope:
        //
        //     theta = acos(|n_ground . up_gravity|)
        //
        // This quantity is independent of robot heading and Z position.
        // ------------------------------------------------------------
        ground_active_slope_valid_ =
            false;

        ground_active_slope_deg_ =
            0.0;

        const LioState ground_active_init_state =
            ieskf_.State();

        if (ground_active_init_state.gravity_W.allFinite() &&
            ground_active_init_state.gravity_W.norm() >
                1.0e-6)
        {
            const Eigen::Vector3d active_up_W =
                -ground_active_init_state
                     .gravity_W
                     .normalized();

            const double active_slope_cosine =
                std::clamp(
                    std::abs(
                        ground_reference_normal_W_
                            .normalized()
                            .dot(active_up_W)),
                    0.0,
                    1.0);

            ground_active_slope_deg_ =
                std::acos(
                    active_slope_cosine) *
                57.29577951308232;

            ground_active_slope_valid_ =
                std::isfinite(
                    ground_active_slope_deg_);

            std::cout
                << "LIO_GROUND_ACTIVE_SLOPE_INITIALIZED"
                << " | piece="
                << ground_active_piece_id_
                << " | slope_deg="
                << ground_active_slope_deg_
                << std::endl;
        }

        reset_pending_ground();

        std::cout
            << "LIO_GROUND_ACTIVE_INITIALIZED"
            << " | piece="
            << ground_active_piece_id_
            << " | n_W=["
            << ground_reference_normal_W_.transpose()
            << "]"
            << " | anchor_W=["
            << ground_active_anchor_W_.transpose()
            << "]"
            << std::endl;
    }

    // ------------------------------------------------------------------------
    // Evaluate CURRENT Ground against persistent ACTIVE Ground.
    // ------------------------------------------------------------------------
    const bool current_ground_candidate_valid =
        ground_reference_frozen_ &&
        ground_active_anchor_valid_ &&
        last_ground_segmentation_valid_ &&
        last_ground_segmentation_result_.success &&
        last_ground_segmentation_result_
            .support_plane_valid &&
        last_ground_segmentation_result_
            .support_constraint_valid &&
        last_ground_segmentation_result_
            .support_ground_normal_L.allFinite() &&
        std::isfinite(
            last_ground_segmentation_result_
                .support_ground_plane_d);

    if (current_ground_candidate_valid)
    {
        Eigen::Vector3d current_normal_L =
            last_ground_segmentation_result_
                .support_ground_normal_L;

        const double current_normal_L_norm =
            current_normal_L.norm();

        if (!std::isfinite(
                current_normal_L_norm) ||
            current_normal_L_norm <=
                1.0e-12)
        {
            ground_joint_measurement_allowed =
                false;
        }
        else
        {
            current_normal_L /=
                current_normal_L_norm;

            double current_plane_d_L =
                last_ground_segmentation_result_
                    .support_ground_plane_d /
                current_normal_L_norm;

            const LioState ground_lifecycle_state =
                ieskf_.State();

            const Eigen::Isometry3d T_WL_ground_lifecycle =
                StateToLidarPose(
                    ground_lifecycle_state);

            if (!T_WL_ground_lifecycle
                     .matrix()
                     .allFinite() ||
                !std::isfinite(
                    current_plane_d_L))
            {
                ground_joint_measurement_allowed =
                    false;
            }
            else
            {
                Eigen::Vector3d current_normal_W =
                    T_WL_ground_lifecycle.rotation() *
                    current_normal_L;

                const double current_normal_W_norm =
                    current_normal_W.norm();

                if (!current_normal_W.allFinite() ||
                    !std::isfinite(
                        current_normal_W_norm) ||
                    current_normal_W_norm <=
                        1.0e-12)
                {
                    ground_joint_measurement_allowed =
                        false;
                }
                else
                {
                    current_normal_W /=
                        current_normal_W_norm;

                    // Deterministic hemisphere relative to ACTIVE.
                    if (current_normal_W.dot(
                            ground_reference_normal_W_) <
                        0.0)
                    {
                        current_normal_W =
                            -current_normal_W;

                        current_plane_d_L =
                            -current_plane_d_L;

                        current_normal_L =
                            -current_normal_L;
                    }

                    const Eigen::Vector3d current_anchor_L =
                        -current_plane_d_L *
                        current_normal_L;

                    const Eigen::Vector3d current_anchor_W =
                        T_WL_ground_lifecycle *
                        current_anchor_L;

                    if (!current_anchor_W.allFinite())
                    {
                        ground_joint_measurement_allowed =
                            false;
                    }
                    else
                    {
                        const Eigen::Vector3d active_normal_W =
                            ground_reference_normal_W_
                                .normalized();

                        const double normal_dot =
                            std::clamp(
                                active_normal_W.dot(
                                    current_normal_W),
                                -1.0,
                                1.0);

                        const double active_normal_error_deg =
                            std::acos(
                                normal_dot) *
                            57.29577951308232;

                        const double active_height_error_m =
                            active_normal_W.dot(
                                current_anchor_W -
                                ground_active_anchor_W_);

                        // ====================================================
                        // FR_GROUND_PHYSICAL_SLOPE_SWITCH_FINAL
                        //
                        // TERRAIN CHANGE must be detected geometrically,
                        // never from height residual.
                        //
                        // Compare measured Ground normal with estimated
                        // gravity in the SAME LiDAR frame.
                        // ====================================================
                        bool current_physical_slope_valid =
                            false;

                        double current_physical_slope_deg =
                            0.0;

                        Eigen::Vector3d physical_up_W =
                            Eigen::Vector3d::UnitZ();

                        if (ground_lifecycle_state
                                .gravity_W
                                .allFinite() &&
                            ground_lifecycle_state
                                .gravity_W
                                .norm() >
                                1.0e-6)
                        {
                            physical_up_W =
                                -ground_lifecycle_state
                                     .gravity_W
                                     .normalized();

                            Eigen::Vector3d physical_up_L =
                                T_WL_ground_lifecycle
                                    .rotation()
                                    .transpose() *
                                physical_up_W;

                            const double physical_up_L_norm =
                                physical_up_L.norm();

                            if (physical_up_L.allFinite() &&
                                std::isfinite(
                                    physical_up_L_norm) &&
                                physical_up_L_norm >
                                    1.0e-9)
                            {
                                physical_up_L /=
                                    physical_up_L_norm;

                                const double slope_cosine =
                                    std::clamp(
                                        std::abs(
                                            current_normal_L.dot(
                                                physical_up_L)),
                                        0.0,
                                        1.0);

                                current_physical_slope_deg =
                                    std::acos(
                                        slope_cosine) *
                                    57.29577951308232;

                                current_physical_slope_valid =
                                    std::isfinite(
                                        current_physical_slope_deg);
                            }
                        }

                        // If ACTIVE slope was not available during initial
                        // bootstrap, recover it from the persistent Active
                        // normal and the current gravity direction.
                        if (!ground_active_slope_valid_ &&
                            current_physical_slope_valid)
                        {
                            const double active_slope_cosine =
                                std::clamp(
                                    std::abs(
                                        active_normal_W.dot(
                                            physical_up_W)),
                                    0.0,
                                    1.0);

                            ground_active_slope_deg_ =
                                std::acos(
                                    active_slope_cosine) *
                                57.29577951308232;

                            ground_active_slope_valid_ =
                                std::isfinite(
                                    ground_active_slope_deg_);
                        }


                        // ----------------------------------------------------
                        // ACTIVE consistency gates.
                        //
                        // MATCH:
                        //   same Ground piece.
                        //
                        // SAFE:
                        //   likely transition but still safe enough to keep
                        //   the old Active measurement while Pending matures.
                        //
                        // GROSS:
                        //   keep Active reference in memory but suspend this
                        //   frame's Ground observation.
                        // ----------------------------------------------------
                        // ----------------------------------------------------
                        // FINAL terrain-switch thresholds.
                        //
                        // A new Ground Piece requires BOTH:
                        //
                        //   1. Active-vs-current normal changes enough
                        //   2. Either old or new surface is physically sloped
                        //      relative to gravity.
                        //
                        // Height residual is intentionally NOT present.
                        // ----------------------------------------------------
                        constexpr double
                            kSwitchNormalDeg =
                                2.00;

                        constexpr double
                            kSwitchPhysicalSlopeDeg =
                                2.50;

                        

                        constexpr double
                            kActiveSafeNormalDeg =
                                5.00;

                        

                        constexpr std::size_t
                            kPendingRequiredFrames =
                                8U;

                        constexpr double
                            kPendingNormalConsistencyDeg =
                                1.00;

                        constexpr double
                            kPendingPlaneConsistencyM =
                                0.05;

                        constexpr double
                            kPendingMaximumNormalDeviationDeg =
                                1.00;

                        constexpr double
                            kPendingMaximumPlaneRmseM =
                                0.03;

                        // FR_GROUND_PIECEWISE_V2
                        //
                        // Ground is only LOCALLY planar.
                        // The complete trajectory is NOT constrained to one
                        // global plane.
                        //
                        // IMPORTANT:
                        //   Height residual must NEVER trigger a terrain switch.
                        //   A Z-estimation error can create a large height
                        //   residual even when the physical Ground did not
                        //   change.
                        //
                        // A new terrain piece is opened only when:
                        //
                        //   1. Ground orientation disagrees with ACTIVE;
                        //   2. physical slope changes consistently, OR the
                        //      normal change is large enough to represent a
                        //      direction-changing slope;
                        //   3. gravity direction is still trustworthy.
                        //
                        // PENDING temporal consistency and the existing atomic
                        // switch remain responsible for final acceptance.
                        constexpr double
                            kTerrainNormalTriggerDeg =
                                1.50;

                        constexpr double
                            kTerrainSlopeDeltaTriggerDeg =
                                1.00;

                        constexpr double
                            kTerrainHardNormalTriggerDeg =
                                3.50;

                        constexpr double
                            kTerrainMaximumGravityTiltDeg =
                                1.50;

                        double gravity_tilt_deg =
                            std::numeric_limits<double>::infinity();

                        if (current_physical_slope_valid)
                        {
                            const double gravity_up_cosine =
                                std::clamp(
                                    std::abs(
                                        physical_up_W.dot(
                                            Eigen::Vector3d::UnitZ())),
                                    0.0,
                                    1.0);

                            gravity_tilt_deg =
                                std::acos(
                                    gravity_up_cosine) *
                                57.29577951308232;
                        }

                        const bool gravity_direction_reliable =
                            current_physical_slope_valid &&
                            std::isfinite(
                                gravity_tilt_deg) &&
                            gravity_tilt_deg <=
                                kTerrainMaximumGravityTiltDeg;

                        const bool physical_slope_delta_valid =
                            current_physical_slope_valid &&
                            ground_active_slope_valid_;

                        const double physical_slope_delta_deg =
                            physical_slope_delta_valid
                                ? std::abs(
                                      current_physical_slope_deg -
                                      ground_active_slope_deg_)
                                : 0.0;

                        const bool terrain_change_candidate =
                            gravity_direction_reliable &&
                            active_normal_error_deg >=
                                kTerrainNormalTriggerDeg &&
                            (
                                (
                                    physical_slope_delta_valid &&
                                    physical_slope_delta_deg >=
                                        kTerrainSlopeDeltaTriggerDeg
                                )
                                ||
                                active_normal_error_deg >=
                                    kTerrainHardNormalTriggerDeg
                            );

                        if (terrain_change_candidate)
                        {
                            std::cout
                                << "LIO_GROUND_TERRAIN_CANDIDATE"
                                << " | active_piece="
                                << ground_active_piece_id_
                                << " | normal_delta_deg="
                                << active_normal_error_deg
                                << " | slope_current_deg="
                                << current_physical_slope_deg
                                << " | slope_active_deg="
                                << ground_active_slope_deg_
                                << " | slope_delta_deg="
                                << physical_slope_delta_deg
                                << " | gravity_tilt_deg="
                                << gravity_tilt_deg
                                << std::endl;
                        }

                        // Same physical Ground piece:
                        //
                        // Height can be arbitrarily non-zero here.
                        // Height means "correct Z", NOT "change datum".
                        const bool active_match =
                            !terrain_change_candidate;

                        // ACTIVE reference is retained regardless of height error.
                        //
                        // During a genuine steep transition we may suspend
                        // the old Ground measurement if angular mismatch
                        // becomes too large, but HEIGHT never causes suspend.
                        const bool active_safe =
                            active_normal_error_deg <=
                                kActiveSafeNormalDeg;

                        if (active_match)
                        {
                            // Same piece: normal operation.
                            ground_joint_measurement_allowed =
                                true;

                            if (ground_pending_active_)
                            {
                                std::cout
                                    << "LIO_GROUND_PENDING_CANCEL"
                                    << " | reason=ACTIVE_MATCH"
                                    << " | count="
                                    << ground_pending_sample_count_
                                    << std::endl;
                            }

                            reset_pending_ground();
                        }
                        else
                        {
                            // Active reference remains persistent.
                            //
                            // If discrepancy is still within a safe
                            // transition band, keep constraining with ACTIVE.
                            // Otherwise suspend Ground for THIS frame only.
                            ground_joint_measurement_allowed =
                                active_safe;

                            // ------------------------------------------------
                            // Start/update Pending candidate.
                            // ------------------------------------------------
                            bool append_to_pending =
                                false;

                            if (!ground_pending_active_ ||
                                ground_pending_normals_W_.empty() ||
                                ground_pending_anchors_W_.empty())
                            {
                                reset_pending_ground();

                                ground_pending_active_ =
                                    true;

                                append_to_pending =
                                    true;
                            }
                            else
                            {
                                Eigen::Vector3d pending_normal_sum =
                                    Eigen::Vector3d::Zero();

                                for (const Eigen::Vector3d &sample_normal :
                                     ground_pending_normals_W_)
                                {
                                    Eigen::Vector3d aligned_normal =
                                        sample_normal;

                                    if (aligned_normal.dot(
                                            current_normal_W) <
                                        0.0)
                                    {
                                        aligned_normal =
                                            -aligned_normal;
                                    }

                                    pending_normal_sum +=
                                        aligned_normal;
                                }

                                const double pending_sum_norm =
                                    pending_normal_sum.norm();

                                if (pending_sum_norm >
                                        1.0e-12 &&
                                    std::isfinite(
                                        pending_sum_norm))
                                {
                                    const Eigen::Vector3d
                                        pending_mean_normal =
                                            pending_normal_sum /
                                            pending_sum_norm;

                                    const double
                                        pending_normal_error_deg =
                                            std::acos(
                                                std::clamp(
                                                    pending_mean_normal.dot(
                                                        current_normal_W),
                                                    -1.0,
                                                    1.0)) *
                                            57.29577951308232;

                                    // Point-to-candidate-plane consistency.
                                    //
                                    // Horizontal travel along the same plane
                                    // does NOT increase this quantity.
                                    const double
                                        pending_plane_error_m =
                                            std::abs(
                                                pending_mean_normal.dot(
                                                    current_anchor_W -
                                                    ground_pending_anchors_W_
                                                        .front()));

                                    append_to_pending =
                                        pending_normal_error_deg <=
                                            kPendingNormalConsistencyDeg &&
                                        pending_plane_error_m <=
                                            kPendingPlaneConsistencyM;
                                }
                            }

                            if (!append_to_pending)
                            {
                                std::cout
                                    << "LIO_GROUND_PENDING_RESTART"
                                    << " | previous_count="
                                    << ground_pending_sample_count_
                                    << " | active_normal_deg="
                                    << active_normal_error_deg
                                    << " | active_height="
                                    << active_height_error_m
                                    << std::endl;

                                reset_pending_ground();

                                ground_pending_active_ =
                                    true;
                            }

                            ground_pending_normals_W_.push_back(
                                current_normal_W);

                            ground_pending_anchors_W_.push_back(
                                current_anchor_W);

                            ground_pending_sample_count_ =
                                ground_pending_normals_W_.size();

                            if (ground_pending_sample_count_ == 1U ||
                                ground_pending_sample_count_ ==
                                    kPendingRequiredFrames)
                            {
                                std::cout
                                    << "LIO_GROUND_PENDING"
                                    << " | active_piece="
                                    << ground_active_piece_id_
                                    << " | count="
                                    << ground_pending_sample_count_
                                    << " | active_normal_deg="
                                    << active_normal_error_deg
                                    << " | active_height="
                                    << active_height_error_m
                                    << " | current_slope_deg="
                                    << current_physical_slope_deg
                                    << " | active_slope_deg="
                                    << ground_active_slope_deg_
                                    << " | active_measurement="
                                    << (ground_joint_measurement_allowed
                                            ? 1
                                            : 0)
                                    << std::endl;
                            }

                            // ------------------------------------------------
                            // Mature Pending candidate.
                            // ------------------------------------------------
                            if (ground_pending_sample_count_ >=
                                kPendingRequiredFrames)
                            {
                                Eigen::Vector3d candidate_normal_sum =
                                    Eigen::Vector3d::Zero();

                                for (Eigen::Vector3d sample_normal :
                                     ground_pending_normals_W_)
                                {
                                    if (sample_normal.dot(
                                            active_normal_W) <
                                        0.0)
                                    {
                                        sample_normal =
                                            -sample_normal;
                                    }

                                    candidate_normal_sum +=
                                        sample_normal;
                                }

                                const double candidate_normal_sum_norm =
                                    candidate_normal_sum.norm();

                                bool pending_stable =
                                    std::isfinite(
                                        candidate_normal_sum_norm) &&
                                    candidate_normal_sum_norm >
                                        1.0e-12;

                                Eigen::Vector3d candidate_normal_W =
                                    active_normal_W;

                                double maximum_normal_deviation_deg =
                                    0.0;

                                double candidate_plane_rmse_m =
                                    0.0;

                                if (pending_stable)
                                {
                                    candidate_normal_W =
                                        candidate_normal_sum /
                                        candidate_normal_sum_norm;

                                    if (candidate_normal_W.dot(
                                            active_normal_W) <
                                        0.0)
                                    {
                                        candidate_normal_W =
                                            -candidate_normal_W;
                                    }

                                    double mean_projection =
                                        0.0;

                                    for (const Eigen::Vector3d &anchor :
                                         ground_pending_anchors_W_)
                                    {
                                        mean_projection +=
                                            candidate_normal_W.dot(
                                                anchor);
                                    }

                                    mean_projection /=
                                        static_cast<double>(
                                            ground_pending_anchors_W_
                                                .size());

                                    double squared_plane_error_sum =
                                        0.0;

                                    for (std::size_t i = 0;
                                         i <
                                         ground_pending_normals_W_.size();
                                         ++i)
                                    {
                                        Eigen::Vector3d sample_normal =
                                            ground_pending_normals_W_[i];

                                        if (sample_normal.dot(
                                                candidate_normal_W) <
                                            0.0)
                                        {
                                            sample_normal =
                                                -sample_normal;
                                        }

                                        const double deviation_deg =
                                            std::acos(
                                                std::clamp(
                                                    sample_normal.dot(
                                                        candidate_normal_W),
                                                    -1.0,
                                                    1.0)) *
                                            57.29577951308232;

                                        maximum_normal_deviation_deg =
                                            std::max(
                                                maximum_normal_deviation_deg,
                                                deviation_deg);

                                        const double plane_error =
                                            candidate_normal_W.dot(
                                                ground_pending_anchors_W_[i]) -
                                            mean_projection;

                                        squared_plane_error_sum +=
                                            plane_error *
                                            plane_error;
                                    }

                                    candidate_plane_rmse_m =
                                        std::sqrt(
                                            squared_plane_error_sum /
                                            static_cast<double>(
                                                ground_pending_anchors_W_
                                                    .size()));

                                    pending_stable =
                                        maximum_normal_deviation_deg <=
                                            kPendingMaximumNormalDeviationDeg &&
                                        candidate_plane_rmse_m <=
                                            kPendingMaximumPlaneRmseM;
                                }

                                if (!pending_stable)
                                {
                                    std::cout
                                        << "LIO_GROUND_PENDING_REJECT"
                                        << " | count="
                                        << ground_pending_sample_count_
                                        << " | normal_max_deg="
                                        << maximum_normal_deviation_deg
                                        << " | plane_rmse="
                                        << candidate_plane_rmse_m
                                        << std::endl;

                                    // Keep newest sample as beginning of a
                                    // fresh candidate.
                                    const Eigen::Vector3d newest_normal =
                                        current_normal_W;

                                    const Eigen::Vector3d newest_anchor =
                                        current_anchor_W;

                                    reset_pending_ground();

                                    ground_pending_active_ =
                                        true;

                                    ground_pending_normals_W_.push_back(
                                        newest_normal);

                                    ground_pending_anchors_W_.push_back(
                                        newest_anchor);

                                    ground_pending_sample_count_ =
                                        1U;
                                }
                                else
                                {
                                    // ========================================
                                    // CONTINUITY INHERITANCE
                                    //
                                    // Project CURRENT boundary observation onto
                                    // OLD ACTIVE plane:
                                    //
                                    // b_A =
                                    //   b_W - n_A * n_A^T(b_W-a_A)
                                    //
                                    // New piece uses:
                                    //
                                    //     a_B = b_A
                                    //
                                    // Therefore piece orientation may change,
                                    // but position is continuous at switch.
                                    // ========================================
                                    const Eigen::Vector3d
                                        continuity_anchor_W =
                                            current_anchor_W -
                                            active_normal_W *
                                            active_normal_W.dot(
                                                current_anchor_W -
                                                ground_active_anchor_W_);

                                    if (!continuity_anchor_W.allFinite() ||
                                        !candidate_normal_W.allFinite())
                                    {
                                        std::cerr
                                            << "LIO_GROUND_SWITCH_INVALID"
                                            << std::endl;

                                        ground_joint_measurement_allowed =
                                            false;
                                    }
                                    else
                                    {
                                        const std::size_t old_piece_id =
                                            ground_active_piece_id_;

                                        // Atomic switch.
                                        ground_reference_normal_W_ =
                                            candidate_normal_W.normalized();

                                        ground_active_anchor_W_ =
                                            continuity_anchor_W;

                                        ground_reference_plane_d_W_ =
                                            -ground_reference_normal_W_.dot(
                                                ground_active_anchor_W_);

                                        ground_reference_frozen_ =
                                            true;

                                        ground_active_anchor_valid_ =
                                            true;

                                        ++ground_active_piece_id_;

                                        // New ACTIVE physical slope is derived
                                        // from the candidate normal and
                                        // gravity at the atomic switch.
                                        if (current_physical_slope_valid)
                                        {
                                            const double
                                                new_active_slope_cosine =
                                                    std::clamp(
                                                        std::abs(
                                                            ground_reference_normal_W_
                                                                .dot(
                                                                    physical_up_W)),
                                                        0.0,
                                                        1.0);

                                            ground_active_slope_deg_ =
                                                std::acos(
                                                    new_active_slope_cosine) *
                                                57.29577951308232;

                                            ground_active_slope_valid_ =
                                                std::isfinite(
                                                    ground_active_slope_deg_);
                                        }
                                        else
                                        {
                                            ground_active_slope_valid_ =
                                                false;

                                            ground_active_slope_deg_ =
                                                0.0;
                                        }

                                        ground_joint_measurement_allowed =
                                            true;

                                        std::cout
                                            << "LIO_GROUND_ATOMIC_SWITCH"
                                            << " | old_piece="
                                            << old_piece_id
                                            << " | new_piece="
                                            << ground_active_piece_id_
                                            << " | samples="
                                            << ground_pending_sample_count_
                                            << " | normal_max_deg="
                                            << maximum_normal_deviation_deg
                                            << " | plane_rmse="
                                            << candidate_plane_rmse_m
                                            << " | n_W=["
                                            << ground_reference_normal_W_
                                                   .transpose()
                                            << "]"
                                            << " | anchor_W=["
                                            << ground_active_anchor_W_
                                                   .transpose()
                                            << "]"
                                            << std::endl;

                                        reset_pending_ground();
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Freeze the observation topology for this LiDAR frame.
    //
    // IMPORTANT:
    //   - current Ground plane comes from this frame
    //   - reference is the current active frozen piece
    //   - these snapshots do NOT change during IEKF iterations
    
    // ========================================================================
    // FR_JOINT_WALL_FINAL_PREP
    //
    // FINAL structural Wall observation:
    //
    //   Persistent association
    //      -> geometry/quality gate
    //      -> rank-aware tangent translation proposal
    //      -> yaw proposal
    //      -> temporal confirmation + hysteresis + EMA
    //      -> frozen target for the complete IEKF inner iteration
    //
    // Ground is NOT required.
    //
    // Up-axis priority:
    //
    //   1. trusted frozen Ground normal
    //   2. negative estimated gravity direction
    //   3. odom +Z emergency fallback
    //
    // No pose/state is injected here.
    // ========================================================================

    bool joint_wall_final_ready =
        false;

    bool joint_wall_final_translation_active =
        false;

    bool joint_wall_final_yaw_active =
        false;

    std::size_t joint_wall_final_rank =
        0;

    Eigen::Vector3d joint_wall_final_target_delta_t_S =
        Eigen::Vector3d::Zero();

    double joint_wall_final_target_yaw_rad =
        0.0;

    bool joint_wall_final_up_from_ground =
        false;

    const auto reset_wall_temporal_state =
        [this]()
        {
            wall_temporal_initialized_ =
                false;

            wall_consistent_frames_ =
                0;

            wall_temporal_last_frame_index_ =
                0;

            wall_previous_delta_t_S_.setZero();

            wall_previous_yaw_rad_ =
                0.0;

            wall_filtered_delta_t_S_.setZero();

            wall_filtered_yaw_rad_ =
                0.0;
        };

    bool joint_wall_final_proposal_valid =
        false;

    if (joint_wall_ready &&
        config_.wall_constraint_enable &&
        submap_context.valid &&
        submap_context.T_O_S_creation.matrix().allFinite() &&
        !joint_wall_association.active_static_walls.empty())
    {
        // ------------------------------------------------------------
        // Final up axis.
        // ------------------------------------------------------------
        const LioState wall_snapshot_state =
            ieskf_.State();

        Eigen::Vector3d up_O =
            Eigen::Vector3d::UnitZ();

        if (ground_reference_frozen_ &&
            ground_reference_normal_W_.allFinite() &&
            ground_reference_normal_W_.norm() >
                1.0e-9)
        {
            up_O =
                ground_reference_normal_W_.normalized();

            joint_wall_final_up_from_ground =
                true;
        }
        else if (wall_snapshot_state.gravity_W.allFinite() &&
                 wall_snapshot_state.gravity_W.norm() >
                     1.0e-6)
        {
            // gravity_W points downward.
            up_O =
                -wall_snapshot_state.gravity_W.normalized();

            joint_wall_final_up_from_ground =
                false;
        }

        joint_wall_up_S =
            submap_context
                .T_O_S_creation
                .rotation()
                .transpose() *
            up_O;

        if (joint_wall_up_S.allFinite() &&
            joint_wall_up_S.norm() >
                1.0e-9)
        {
            joint_wall_up_S.normalize();

            constexpr double
                kMinimumQuality =
                    0.85;

            constexpr double
                kMaximumNormalDifferenceDeg =
                    1.5;

            constexpr double
                kMaximumPlaneDistanceDifferenceM =
                    0.10;

            constexpr double
                kMaximumTranslationCorrectionM =
                    0.03;

            constexpr double
                kMaximumYawCorrectionDeg =
                    0.50;

            constexpr double
                kObservableRelativeEigenThreshold =
                    0.10;

            Eigen::Matrix3d translation_H =
                Eigen::Matrix3d::Zero();

            Eigen::Vector3d translation_b =
                Eigen::Vector3d::Zero();

            double yaw_weighted_sum =
                0.0;

            double yaw_weight_sum =
                0.0;

            std::size_t used_walls =
                0;

            for (const fr_slam::ActiveWallAssociation &wall :
                 joint_wall_association.active_static_walls)
            {
                if (!wall.reference_normal_A.allFinite() ||
                    !wall.observed_normal_A.allFinite() ||
                    !std::isfinite(wall.reference_d_A) ||
                    !std::isfinite(wall.observed_d_A) ||
                    !std::isfinite(wall.quality) ||
                    !std::isfinite(
                        wall.normal_difference_deg) ||
                    !std::isfinite(
                        wall.plane_distance_difference_m))
                {
                    continue;
                }

                if (wall.quality <
                        kMinimumQuality ||
                    wall.normal_difference_deg >
                        kMaximumNormalDifferenceDeg ||
                    std::abs(
                        wall.plane_distance_difference_m) >
                        kMaximumPlaneDistanceDifferenceM)
                {
                    continue;
                }

                Eigen::Vector3d reference_normal =
                    wall.reference_normal_A -
                    joint_wall_up_S *
                        joint_wall_up_S.dot(
                            wall.reference_normal_A);

                Eigen::Vector3d observed_normal =
                    wall.observed_normal_A -
                    joint_wall_up_S *
                        joint_wall_up_S.dot(
                            wall.observed_normal_A);

                const double reference_norm =
                    reference_normal.norm();

                const double observed_norm =
                    observed_normal.norm();

                if (!std::isfinite(reference_norm) ||
                    !std::isfinite(observed_norm) ||
                    reference_norm <= 1.0e-6 ||
                    observed_norm <= 1.0e-6)
                {
                    continue;
                }

                reference_normal /=
                    reference_norm;

                observed_normal /=
                    observed_norm;

                if (reference_normal.dot(
                        observed_normal) < 0.0)
                {
                    observed_normal =
                        -observed_normal;
                }

                const double weight =
                    std::clamp(
                        wall.quality,
                        0.0,
                        1.0);

                if (weight <= 1.0e-6)
                {
                    continue;
                }

                const double distance_residual =
                    wall.observed_d_A -
                    wall.reference_d_A;

                const double yaw_sine =
                    joint_wall_up_S.dot(
                        observed_normal.cross(
                            reference_normal));

                const double yaw_cosine =
                    std::clamp(
                        observed_normal.dot(
                            reference_normal),
                        -1.0,
                        1.0);

                const double yaw_residual =
                    std::atan2(
                        yaw_sine,
                        yaw_cosine);

                if (!std::isfinite(distance_residual) ||
                    !std::isfinite(yaw_residual))
                {
                    continue;
                }

                translation_H +=
                    weight *
                    reference_normal *
                    reference_normal.transpose();

                translation_b +=
                    weight *
                    distance_residual *
                    reference_normal;

                yaw_weighted_sum +=
                    weight *
                    yaw_residual;

                yaw_weight_sum +=
                    weight;

                ++used_walls;
            }

            if (used_walls > 0 &&
                yaw_weight_sum > 1.0e-12)
            {
                Eigen::SelfAdjointEigenSolver<
                    Eigen::Matrix3d>
                    eigen_solver(
                        translation_H);

                if (eigen_solver.info() ==
                    Eigen::Success)
                {
                    const Eigen::Vector3d eigenvalues =
                        eigen_solver.eigenvalues();

                    const Eigen::Matrix3d eigenvectors =
                        eigen_solver.eigenvectors();

                    const double maximum_eigenvalue =
                        eigenvalues.maxCoeff();

                    if (eigenvalues.allFinite() &&
                        eigenvectors.allFinite() &&
                        std::isfinite(
                            maximum_eigenvalue) &&
                        maximum_eigenvalue >
                            1.0e-12)
                    {
                        const double
                            eigen_threshold =
                                std::max(
                                    1.0e-6,
                                    maximum_eigenvalue *
                                        kObservableRelativeEigenThreshold);

                        Eigen::Vector3d
                            inverse_eigenvalues =
                                Eigen::Vector3d::Zero();

                        std::size_t observable_rank =
                            0;

                        for (int index = 0;
                             index < 3;
                             ++index)
                        {
                            if (eigenvalues(index) >
                                eigen_threshold)
                            {
                                inverse_eigenvalues(index) =
                                    1.0 /
                                    eigenvalues(index);

                                ++observable_rank;
                            }
                        }

                        // Tangent-plane Wall geometry has at most rank 2.
                        observable_rank =
                            std::min<std::size_t>(
                                observable_rank,
                                2U);

                        Eigen::Vector3d
                            raw_delta_t_S =
                                eigenvectors *
                                inverse_eigenvalues
                                    .asDiagonal() *
                                eigenvectors.transpose() *
                                translation_b;

                        raw_delta_t_S -=
                            joint_wall_up_S *
                            joint_wall_up_S.dot(
                                raw_delta_t_S);

                        const double raw_translation_norm =
                            raw_delta_t_S.norm();

                        if (raw_delta_t_S.allFinite() &&
                            std::isfinite(
                                raw_translation_norm))
                        {
                            Eigen::Vector3d
                                bounded_delta_t_S =
                                    raw_delta_t_S;

                            if (raw_translation_norm >
                                    kMaximumTranslationCorrectionM &&
                                raw_translation_norm >
                                    1.0e-12)
                            {
                                bounded_delta_t_S *=
                                    kMaximumTranslationCorrectionM /
                                    raw_translation_norm;
                            }

                            const double raw_yaw_rad =
                                yaw_weighted_sum /
                                yaw_weight_sum;

                            const double maximum_yaw_rad =
                                kMaximumYawCorrectionDeg *
                                0.017453292519943295;

                            const double bounded_yaw_rad =
                                std::clamp(
                                    raw_yaw_rad,
                                    -maximum_yaw_rad,
                                    maximum_yaw_rad);

                            if (bounded_delta_t_S.allFinite() &&
                                std::isfinite(
                                    raw_yaw_rad) &&
                                std::isfinite(
                                    bounded_yaw_rad))
                            {
                                joint_wall_final_proposal_valid =
                                    true;

                                joint_wall_final_rank =
                                    observable_rank;

                                // ------------------------------------------------
                                // Temporal validation.
                                //
                                // Activation:
                                //   translation >= 5 mm
                                //   yaw         >= 0.05 deg
                                //
                                // Once confirmed, lower deactivation thresholds
                                // provide hysteresis:
                                //   translation >= 2.5 mm
                                //   yaw         >= 0.025 deg
                                // ------------------------------------------------
                                constexpr double
                                    kTranslationActivateM =
                                        0.005;

                                constexpr double
                                    kTranslationHoldM =
                                        0.0025;

                                constexpr double
                                    kYawActivateDeg =
                                        0.05;

                                constexpr double
                                    kYawHoldDeg =
                                        0.025;

                                constexpr double
                                    kDirectionCosineThreshold =
                                        0.80;

                                constexpr std::size_t
                                    kRequiredConsistentFrames =
                                        3;

                                constexpr double
                                    kEmaAlpha =
                                        0.40;

                                const bool was_confirmed =
                                    wall_temporal_initialized_ &&
                                    wall_consistent_frames_ >=
                                        kRequiredConsistentFrames;

                                const double
                                    translation_deadband =
                                        was_confirmed
                                            ? kTranslationHoldM
                                            : kTranslationActivateM;

                                const double yaw_deadband_rad =
                                    (was_confirmed
                                         ? kYawHoldDeg
                                         : kYawActivateDeg) *
                                    0.017453292519943295;

                                const bool translation_active =
                                    observable_rank > 0 &&
                                    bounded_delta_t_S.norm() >=
                                        translation_deadband;

                                const bool yaw_active =
                                    std::abs(
                                        bounded_yaw_rad) >=
                                    yaw_deadband_rad;

                                if (!translation_active &&
                                    !yaw_active)
                                {
                                    reset_wall_temporal_state();
                                }
                                else
                                {
                                    const Eigen::Vector3d
                                        temporal_delta_t_S =
                                            translation_active
                                                ? bounded_delta_t_S
                                                : Eigen::Vector3d::Zero();

                                    const double temporal_yaw_rad =
                                        yaw_active
                                            ? bounded_yaw_rad
                                            : 0.0;

                                    const bool consecutive =
                                        wall_temporal_initialized_ &&
                                        wall_temporal_last_frame_index_ >
                                            0 &&
                                        wall_association_frame_index_ ==
                                            wall_temporal_last_frame_index_ +
                                                1;

                                    const double
                                        previous_translation_norm =
                                            wall_previous_delta_t_S_
                                                .norm();

                                    const bool
                                        previous_translation_active =
                                            previous_translation_norm >=
                                            translation_deadband;

                                    const bool
                                        previous_yaw_active =
                                            std::abs(
                                                wall_previous_yaw_rad_) >=
                                            yaw_deadband_rad;

                                    bool translation_consistent =
                                        true;

                                    if (translation_active !=
                                        previous_translation_active)
                                    {
                                        translation_consistent =
                                            false;
                                    }
                                    else if (translation_active &&
                                             previous_translation_active)
                                    {
                                        const double denominator =
                                            temporal_delta_t_S.norm() *
                                            wall_previous_delta_t_S_
                                                .norm();

                                        if (!std::isfinite(
                                                denominator) ||
                                            denominator <=
                                                1.0e-12)
                                        {
                                            translation_consistent =
                                                false;
                                        }
                                        else
                                        {
                                            const double
                                                direction_cosine =
                                                    temporal_delta_t_S.dot(
                                                        wall_previous_delta_t_S_) /
                                                    denominator;

                                            translation_consistent =
                                                std::isfinite(
                                                    direction_cosine) &&
                                                direction_cosine >=
                                                    kDirectionCosineThreshold;
                                        }
                                    }

                                    bool yaw_consistent =
                                        true;

                                    if (yaw_active !=
                                        previous_yaw_active)
                                    {
                                        yaw_consistent =
                                            false;
                                    }
                                    else if (yaw_active &&
                                             previous_yaw_active)
                                    {
                                        yaw_consistent =
                                            temporal_yaw_rad *
                                                wall_previous_yaw_rad_ >
                                            0.0;
                                    }

                                    if (!wall_temporal_initialized_ ||
                                        !consecutive ||
                                        !translation_consistent ||
                                        !yaw_consistent)
                                    {
                                        wall_temporal_initialized_ =
                                            true;

                                        wall_consistent_frames_ =
                                            1;

                                        wall_filtered_delta_t_S_ =
                                            temporal_delta_t_S;

                                        wall_filtered_yaw_rad_ =
                                            temporal_yaw_rad;
                                    }
                                    else
                                    {
                                        wall_consistent_frames_ =
                                            std::min<std::size_t>(
                                                wall_consistent_frames_ +
                                                    1U,
                                                kRequiredConsistentFrames);

                                        wall_filtered_delta_t_S_ =
                                            (1.0 - kEmaAlpha) *
                                                wall_filtered_delta_t_S_ +
                                            kEmaAlpha *
                                                temporal_delta_t_S;

                                        wall_filtered_yaw_rad_ =
                                            (1.0 - kEmaAlpha) *
                                                wall_filtered_yaw_rad_ +
                                            kEmaAlpha *
                                                temporal_yaw_rad;
                                    }

                                    wall_temporal_last_frame_index_ =
                                        wall_association_frame_index_;

                                    wall_previous_delta_t_S_ =
                                        temporal_delta_t_S;

                                    wall_previous_yaw_rad_ =
                                        temporal_yaw_rad;

                                    if (wall_consistent_frames_ >=
                                        kRequiredConsistentFrames)
                                    {
                                        joint_wall_final_target_delta_t_S =
                                            wall_filtered_delta_t_S_;

                                        joint_wall_final_target_yaw_rad =
                                            wall_filtered_yaw_rad_;

                                        joint_wall_final_translation_active =
                                            joint_wall_final_rank > 0 &&
                                            joint_wall_final_target_delta_t_S
                                                    .norm() >=
                                                kTranslationHoldM;

                                        joint_wall_final_yaw_active =
                                            std::abs(
                                                joint_wall_final_target_yaw_rad) >=
                                            kYawHoldDeg *
                                                0.017453292519943295;

                                        joint_wall_final_ready =
                                            joint_wall_final_translation_active ||
                                            joint_wall_final_yaw_active;

                                        if (joint_wall_final_ready &&
                                            (!was_confirmed ||
                                             (wall_association_frame_index_ %
                                                  20U) == 0U))
                                        {
                                            std::cout
                                                << "LIO_JOINT_WALL_TEMPORAL_READY"
                                                << " | submap="
                                                << submap_context
                                                       .primary_submap_id
                                                << " | frame="
                                                << wall_association_frame_index_
                                                << " | confirmed="
                                                << wall_consistent_frames_
                                                << " | rank="
                                                << joint_wall_final_rank
                                                << " | dt_S=["
                                                << joint_wall_final_target_delta_t_S.x()
                                                << ","
                                                << joint_wall_final_target_delta_t_S.y()
                                                << ","
                                                << joint_wall_final_target_delta_t_S.z()
                                                << "]"
                                                << " | dt_norm="
                                                << joint_wall_final_target_delta_t_S.norm()
                                                << " | yaw_deg="
                                                << joint_wall_final_target_yaw_rad *
                                                       57.29577951308232
                                                << " | up_source="
                                                << (joint_wall_final_up_from_ground
                                                        ? "GROUND"
                                                        : "GRAVITY")
                                                << std::endl;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if (!joint_wall_final_proposal_valid)
    {
        reset_wall_temporal_state();
    }

    // ------------------------------------------------------------------------
    // Disable the old Joint Wall V1 branch.
    //
    // The existing joint_observation_builder below is intentionally retained
    // as Ground-only. FINAL Wall is added by a wrapper builder immediately
    // before IteratedLidarUpdate().
    // ------------------------------------------------------------------------
    joint_wall_ready =
        false;

const bool joint_ground_ready =
        ground_joint_measurement_allowed &&
        config_.ground.enabled &&
        config_.ground.mode != "off" &&
        config_.ground.mode != "disabled" &&
        last_ground_segmentation_valid_ &&
        ground_reference_frozen_ &&
        last_ground_segmentation_result_.success &&
        last_ground_segmentation_result_.support_plane_valid &&
        last_ground_segmentation_result_.support_constraint_valid &&
        last_ground_segmentation_result_.support_ground_cloud &&
        !last_ground_segmentation_result_.support_ground_cloud->empty() &&
        last_ground_segmentation_result_
            .support_ground_normal_L.allFinite() &&
        std::isfinite(
            last_ground_segmentation_result_
                .support_ground_plane_d) &&
        ground_reference_normal_W_.allFinite() &&
        ground_reference_normal_W_.norm() > 1.0e-12 &&
        std::isfinite(
            ground_reference_plane_d_W_);

    // ========================================================================
    // FR_GROUND_DEGRADED_HEIGHT_V1
    //
    // If HIGH_RMSE is the ONLY failed support gate, retain a WEAK
    // height-only Ground observation.
    //
    // We deliberately do NOT trust the noisy normal:
    //
    //     normal measurement = OFF
    //     height measurement = weak ON
    //
    // All other rejection masks remain measurement OFF.
    // Persistent Ground ownership remains responsible for rank-3 ownership.
    // ========================================================================
    const bool joint_ground_degraded_height_ready =
        ground_joint_measurement_allowed &&
        config_.ground.enabled &&
        config_.ground.mode != "off" &&
        config_.ground.mode != "disabled" &&
        last_ground_segmentation_valid_ &&
        ground_reference_frozen_ &&
        last_ground_segmentation_result_.success &&
        last_ground_segmentation_result_.support_plane_valid &&
        !last_ground_segmentation_result_.support_constraint_valid &&
        last_ground_segmentation_result_
                .support_constraint_rejection_mask ==
            fr_slam::SUPPORT_CONSTRAINT_REJECT_HIGH_RMSE &&
        last_ground_segmentation_result_.support_ground_cloud &&
        !last_ground_segmentation_result_.support_ground_cloud->empty() &&
        last_ground_segmentation_result_
            .support_ground_normal_L.allFinite() &&
        std::isfinite(
            last_ground_segmentation_result_
                .support_ground_plane_d) &&
        ground_reference_normal_W_.allFinite() &&
        ground_reference_normal_W_.norm() > 1.0e-12 &&
        std::isfinite(
            ground_reference_plane_d_W_);

    const auto joint_ground_result =
        last_ground_segmentation_result_;

    Eigen::Vector3d joint_ground_reference_normal_W =
        Eigen::Vector3d::UnitZ();

    double joint_ground_reference_plane_d_W =
        0.0;

    if (joint_ground_ready ||
        joint_ground_degraded_height_ready)
    {
        joint_ground_reference_normal_W =
            ground_reference_normal_W_.normalized();

        joint_ground_reference_plane_d_W =
            ground_reference_plane_d_W_;
    }

    if (joint_ground_degraded_height_ready)
    {
        std::cout
            << "LIO_GROUND_DEGRADED_HEIGHT"
            << " | mask="
            << last_ground_segmentation_result_
                   .support_constraint_rejection_mask
            << " | rmse="
            << last_ground_segmentation_result_
                   .support_plane_rmse_m
            << " | sigma=0.05"
            << std::endl;
    }

    // ========================================================================
    // FR_PERSISTENT_HEIGHT_HOLD_V1
    //
    // Cache ONLY trusted Ground geometry.
    //
    // During a short Ground dropout we reuse this trusted height target.
    // Rejected current Ground geometry is NEVER used by this hold.
    // ========================================================================
    static bool
        persistent_height_hold_valid =
            false;

    static double
        persistent_height_hold_target_lidar_normal_W =
            0.0;

    static double
        persistent_height_hold_timestamp =
            0.0;

    if (!ground_reference_frozen_)
    {
        persistent_height_hold_valid =
            false;
    }

    if (joint_ground_ready)
    {
        const LioState hold_source_state =
            ieskf_.State();

        const Eigen::Isometry3d hold_T_WL =
            StateToLidarPose(
                hold_source_state);

        Eigen::Vector3d hold_normal_L =
            joint_ground_result
                .support_ground_normal_L;

        const double hold_normal_norm =
            hold_normal_L.norm();

        if (hold_T_WL.matrix().allFinite() &&
            hold_normal_L.allFinite() &&
            std::isfinite(hold_normal_norm) &&
            hold_normal_norm > 1.0e-12 &&
            std::isfinite(
                joint_ground_result
                    .support_ground_plane_d))
        {
            double hold_plane_d_L =
                joint_ground_result
                    .support_ground_plane_d /
                hold_normal_norm;

            hold_normal_L /=
                hold_normal_norm;

            Eigen::Vector3d hold_reference_normal_W =
                ground_reference_normal_W_.normalized();

            const Eigen::Vector3d
                hold_reference_normal_L =
                    hold_T_WL.rotation().transpose() *
                    hold_reference_normal_W;

            const double hold_denominator =
                hold_normal_L.dot(
                    hold_reference_normal_L);

            if (std::isfinite(hold_denominator) &&
                std::abs(hold_denominator) >
                    0.50)
            {
                const double hold_clearance_m =
                    hold_plane_d_L /
                    hold_denominator;

                const double hold_target =
                    -ground_reference_plane_d_W_ +
                    hold_clearance_m;

                if (std::isfinite(hold_clearance_m) &&
                    std::isfinite(hold_target) &&
                    hold_clearance_m > 0.05 &&
                    hold_clearance_m < 2.0)
                {
                    persistent_height_hold_target_lidar_normal_W =
                        hold_target;

                    persistent_height_hold_timestamp =
                        hold_source_state.timestamp;

                    persistent_height_hold_valid =
                        true;
                }
            }
        }
    }

    const double persistent_height_hold_age_s =
        persistent_height_hold_valid
            ? ieskf_.State().timestamp -
                  persistent_height_hold_timestamp
            : 1.0e9;

    constexpr double
        kPersistentHeightHoldMaximumAgeS =
            1.0;

    const bool joint_ground_persistent_height_hold_ready =
        !joint_ground_ready &&
        ground_reference_frozen_ &&
        persistent_height_hold_valid &&
        std::isfinite(
            persistent_height_hold_age_s) &&
        persistent_height_hold_age_s >= 0.0 &&
        persistent_height_hold_age_s <=
            kPersistentHeightHoldMaximumAgeS;

    Eigen::Vector3d persistent_height_hold_normal_W =
        Eigen::Vector3d::UnitZ();

    if (ground_reference_frozen_ &&
        ground_reference_normal_W_.allFinite() &&
        ground_reference_normal_W_.norm() >
            1.0e-12)
    {
        persistent_height_hold_normal_W =
            ground_reference_normal_W_.normalized();
    }

    if (joint_ground_persistent_height_hold_ready)
    {
        std::cout
            << "LIO_GROUND_PERSISTENT_HEIGHT_HOLD"
            << " | age="
            << persistent_height_hold_age_s
            << " | target="
            << persistent_height_hold_target_lidar_normal_W
            << " | sigma=0.03"
            << std::endl;
    }

    // ========================================================================
    // FR_GROUND_TRACE_V1
    //
    // DIAGNOSTICS ONLY.
    //
    // Record the exact local support-plane measurement used by Joint Ground
    // together with its local/world anchor and frozen-plane height residual.
    //
    // This block MUST NOT modify segmentation, state, covariance, Ground
    // ownership or the IESKF measurement.
    // ========================================================================
    if (joint_ground_ready)
    {
        const LioState ground_trace_state =
            ieskf_.State();

        const Eigen::Isometry3d ground_trace_T_WL =
            StateToLidarPose(
                ground_trace_state);

        Eigen::Vector3d ground_trace_normal_L =
            joint_ground_result
                .support_ground_normal_L;

        const double ground_trace_normal_norm =
            ground_trace_normal_L.norm();

        if (ground_trace_T_WL.matrix().allFinite() &&
            ground_trace_normal_L.allFinite() &&
            std::isfinite(ground_trace_normal_norm) &&
            ground_trace_normal_norm > 1.0e-12)
        {
            ground_trace_normal_L /=
                ground_trace_normal_norm;

            const double ground_trace_plane_d_L =
                joint_ground_result
                    .support_ground_plane_d /
                ground_trace_normal_norm;

            if (std::isfinite(
                    ground_trace_plane_d_L))
            {
                // Plane:
                //
                //     n_L^T p_L + d_L = 0
                //
                // Closest point to LiDAR origin:
                //
                //     p0_L = -d_L n_L
                const Eigen::Vector3d
                    ground_trace_anchor_L =
                        -ground_trace_plane_d_L *
                        ground_trace_normal_L;

                const Eigen::Vector3d
                    ground_trace_anchor_W =
                        ground_trace_T_WL *
                        ground_trace_anchor_L;

                Eigen::Vector3d ground_trace_normal_W =
                    ground_trace_T_WL.rotation() *
                    ground_trace_normal_L;

                const double ground_trace_normal_W_norm =
                    ground_trace_normal_W.norm();

                if (ground_trace_anchor_L.allFinite() &&
                    ground_trace_anchor_W.allFinite() &&
                    ground_trace_normal_W.allFinite() &&
                    std::isfinite(
                        ground_trace_normal_W_norm) &&
                    ground_trace_normal_W_norm >
                        1.0e-12)
                {
                    ground_trace_normal_W /=
                        ground_trace_normal_W_norm;

                    if (ground_trace_normal_W.dot(
                            joint_ground_reference_normal_W) <
                        0.0)
                    {
                        ground_trace_normal_W =
                            -ground_trace_normal_W;
                    }

                    const double ground_trace_height_residual =
                        -(
                            joint_ground_reference_normal_W.dot(
                                ground_trace_anchor_W) +
                            joint_ground_reference_plane_d_W);

                    const double ground_trace_normal_cos =
                        std::clamp(
                            ground_trace_normal_W.dot(
                                joint_ground_reference_normal_W),
                            -1.0,
                            1.0);

                    const double
                        ground_trace_normal_error_deg =
                            std::acos(
                                ground_trace_normal_cos) *
                            57.29577951308232;

                    std::cout
                        << "LIO_GROUND_TRACE_V1"
                        << " | t="
                        << ground_trace_state.timestamp
                        << " | state_pz="
                        << ground_trace_state.P_WI.z()
                        << " | lidar_z="
                        << ground_trace_T_WL
                               .translation()
                               .z()
                        << " | plane_d_raw="
                        << joint_ground_result
                               .support_ground_plane_d
                        << " | plane_d_L="
                        << ground_trace_plane_d_L
                        << " | distance="
                        << joint_ground_result
                               .support_ground_distance_m
                        << " | normal_L=["
                        << ground_trace_normal_L.transpose()
                        << "]"
                        << " | anchor_L=["
                        << ground_trace_anchor_L.transpose()
                        << "]"
                        << " | anchor_W=["
                        << ground_trace_anchor_W.transpose()
                        << "]"
                        << " | height_res="
                        << ground_trace_height_residual
                        << " | normal_err_deg="
                        << ground_trace_normal_error_deg
                        << " | rmse="
                        << joint_ground_result
                               .support_plane_rmse_m
                        << " | inlier="
                        << joint_ground_result
                               .support_plane_inlier_ratio
                        << " | tilt="
                        << joint_ground_result
                               .support_ground_tilt_deg
                        << " | ref_d="
                        << joint_ground_reference_plane_d_W
                        << std::endl;
                }
            }
        }
    }


    
    std::function<
        bool(
            const LioState &,
            Ieskf::StateMatrix &,
            Ieskf::StateVector &)>

        joint_ground_observation_builder
 =

        [this,
         joint_ground_ready,
         joint_ground_degraded_height_ready,
         joint_ground_persistent_height_hold_ready,
         persistent_height_hold_target_lidar_normal_W,
         persistent_height_hold_normal_W,
         joint_ground_result,
         joint_ground_reference_normal_W,
         joint_ground_reference_plane_d_W](
            const LioState &linearization_state,
            Ieskf::StateMatrix &joint_information,
            Ieskf::StateVector &joint_gradient)
        {
            joint_information.setZero();
            joint_gradient.setZero();

            // ============================================================
            // FR_PERSISTENT_HEIGHT_HOLD_V1
            //
            // Invalid current Ground must NEVER directly drive the state.
            //
            // For a short dropout, use the LAST TRUSTED Ground height target.
            // ============================================================
            if (!joint_ground_ready)
            {
                if (joint_ground_persistent_height_hold_ready)
                {
                    const Eigen::Isometry3d hold_T_WL =
                        StateToLidarPose(
                            linearization_state);

                    if (!hold_T_WL.matrix().allFinite())
                    {
                        return false;
                    }

                    const double current_lidar_normal_position =
                        persistent_height_hold_normal_W.dot(
                            hold_T_WL.translation());

                    const double hold_residual =
                        persistent_height_hold_target_lidar_normal_W -
                        current_lidar_normal_position;

                    if (!std::isfinite(
                            current_lidar_normal_position) ||
                        !std::isfinite(
                            hold_residual))
                    {
                        return false;
                    }

                    Ieskf::StateVector hold_jacobian =
                        Ieskf::StateVector::Zero();

                    hold_jacobian.segment<3>(
                        LioStateIndex::POSITION) =
                        -persistent_height_hold_normal_W;

                    constexpr double
                        kPersistentHeightHoldSigmaM =
                            0.03;

                    const double hold_information =
                        1.0 /
                        (
                            kPersistentHeightHoldSigmaM *
                            kPersistentHeightHoldSigmaM
                        );

                    joint_information.noalias() +=
                        hold_information *
                        hold_jacobian *
                        hold_jacobian.transpose();

                    joint_gradient.noalias() +=
                        hold_information *
                        hold_jacobian *
                        hold_residual;

                    return
                        joint_information.allFinite() &&
                        joint_gradient.allFinite();
                }

                // Keep the previous HIGH_RMSE degraded fallback only if the
                // trusted-height hold has expired / is unavailable.
                if (!joint_ground_degraded_height_ready)
                {
                    return true;
                }
            }

            // ------------------------------------------------------------
            // Evaluate the 3-D Piecewise Frozen Ground residual:
            //
            //   r_G =
            //     [ tangent normal residual 1
            //       tangent normal residual 2
            //       frozen-plane distance residual ]
            //
            // No direct velocity measurement exists.
            // ------------------------------------------------------------
            const auto evaluate_ground_residual =
                [this,
                 &joint_ground_result,
                 &joint_ground_reference_normal_W,
                 joint_ground_reference_plane_d_W](
                    const LioState &test_state,
                    Eigen::Vector3d &residual,
                    double &normal_error_deg,
                    double &height_error_m)
                {
                    Eigen::Vector3d normal_L =
                        joint_ground_result
                            .support_ground_normal_L;

                    const double normal_L_norm =
                        normal_L.norm();

                    if (!normal_L.allFinite() ||
                        !std::isfinite(normal_L_norm) ||
                        normal_L_norm <= 1.0e-12)
                    {
                        return false;
                    }

                    normal_L /=
                        normal_L_norm;

                    double plane_d_L =
                        joint_ground_result
                            .support_ground_plane_d /
                        normal_L_norm;

                    if (!std::isfinite(plane_d_L))
                    {
                        return false;
                    }

                    const Eigen::Isometry3d T_WL =
                        StateToLidarPose(
                            test_state);

                    if (!T_WL.matrix().allFinite())
                    {
                        return false;
                    }

                    Eigen::Vector3d normal_W =
                        T_WL.rotation() *
                        normal_L;

                    const double normal_W_norm =
                        normal_W.norm();

                    if (!normal_W.allFinite() ||
                        !std::isfinite(normal_W_norm) ||
                        normal_W_norm <= 1.0e-12)
                    {
                        return false;
                    }

                    normal_W /=
                        normal_W_norm;

                    // Deterministic hemisphere relative to ACTIVE piece.
                    if (normal_W.dot(
                            joint_ground_reference_normal_W) <
                        0.0)
                    {
                        normal_W =
                            -normal_W;

                        plane_d_L =
                            -plane_d_L;
                    }

                    // Plane equation:
                    //
                    //     n_W^T x_W + d_W = 0
                    //
                    // For x_W = R_WL x_L + t_WL:
                    //
                    //     d_W = d_L - n_W^T t_WL
                    const double plane_d_W =
                        plane_d_L -
                        normal_W.dot(
                            T_WL.translation());

                    if (!std::isfinite(plane_d_W))
                    {
                        return false;
                    }

                    // Tangent basis around current active frozen normal.
                    Eigen::Vector3d tangent_1 =
                        joint_ground_reference_normal_W
                            .unitOrthogonal();

                    const double tangent_1_norm =
                        tangent_1.norm();

                    if (!tangent_1.allFinite() ||
                        tangent_1_norm <= 1.0e-12)
                    {
                        return false;
                    }

                    tangent_1 /=
                        tangent_1_norm;

                    Eigen::Vector3d tangent_2 =
                        joint_ground_reference_normal_W
                            .cross(tangent_1);

                    const double tangent_2_norm =
                        tangent_2.norm();

                    if (!tangent_2.allFinite() ||
                        tangent_2_norm <= 1.0e-12)
                    {
                        return false;
                    }

                    tangent_2 /=
                        tangent_2_norm;

                    const Eigen::Vector3d normal_difference =
                        normal_W -
                        joint_ground_reference_normal_W;

                    residual(0) =
                        tangent_1.dot(
                            normal_difference);

                    residual(1) =
                        tangent_2.dot(
                            normal_difference);

                    // ----------------------------------------------------
                    // GROUND FINAL: LOCAL observed-plane anchor.
                    //
                    // Current Ground plane in LiDAR coordinates:
                    //
                    //     n_L^T p_L + d_L = 0
                    //
                    // Closest point to the LiDAR origin:
                    //
                    //     p0_L = -d_L * n_L
                    //
                    // This lever arm stays LOCAL (~sensor height),
                    // instead of growing with global XY distance.
                    // ----------------------------------------------------
                    const Eigen::Vector3d observed_anchor_L =
                        -plane_d_L *
                        normal_L;

                    if (!observed_anchor_L.allFinite())
                    {
                        return false;
                    }

                    const Eigen::Vector3d observed_anchor_W =
                        T_WL *
                        observed_anchor_L;

                    if (!observed_anchor_W.allFinite())
                    {
                        return false;
                    }

                    // Signed point-to-FROZEN-plane residual.
                    //
                    // The sign is chosen to preserve the old correction
                    // direction for parallel Ground planes.
                    residual(2) =
                        -(
                            joint_ground_reference_normal_W.dot(
                                observed_anchor_W) +
                            joint_ground_reference_plane_d_W
                         );

                    const double normal_cosine =
                        std::clamp(
                            normal_W.dot(
                                joint_ground_reference_normal_W),
                            -1.0,
                            1.0);

                    normal_error_deg =
                        std::acos(
                            normal_cosine) *
                        57.29577951308232;

                    height_error_m =
                        std::abs(
                            residual(2));

                    return
                        residual.allFinite() &&
                        std::isfinite(
                            normal_error_deg) &&
                        std::isfinite(
                            height_error_m);
                };

            Eigen::Vector3d residual =
                Eigen::Vector3d::Zero();

            double normal_error_deg =
                0.0;

            double height_error_m =
                0.0;

            if (!evaluate_ground_residual(
                    linearization_state,
                    residual,
                    normal_error_deg,
                    height_error_m))
            {
                return true;
            }

            // HARD validity gate.
            if (joint_ground_ready)
            {
                if (normal_error_deg >
                        config_.ground
                            .maximum_normal_residual_deg ||
                    height_error_m >
                        config_.ground
                            .maximum_height_residual_m)
                {
                    return true;
                }
            }
            else
            {
                // FR_GROUND_DEGRADED_HEIGHT_V1
                //
                // No normal observation is used in degraded mode.
                // Height safety gate remains mandatory.
                if (height_error_m >
                    config_.ground
                        .maximum_height_residual_m)
                {
                    return true;
                }
            }

            const double normal_sigma_rad =
                config_.ground.normal_sigma_deg *
                0.017453292519943295;

            const double height_sigma_m =
                config_.ground.height_sigma_m;

            if (!std::isfinite(normal_sigma_rad) ||
                !std::isfinite(height_sigma_m) ||
                normal_sigma_rad <= 0.0 ||
                height_sigma_m <= 0.0)
            {
                return true;
            }

            Eigen::Matrix3d information_R =
                Eigen::Matrix3d::Zero();

            information_R(0, 0) =
                1.0 /
                (normal_sigma_rad *
                 normal_sigma_rad);

            information_R(1, 1) =
                1.0 /
                (normal_sigma_rad *
                 normal_sigma_rad);

            information_R(2, 2) =
                1.0 /
                (height_sigma_m *
                 height_sigma_m);

            if (joint_ground_degraded_height_ready)
            {
                constexpr double
                    kDegradedHeightSigmaM =
                        0.05;

                information_R(0, 0) =
                    0.0;

                information_R(1, 1) =
                    0.0;

                information_R(2, 2) =
                    1.0 /
                    (
                        kDegradedHeightSigmaM *
                        kDegradedHeightSigmaM
                    );

                // Normal residual is explicitly inactive.
                residual(0) =
                    0.0;

                residual(1) =
                    0.0;
            }

            // ------------------------------------------------------------
            // Numerical Jacobian in the EXACT project error-state convention.
            //
            // State rotation uses RIGHT perturbation:
            //
            //     R_new = R * Exp(dtheta)
            //
            // Position is additive in W.
            //
            // Therefore H_G(:,V) = 0 exactly.
            // ------------------------------------------------------------
            Eigen::MatrixXd ground_jacobian =
                Eigen::MatrixXd::Zero(
                    3,
                    joint_information.rows());

            constexpr double kRotationEpsilon =
                1.0e-6;

            constexpr double kPositionEpsilon =
                1.0e-5;

            for (int axis = 0;
                 axis < 3;
                 ++axis)
            {
                Eigen::Vector3d axis_vector =
                    Eigen::Vector3d::Zero();

                axis_vector(axis) =
                    1.0;

                LioState perturbed_state =
                    linearization_state;

                Eigen::Quaterniond delta_q(
                    Eigen::AngleAxisd(
                        kRotationEpsilon,
                        axis_vector));

                perturbed_state.Q_WI =
                    perturbed_state.Q_WI *
                    delta_q;

                perturbed_state.Q_WI.normalize();

                Eigen::Vector3d perturbed_residual =
                    Eigen::Vector3d::Zero();

                double unused_normal_error_deg =
                    0.0;

                double unused_height_error_m =
                    0.0;

                if (!evaluate_ground_residual(
                        perturbed_state,
                        perturbed_residual,
                        unused_normal_error_deg,
                        unused_height_error_m))
                {
                    return true;
                }

                ground_jacobian.col(
                    LioStateIndex::ROTATION +
                    axis) =
                    (perturbed_residual -
                     residual) /
                    kRotationEpsilon;
            }

            for (int axis = 0;
                 axis < 3;
                 ++axis)
            {
                LioState perturbed_state =
                    linearization_state;

                perturbed_state.P_WI(axis) +=
                    kPositionEpsilon;

                Eigen::Vector3d perturbed_residual =
                    Eigen::Vector3d::Zero();

                double unused_normal_error_deg =
                    0.0;

                double unused_height_error_m =
                    0.0;

                if (!evaluate_ground_residual(
                        perturbed_state,
                        perturbed_residual,
                        unused_normal_error_deg,
                        unused_height_error_m))
                {
                    return true;
                }

                ground_jacobian.col(
                    LioStateIndex::POSITION +
                    axis) =
                    (perturbed_residual -
                     residual) /
                    kPositionEpsilon;
            }

            
            // ================================================================
            // GROUND FINAL DOF OWNERSHIP
            //
            //   normal residuals:
            //       own ROLL/PITCH geometry through ROTATION
            //
            //   height residual:
            //       owns TRANSLATION along frozen Ground normal
            //
            // The 1-cm height observation MUST NOT create artificial
            // roll/pitch information through a geometric lever arm.
            //
            // V / bias / gravity may still change INDIRECTLY through
            // IESKF covariance coupling; they have no direct Ground row.
            // ================================================================
            for (int axis = 0;
                 axis < 3;
                 ++axis)
            {
                // Height does not directly own orientation.
                ground_jacobian(
                    2,
                    LioStateIndex::ROTATION +
                    axis) =
                    0.0;

                // Ground-normal residual does not own translation.
                ground_jacobian(
                    0,
                    LioStateIndex::POSITION +
                    axis) =
                    0.0;

                ground_jacobian(
                    1,
                    LioStateIndex::POSITION +
                    axis) =
                    0.0;

                // Exact Jacobian of:
                //
                // r_h = -(n_ref^T p + ...)
                ground_jacobian(
                    2,
                    LioStateIndex::POSITION +
                    axis) =
                    -joint_ground_reference_normal_W(
                        axis);
            }

            if (joint_ground_degraded_height_ready)
            {
                ground_jacobian.row(0).setZero();
                ground_jacobian.row(1).setZero();
            }

            if (!ground_jacobian.allFinite())
            {
                return true;
            }

            // ============================================================
            // FR_GROUND_HEIGHT_ROBUST_INNOVATION_V1
            //
            // Ground height directly observes POSITION only.
            // Vz may still change through the legitimate prior covariance
            // P(z,vz).  When a VALID Ground frame arrives with a very large
            // height innovation, however, the full 1-cm information can
            // create a violent one-frame Z/Vz correction.
            //
            // Use a Huber M-estimator ONLY on the VALID Ground height row:
            //
            //     delta_h = 3 * sigma_h
            //
            //     w = 1                         |r_h| <= delta_h
            //         delta_h / |r_h|           otherwise
            //
            // Both information and gradient therefore receive the same
            // IRLS weight.  Small residuals are EXACTLY unchanged.
            //
            // IMPORTANT:
            //   - ownership is unchanged
            //   - Dense LiDAR still cannot reclaim Ground Z
            //   - no direct velocity measurement is introduced
            //   - covariance coupling remains intact
            //   - Persistent Height Hold is unchanged
            // ============================================================
            double ground_height_robust_weight =
                1.0;

            if (joint_ground_ready)
            {
                constexpr double
                    kGroundHeightHuberSigma =
                        3.0;

                const double height_huber_delta_m =
                    kGroundHeightHuberSigma *
                    height_sigma_m;

                const double absolute_height_residual_m =
                    std::abs(
                        residual(2));

                if (std::isfinite(
                        height_huber_delta_m) &&
                    std::isfinite(
                        absolute_height_residual_m) &&
                    height_huber_delta_m > 0.0 &&
                    absolute_height_residual_m >
                        height_huber_delta_m)
                {
                    ground_height_robust_weight =
                        height_huber_delta_m /
                        absolute_height_residual_m;

                    information_R(2, 2) *=
                        ground_height_robust_weight;

                    static std::size_t
                        ground_height_robust_counter =
                            0U;

                    ++ground_height_robust_counter;

                    if ((ground_height_robust_counter %
                         20U) == 1U)
                    {
                        std::cout
                            << "LIO_GROUND_HEIGHT_ROBUST"
                            << " | r_h="
                            << residual(2)
                            << " | sigma="
                            << height_sigma_m
                            << " | delta="
                            << height_huber_delta_m
                            << " | weight="
                            << ground_height_robust_weight
                            << " | info_z="
                            << information_R(2, 2)
                            << std::endl;
                    }
                }
            }

            // Information-form contribution:
            //
            //   Lambda_G = H_G^T R_G^-1 H_G
            //   g_G      = H_G^T R_G^-1 r_G
            //
            // The IESKF solves:
            //
            //   delta = -(Lambda_prior + Lambda_L + Lambda_G)^-1
            //           (g_prior + g_L + g_G)
            joint_information +=
                ground_jacobian.transpose() *
                information_R *
                ground_jacobian;

            joint_gradient +=
                ground_jacobian.transpose() *
                information_R *
                residual;

            static std::size_t
                joint_ground_counter = 0;

            ++joint_ground_counter;

            if ((joint_ground_counter %
                 100U) == 0U)
            {
                std::cout
                    << "LIO_JOINT_GROUND"
                    << " | r=["
                    << residual.transpose()
                    << "]"
                    << " | normal_deg="
                    << normal_error_deg
                    << " | height="
                    << residual(2)
                    << " | info_diag=["
                    << joint_information(
                           LioStateIndex::ROTATION + 0,
                           LioStateIndex::ROTATION + 0)
                    << " "
                    << joint_information(
                           LioStateIndex::ROTATION + 1,
                           LioStateIndex::ROTATION + 1)
                    << " "
                    << joint_information(
                           LioStateIndex::POSITION + 2,
                           LioStateIndex::POSITION + 2)
                    << "]"
                    << std::endl;
            }

            return true;
        };

    // ========================================================================
    // FR_JOINT_WALL_V1
    //
    // Combined structural callback:
    //
    //      Lambda_struct = Lambda_G + Lambda_W
    //      g_struct      = g_G      + g_W
    //
    // Wall owns:
    //      - translation along trusted wall normals
    //      - yaw around the Ground/world-up axis
    //
    // Wall does NOT directly observe:
    //      z / roll / pitch / velocity / bias / gravity
    // ========================================================================
    std::function<
        bool(
            const LioState &,
            Ieskf::StateMatrix &,
            Ieskf::StateVector &)>
        joint_observation_builder =
            [this,
             &joint_ground_observation_builder,
             joint_wall_ready,
             joint_wall_association,
             joint_wall_baseline_T_SL,
             joint_wall_up_S,
             joint_wall_T_O_S_creation =
                 submap_context.T_O_S_creation](
                const LioState &linearization_state,
                Ieskf::StateMatrix &joint_information,
                Ieskf::StateVector &joint_gradient)
            {
                joint_information.setZero();
                joint_gradient.setZero();

                // ------------------------------------------------------------
                // Ground contribution.
                // ------------------------------------------------------------
                Ieskf::StateMatrix ground_information =
                    Ieskf::StateMatrix::Zero();

                Ieskf::StateVector ground_gradient =
                    Ieskf::StateVector::Zero();

                if (!joint_ground_observation_builder(
                        linearization_state,
                        ground_information,
                        ground_gradient))
                {
                    return false;
                }

                joint_information +=
                    ground_information;

                joint_gradient +=
                    ground_gradient;

                if (!joint_wall_ready)
                {
                    return true;
                }

                constexpr double kMinimumQuality =
                    0.85;

                constexpr double kMaximumNormalDifferenceDeg =
                    1.5;

                constexpr double kMaximumPlaneDifferenceM =
                    0.10;

                constexpr double kMaximumTranslationResidualM =
                    0.03;

                constexpr double kMaximumYawResidualDeg =
                    0.50;

                // Conservative V1 measurement noise.
                constexpr double kWallTranslationSigmaM =
                    0.03;

                constexpr double kWallYawSigmaDeg =
                    1.00;

                constexpr double kRotationEpsilon =
                    1.0e-6;

                constexpr double kPositionEpsilon =
                    1.0e-5;

                const double wall_yaw_sigma_rad =
                    kWallYawSigmaDeg *
                    0.017453292519943295;

                const double maximum_yaw_rad =
                    kMaximumYawResidualDeg *
                    0.017453292519943295;

                double yaw_weighted_target =
                    0.0;

                double yaw_weight_sum =
                    0.0;

                std::size_t used_walls =
                    0U;

                // ============================================================
                // Translation constraints.
                // Each trusted wall contributes ONE scalar normal-direction
                // observation. Parallel walls naturally remain rank-1.
                // ============================================================
                for (const fr_slam::ActiveWallAssociation &wall :
                     joint_wall_association.active_static_walls)
                {
                    if (!wall.reference_normal_A.allFinite() ||
                        !wall.observed_normal_A.allFinite() ||
                        !std::isfinite(wall.reference_d_A) ||
                        !std::isfinite(wall.observed_d_A) ||
                        !std::isfinite(wall.quality) ||
                        !std::isfinite(
                            wall.normal_difference_deg) ||
                        !std::isfinite(
                            wall.plane_distance_difference_m))
                    {
                        continue;
                    }

                    if (wall.quality <
                            kMinimumQuality ||
                        wall.normal_difference_deg >
                            kMaximumNormalDifferenceDeg ||
                        std::abs(
                            wall.plane_distance_difference_m) >
                            kMaximumPlaneDifferenceM)
                    {
                        continue;
                    }

                    Eigen::Vector3d reference_normal =
                        wall.reference_normal_A -
                        joint_wall_up_S *
                            joint_wall_up_S.dot(
                                wall.reference_normal_A);

                    Eigen::Vector3d observed_normal =
                        wall.observed_normal_A -
                        joint_wall_up_S *
                            joint_wall_up_S.dot(
                                wall.observed_normal_A);

                    const double reference_norm =
                        reference_normal.norm();

                    const double observed_norm =
                        observed_normal.norm();

                    if (!std::isfinite(reference_norm) ||
                        !std::isfinite(observed_norm) ||
                        reference_norm <= 1.0e-6 ||
                        observed_norm <= 1.0e-6)
                    {
                        continue;
                    }

                    reference_normal /=
                        reference_norm;

                    observed_normal /=
                        observed_norm;

                    if (reference_normal.dot(
                            observed_normal) < 0.0)
                    {
                        observed_normal =
                            -observed_normal;
                    }

                    const double quality =
                        std::clamp(
                            wall.quality,
                            0.0,
                            1.0);

                    if (quality <= 1.0e-6)
                    {
                        continue;
                    }

                    const double desired_distance =
                        std::clamp(
                            wall.observed_d_A -
                                wall.reference_d_A,
                            -kMaximumTranslationResidualM,
                            kMaximumTranslationResidualM);

                    const double yaw_sine =
                        joint_wall_up_S.dot(
                            observed_normal.cross(
                                reference_normal));

                    const double yaw_cosine =
                        std::clamp(
                            observed_normal.dot(
                                reference_normal),
                            -1.0,
                            1.0);

                    const double desired_yaw =
                        std::clamp(
                            std::atan2(
                                yaw_sine,
                                yaw_cosine),
                            -maximum_yaw_rad,
                            maximum_yaw_rad);

                    if (!std::isfinite(desired_distance) ||
                        !std::isfinite(desired_yaw))
                    {
                        continue;
                    }

                    auto evaluate_translation_residual =
                        [this,
                         &joint_wall_T_O_S_creation,
                         &joint_wall_baseline_T_SL,
                         &reference_normal,
                         desired_distance](
                            const LioState &test_state,
                            double &residual)
                        {
                            const Eigen::Isometry3d T_OL =
                                StateToLidarPose(
                                    test_state);

                            if (!T_OL.matrix().allFinite())
                            {
                                return false;
                            }

                            const Eigen::Isometry3d T_SL =
                                joint_wall_T_O_S_creation
                                    .inverse() *
                                T_OL;

                            if (!T_SL.matrix().allFinite())
                            {
                                return false;
                            }

                            const Eigen::Vector3d delta_t_S =
                                T_SL.translation() -
                                joint_wall_baseline_T_SL
                                    .translation();

                            residual =
                                reference_normal.dot(
                                    delta_t_S) -
                                desired_distance;

                            return std::isfinite(
                                residual);
                        };

                    double residual =
                        0.0;

                    if (!evaluate_translation_residual(
                            linearization_state,
                            residual))
                    {
                        continue;
                    }

                    Eigen::Matrix<double, 1, Ieskf::STATE_DIM>
                        H =
                            Eigen::Matrix<
                                double,
                                1,
                                Ieskf::STATE_DIM>::Zero();

                    for (int axis = 0;
                         axis < 3;
                         ++axis)
                    {
                        Eigen::Vector3d axis_vector =
                            Eigen::Vector3d::Zero();

                        axis_vector(axis) =
                            1.0;

                        LioState perturbed_state =
                            linearization_state;

                        const Eigen::Quaterniond dq(
                            Eigen::AngleAxisd(
                                kRotationEpsilon,
                                axis_vector));

                        perturbed_state.Q_WI =
                            (
                                linearization_state.Q_WI *
                                dq
                            ).normalized();

                        double perturbed_residual =
                            0.0;

                        if (evaluate_translation_residual(
                                perturbed_state,
                                perturbed_residual))
                        {
                            H(
                                0,
                                LioStateIndex::ROTATION +
                                    axis) =
                                (
                                    perturbed_residual -
                                    residual
                                ) /
                                kRotationEpsilon;
                        }
                    }

                    for (int axis = 0;
                         axis < 3;
                         ++axis)
                    {
                        LioState perturbed_state =
                            linearization_state;

                        perturbed_state.P_WI(axis) +=
                            kPositionEpsilon;

                        double perturbed_residual =
                            0.0;

                        if (evaluate_translation_residual(
                                perturbed_state,
                                perturbed_residual))
                        {
                            H(
                                0,
                                LioStateIndex::POSITION +
                                    axis) =
                                (
                                    perturbed_residual -
                                    residual
                                ) /
                                kPositionEpsilon;
                        }
                    }

                    const double translation_information =
                        quality /
                        (
                            kWallTranslationSigmaM *
                            kWallTranslationSigmaM
                        );

                    joint_information +=
                        H.transpose() *
                        translation_information *
                        H;

                    joint_gradient +=
                        H.transpose() *
                        translation_information *
                        residual;

                    yaw_weighted_target +=
                        quality *
                        desired_yaw;

                    yaw_weight_sum +=
                        quality;

                    ++used_walls;
                }

                // ============================================================
                // ONE combined yaw constraint.
                // ============================================================
                if (used_walls > 0U &&
                    yaw_weight_sum > 1.0e-12)
                {
                    const double desired_yaw =
                        yaw_weighted_target /
                        yaw_weight_sum;

                    auto evaluate_yaw_residual =
                        [this,
                         &joint_wall_T_O_S_creation,
                         &joint_wall_baseline_T_SL,
                         &joint_wall_up_S,
                         desired_yaw](
                            const LioState &test_state,
                            double &residual)
                        {
                            const Eigen::Isometry3d T_OL =
                                StateToLidarPose(
                                    test_state);

                            if (!T_OL.matrix().allFinite())
                            {
                                return false;
                            }

                            const Eigen::Isometry3d T_SL =
                                joint_wall_T_O_S_creation
                                    .inverse() *
                                T_OL;

                            if (!T_SL.matrix().allFinite())
                            {
                                return false;
                            }

                            const Eigen::Matrix3d R_delta =
                                T_SL.rotation() *
                                joint_wall_baseline_T_SL
                                    .rotation()
                                    .transpose();

                            Eigen::AngleAxisd aa(
                                R_delta);

                            if (!std::isfinite(aa.angle()) ||
                                !aa.axis().allFinite())
                            {
                                return false;
                            }

                            const Eigen::Vector3d rotation_vector =
                                aa.axis() *
                                aa.angle();

                            const double yaw_increment =
                                joint_wall_up_S.dot(
                                    rotation_vector);

                            residual =
                                yaw_increment -
                                desired_yaw;

                            return std::isfinite(
                                residual);
                        };

                    double yaw_residual =
                        0.0;

                    if (evaluate_yaw_residual(
                            linearization_state,
                            yaw_residual))
                    {
                        Eigen::Matrix<
                            double,
                            1,
                            Ieskf::STATE_DIM>
                            H_yaw =
                                Eigen::Matrix<
                                    double,
                                    1,
                                    Ieskf::STATE_DIM>::Zero();

                        for (int axis = 0;
                             axis < 3;
                             ++axis)
                        {
                            Eigen::Vector3d axis_vector =
                                Eigen::Vector3d::Zero();

                            axis_vector(axis) =
                                1.0;

                            LioState perturbed_state =
                                linearization_state;

                            const Eigen::Quaterniond dq(
                                Eigen::AngleAxisd(
                                    kRotationEpsilon,
                                    axis_vector));

                            perturbed_state.Q_WI =
                                (
                                    linearization_state.Q_WI *
                                    dq
                                ).normalized();

                            double perturbed_residual =
                                0.0;

                            if (evaluate_yaw_residual(
                                    perturbed_state,
                                    perturbed_residual))
                            {
                                H_yaw(
                                    0,
                                    LioStateIndex::ROTATION +
                                        axis) =
                                    (
                                        perturbed_residual -
                                        yaw_residual
                                    ) /
                                    kRotationEpsilon;
                            }
                        }

                        const double yaw_information =
                            1.0 /
                            (
                                wall_yaw_sigma_rad *
                                wall_yaw_sigma_rad
                            );

                        joint_information +=
                            H_yaw.transpose() *
                            yaw_information *
                            H_yaw;

                        joint_gradient +=
                            H_yaw.transpose() *
                            yaw_information *
                            yaw_residual;

                        static std::size_t
                            joint_wall_counter = 0U;

                        ++joint_wall_counter;

                        if ((joint_wall_counter %
                             100U) == 0U)
                        {
                            std::cout
                                << "LIO_JOINT_WALL"
                                << " | used="
                                << used_walls
                                << " | yaw_target_deg="
                                << desired_yaw *
                                       57.29577951308232
                                << " | yaw_res_deg="
                                << yaw_residual *
                                       57.29577951308232
                                << " | info_pos_diag=["
                                << joint_information(
                                       LioStateIndex::POSITION + 0,
                                       LioStateIndex::POSITION + 0)
                                << " "
                                << joint_information(
                                       LioStateIndex::POSITION + 1,
                                       LioStateIndex::POSITION + 1)
                                << " "
                                << joint_information(
                                       LioStateIndex::POSITION + 2,
                                       LioStateIndex::POSITION + 2)
                                << "]"
                                << std::endl;
                        }
                    }
                }

                return
                    joint_information.allFinite() &&
                    joint_gradient.allFinite();
            };

    
    // ========================================================================
    // FR_JOINT_WALL_FINAL_BUILDER
    //
    // Existing joint_observation_builder = Ground only.
    //
    // FINAL Wall contributes:
    //
    //   translation:
    //       rank-1 or rank-2 basis measurement in the Ground/gravity tangent
    //       plane. Parallel walls DO NOT multiply the same information.
    //
    //   rotation:
    //       one yaw measurement around the frozen up axis.
    //
    // Topology / basis / target are frozen before IEKF iteration.
    // Only residual/Jacobian are relinearized at the current IEKF state.
    // ========================================================================

    std::function<
        bool(
            const LioState &,
            Ieskf::StateMatrix &,
            Ieskf::StateVector &)>
        joint_observation_builder_final =
            [this,
             &joint_observation_builder,
             &submap_context,
             joint_wall_final_ready,
             joint_wall_final_translation_active,
             joint_wall_final_yaw_active,
             joint_wall_final_rank,
             joint_wall_final_target_delta_t_S,
             joint_wall_final_target_yaw_rad,
             joint_wall_association,
             joint_wall_baseline_T_SL,
             joint_wall_up_S](
                const LioState &linearization_state,
                Ieskf::StateMatrix &joint_information,
                Ieskf::StateVector &joint_gradient)
            {
                if (!joint_observation_builder(
                        linearization_state,
                        joint_information,
                        joint_gradient))
                {
                    return false;
                }

                if (!joint_wall_final_ready)
                {
                    return true;
                }

                Ieskf::StateMatrix wall_information =
                    Ieskf::StateMatrix::Zero();

                Ieskf::StateVector wall_gradient =
                    Ieskf::StateVector::Zero();

                constexpr double
                    kMinimumQuality =
                        0.85;

                constexpr double
                    kMaximumNormalDifferenceDeg =
                        1.5;

                constexpr double
                    kMaximumPlaneDistanceDifferenceM =
                        0.10;

                constexpr double
                    kObservableRelativeEigenThreshold =
                        0.10;

                constexpr double
                    kTranslationSigmaM =
                        0.03;

                constexpr double
                    kYawSigmaRad =
                        1.0 *
                        0.017453292519943295;

                Eigen::Matrix3d translation_H =
                    Eigen::Matrix3d::Zero();

                for (const fr_slam::ActiveWallAssociation &wall :
                     joint_wall_association.active_static_walls)
                {
                    if (!wall.reference_normal_A.allFinite() ||
                        !std::isfinite(wall.quality) ||
                        !std::isfinite(
                            wall.normal_difference_deg) ||
                        !std::isfinite(
                            wall.plane_distance_difference_m))
                    {
                        continue;
                    }

                    if (wall.quality <
                            kMinimumQuality ||
                        wall.normal_difference_deg >
                            kMaximumNormalDifferenceDeg ||
                        std::abs(
                            wall.plane_distance_difference_m) >
                            kMaximumPlaneDistanceDifferenceM)
                    {
                        continue;
                    }

                    Eigen::Vector3d reference_normal =
                        wall.reference_normal_A -
                        joint_wall_up_S *
                            joint_wall_up_S.dot(
                                wall.reference_normal_A);

                    const double normal_norm =
                        reference_normal.norm();

                    if (!reference_normal.allFinite() ||
                        !std::isfinite(normal_norm) ||
                        normal_norm <= 1.0e-6)
                    {
                        continue;
                    }

                    reference_normal /=
                        normal_norm;

                    const double weight =
                        std::clamp(
                            wall.quality,
                            0.0,
                            1.0);

                    translation_H +=
                        weight *
                        reference_normal *
                        reference_normal.transpose();
                }

                Eigen::SelfAdjointEigenSolver<
                    Eigen::Matrix3d>
                    eigen_solver(
                        translation_H);

                if (eigen_solver.info() !=
                    Eigen::Success)
                {
                    return true;
                }

                const Eigen::Vector3d eigenvalues =
                    eigen_solver.eigenvalues();

                const Eigen::Matrix3d eigenvectors =
                    eigen_solver.eigenvectors();

                const double maximum_eigenvalue =
                    eigenvalues.maxCoeff();

                if (!eigenvalues.allFinite() ||
                    !eigenvectors.allFinite() ||
                    !std::isfinite(
                        maximum_eigenvalue) ||
                    maximum_eigenvalue <=
                        1.0e-12)
                {
                    return true;
                }

                const double eigen_threshold =
                    std::max(
                        1.0e-6,
                        maximum_eigenvalue *
                            kObservableRelativeEigenThreshold);

                const auto evaluate_translation_delta_S =
                    [this,
                     &submap_context,
                     &joint_wall_baseline_T_SL](
                        const LioState &test_state,
                        Eigen::Vector3d &delta_t_S)
                    {
                        const Eigen::Isometry3d T_OL =
                            StateToLidarPose(
                                test_state);

                        if (!T_OL.matrix().allFinite())
                        {
                            return false;
                        }

                        const Eigen::Isometry3d T_SL =
                            submap_context
                                .T_O_S_creation
                                .inverse() *
                            T_OL;

                        if (!T_SL.matrix().allFinite())
                        {
                            return false;
                        }

                        delta_t_S =
                            T_SL.translation() -
                            joint_wall_baseline_T_SL
                                .translation();

                        return delta_t_S.allFinite();
                    };

                const auto make_rotation_perturbed_state =
                    [](
                        const LioState &state,
                        int axis,
                        double delta)
                    {
                        LioState perturbed =
                            state;

                        Eigen::Vector3d axis_vector =
                            Eigen::Vector3d::Zero();

                        axis_vector(axis) =
                            1.0;

                        const Eigen::Quaterniond dq(
                            Eigen::AngleAxisd(
                                delta,
                                axis_vector));

                        perturbed.Q_WI =
                            (
                                state.Q_WI.normalized() *
                                dq
                            ).normalized();

                        return perturbed;
                    };

                // ------------------------------------------------------------
                // Rank-aware translation.
                //
                // Use ONE scalar observation per observable eigen-direction.
                // Parallel duplicate walls therefore remain rank-1.
                // ------------------------------------------------------------
                std::size_t used_rank =
                    0;

                if (joint_wall_final_translation_active &&
                    joint_wall_final_rank > 0)
                {
                    Eigen::Vector3d current_delta_t_S;

                    if (!evaluate_translation_delta_S(
                            linearization_state,
                            current_delta_t_S))
                    {
                        return false;
                    }

                    for (int eigen_index = 2;
                         eigen_index >= 0;
                         --eigen_index)
                    {
                        if (used_rank >=
                            joint_wall_final_rank)
                        {
                            break;
                        }

                        if (eigenvalues(eigen_index) <=
                            eigen_threshold)
                        {
                            continue;
                        }

                        Eigen::Vector3d basis_S =
                            eigenvectors.col(
                                eigen_index);

                        basis_S -=
                            joint_wall_up_S *
                            joint_wall_up_S.dot(
                                basis_S);

                        const double basis_norm =
                            basis_S.norm();

                        if (!basis_S.allFinite() ||
                            !std::isfinite(basis_norm) ||
                            basis_norm <= 1.0e-9)
                        {
                            continue;
                        }

                        basis_S /=
                            basis_norm;

                        const double residual =
                            basis_S.dot(
                                current_delta_t_S -
                                joint_wall_final_target_delta_t_S);

                        if (!std::isfinite(residual))
                        {
                            continue;
                        }

                        Ieskf::StateVector jacobian =
                            Ieskf::StateVector::Zero();

                        constexpr double
                            kRotationEpsilon =
                                1.0e-6;

                        constexpr double
                            kPositionEpsilon =
                                1.0e-5;

                        // Right-perturb IMU rotation.
                        for (int axis = 0;
                             axis < 3;
                             ++axis)
                        {
                            const LioState plus_state =
                                make_rotation_perturbed_state(
                                    linearization_state,
                                    axis,
                                    kRotationEpsilon);

                            const LioState minus_state =
                                make_rotation_perturbed_state(
                                    linearization_state,
                                    axis,
                                    -kRotationEpsilon);

                            Eigen::Vector3d plus_delta_t_S;
                            Eigen::Vector3d minus_delta_t_S;

                            if (!evaluate_translation_delta_S(
                                    plus_state,
                                    plus_delta_t_S) ||
                                !evaluate_translation_delta_S(
                                    minus_state,
                                    minus_delta_t_S))
                            {
                                return false;
                            }

                            const double plus_residual =
                                basis_S.dot(
                                    plus_delta_t_S -
                                    joint_wall_final_target_delta_t_S);

                            const double minus_residual =
                                basis_S.dot(
                                    minus_delta_t_S -
                                    joint_wall_final_target_delta_t_S);

                            jacobian(
                                LioStateIndex::ROTATION +
                                axis) =
                                    (
                                        plus_residual -
                                        minus_residual
                                    ) /
                                    (
                                        2.0 *
                                        kRotationEpsilon
                                    );
                        }

                        // World-frame position perturbation.
                        for (int axis = 0;
                             axis < 3;
                             ++axis)
                        {
                            LioState plus_state =
                                linearization_state;

                            LioState minus_state =
                                linearization_state;

                            plus_state.P_WI(axis) +=
                                kPositionEpsilon;

                            minus_state.P_WI(axis) -=
                                kPositionEpsilon;

                            Eigen::Vector3d plus_delta_t_S;
                            Eigen::Vector3d minus_delta_t_S;

                            if (!evaluate_translation_delta_S(
                                    plus_state,
                                    plus_delta_t_S) ||
                                !evaluate_translation_delta_S(
                                    minus_state,
                                    minus_delta_t_S))
                            {
                                return false;
                            }

                            const double plus_residual =
                                basis_S.dot(
                                    plus_delta_t_S -
                                    joint_wall_final_target_delta_t_S);

                            const double minus_residual =
                                basis_S.dot(
                                    minus_delta_t_S -
                                    joint_wall_final_target_delta_t_S);

                            jacobian(
                                LioStateIndex::POSITION +
                                axis) =
                                    (
                                        plus_residual -
                                        minus_residual
                                    ) /
                                    (
                                        2.0 *
                                        kPositionEpsilon
                                    );
                        }

                        const double
                            relative_observability =
                                std::clamp(
                                    eigenvalues(
                                        eigen_index) /
                                        maximum_eigenvalue,
                                    0.0,
                                    1.0);

                        const double information_weight =
                            relative_observability /
                            (
                                kTranslationSigmaM *
                                kTranslationSigmaM
                            );

                        wall_information +=
                            information_weight *
                            jacobian *
                            jacobian.transpose();

                        wall_gradient +=
                            information_weight *
                            jacobian *
                            residual;

                        ++used_rank;
                    }
                }

                // ------------------------------------------------------------
                // One yaw observation around the frozen up axis.
                // ------------------------------------------------------------
                double yaw_residual =
                    0.0;

                if (joint_wall_final_yaw_active)
                {
                    const auto evaluate_yaw =
                        [this,
                         &submap_context,
                         &joint_wall_baseline_T_SL,
                         &joint_wall_up_S](
                            const LioState &test_state,
                            double &yaw_rad)
                        {
                            const Eigen::Isometry3d T_OL =
                                StateToLidarPose(
                                    test_state);

                            if (!T_OL.matrix().allFinite())
                            {
                                return false;
                            }

                            const Eigen::Isometry3d T_SL =
                                submap_context
                                    .T_O_S_creation
                                    .inverse() *
                                T_OL;

                            if (!T_SL.matrix().allFinite())
                            {
                                return false;
                            }

                            const Eigen::Matrix3d relative_R =
                                joint_wall_baseline_T_SL
                                    .rotation()
                                    .transpose() *
                                T_SL.rotation();

                            if (!relative_R.allFinite())
                            {
                                return false;
                            }

                            Eigen::AngleAxisd angle_axis(
                                relative_R);

                            Eigen::Vector3d rotation_vector =
                                Eigen::Vector3d::Zero();

                            if (std::isfinite(
                                    angle_axis.angle()) &&
                                std::abs(
                                    angle_axis.angle()) >
                                    1.0e-12 &&
                                angle_axis.axis().allFinite())
                            {
                                rotation_vector =
                                    angle_axis.axis() *
                                    angle_axis.angle();
                            }

                            yaw_rad =
                                joint_wall_up_S.dot(
                                    rotation_vector);

                            return std::isfinite(
                                yaw_rad);
                        };

                    double current_yaw =
                        0.0;

                    if (!evaluate_yaw(
                            linearization_state,
                            current_yaw))
                    {
                        return false;
                    }

                    yaw_residual =
                        current_yaw -
                        joint_wall_final_target_yaw_rad;

                    Ieskf::StateVector yaw_jacobian =
                        Ieskf::StateVector::Zero();

                    constexpr double
                        kRotationEpsilon =
                            1.0e-6;

                    for (int axis = 0;
                         axis < 3;
                         ++axis)
                    {
                        const LioState plus_state =
                            make_rotation_perturbed_state(
                                linearization_state,
                                axis,
                                kRotationEpsilon);

                        const LioState minus_state =
                            make_rotation_perturbed_state(
                                linearization_state,
                                axis,
                                -kRotationEpsilon);

                        double plus_yaw =
                            0.0;

                        double minus_yaw =
                            0.0;

                        if (!evaluate_yaw(
                                plus_state,
                                plus_yaw) ||
                            !evaluate_yaw(
                                minus_state,
                                minus_yaw))
                        {
                            return false;
                        }

                        yaw_jacobian(
                            LioStateIndex::ROTATION +
                            axis) =
                                (
                                    plus_yaw -
                                    minus_yaw
                                ) /
                                (
                                    2.0 *
                                    kRotationEpsilon
                                );
                    }

                    const double yaw_information =
                        1.0 /
                        (
                            kYawSigmaRad *
                            kYawSigmaRad
                        );

                    wall_information +=
                        yaw_information *
                        yaw_jacobian *
                        yaw_jacobian.transpose();

                    wall_gradient +=
                        yaw_information *
                        yaw_jacobian *
                        yaw_residual;
                }

                if (!wall_information.allFinite() ||
                    !wall_gradient.allFinite())
                {
                    return false;
                }

                wall_information =
                    0.5 *
                    (
                        wall_information +
                        wall_information.transpose()
                    );

                joint_information +=
                    wall_information;

                joint_gradient +=
                    wall_gradient;

                static std::size_t
                    joint_wall_final_counter =
                        0;

                ++joint_wall_final_counter;

                if ((joint_wall_final_counter %
                     100U) == 1U)
                {
                    std::cout
                        << "LIO_JOINT_WALL_FINAL"
                        << " | rank="
                        << used_rank
                        << " | translation="
                        << (
                               joint_wall_final_translation_active
                                   ? 1
                                   : 0
                           )
                        << " | yaw="
                        << (
                               joint_wall_final_yaw_active
                                   ? 1
                                   : 0
                           )
                        << " | target_dt_S=["
                        << joint_wall_final_target_delta_t_S.x()
                        << ","
                        << joint_wall_final_target_delta_t_S.y()
                        << ","
                        << joint_wall_final_target_delta_t_S.z()
                        << "]"
                        << " | target_yaw_deg="
                        << joint_wall_final_target_yaw_rad *
                               57.29577951308232
                        << " | yaw_res_deg="
                        << yaw_residual *
                               57.29577951308232
                        << " | wall_info_pos_diag=["
                        << wall_information(
                               LioStateIndex::POSITION,
                               LioStateIndex::POSITION)
                        << " "
                        << wall_information(
                               LioStateIndex::POSITION + 1,
                               LioStateIndex::POSITION + 1)
                        << " "
                        << wall_information(
                               LioStateIndex::POSITION + 2,
                               LioStateIndex::POSITION + 2)
                        << "]"
                        << std::endl;
                }

                return true;
            };


    // ========================================================================
    // FR_PERSISTENT_GROUND_OWNERSHIP_V4
    //
    // Once Ground reference is frozen:
    //
    //   current Ground VALID:
    //       Ground measurement supplies its own rank-3 ownership.
    //
    //   current Ground INVALID:
    //       no Ground residual is injected,
    //       but Dense LiDAR still cannot reclaim tilt + vertical position.
    //
    // This block creates OWNERSHIP ONLY.
    // It contains no residual and no gradient.
    // ========================================================================
    const bool persistent_ground_owner_active =
        config_.ground.enabled &&
        config_.ground.mode != "off" &&
        config_.ground.mode != "disabled" &&
        ground_reference_frozen_ &&
        ground_reference_normal_W_.allFinite() &&
        ground_reference_normal_W_.norm() > 1.0e-12;

    Eigen::Vector3d persistent_ground_normal_W =
        Eigen::Vector3d::UnitZ();

    if (ground_reference_frozen_ &&
        ground_reference_normal_W_.allFinite() &&
        ground_reference_normal_W_.norm() > 1.0e-12)
    {
        persistent_ground_normal_W =
            ground_reference_normal_W_.normalized();
    }

    std::function<
        bool(
            const LioState &,
            Ieskf::StateMatrix &)>
        structural_ownership_builder =
            [persistent_ground_owner_active,
             persistent_ground_normal_W](
                const LioState &linearization_state,
                Ieskf::StateMatrix &ownership_information)
            {
                ownership_information.setZero();

                if (!persistent_ground_owner_active)
                {
                    return true;
                }

                if (!linearization_state.Q_WI.coeffs().allFinite())
                {
                    return false;
                }

                const double quaternion_norm =
                    linearization_state.Q_WI.norm();

                if (!std::isfinite(quaternion_norm) ||
                    quaternion_norm <= 1.0e-12)
                {
                    return false;
                }

                const Eigen::Matrix3d R_WI =
                    linearization_state.Q_WI
                        .normalized()
                        .toRotationMatrix();

                // State rotation uses RIGHT perturbation:
                //
                //     R_new = R * Exp(delta_theta)
                //
                // Therefore a world Ground direction must be expressed
                // in the IMU tangent frame for ROTATION ownership.
                Eigen::Vector3d ground_normal_I =
                    R_WI.transpose() *
                    persistent_ground_normal_W;

                const double ground_normal_I_norm =
                    ground_normal_I.norm();

                if (!ground_normal_I.allFinite() ||
                    !std::isfinite(ground_normal_I_norm) ||
                    ground_normal_I_norm <= 1.0e-12)
                {
                    return false;
                }

                ground_normal_I /=
                    ground_normal_I_norm;

                constexpr double kOwnershipWeight =
                    10000.0;

                // --------------------------------------------------------
                // ROTATION:
                //
                // Ground owns the two directions perpendicular to
                // Ground normal. Rotation about Ground normal = yaw,
                // therefore it remains available to Dense LiDAR.
                // --------------------------------------------------------
                const Eigen::Matrix3d ground_tilt_projector_I =
                    Eigen::Matrix3d::Identity() -
                    ground_normal_I *
                    ground_normal_I.transpose();

                ownership_information.block<3, 3>(
                    LioStateIndex::ROTATION,
                    LioStateIndex::ROTATION) =
                    kOwnershipWeight *
                    ground_tilt_projector_I;

                // --------------------------------------------------------
                // POSITION:
                //
                // Position error is represented in WORLD coordinates.
                // Ground owns translation along frozen Ground normal.
                // --------------------------------------------------------
                ownership_information.block<3, 3>(
                    LioStateIndex::POSITION,
                    LioStateIndex::POSITION) =
                    kOwnershipWeight *
                    (
                        persistent_ground_normal_W *
                        persistent_ground_normal_W.transpose()
                    );

                ownership_information =
                    0.5 *
                    (
                        ownership_information +
                        ownership_information.transpose()
                    );

                return ownership_information.allFinite();
            };

    // One line per degraded frame -- NOT per IEKF iteration.
    if (persistent_ground_owner_active &&
        !joint_ground_ready)
    {
        std::cout
            << "LIO_PERSISTENT_GROUND_OWNERSHIP"
            << " | measurement=0"
            << " | ownership_rank=3"
            << " | n_W=["
            << persistent_ground_normal_W.transpose()
            << "]"
            << std::endl;
    }

if (!ieskf_.IteratedLidarUpdate(
            result.processed_frame.cloud,
            *prepared_target,
            measurement_builder_,
            result.lidar_update,
            &joint_observation_builder_final,
            &structural_ownership_builder))
    {
        std::cerr
            << "LIO_REJECT | stage=JOINT_LIDAR_GROUND_UPDATE"
            << std::endl;
        return false;
    }

    // ========================================================================
    // 5.1 Ground measurement.
    //
    // Before the frozen reference exists this call only accumulates trusted
    // support planes for BOOTSTRAP.  Once frozen, it performs the Ground
    // IESKF update on roll / pitch / z.
    // ========================================================================
    

    
// Ground is already fused inside FR_JOINT_GROUND_V1.
//
// DO NOT perform a second GroundStateUpdate / GroundPoseUpdate here.
// That would double-count the same current-frame Ground geometry.


    // ========================================================================
    // 5.2 Final V1 Wall Association -- DIAGNOSTICS ONLY.
    //
    // IMPORTANT:
    //   Ground has already modified/injected the pose.
    //
    //   T_O_L = final Ground-corrected LIO pose
    //   T_O_S = immutable PRIMARY Submap creation pose
    //
    // therefore:
    //
    //   T_S_L = T_O_S^-1 * T_O_L
    //
    // Wall persistent geometry is stored entirely in S.
    //
    // NO Wall pose correction is applied in this version.
    // ========================================================================
    // OLD post-IESKF Wall path disabled by FR_JOINT_WALL_V1.
    // Wall is now prepared BEFORE the IEKF and fused through the
    // structural joint_observation_builder above.
    if (false &&
        config_.wall_constraint_enable &&
        submap_context.valid &&
        submap_context.T_O_S_creation.matrix().allFinite())
    {
        if (wall_association_submap_id_ !=
            submap_context.primary_submap_id)
        {
            wall_association_.Reset();

            wall_association_frame_index_ = 0;

            wall_temporal_initialized_ = false;
            wall_consistent_frames_ = 0;
            wall_temporal_last_frame_index_ = 0;

            wall_previous_delta_t_S_.setZero();
            wall_previous_yaw_rad_ = 0.0;

            wall_filtered_delta_t_S_.setZero();
            wall_filtered_yaw_rad_ = 0.0;

            wall_association_submap_id_ =
                submap_context.primary_submap_id;

            std::cout
                << "LIO_WALL_SUBMAP_RESET"
                << " | submap="
                << wall_association_submap_id_
                << std::endl;
        }

        if (last_ground_segmentation_valid_ &&
            ground_measurement_cloud &&
            !ground_measurement_cloud->empty())
        {
            const Eigen::Isometry3d T_OL_ground =
                StateToLidarPose(
                    ieskf_.State());

            if (T_OL_ground.matrix().allFinite())
            {
                const fr_slam::MultiPlaneExtractor
                    multi_plane_extractor;

                Eigen::Vector3d up_direction_L =
                    T_OL_ground.rotation().transpose() *
                    Eigen::Vector3d::UnitZ();

                if (!up_direction_L.allFinite() ||
                    up_direction_L.norm() < 1.0e-9)
                {
                    up_direction_L =
                        Eigen::Vector3d::UnitZ();
                }

                fr_slam::MultiPlaneExtractionResult
                    multi_plane_result =
                        multi_plane_extractor.Extract(
                            ground_measurement_cloud,
                            up_direction_L);

                fr_slam::RefinePlaneConstraintEligibility(
                    multi_plane_result,
                    last_ground_segmentation_result_,
                    multi_plane_extractor.GetConfig(),
                    T_OL_ground.rotation(),
                    T_OL_ground.translation());

                // Keep existing RViz MultiPlane visualization working in LIO.
                fr_slam::StoreLatestMultiPlaneResult(
                    multi_plane_result);

                const Eigen::Isometry3d T_SL =
                    submap_context
                        .T_O_S_creation
                        .inverse() *
                    T_OL_ground;

                if (T_SL.matrix().allFinite())
                {
                    ++wall_association_frame_index_;

                    result.wall_association =
                        wall_association_.Update(
                            multi_plane_result,
                            T_SL,
                            wall_association_frame_index_);


                    ProcessWallMeasurement(
                            result.wall_association,
                            result.processed_frame.cloud,
                            prepared_target,
                            submap_context);

                    if ((wall_association_frame_index_ % 10U) == 0U ||
                        result.wall_association
                            .active_static_wall_count > 0U)
                    {
                        std::cout
                            << "LIO_WALL_ASSOC"
                            << " | submap="
                            << wall_association_submap_id_
                            << " | frame="
                            << wall_association_frame_index_
                            << " | raw="
                            << result.wall_association
                                   .raw_wall_candidates
                            << " | physical="
                            << result.wall_association
                                   .physical_wall_candidates
                            << " | persistent="
                            << result.wall_association
                                   .persistent_walls
                            << " | active_static="
                            << result.wall_association
                                   .active_static_wall_count
                            << std::endl;

                        for (const fr_slam::ActiveWallAssociation &wall :
                             result.wall_association
                                 .active_static_walls)
                        {
                            std::cout
                                << "LIO_WALL_ACTIVE"
                                << " | submap="
                                << wall_association_submap_id_
                                << " | id="
                                << wall.persistent_wall_id
                                << " | ref_n=["
                                << wall.reference_normal_A.x()
                                << ","
                                << wall.reference_normal_A.y()
                                << ","
                                << wall.reference_normal_A.z()
                                << "]"
                                << " | obs_n=["
                                << wall.observed_normal_A.x()
                                << ","
                                << wall.observed_normal_A.y()
                                << ","
                                << wall.observed_normal_A.z()
                                << "]"
                                << " | ref_d="
                                << wall.reference_d_A
                                << " | obs_d="
                                << wall.observed_d_A
                                << " | normal_diff_deg="
                                << wall.normal_difference_deg
                                << " | plane_diff_m="
                                << wall.plane_distance_difference_m
                                << " | quality="
                                << wall.quality
                                << std::endl;
                        }
                    }
                }
            }
        }
    }



    result.state =
        ieskf_.State();

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
