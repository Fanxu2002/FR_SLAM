#include "fr_slam/frontend/ieskf.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

#include "fr_slam/frontend/lio_math.hpp"
#include "fr_slam/frontend/lio_measurement.hpp"

namespace
{

    constexpr double GRAVITY_TRANSPORT_EPSILON =
        1.0e-6;


    Eigen::Matrix3d SkewSymmetric(const Eigen::Vector3d &v)
    {
        Eigen::Matrix3d skew = Eigen::Matrix3d::Zero();

        skew <<
            0.0, -v.z(), v.y(),
            v.z(), 0.0, -v.x(),
            -v.y(), v.x(), 0.0;

        return skew;
    }

    // ========================================================================
    // BuildLioPoseGraphInformation()
    //
    // FR_LIO_ODOM_INFORMATION_V2
    //
    // Convert the final ACTUAL LiDAR/Ground/Wall measurement information from
    // the IESKF tangent space into a robust dynamic 6x6 PoseGraph odometry-edge
    // information matrix.
    //
    // Design rules:
    //   1. Use measurement information, not posterior covariance.
    //   2. Persistent structural ownership is NOT a measurement.
    //   3. If online extrinsic estimation is enabled, marginalize the
    //      extrinsic block with a Moore-Penrose Schur complement.
    //   4. Transform IMU [right-rotation, world-position] perturbations into
    //      local LiDAR [rotation, translation], including the lever arm.
    //   5. Reorder to g2o::EdgeSE3 [translation, rotation].
    //   6. Build directional confidence from marginal covariance, NORMALIZE
    //      TRANSLATION AND ROTATION SEPARATELY, then clamp to [0.01, 1.0].
    //      This avoids the V1 saturation bug where absolute Hessian scale made
    //      tx/ty/tz/roll/pitch all clamp to 1 before normalization.
    //   7. Preserve full 6x6 precision coupling, but shrink its standardized
    //      correlation toward Identity:
    //
    //          J_safe = alpha * J + (1-alpha) * I, alpha = 0.75
    //
    //      This retains 75% of the measured coupling while guaranteeing a
    //      finite SPD margin. With the 0.01 confidence floor, the resulting
    //      condition number is conservatively bounded to O(1e3), instead of
    //      the O(1e4~1e5) near-singular V1 behavior observed in real runs.
    // ========================================================================
    bool BuildLioPoseGraphInformation(
        const Ieskf::StateMatrix &measurement_information,
        const LioState &state,
        const bool estimate_extrinsic,
        Eigen::Matrix<double, 6, 6> &pose_graph_information)
    {
        using Matrix6d = Eigen::Matrix<double, 6, 6>;
        using Vector6d = Eigen::Matrix<double, 6, 1>;

        pose_graph_information = Matrix6d::Identity();

        if (!measurement_information.allFinite() ||
            !state.Q_WI.coeffs().allFinite() ||
            state.Q_WI.norm() <= 1.0e-12 ||
            !state.Q_IL.coeffs().allFinite() ||
            state.Q_IL.norm() <= 1.0e-12 ||
            !state.P_IL.allFinite())
        {
            return false;
        }

        Matrix6d lambda_pose =
            measurement_information.block<6, 6>(
                LioStateIndex::ROTATION,
                LioStateIndex::ROTATION);

        lambda_pose =
            0.5 *
            (lambda_pose + lambda_pose.transpose());

        if (!lambda_pose.allFinite())
        {
            return false;
        }

        // ---------------------------------------------------------------
        // Marginalize online extrinsic variables when they are estimated.
        // ---------------------------------------------------------------
        if (estimate_extrinsic)
        {
            const Matrix6d lambda_pose_extrinsic =
                measurement_information.block<6, 6>(
                    LioStateIndex::ROTATION,
                    LioStateIndex::EXTRINSIC_ROTATION);

            Matrix6d lambda_extrinsic =
                measurement_information.block<6, 6>(
                    LioStateIndex::EXTRINSIC_ROTATION,
                    LioStateIndex::EXTRINSIC_ROTATION);

            lambda_extrinsic =
                0.5 *
                (lambda_extrinsic + lambda_extrinsic.transpose());

            if (!lambda_pose_extrinsic.allFinite() ||
                !lambda_extrinsic.allFinite())
            {
                return false;
            }

            Eigen::SelfAdjointEigenSolver<Matrix6d> extrinsic_solver(
                lambda_extrinsic);

            if (extrinsic_solver.info() != Eigen::Success ||
                !extrinsic_solver.eigenvalues().allFinite() ||
                !extrinsic_solver.eigenvectors().allFinite())
            {
                return false;
            }

            const Vector6d extrinsic_eigenvalues =
                extrinsic_solver.eigenvalues();

            const double maximum_extrinsic_eigenvalue =
                std::max(0.0, extrinsic_eigenvalues.maxCoeff());

            Matrix6d lambda_extrinsic_pinv = Matrix6d::Zero();

            if (maximum_extrinsic_eigenvalue > 1.0e-12)
            {
                const double threshold =
                    std::max(
                        1.0e-10,
                        maximum_extrinsic_eigenvalue * 1.0e-8);

                Matrix6d inverse_eigenvalues = Matrix6d::Zero();

                for (int i = 0; i < 6; ++i)
                {
                    if (extrinsic_eigenvalues(i) > threshold)
                    {
                        inverse_eigenvalues(i, i) =
                            1.0 / extrinsic_eigenvalues(i);
                    }
                }

                lambda_extrinsic_pinv =
                    extrinsic_solver.eigenvectors() *
                    inverse_eigenvalues *
                    extrinsic_solver.eigenvectors().transpose();
            }

            lambda_pose -=
                lambda_pose_extrinsic *
                lambda_extrinsic_pinv *
                lambda_pose_extrinsic.transpose();

            lambda_pose =
                0.5 *
                (lambda_pose + lambda_pose.transpose());
        }

        // ---------------------------------------------------------------
        // Convert the possibly rank-deficient pose information into a finite
        // covariance. Unobservable modes get a large but finite variance.
        // ---------------------------------------------------------------
        Eigen::SelfAdjointEigenSolver<Matrix6d> pose_solver(
            lambda_pose);

        if (pose_solver.info() != Eigen::Success ||
            !pose_solver.eigenvalues().allFinite() ||
            !pose_solver.eigenvectors().allFinite())
        {
            return false;
        }

        const Vector6d pose_eigenvalues =
            pose_solver.eigenvalues();

        const double maximum_pose_eigenvalue =
            std::max(0.0, pose_eigenvalues.maxCoeff());

        if (!std::isfinite(maximum_pose_eigenvalue) ||
            maximum_pose_eigenvalue <= 1.0e-12)
        {
            return false;
        }

        const double information_floor =
            std::max(
                1.0e-9,
                maximum_pose_eigenvalue * 1.0e-6);

        Matrix6d inverse_pose_eigenvalues = Matrix6d::Zero();

        for (int i = 0; i < 6; ++i)
        {
            const double clipped_information =
                std::max(
                    information_floor,
                    pose_eigenvalues(i));

            inverse_pose_eigenvalues(i, i) =
                1.0 / clipped_information;
        }

        Matrix6d covariance_imu_rp =
            pose_solver.eigenvectors() *
            inverse_pose_eigenvalues *
            pose_solver.eigenvectors().transpose();

        covariance_imu_rp =
            0.5 *
            (covariance_imu_rp + covariance_imu_rp.transpose());

        if (!covariance_imu_rp.allFinite())
        {
            return false;
        }

        // ---------------------------------------------------------------
        // IESKF tangent:
        //   x = [delta_theta_I(right), delta_p_W]
        //
        // Local LiDAR tangent:
        //   y = [delta_theta_L, delta_t_L]
        //
        //   delta_theta_L = R_LI * delta_theta_I
        //   delta_t_L     = R_LW * delta_p_W
        //                     - R_LI * [p_IL]x * delta_theta_I
        // ---------------------------------------------------------------
        const Eigen::Matrix3d R_WI =
            state.Q_WI.normalized().toRotationMatrix();

        const Eigen::Matrix3d R_IL =
            state.Q_IL.normalized().toRotationMatrix();

        const Eigen::Matrix3d R_LI =
            R_IL.transpose();

        const Eigen::Matrix3d R_WL =
            R_WI * R_IL;

        const Eigen::Matrix3d R_LW =
            R_WL.transpose();

        Matrix6d imu_pose_to_lidar_local = Matrix6d::Zero();

        imu_pose_to_lidar_local.block<3, 3>(0, 0) =
            R_LI;

        imu_pose_to_lidar_local.block<3, 3>(3, 0) =
            -R_LI * SkewSymmetric(state.P_IL);

        imu_pose_to_lidar_local.block<3, 3>(3, 3) =
            R_LW;

        Matrix6d covariance_lidar_rt =
            imu_pose_to_lidar_local *
            covariance_imu_rp *
            imu_pose_to_lidar_local.transpose();

        covariance_lidar_rt =
            0.5 *
            (covariance_lidar_rt + covariance_lidar_rt.transpose());

        if (!covariance_lidar_rt.allFinite())
        {
            return false;
        }

        // [rx ry rz tx ty tz] -> g2o [tx ty tz rx ry rz].
        Matrix6d rt_to_tr = Matrix6d::Zero();
        rt_to_tr.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
        rt_to_tr.block<3, 3>(3, 0) = Eigen::Matrix3d::Identity();

        Matrix6d covariance_tr =
            rt_to_tr *
            covariance_lidar_rt *
            rt_to_tr.transpose();

        covariance_tr =
            0.5 *
            (covariance_tr + covariance_tr.transpose());

        if (!covariance_tr.allFinite())
        {
            return false;
        }

        // ---------------------------------------------------------------
        // V2 directional confidence.
        //
        // IMPORTANT: meters and radians are different units, therefore the
        // translation group and rotation group are normalized independently.
        // We normalize FIRST and clamp SECOND.
        // ---------------------------------------------------------------
        constexpr double minimum_directional_confidence = 0.01;

        Vector6d raw_directional_information = Vector6d::Zero();
        Vector6d confidence_tr = Vector6d::Ones();

        for (int i = 0; i < 6; ++i)
        {
            const double variance = covariance_tr(i, i);

            if (!std::isfinite(variance) || variance <= 0.0)
            {
                return false;
            }

            raw_directional_information(i) =
                1.0 / variance;

            if (!std::isfinite(raw_directional_information(i)) ||
                raw_directional_information(i) <= 0.0)
            {
                return false;
            }
        }

        for (int group_start : {0, 3})
        {
            double group_maximum = 0.0;

            for (int i = group_start;
                 i < group_start + 3;
                 ++i)
            {
                group_maximum =
                    std::max(
                        group_maximum,
                        raw_directional_information(i));
            }

            if (!std::isfinite(group_maximum) ||
                group_maximum <= 0.0)
            {
                return false;
            }

            for (int i = group_start;
                 i < group_start + 3;
                 ++i)
            {
                confidence_tr(i) =
                    std::clamp(
                        raw_directional_information(i) /
                            group_maximum,
                        minimum_directional_confidence,
                        1.0);
            }
        }

        // ---------------------------------------------------------------
        // Recover the full precision shape Q = C^-1.
        // ---------------------------------------------------------------
        Eigen::SelfAdjointEigenSolver<Matrix6d> covariance_solver(
            covariance_tr);

        if (covariance_solver.info() != Eigen::Success ||
            !covariance_solver.eigenvalues().allFinite() ||
            !covariance_solver.eigenvectors().allFinite() ||
            covariance_solver.eigenvalues().minCoeff() <= 1.0e-15)
        {
            return false;
        }

        Matrix6d inverse_covariance_eigenvalues = Matrix6d::Zero();

        for (int i = 0; i < 6; ++i)
        {
            inverse_covariance_eigenvalues(i, i) =
                1.0 / covariance_solver.eigenvalues()(i);
        }

        Matrix6d precision_shape =
            covariance_solver.eigenvectors() *
            inverse_covariance_eigenvalues *
            covariance_solver.eigenvectors().transpose();

        precision_shape =
            0.5 *
            (precision_shape + precision_shape.transpose());

        if (!precision_shape.allFinite())
        {
            return false;
        }

        // ---------------------------------------------------------------
        // Standardize precision to unit diagonal:
        //
        //     J_ij = Q_ij / sqrt(Q_ii * Q_jj)
        //
        // This is a diagonal congruence transform and preserves SPD.
        // ---------------------------------------------------------------
        Matrix6d standardized_precision = Matrix6d::Zero();

        for (int i = 0; i < 6; ++i)
        {
            if (!std::isfinite(precision_shape(i, i)) ||
                precision_shape(i, i) <= 0.0)
            {
                return false;
            }

            for (int j = 0; j < 6; ++j)
            {
                if (!std::isfinite(precision_shape(j, j)) ||
                    precision_shape(j, j) <= 0.0)
                {
                    return false;
                }

                const double denominator =
                    std::sqrt(
                        precision_shape(i, i) *
                        precision_shape(j, j));

                if (!std::isfinite(denominator) ||
                    denominator <= 0.0)
                {
                    return false;
                }

                standardized_precision(i, j) =
                    precision_shape(i, j) / denominator;
            }
        }

        standardized_precision =
            0.5 *
            (standardized_precision + standardized_precision.transpose());

        if (!standardized_precision.allFinite())
        {
            return false;
        }

        // ---------------------------------------------------------------
        // V2 coupling regularization.
        //
        // Keep 75% of the measured correlation shape and inject 25% Identity.
        // Because standardized_precision is SPD with unit diagonal, this keeps
        // the diagonal exactly one and gives a strict SPD eigenvalue margin.
        // ---------------------------------------------------------------
        constexpr double coupling_retention = 0.75;

        Matrix6d regularized_precision_shape =
            coupling_retention * standardized_precision +
            (1.0 - coupling_retention) * Matrix6d::Identity();

        regularized_precision_shape =
            0.5 *
            (regularized_precision_shape +
             regularized_precision_shape.transpose());

        Eigen::SelfAdjointEigenSolver<Matrix6d> shape_solver(
            regularized_precision_shape,
            Eigen::EigenvaluesOnly);

        if (shape_solver.info() != Eigen::Success ||
            !shape_solver.eigenvalues().allFinite() ||
            shape_solver.eigenvalues().minCoeff() <= 1.0e-6)
        {
            return false;
        }

        // ---------------------------------------------------------------
        // Restore the V2 directional confidence by congruence scaling:
        //
        //     Omega = W * J_safe * W
        //     W = diag(sqrt(confidence_tr))
        //
        // Since diag(J_safe) == 1, diag(Omega) == confidence_tr exactly.
        // ---------------------------------------------------------------
        Matrix6d confidence_scale = Matrix6d::Zero();

        for (int i = 0; i < 6; ++i)
        {
            confidence_scale(i, i) =
                std::sqrt(confidence_tr(i));
        }

        pose_graph_information =
            confidence_scale *
            regularized_precision_shape *
            confidence_scale;

        pose_graph_information =
            0.5 *
            (pose_graph_information + pose_graph_information.transpose());

        if (!pose_graph_information.allFinite())
        {
            return false;
        }

        // ---------------------------------------------------------------
        // Final strict SPD + conditioning guard. Failure here only causes the
        // mapping bridge to use the legacy identity fallback for this keyframe;
        // it never rejects the valid LIO frame itself.
        // ---------------------------------------------------------------
        Eigen::SelfAdjointEigenSolver<Matrix6d> final_solver(
            pose_graph_information,
            Eigen::EigenvaluesOnly);

        if (final_solver.info() != Eigen::Success ||
            !final_solver.eigenvalues().allFinite())
        {
            return false;
        }

        const double minimum_eigenvalue =
            final_solver.eigenvalues().minCoeff();

        const double maximum_eigenvalue =
            final_solver.eigenvalues().maxCoeff();

        if (!std::isfinite(minimum_eigenvalue) ||
            !std::isfinite(maximum_eigenvalue) ||
            minimum_eigenvalue <= 1.0e-8 ||
            maximum_eigenvalue <= minimum_eigenvalue)
        {
            return false;
        }

        const double condition_number =
            maximum_eigenvalue / minimum_eigenvalue;

        constexpr double maximum_condition_number = 2500.0;

        if (!std::isfinite(condition_number) ||
            condition_number > maximum_condition_number)
        {
            return false;
        }

        return true;
    }

    bool InitialUncertaintyIsValid(
        const IeskfInitialUncertainty &uncertainty)
    {
        return std::isfinite(uncertainty.rotation_std) &&
               uncertainty.rotation_std > 0.0 &&

               std::isfinite(uncertainty.position_std) &&
               uncertainty.position_std > 0.0 &&

               std::isfinite(uncertainty.velocity_std) &&
               uncertainty.velocity_std > 0.0 &&

               std::isfinite(uncertainty.gyro_bias_std) &&
               uncertainty.gyro_bias_std > 0.0 &&

               std::isfinite(uncertainty.accel_bias_std) &&
               uncertainty.accel_bias_std > 0.0 &&

               std::isfinite(uncertainty.gravity_std) &&
               uncertainty.gravity_std > 0.0 &&

               std::isfinite(uncertainty.extrinsic_rotation_std) &&
               uncertainty.extrinsic_rotation_std > 0.0 &&

               std::isfinite(uncertainty.extrinsic_position_std) &&
               uncertainty.extrinsic_position_std > 0.0;
    }

    bool ProcessNoiseIsValid(
        const IeskfConfig &config)
    {
        return std::isfinite(config.gyro_noise_std) &&
               config.gyro_noise_std >= 0.0 &&

               std::isfinite(config.accel_noise_std) &&
               config.accel_noise_std >= 0.0 &&

               std::isfinite(config.gyro_bias_random_walk_std) &&
               config.gyro_bias_random_walk_std >= 0.0 &&

               std::isfinite(config.accel_bias_random_walk_std) &&
               config.accel_bias_random_walk_std >= 0.0;
    }

    bool ImuDataIsFinite(
        const IMU_DATA &imu)
    {
        return std::isfinite(imu.timestamp) &&
               imu.gyro.allFinite() &&
               imu.accelerometer.allFinite();
    }

    bool LidarIterationConfigIsValid(
        const IeskfConfig &config)
    {
        return config.max_lidar_iterations > 0 &&

               std::isfinite(
                   config.lidar_convergence_rotation_rad) &&
               config.lidar_convergence_rotation_rad > 0.0 &&

               std::isfinite(
                   config.lidar_convergence_position_m) &&
               config.lidar_convergence_position_m > 0.0;
    }

} // namespace

Ieskf::Ieskf()
{
}

Ieskf::Ieskf(
    const IeskfConfig &config)
    : config_(config)
{
}

bool Ieskf::SetConfig(
    const IeskfConfig &config)
{
    if (!ConfigIsValid(config))
    {
        return false;
    }

    config_ =
        config;

    return true;
}

const IeskfConfig &Ieskf::Config() const
{
    return config_;
}

bool Ieskf::Initialize(
    const LioState &initial_state)
{
    if (!StateIsFinite(initial_state))
    {
        return false;
    }

    if (!InitialUncertaintyIsValid(
            config_.initial_uncertainty))
    {
        return false;
    }

    LioState normalized_state =
        initial_state;

    normalized_state.Q_WI =
        NormalizeQuaternion(
            normalized_state.Q_WI);

    normalized_state.Q_IL =
        NormalizeQuaternion(
            normalized_state.Q_IL);

    const double gravity_magnitude =
        normalized_state.gravity_W.norm();

    if (!std::isfinite(gravity_magnitude) ||
        gravity_magnitude < 1.0e-12)
    {
        return false;
    }

    StateMatrix initial_covariance =
        BuildInitialCovariance();

    initial_covariance =
        0.5 *
        (initial_covariance +
         initial_covariance.transpose());

    if (!CovarianceIsFinite(initial_covariance))
    {
        return false;
    }

    state_ =
        normalized_state;

    covariance_ =
        initial_covariance;

    initialized_ =
        true;

    return true;
}

bool Ieskf::IsInitialized() const
{
    return initialized_;
}

const LioState &Ieskf::State() const
{
    return state_;
}

const Ieskf::StateMatrix &
Ieskf::Covariance() const
{
    return covariance_;
}

Ieskf::StateMatrix
Ieskf::BuildInitialCovariance() const
{
    StateMatrix covariance =
        StateMatrix::Zero();

    const IeskfInitialUncertainty &uncertainty =
        config_.initial_uncertainty;

    covariance.block<3, 3>(
        LioStateIndex::ROTATION,
        LioStateIndex::ROTATION) =
        Eigen::Matrix3d::Identity() *
        uncertainty.rotation_std *
        uncertainty.rotation_std;

    covariance.block<3, 3>(
        LioStateIndex::POSITION,
        LioStateIndex::POSITION) =
        Eigen::Matrix3d::Identity() *
        uncertainty.position_std *
        uncertainty.position_std;

    covariance.block<3, 3>(
        LioStateIndex::VELOCITY,
        LioStateIndex::VELOCITY) =
        Eigen::Matrix3d::Identity() *
        uncertainty.velocity_std *
        uncertainty.velocity_std;

    covariance.block<3, 3>(
        LioStateIndex::GYRO_BIAS,
        LioStateIndex::GYRO_BIAS) =
        Eigen::Matrix3d::Identity() *
        uncertainty.gyro_bias_std *
        uncertainty.gyro_bias_std;

    covariance.block<3, 3>(
        LioStateIndex::ACCEL_BIAS,
        LioStateIndex::ACCEL_BIAS) =
        Eigen::Matrix3d::Identity() *
        uncertainty.accel_bias_std *
        uncertainty.accel_bias_std;

    covariance.block<2, 2>(
        LioStateIndex::GRAVITY,
        LioStateIndex::GRAVITY) =
        Eigen::Matrix2d::Identity() *
        uncertainty.gravity_std *
        uncertainty.gravity_std;

    covariance.block<3, 3>(
        LioStateIndex::EXTRINSIC_ROTATION,
        LioStateIndex::EXTRINSIC_ROTATION) =
        Eigen::Matrix3d::Identity() *
        uncertainty.extrinsic_rotation_std *
        uncertainty.extrinsic_rotation_std;

    covariance.block<3, 3>(
        LioStateIndex::EXTRINSIC_POSITION,
        LioStateIndex::EXTRINSIC_POSITION) =
        Eigen::Matrix3d::Identity() *
        uncertainty.extrinsic_position_std *
        uncertainty.extrinsic_position_std;

    return covariance;
}

void Ieskf::BoxPlus(
    LioState &state,
    const StateVector &delta_x)
{
    const Eigen::Vector3d delta_theta_WI =
        delta_x.segment<3>(
            LioStateIndex::ROTATION);

    const Eigen::Matrix3d R_WI_new =
        state.Q_WI.normalized().toRotationMatrix() *
        ExpSO3(delta_theta_WI);

    state.Q_WI =
        NormalizeQuaternion(
            Eigen::Quaterniond(
                R_WI_new));

    state.P_WI +=
        delta_x.segment<3>(
            LioStateIndex::POSITION);

    state.V_WI +=
        delta_x.segment<3>(
            LioStateIndex::VELOCITY);

    state.gyro_bias +=
        delta_x.segment<3>(
            LioStateIndex::GYRO_BIAS);

    state.accel_bias +=
        delta_x.segment<3>(
            LioStateIndex::ACCEL_BIAS);

    state.gravity_W =
        GravityBoxPlus(
            state.gravity_W,
            delta_x.segment<2>(
                LioStateIndex::GRAVITY));

    const Eigen::Vector3d delta_theta_IL =
        delta_x.segment<3>(
            LioStateIndex::EXTRINSIC_ROTATION);

    const Eigen::Matrix3d R_IL_new =
        state.Q_IL.normalized().toRotationMatrix() *
        ExpSO3(delta_theta_IL);

    state.Q_IL =
        NormalizeQuaternion(
            Eigen::Quaterniond(
                R_IL_new));

    state.P_IL +=
        delta_x.segment<3>(
            LioStateIndex::EXTRINSIC_POSITION);
}

Ieskf::StateVector
Ieskf::BoxMinus(
    const LioState &target,
    const LioState &reference)
{
    StateVector delta_x =
        StateVector::Zero();

    const Eigen::Matrix3d R_WI_reference =
        reference.Q_WI.normalized().toRotationMatrix();

    const Eigen::Matrix3d R_WI_target =
        target.Q_WI.normalized().toRotationMatrix();

    delta_x.segment<3>(
        LioStateIndex::ROTATION) =
        LogSO3(
            R_WI_reference.transpose() *
            R_WI_target);

    delta_x.segment<3>(
        LioStateIndex::POSITION) =
        target.P_WI -
        reference.P_WI;

    delta_x.segment<3>(
        LioStateIndex::VELOCITY) =
        target.V_WI -
        reference.V_WI;

    delta_x.segment<3>(
        LioStateIndex::GYRO_BIAS) =
        target.gyro_bias -
        reference.gyro_bias;

    delta_x.segment<3>(
        LioStateIndex::ACCEL_BIAS) =
        target.accel_bias -
        reference.accel_bias;

    delta_x.segment<2>(
        LioStateIndex::GRAVITY) =
        GravityBoxMinus(
            target.gravity_W,
            reference.gravity_W);

    const Eigen::Matrix3d R_IL_reference =
        reference.Q_IL.normalized().toRotationMatrix();

    const Eigen::Matrix3d R_IL_target =
        target.Q_IL.normalized().toRotationMatrix();

    delta_x.segment<3>(
        LioStateIndex::EXTRINSIC_ROTATION) =
        LogSO3(
            R_IL_reference.transpose() *
            R_IL_target);

    delta_x.segment<3>(
        LioStateIndex::EXTRINSIC_POSITION) =
        target.P_IL -
        reference.P_IL;

    return delta_x;
}

Ieskf::StateMatrix
Ieskf::BuildErrorStateJacobian(
    const LioState &linearization_state,
    const Eigen::Vector3d &omega_unbiased,
    const Eigen::Vector3d &accel_unbiased) const
{
    StateMatrix F =
        StateMatrix::Zero();

    const Eigen::Matrix3d R_WI =
        linearization_state.Q_WI
            .normalized()
            .toRotationMatrix();

    F.block<3, 3>(
        LioStateIndex::ROTATION,
        LioStateIndex::ROTATION) =
        -Skew(
            omega_unbiased);

    F.block<3, 3>(
        LioStateIndex::ROTATION,
        LioStateIndex::GYRO_BIAS) =
        -Eigen::Matrix3d::Identity();

    F.block<3, 3>(
        LioStateIndex::POSITION,
        LioStateIndex::VELOCITY) =
        Eigen::Matrix3d::Identity();

    F.block<3, 3>(
        LioStateIndex::VELOCITY,
        LioStateIndex::ROTATION) =
        -R_WI *
        Skew(
            accel_unbiased);

    F.block<3, 3>(
        LioStateIndex::VELOCITY,
        LioStateIndex::ACCEL_BIAS) =
        -R_WI;

    F.block<3, 2>(
        LioStateIndex::VELOCITY,
        LioStateIndex::GRAVITY) =
        GravityErrorJacobian(
            linearization_state.gravity_W);

    return F;
}

Ieskf::NoiseJacobian
Ieskf::BuildNoiseJacobian(
    const LioState &linearization_state) const
{
    NoiseJacobian G =
        NoiseJacobian::Zero();

    const Eigen::Matrix3d R_WI =
        linearization_state.Q_WI
            .normalized()
            .toRotationMatrix();

    constexpr int GYRO_NOISE = 0;
    constexpr int ACCEL_NOISE = 3;
    constexpr int GYRO_BIAS_RANDOM_WALK = 6;
    constexpr int ACCEL_BIAS_RANDOM_WALK = 9;

    G.block<3, 3>(
        LioStateIndex::ROTATION,
        GYRO_NOISE) =
        -Eigen::Matrix3d::Identity();

    G.block<3, 3>(
        LioStateIndex::VELOCITY,
        ACCEL_NOISE) =
        -R_WI;

    G.block<3, 3>(
        LioStateIndex::GYRO_BIAS,
        GYRO_BIAS_RANDOM_WALK) =
        Eigen::Matrix3d::Identity();

    G.block<3, 3>(
        LioStateIndex::ACCEL_BIAS,
        ACCEL_BIAS_RANDOM_WALK) =
        Eigen::Matrix3d::Identity();

    return G;
}

Ieskf::NoiseMatrix
Ieskf::BuildContinuousNoiseCovariance() const
{
    NoiseMatrix Q_c =
        NoiseMatrix::Zero();

    constexpr int GYRO_NOISE = 0;
    constexpr int ACCEL_NOISE = 3;
    constexpr int GYRO_BIAS_RANDOM_WALK = 6;
    constexpr int ACCEL_BIAS_RANDOM_WALK = 9;

    const double gyro_noise_variance =
        config_.gyro_noise_std *
        config_.gyro_noise_std;

    const double accel_noise_variance =
        config_.accel_noise_std *
        config_.accel_noise_std;

    const double gyro_bias_random_walk_variance =
        config_.gyro_bias_random_walk_std *
        config_.gyro_bias_random_walk_std;

    const double accel_bias_random_walk_variance =
        config_.accel_bias_random_walk_std *
        config_.accel_bias_random_walk_std;

    Q_c.block<3, 3>(
        GYRO_NOISE,
        GYRO_NOISE) =
        Eigen::Matrix3d::Identity() *
        gyro_noise_variance;

    Q_c.block<3, 3>(
        ACCEL_NOISE,
        ACCEL_NOISE) =
        Eigen::Matrix3d::Identity() *
        accel_noise_variance;

    Q_c.block<3, 3>(
        GYRO_BIAS_RANDOM_WALK,
        GYRO_BIAS_RANDOM_WALK) =
        Eigen::Matrix3d::Identity() *
        gyro_bias_random_walk_variance;

    Q_c.block<3, 3>(
        ACCEL_BIAS_RANDOM_WALK,
        ACCEL_BIAS_RANDOM_WALK) =
        Eigen::Matrix3d::Identity() *
        accel_bias_random_walk_variance;

    return Q_c;
}

bool Ieskf::Propagate(
    const IMU_DATA &imu_k,
    const IMU_DATA &imu_next)
{
    if (!initialized_ ||
        !ProcessNoiseIsValid(config_) ||
        !ImuDataIsFinite(imu_k) ||
        !ImuDataIsFinite(imu_next))
    {
        return false;
    }

    const double dt =
        imu_next.timestamp -
        imu_k.timestamp;

    if (!std::isfinite(dt) ||
        dt <= 0.0 ||
        dt > config_.max_imu_dt)
    {
        return false;
    }

    const LioState state_before =
        state_;

    const StateMatrix covariance_before =
        covariance_;

    if (!StateIsFinite(state_before) ||
        !CovarianceIsFinite(covariance_before))
    {
        return false;
    }

    const Eigen::Vector3d omega_k =
        imu_k.gyro -
        state_before.gyro_bias;

    const Eigen::Vector3d omega_next =
        imu_next.gyro -
        state_before.gyro_bias;

    const Eigen::Vector3d omega_mid =
        0.5 *
        (omega_k +
         omega_next);

    const Eigen::Vector3d accel_k_I =
        imu_k.accelerometer -
        state_before.accel_bias;

    const Eigen::Vector3d accel_next_I =
        imu_next.accelerometer -
        state_before.accel_bias;

    const Eigen::Vector3d accel_mid_I =
        0.5 *
        (accel_k_I +
         accel_next_I);

    if (!omega_mid.allFinite() ||
        !accel_mid_I.allFinite())
    {
        return false;
    }

    const Eigen::Matrix3d R_WI_k =
        state_before.Q_WI
            .normalized()
            .toRotationMatrix();

    const Eigen::Matrix3d R_WI_next =
        R_WI_k *
        ExpSO3(
            omega_mid *
            dt);

    const Eigen::Vector3d accel_k_W =
        R_WI_k *
            accel_k_I +
        state_before.gravity_W;

    const Eigen::Vector3d accel_next_W =
        R_WI_next *
            accel_next_I +
        state_before.gravity_W;

    const Eigen::Vector3d accel_mid_W =
        0.5 *
        (accel_k_W +
         accel_next_W);

    LioState next_state =
        state_before;

    next_state.timestamp =
        imu_next.timestamp;

    next_state.Q_WI =
        NormalizeQuaternion(
            Eigen::Quaterniond(
                R_WI_next));

    next_state.P_WI =
        state_before.P_WI +
        state_before.V_WI * dt +
        0.5 * accel_mid_W * dt * dt;

    next_state.V_WI =
        state_before.V_WI +
        accel_mid_W * dt;

    if (!StateIsFinite(next_state))
    {
        return false;
    }

    LioState linearization_state =
        state_before;

    linearization_state.Q_WI =
        NormalizeQuaternion(
            Eigen::Quaterniond(
                R_WI_k *
                ExpSO3(
                    0.5 *
                    omega_mid *
                    dt)));

    const StateMatrix F =
        BuildErrorStateJacobian(
            linearization_state,
            omega_mid,
            accel_mid_I);

    const NoiseJacobian G =
        BuildNoiseJacobian(
            linearization_state);

    const NoiseMatrix Q_c =
        BuildContinuousNoiseCovariance();

    if (!F.allFinite() ||
        !G.allFinite() ||
        !Q_c.allFinite())
    {
        return false;
    }

    const StateMatrix F2 =
        F *
        F;

    const StateMatrix F3 =
        F2 *
        F;

    const double dt2 =
        dt *
        dt;

    const double dt3 =
        dt2 *
        dt;

    const StateMatrix Phi =
        StateMatrix::Identity() +
        F * dt +
        0.5 * F2 * dt2 +
        (1.0 / 6.0) * F3 * dt3;

    const StateMatrix L =
        G *
        Q_c *
        G.transpose();

    StateMatrix Q_d =
        L * dt +
        0.5 *
            (F * L +
             L * F.transpose()) *
            dt2 +
        (1.0 / 6.0) *
            (F2 * L +
             2.0 * F * L * F.transpose() +
             L * F2.transpose()) *
            dt3;

    Q_d =
        0.5 *
        (Q_d +
         Q_d.transpose());

    StateMatrix next_covariance =
        Phi *
            covariance_before *
            Phi.transpose() +
        Q_d;

    next_covariance =
        0.5 *
        (next_covariance +
         next_covariance.transpose());

    if (!CovarianceIsFinite(next_covariance) ||
        (next_covariance.diagonal().array() <
         -1.0e-12)
            .any())
    {
        return false;
    }

    state_ =
        next_state;

    covariance_ =
        next_covariance;

    return true;
}

Eigen::Matrix2d
Ieskf::BuildGravityTransportJacobian(
    const Eigen::Vector3d &current_gravity,
    const Eigen::Vector3d &prior_gravity) const
{
    Eigen::Matrix2d jacobian =
        Eigen::Matrix2d::Zero();

    for (int column = 0;
         column < 2;
         ++column)
    {
        Eigen::Vector2d positive_increment =
            Eigen::Vector2d::Zero();

        Eigen::Vector2d negative_increment =
            Eigen::Vector2d::Zero();

        positive_increment(column) =
            GRAVITY_TRANSPORT_EPSILON;

        negative_increment(column) =
            -GRAVITY_TRANSPORT_EPSILON;

        const Eigen::Vector2d delta_positive =
            GravityBoxMinus(
                GravityBoxPlus(
                    current_gravity,
                    positive_increment),
                prior_gravity);

        const Eigen::Vector2d delta_negative =
            GravityBoxMinus(
                GravityBoxPlus(
                    current_gravity,
                    negative_increment),
                prior_gravity);

        jacobian.col(column) =
            (delta_positive -
             delta_negative) /
            (2.0 *
             GRAVITY_TRANSPORT_EPSILON);
    }

    return jacobian;
}

Ieskf::StateMatrix
Ieskf::BuildPriorTransportJacobian(
    const LioState &current_state,
    const LioState &prior_state) const
{
    StateMatrix jacobian =
        StateMatrix::Identity();

    const StateVector prior_delta =
        BoxMinus(
            current_state,
            prior_state);

    jacobian.block<3, 3>(
        LioStateIndex::ROTATION,
        LioStateIndex::ROTATION) =
        RightJacobianInverseSO3(
            prior_delta.segment<3>(
                LioStateIndex::ROTATION));

    jacobian.block<2, 2>(
        LioStateIndex::GRAVITY,
        LioStateIndex::GRAVITY) =
        BuildGravityTransportJacobian(
            current_state.gravity_W,
            prior_state.gravity_W);

    jacobian.block<3, 3>(
        LioStateIndex::EXTRINSIC_ROTATION,
        LioStateIndex::EXTRINSIC_ROTATION) =
        RightJacobianInverseSO3(
            prior_delta.segment<3>(
                LioStateIndex::EXTRINSIC_ROTATION));

    return jacobian;
}

void Ieskf::ApplyEstimationMask(
    StateMatrix &system_matrix,
    StateVector &right_hand_side) const
{
    if (!config_.estimate_gravity)
    {
        for (int index =
                 LioStateIndex::GRAVITY;
             index <
             LioStateIndex::GRAVITY + 2;
             ++index)
        {
            system_matrix.row(index).setZero();
            system_matrix.col(index).setZero();
            system_matrix(index, index) =
                1.0;

            right_hand_side(index) =
                0.0;
        }
    }

    if (!config_.estimate_extrinsic)
    {
        for (int index =
                 LioStateIndex::EXTRINSIC_ROTATION;
             index <
             LioStateIndex::EXTRINSIC_POSITION + 3;
             ++index)
        {
            system_matrix.row(index).setZero();
            system_matrix.col(index).setZero();
            system_matrix(index, index) =
                1.0;

            right_hand_side(index) =
                0.0;
        }
    }
}

bool Ieskf::IteratedLidarUpdate(
    const pcl::PointCloud<LIDAR_POINT>::ConstPtr &deskewed_scan_L,
    const PreparedLidarTarget &target,
    const LioMeasurementBuilder &measurement_builder,
    IeskfLidarUpdateResult &result,
    const std::function<
        bool(
            const LioState &,
            Ieskf::StateMatrix &,
            Ieskf::StateVector &)> *joint_observation_builder,
    // FR_PERSISTENT_GROUND_OWNERSHIP_V4
    const std::function<
        bool(
            const LioState &,
            Ieskf::StateMatrix &)> *structural_ownership_builder)
{
    result =
        IeskfLidarUpdateResult();

    if (!initialized_ ||
        !LidarIterationConfigIsValid(config_) ||
        !deskewed_scan_L ||
        deskewed_scan_L->empty() ||
        !StateIsFinite(state_) ||
        !CovarianceIsFinite(covariance_))
    {
        return false;
    }

    const LioState prior_state =
        state_;

    const StateMatrix prior_covariance =
        0.5 *
        (covariance_ +
         covariance_.transpose());

    Eigen::LDLT<StateMatrix> prior_ldlt(
        prior_covariance);

    if (prior_ldlt.info() != Eigen::Success ||
        !prior_ldlt.isPositive())
    {
        return false;
    }

    const StateMatrix prior_information =
        prior_ldlt.solve(
            StateMatrix::Identity());

    if (!prior_information.allFinite())
    {
        return false;
    }

    LioState current_state =
        prior_state;

    // FR_B1_ADDITIVE_GROUND_ABLATION_V1
    // Keep full Dense LiDAR information and ADD structural measurements.
    // Ground / Wall state machines and residuals are unchanged.
    constexpr bool kEnableStructuralSubspaceOwnershipProjection = false;

    for (int iteration = 0;
         iteration <
         config_.max_lidar_iterations;
         ++iteration)
    {
        LioMeasurementResult measurement;

        if (!measurement_builder.Build(
                current_state,
                deskewed_scan_L,
                target,
                measurement))
        {
            return false;
        }

        // ================================================================
        // JOINT_STRUCTURAL_ITERATION
        //
        // Add frozen structural observations at the SAME IEKF
        // linearization state as the general LiDAR measurement.
        // ================================================================
        if (joint_observation_builder != nullptr)
        {
            StateMatrix joint_information =
                StateMatrix::Zero();

            StateVector joint_gradient =
                StateVector::Zero();

            if (!(*joint_observation_builder)(
                    current_state,
                    joint_information,
                    joint_gradient))
            {
                return false;
            }

            joint_information =
                0.5 *
                (joint_information +
                 joint_information.transpose());

            if (!joint_information.allFinite() ||
                !joint_gradient.allFinite())
            {
                return false;
            }

            // ============================================================
            // FR_PERSISTENT_GROUND_OWNERSHIP_V4
            //
            // Measurement validity != ownership validity.
            //
            // joint_information:
            //     current valid Ground / Wall measurement.
            //
            // persistent_ownership_information:
            //     ownership only; NEVER added as a measurement.
            // ============================================================
            StateMatrix structural_ownership_information =
                joint_information;

            if (structural_ownership_builder != nullptr)
            {
                StateMatrix persistent_ownership_information =
                    StateMatrix::Zero();

                if (!(*structural_ownership_builder)(
                        current_state,
                        persistent_ownership_information))
                {
                    return false;
                }

                persistent_ownership_information =
                    0.5 *
                    (
                        persistent_ownership_information +
                        persistent_ownership_information.transpose()
                    );

                if (!persistent_ownership_information.allFinite())
                {
                    return false;
                }

                structural_ownership_information +=
                    persistent_ownership_information;
            }

            structural_ownership_information =
                0.5 *
                (
                    structural_ownership_information +
                    structural_ownership_information.transpose()
                );

            // ============================================================
            // FR_STRUCTURAL_SUBSPACE_OWNERSHIP_ITERATION
            //
            // Structural observations own their observable subspace.
            //
            // Dense LiDAR information is projected OUT of that subspace
            // before Ground / Wall information is added.
            //
            // IMPORTANT:
            //   - prior information is NOT projected
            //   - only current LiDAR measurement is projected
            // ============================================================
            {
                const StateMatrix structural_information =
                    structural_ownership_information;

                Eigen::SelfAdjointEigenSolver<StateMatrix>
                    ownership_solver(
                        structural_information);

                if (ownership_solver.info() !=
                    Eigen::Success)
                {
                    return false;
                }

                const auto ownership_values =
                    ownership_solver.eigenvalues();

                const auto ownership_vectors =
                    ownership_solver.eigenvectors();

                if (!ownership_values.allFinite() ||
                    !ownership_vectors.allFinite())
                {
                    return false;
                }

                const double maximum_structural_eigenvalue =
                    ownership_values.maxCoeff();

                if (std::isfinite(
                        maximum_structural_eigenvalue) &&
                    maximum_structural_eigenvalue >
                        1.0e-12)
                {
                    const double ownership_threshold =
                        std::max(
                            1.0e-8,
                            maximum_structural_eigenvalue *
                                1.0e-6);

                    StateMatrix owned_projector =
                        StateMatrix::Zero();

                    std::size_t owned_rank =
                        0;

                    for (int index = 0;
                         index < STATE_DIM;
                         ++index)
                    {
                        if (ownership_values(index) >
                            ownership_threshold)
                        {
                            const StateVector direction =
                                ownership_vectors.col(
                                    index);

                            owned_projector.noalias() +=
                                direction *
                                direction.transpose();

                            ++owned_rank;
                        }
                    }

                    if (kEnableStructuralSubspaceOwnershipProjection && owned_rank > 0)
                    {
                        const StateMatrix keep_projector =
                            StateMatrix::Identity() -
                            owned_projector;

                        measurement.information =
                            keep_projector.transpose() *
                            measurement.information *
                            keep_projector;

                        measurement.gradient =
                            keep_projector.transpose() *
                            measurement.gradient;

                        measurement.information =
                            0.5 *
                            (
                                measurement.information +
                                measurement.information.transpose()
                            );

                        if (!measurement.information.allFinite() ||
                            !measurement.gradient.allFinite())
                        {
                            return false;
                        }

                        static std::size_t
                            structural_ownership_iteration_counter =
                                0;

                        ++structural_ownership_iteration_counter;

                        if ((structural_ownership_iteration_counter %
                             100U) == 0U)
                        {
                            std::cout
                                << "LIO_STRUCTURAL_OWNERSHIP"
                                << " | stage=ITERATION"
                                << " | rank="
                                << owned_rank
                                << " | max_eig="
                                << maximum_structural_eigenvalue
                                << std::endl;
                        }
                    }
                }
            }

            measurement.information +=
                joint_information;

            measurement.gradient +=
                joint_gradient;
        }

        if (iteration == 0)
        {
            result.initial_rmse =
                measurement.rmse;

            result.initial_robust_rmse =
                measurement.robust_rmse;
        }

        const StateVector prior_delta =
            BoxMinus(
                current_state,
                prior_state);

        const StateMatrix prior_transport =
            BuildPriorTransportJacobian(
                current_state,
                prior_state);

        if (!prior_delta.allFinite() ||
            !prior_transport.allFinite())
        {
            return false;
        }

        StateMatrix system_matrix =
            prior_transport.transpose() *
                prior_information *
                prior_transport +
            measurement.information;

        StateVector right_hand_side =
            prior_transport.transpose() *
                prior_information *
                prior_delta +
            measurement.gradient;

        system_matrix =
            0.5 *
            (system_matrix +
             system_matrix.transpose());

        if (!system_matrix.allFinite() ||
            !right_hand_side.allFinite())
        {
            return false;
        }

        ApplyEstimationMask(
            system_matrix,
            right_hand_side);

        Eigen::LDLT<StateMatrix> system_ldlt(
            system_matrix);

        if (system_ldlt.info() != Eigen::Success ||
            !system_ldlt.isPositive())
        {
            return false;
        }

        const StateVector increment =
            -system_ldlt.solve(
                right_hand_side);

        if (!increment.allFinite())
        {
            return false;
        }

        // ================================================================
        // LiDAR observability / state-coupling diagnostics.
        //
        // iteration == 0:
        //
        //     current_state == prior_state
        //     prior_delta   == 0
        //
        // Therefore the first increment is driven by the LiDAR
        // measurement through the prior covariance coupling.
        //
        // Diagnostic only. No estimator behavior is changed.
        // ================================================================
        if (iteration == 0)
        {
            constexpr int kRotationX =
                LioStateIndex::ROTATION;

            constexpr int kPositionX =
                LioStateIndex::POSITION;

            constexpr int kVelocityZ =
                LioStateIndex::VELOCITY + 2;

            constexpr int kExtrinsicRotationX =
                LioStateIndex::EXTRINSIC_ROTATION;

            constexpr int kExtrinsicPositionX =
                LioStateIndex::EXTRINSIC_POSITION;

            // ------------------------------------------------------------
            // 1. LiDAR translational information.
            // ------------------------------------------------------------
            const Eigen::Matrix3d translation_information =
                measurement.information.block<3, 3>(
                    LioStateIndex::POSITION,
                    LioStateIndex::POSITION);

            const Eigen::Vector3d translation_info_diag =
                translation_information.diagonal();

            const double translation_info_trace =
                translation_info_diag.sum();

            const double translation_z_fraction =
                translation_info_trace > 1.0e-12
                    ? translation_info_diag.z() /
                          translation_info_trace
                    : 0.0;

            Eigen::Vector3d translation_eigenvalues =
                Eigen::Vector3d::Zero();

            Eigen::Vector3d weakest_translation_direction =
                Eigen::Vector3d::Zero();

            bool translation_eigen_valid =
                false;

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>
                translation_eigen_solver(
                    translation_information);

            if (translation_eigen_solver.info() ==
                    Eigen::Success &&
                translation_eigen_solver
                    .eigenvalues()
                    .allFinite() &&
                translation_eigen_solver
                    .eigenvectors()
                    .allFinite())
            {
                translation_eigenvalues =
                    translation_eigen_solver
                        .eigenvalues();

                weakest_translation_direction =
                    translation_eigen_solver
                        .eigenvectors()
                        .col(0);

                translation_eigen_valid =
                    true;
            }

            const double translation_min_relative =
                translation_eigen_valid &&
                        translation_eigenvalues.z() >
                            1.0e-12
                    ? translation_eigenvalues.x() /
                          translation_eigenvalues.z()
                    : 0.0;

            const double weakest_translation_z =
                translation_eigen_valid
                    ? std::abs(
                          weakest_translation_direction.z())
                    : 0.0;

            // ------------------------------------------------------------

            // ------------------------------------------------------------
            // 1b. Translation information after marginalizing rotation.
            //
            // Pose information:
            //
            //     [ L_rr  L_rp ]
            //     [ L_pr  L_pp ]
            //
            // Marginalized translation information:
            //
            //     L_p|r = L_pp - L_pr * L_rr^+ * L_rp
            //
            // where ^+ is a numerically stable pseudo-inverse.
            // Diagnostic only.
            // ------------------------------------------------------------
            const Eigen::Matrix3d rotation_information =
                measurement.information.block<3, 3>(
                    LioStateIndex::ROTATION,
                    LioStateIndex::ROTATION);

            const Eigen::Matrix3d rotation_translation_information =
                measurement.information.block<3, 3>(
                    LioStateIndex::ROTATION,
                    LioStateIndex::POSITION);

            const Eigen::Matrix3d translation_rotation_information =
                measurement.information.block<3, 3>(
                    LioStateIndex::POSITION,
                    LioStateIndex::ROTATION);

            Eigen::Matrix3d rotation_information_pinv =
                Eigen::Matrix3d::Zero();

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>
                rotation_eigen_solver(
                    rotation_information);

            if (rotation_eigen_solver.info() ==
                    Eigen::Success &&
                rotation_eigen_solver.eigenvalues().allFinite() &&
                rotation_eigen_solver.eigenvectors().allFinite())
            {
                const Eigen::Vector3d rotation_eigenvalues =
                    rotation_eigen_solver.eigenvalues();

                const double rotation_max_eigenvalue =
                    rotation_eigenvalues
                        .cwiseAbs()
                        .maxCoeff();

                if (rotation_max_eigenvalue > 1.0e-12)
                {
                    Eigen::Vector3d inverse_rotation_eigenvalues =
                        Eigen::Vector3d::Zero();

                    constexpr double kRotationEigenThreshold =
                        1.0e-9;

                    for (int eigen_index = 0;
                         eigen_index < 3;
                         ++eigen_index)
                    {
                        if (rotation_eigenvalues(eigen_index) >
                            kRotationEigenThreshold *
                                rotation_max_eigenvalue)
                        {
                            inverse_rotation_eigenvalues(
                                eigen_index) =
                                1.0 /
                                rotation_eigenvalues(
                                    eigen_index);
                        }
                    }

                    rotation_information_pinv =
                        rotation_eigen_solver.eigenvectors() *
                        inverse_rotation_eigenvalues.asDiagonal() *
                        rotation_eigen_solver.eigenvectors().transpose();
                }
            }

            Eigen::Matrix3d schur_translation_information =
                translation_information -
                translation_rotation_information *
                    rotation_information_pinv *
                    rotation_translation_information;

            // Suppress tiny numerical asymmetry.
            schur_translation_information =
                0.5 *
                (schur_translation_information +
                 schur_translation_information.transpose());

            const Eigen::Vector3d schur_translation_diag =
                schur_translation_information.diagonal();

            const double schur_translation_trace =
                schur_translation_diag.sum();

            const double schur_z_fraction =
                schur_translation_trace > 1.0e-12
                    ? schur_translation_diag.z() /
                          schur_translation_trace
                    : 0.0;

            Eigen::Vector3d schur_translation_eigenvalues =
                Eigen::Vector3d::Zero();

            Eigen::Vector3d schur_weakest_translation_direction =
                Eigen::Vector3d::Zero();

            bool schur_eigen_valid =
                false;

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>
                schur_eigen_solver(
                    schur_translation_information);

            if (schur_eigen_solver.info() ==
                    Eigen::Success &&
                schur_eigen_solver.eigenvalues().allFinite() &&
                schur_eigen_solver.eigenvectors().allFinite())
            {
                schur_translation_eigenvalues =
                    schur_eigen_solver.eigenvalues();

                schur_weakest_translation_direction =
                    schur_eigen_solver
                        .eigenvectors()
                        .col(0);

                schur_eigen_valid =
                    true;
            }

            const double schur_translation_min_relative =
                schur_eigen_valid &&
                        std::abs(
                            schur_translation_eigenvalues.z()) >
                            1.0e-12
                    ? schur_translation_eigenvalues.x() /
                          schur_translation_eigenvalues.z()
                    : 0.0;

            const double schur_weakest_translation_z =
                schur_eigen_valid
                    ? std::abs(
                          schur_weakest_translation_direction.z())
                    : 0.0;
            // 2. LiDAR has no direct velocity measurement.
            // ------------------------------------------------------------
            const double measurement_vz_information_norm =
                measurement.information
                    .row(kVelocityZ)
                    .norm();

            const double measurement_vz_gradient =
                measurement.gradient(
                    kVelocityZ);

            // ------------------------------------------------------------
            // 3. Prior covariance correlation between Vz and pose.
            //
            // Rotation components are error-state dtheta components,
            // not Euler angles.
            // ------------------------------------------------------------
            const auto covariance_correlation =
                [&prior_covariance](
                    int first_index,
                    int second_index)
            {
                const double first_variance =
                    prior_covariance(
                        first_index,
                        first_index);

                const double second_variance =
                    prior_covariance(
                        second_index,
                        second_index);

                const double variance_product =
                    first_variance *
                    second_variance;

                if (!std::isfinite(
                        variance_product) ||
                    variance_product <=
                        1.0e-30)
                {
                    return 0.0;
                }

                return prior_covariance(
                           first_index,
                           second_index) /
                       std::sqrt(
                           variance_product);
            };

            const Eigen::Vector3d corr_vz_rotation(
                covariance_correlation(
                    kVelocityZ,
                    kRotationX + 0),
                covariance_correlation(
                    kVelocityZ,
                    kRotationX + 1),
                covariance_correlation(
                    kVelocityZ,
                    kRotationX + 2));

            const Eigen::Vector3d corr_vz_position(
                covariance_correlation(
                    kVelocityZ,
                    kPositionX + 0),
                covariance_correlation(
                    kVelocityZ,
                    kPositionX + 1),
                covariance_correlation(
                    kVelocityZ,
                    kPositionX + 2));

            // ------------------------------------------------------------
            // 4. Decompose the first LiDAR-driven Vz increment.
            //
            //     delta =
            //       -A^-1 (
            //           g_rot +
            //           g_pos +
            //           g_ext_rot +
            //           g_ext_pos)
            // ------------------------------------------------------------
            StateVector rotation_rhs =
                StateVector::Zero();

            StateVector position_rhs =
                StateVector::Zero();

            StateVector extrinsic_rotation_rhs =
                StateVector::Zero();

            StateVector extrinsic_position_rhs =
                StateVector::Zero();

            rotation_rhs.segment<3>(
                LioStateIndex::ROTATION) =
                measurement.gradient.segment<3>(
                    LioStateIndex::ROTATION);

            position_rhs.segment<3>(
                LioStateIndex::POSITION) =
                measurement.gradient.segment<3>(
                    LioStateIndex::POSITION);

            if (config_.estimate_extrinsic)
            {
                extrinsic_rotation_rhs.segment<3>(
                    LioStateIndex::EXTRINSIC_ROTATION) =
                    measurement.gradient.segment<3>(
                        LioStateIndex::EXTRINSIC_ROTATION);

                extrinsic_position_rhs.segment<3>(
                    LioStateIndex::EXTRINSIC_POSITION) =
                    measurement.gradient.segment<3>(
                        LioStateIndex::EXTRINSIC_POSITION);
            }

            const StateVector increment_from_rotation =
                -system_ldlt.solve(
                    rotation_rhs);

            const StateVector increment_from_position =
                -system_ldlt.solve(
                    position_rhs);

            const StateVector
                increment_from_extrinsic_rotation =
                    -system_ldlt.solve(
                        extrinsic_rotation_rhs);

            const StateVector
                increment_from_extrinsic_position =
                    -system_ldlt.solve(
                        extrinsic_position_rhs);

            const double vz_from_rotation =
                increment_from_rotation(
                    kVelocityZ);

            const double vz_from_position =
                increment_from_position(
                    kVelocityZ);

            const double vz_from_extrinsic_rotation =
                increment_from_extrinsic_rotation(
                    kVelocityZ);

            const double vz_from_extrinsic_position =
                increment_from_extrinsic_position(
                    kVelocityZ);

            const double decomposed_vz_increment =
                vz_from_rotation +
                vz_from_position +
                vz_from_extrinsic_rotation +
                vz_from_extrinsic_position;

            const double vz_source_check =
                increment(
                    kVelocityZ) -
                decomposed_vz_increment;

            std::cerr
                << "LIO OBS"
                << " | t="
                << prior_state.timestamp
                << " | corr="
                << measurement.correspondences

                << " | trans_info=["
                << translation_info_diag.x()
                << " "
                << translation_info_diag.y()
                << " "
                << translation_info_diag.z()
                << "]"

                << " | z_frac="
                << translation_z_fraction

                << " | trans_eig=["
                << translation_eigenvalues.x()
                << " "
                << translation_eigenvalues.y()
                << " "
                << translation_eigenvalues.z()
                << "]"

                << " | trans_min_rel="
                << translation_min_relative

                << " | weak_dir=["
                << weakest_translation_direction.x()
                << " "
                << weakest_translation_direction.y()
                << " "
                << weakest_translation_direction.z()
                << "]"

                << " | weak_z="
                << weakest_translation_z

                << " | schur_z_frac="
                << schur_z_fraction

                << " | schur_eig=["
                << schur_translation_eigenvalues.x()
                << " "
                << schur_translation_eigenvalues.y()
                << " "
                << schur_translation_eigenvalues.z()
                << "]"

                << " | schur_min_rel="
                << schur_translation_min_relative

                << " | schur_weak_dir=["
                << schur_weakest_translation_direction.x()
                << " "
                << schur_weakest_translation_direction.y()
                << " "
                << schur_weakest_translation_direction.z()
                << "]"

                << " | schur_weak_z="
                << schur_weakest_translation_z
                << " | meas_vz_info="
                << measurement_vz_information_norm

                << " | meas_vz_grad="
                << measurement_vz_gradient

                << " | corr_vz_rot=["
                << corr_vz_rotation.x()
                << " "
                << corr_vz_rotation.y()
                << " "
                << corr_vz_rotation.z()
                << "]"

                << " | corr_vz_pos=["
                << corr_vz_position.x()
                << " "
                << corr_vz_position.y()
                << " "
                << corr_vz_position.z()
                << "]"

                << " | inc0_vz="
                << increment(
                       kVelocityZ)

                << " | vz_src=[rot="
                << vz_from_rotation
                << " pos="
                << vz_from_position
                << " extR="
                << vz_from_extrinsic_rotation
                << " extP="
                << vz_from_extrinsic_position
                << "]"

                << " | src_check="
                << vz_source_check
                << std::endl;
        }
        const double rotation_increment =
            increment.segment<3>(
                         LioStateIndex::ROTATION)
                .norm();

        const double position_increment =
            increment.segment<3>(
                         LioStateIndex::POSITION)
                .norm();

        const double extrinsic_rotation_increment =
            increment.segment<3>(
                         LioStateIndex::EXTRINSIC_ROTATION)
                .norm();

        const double extrinsic_position_increment =
            increment.segment<3>(
                         LioStateIndex::EXTRINSIC_POSITION)
                .norm();

        result.final_rotation_increment =
            rotation_increment;

        result.final_position_increment =
            position_increment;

        result.final_extrinsic_rotation_increment =
            extrinsic_rotation_increment;

        result.final_extrinsic_position_increment =
            extrinsic_position_increment;

        LioState next_state =
            current_state;

        BoxPlus(
            next_state,
            increment);

        if (!StateIsFinite(next_state))
        {
            return false;
        }

        current_state =
            next_state;

        result.iterations =
            iteration + 1;

        const bool pose_converged =
            rotation_increment <=
                config_.lidar_convergence_rotation_rad &&
            position_increment <=
                config_.lidar_convergence_position_m;

        bool extrinsic_converged =
            true;

        if (config_.estimate_extrinsic)
        {
            extrinsic_converged =
                extrinsic_rotation_increment <=
                    config_.lidar_convergence_rotation_rad &&
                extrinsic_position_increment <=
                    config_.lidar_convergence_position_m;
        }

        if (pose_converged &&
            extrinsic_converged)
        {
            result.converged =
                true;

            break;
        }
    }

    LioMeasurementResult final_measurement;

    if (!measurement_builder.Build(
            current_state,
            deskewed_scan_L,
            target,
            final_measurement))
    {
        return false;
    }

    // =====================================================================
    // JOINT_STRUCTURAL_FINAL
    //
    // The posterior covariance must contain the SAME structural
    // information that generated the converged state.
    // =====================================================================
    if (joint_observation_builder != nullptr)
    {
        StateMatrix final_joint_information =
            StateMatrix::Zero();

        StateVector final_joint_gradient =
            StateVector::Zero();

        if (!(*joint_observation_builder)(
                current_state,
                final_joint_information,
                final_joint_gradient))
        {
            return false;
        }

        final_joint_information =
            0.5 *
            (final_joint_information +
             final_joint_information.transpose());

        if (!final_joint_information.allFinite() ||
            !final_joint_gradient.allFinite())
        {
            return false;
        }

        // ================================================================
        // FR_PERSISTENT_GROUND_OWNERSHIP_V4
        //
        // Posterior covariance must use the SAME persistent ownership
        // subspace used during the iterative solve.
        // ================================================================
        StateMatrix final_structural_ownership_information =
            final_joint_information;

        if (structural_ownership_builder != nullptr)
        {
            StateMatrix persistent_ownership_information =
                StateMatrix::Zero();

            if (!(*structural_ownership_builder)(
                    current_state,
                    persistent_ownership_information))
            {
                return false;
            }

            persistent_ownership_information =
                0.5 *
                (
                    persistent_ownership_information +
                    persistent_ownership_information.transpose()
                );

            if (!persistent_ownership_information.allFinite())
            {
                return false;
            }

            final_structural_ownership_information +=
                persistent_ownership_information;
        }

        final_structural_ownership_information =
            0.5 *
            (
                final_structural_ownership_information +
                final_structural_ownership_information.transpose()
            );

        // ================================================================
        // FR_STRUCTURAL_SUBSPACE_OWNERSHIP_FINAL
        //
        // Posterior covariance must use exactly the same ownership rule
        // that generated the converged state.
        // ================================================================
        {
            const StateMatrix structural_information =
                final_structural_ownership_information;

            Eigen::SelfAdjointEigenSolver<StateMatrix>
                ownership_solver(
                    structural_information);

            if (ownership_solver.info() !=
                Eigen::Success)
            {
                return false;
            }

            const auto ownership_values =
                ownership_solver.eigenvalues();

            const auto ownership_vectors =
                ownership_solver.eigenvectors();

            if (!ownership_values.allFinite() ||
                !ownership_vectors.allFinite())
            {
                return false;
            }

            const double maximum_structural_eigenvalue =
                ownership_values.maxCoeff();

            if (std::isfinite(
                    maximum_structural_eigenvalue) &&
                maximum_structural_eigenvalue >
                    1.0e-12)
            {
                const double ownership_threshold =
                    std::max(
                        1.0e-8,
                        maximum_structural_eigenvalue *
                            1.0e-6);

                StateMatrix owned_projector =
                    StateMatrix::Zero();

                std::size_t owned_rank =
                    0;

                for (int index = 0;
                     index < STATE_DIM;
                     ++index)
                {
                    if (ownership_values(index) >
                        ownership_threshold)
                    {
                        const StateVector direction =
                            ownership_vectors.col(
                                index);

                        owned_projector.noalias() +=
                            direction *
                            direction.transpose();

                        ++owned_rank;
                    }
                }

                if (kEnableStructuralSubspaceOwnershipProjection && owned_rank > 0)
                {
                    const StateMatrix keep_projector =
                        StateMatrix::Identity() -
                        owned_projector;

                    final_measurement.information =
                        keep_projector.transpose() *
                        final_measurement.information *
                        keep_projector;

                    final_measurement.gradient =
                        keep_projector.transpose() *
                        final_measurement.gradient;

                    final_measurement.information =
                        0.5 *
                        (
                            final_measurement.information +
                            final_measurement.information.transpose()
                        );

                    if (!final_measurement.information.allFinite() ||
                        !final_measurement.gradient.allFinite())
                    {
                        return false;
                    }

                    static std::size_t
                        structural_ownership_final_counter =
                            0;

                    ++structural_ownership_final_counter;

                    if ((structural_ownership_final_counter %
                         100U) == 0U)
                    {
                        std::cout
                            << "LIO_STRUCTURAL_OWNERSHIP"
                            << " | stage=FINAL"
                            << " | rank="
                            << owned_rank
                            << " | max_eig="
                            << maximum_structural_eigenvalue
                            << std::endl;
                    }
                }
            }
        }

        final_measurement.information +=
            final_joint_information;

        final_measurement.gradient +=
            final_joint_gradient;
    }

    const StateMatrix final_prior_transport =
        BuildPriorTransportJacobian(
            current_state,
            prior_state);

    if (!final_prior_transport.allFinite())
    {
        return false;
    }

    StateMatrix posterior_information =
        final_prior_transport.transpose() *
            prior_information *
            final_prior_transport +
        final_measurement.information;

    posterior_information =
        0.5 *
        (posterior_information +
         posterior_information.transpose());

    StateVector zero_rhs =
        StateVector::Zero();

    ApplyEstimationMask(
        posterior_information,
        zero_rhs);

    Eigen::LDLT<StateMatrix> posterior_ldlt(
        posterior_information);

    if (posterior_ldlt.info() != Eigen::Success ||
        !posterior_ldlt.isPositive())
    {
        return false;
    }

    StateMatrix posterior_covariance =
        posterior_ldlt.solve(
            StateMatrix::Identity());

    posterior_covariance =
        0.5 *
        (posterior_covariance +
         posterior_covariance.transpose());

    if (!CovarianceIsFinite(posterior_covariance) ||
        (posterior_covariance.diagonal().array() <
         -1.0e-12)
            .any())
    {
        return false;
    }
    // =====================================================================
    // LIO_COUPLING_DIAG
    //
    // Diagnostic only:
    //   1. covariance correlations between pose / IMU / extrinsic states;
    //   2. extrinsic information after removing pose-explainable information;
    //   3. weighted LiDAR correction split between pose and extrinsic.
    //
    // This block MUST NOT modify the filter state or covariance.
    // =====================================================================
    {
        constexpr int kRoll =
            LioStateIndex::ROTATION;

        constexpr int kPitch =
            LioStateIndex::ROTATION + 1;

        constexpr int kPositionZ =
            LioStateIndex::POSITION + 2;

        constexpr int kVelocityZ =
            LioStateIndex::VELOCITY + 2;

        constexpr int kAccelBiasZ =
            LioStateIndex::ACCEL_BIAS + 2;

        constexpr int kExtrinsicRoll =
            LioStateIndex::EXTRINSIC_ROTATION;

        constexpr int kExtrinsicPitch =
            LioStateIndex::EXTRINSIC_ROTATION + 1;

        constexpr int kExtrinsicPositionZ =
            LioStateIndex::EXTRINSIC_POSITION + 2;

        const auto covariance_correlation =
            [](
                const StateMatrix &covariance,
                const int row,
                const int column) -> double
        {
            const double variance_row =
                covariance(row, row);

            const double variance_column =
                covariance(column, column);

            if (!std::isfinite(variance_row) ||
                !std::isfinite(variance_column) ||
                variance_row <= 1.0e-18 ||
                variance_column <= 1.0e-18)
            {
                return 0.0;
            }

            const double denominator =
                std::sqrt(
                    variance_row *
                    variance_column);

            if (!std::isfinite(denominator) ||
                denominator <= 1.0e-18)
            {
                return 0.0;
            }

            return covariance(row, column) /
                   denominator;
        };

        // -----------------------------------------------------------------
        // 1. Prior / posterior state correlation.
        // -----------------------------------------------------------------
        const double rho_prior_z_tilz =
            covariance_correlation(
                prior_covariance,
                kPositionZ,
                kExtrinsicPositionZ);

        const double rho_post_z_tilz =
            covariance_correlation(
                posterior_covariance,
                kPositionZ,
                kExtrinsicPositionZ);

        const double rho_prior_pitch_rily =
            covariance_correlation(
                prior_covariance,
                kPitch,
                kExtrinsicPitch);

        const double rho_post_pitch_rily =
            covariance_correlation(
                posterior_covariance,
                kPitch,
                kExtrinsicPitch);

        const double rho_prior_roll_rilx =
            covariance_correlation(
                prior_covariance,
                kRoll,
                kExtrinsicRoll);

        const double rho_post_roll_rilx =
            covariance_correlation(
                posterior_covariance,
                kRoll,
                kExtrinsicRoll);

        const double rho_prior_z_baz =
            covariance_correlation(
                prior_covariance,
                kPositionZ,
                kAccelBiasZ);

        const double rho_post_z_baz =
            covariance_correlation(
                posterior_covariance,
                kPositionZ,
                kAccelBiasZ);

        const double rho_prior_vz_baz =
            covariance_correlation(
                prior_covariance,
                kVelocityZ,
                kAccelBiasZ);

        const double rho_post_vz_baz =
            covariance_correlation(
                posterior_covariance,
                kVelocityZ,
                kAccelBiasZ);

        // -----------------------------------------------------------------
        // 2. Extrinsic conditional LiDAR information.
        //
        // Direct LiDAR measurement observes:
        //
        //     pose      : state [0 ... 5]
        //     extrinsic : state [17 ... 22]
        //
        // We remove the part of extrinsic information explainable by pose:
        //
        // Lambda_ext|pose
        //   = Lambda_ee
        //   - Lambda_ep Lambda_pp^+ Lambda_pe
        //
        // Moore-Penrose inverse is used because corridor pose information
        // itself may be rank deficient.
        // -----------------------------------------------------------------
        using Matrix6d =
            Eigen::Matrix<double, 6, 6>;

        using Vector6d =
            Eigen::Matrix<double, 6, 1>;

        const Matrix6d lambda_pp =
            0.5 *
            (final_measurement.information
                 .block<6, 6>(
                     LioStateIndex::ROTATION,
                     LioStateIndex::ROTATION) +
             final_measurement.information
                 .block<6, 6>(
                     LioStateIndex::ROTATION,
                     LioStateIndex::ROTATION)
                 .transpose());

        const Matrix6d lambda_pe =
            final_measurement.information
                .block<6, 6>(
                    LioStateIndex::ROTATION,
                    LioStateIndex::EXTRINSIC_ROTATION);

        const Matrix6d lambda_ee =
            0.5 *
            (final_measurement.information
                 .block<6, 6>(
                     LioStateIndex::EXTRINSIC_ROTATION,
                     LioStateIndex::EXTRINSIC_ROTATION) +
             final_measurement.information
                 .block<6, 6>(
                     LioStateIndex::EXTRINSIC_ROTATION,
                     LioStateIndex::EXTRINSIC_ROTATION)
                 .transpose());

        Matrix6d lambda_pp_pinv =
            Matrix6d::Zero();

        bool schur_valid =
            false;

        Eigen::SelfAdjointEigenSolver<Matrix6d>
            pose_information_solver(
                lambda_pp);

        if (pose_information_solver.info() ==
                Eigen::Success &&
            pose_information_solver.eigenvalues().allFinite() &&
            pose_information_solver.eigenvectors().allFinite())
        {
            const Vector6d pose_eigenvalues =
                pose_information_solver.eigenvalues();

            const double maximum_pose_eigenvalue =
                std::max(
                    0.0,
                    pose_eigenvalues.maxCoeff());

            const double pseudo_inverse_threshold =
                std::max(
                    1.0e-9,
                    maximum_pose_eigenvalue *
                        1.0e-6);

            Vector6d inverse_eigenvalues =
                Vector6d::Zero();

            for (int index = 0;
                 index < 6;
                 ++index)
            {
                if (pose_eigenvalues(index) >
                    pseudo_inverse_threshold)
                {
                    inverse_eigenvalues(index) =
                        1.0 /
                        pose_eigenvalues(index);
                }
            }

            lambda_pp_pinv =
                pose_information_solver.eigenvectors() *
                inverse_eigenvalues.asDiagonal() *
                pose_information_solver
                    .eigenvectors()
                    .transpose();

            schur_valid =
                lambda_pp_pinv.allFinite();
        }

        Matrix6d lambda_ext_given_pose =
            Matrix6d::Zero();

        Vector6d normalized_ext_eigenvalues =
            Vector6d::Zero();

        double normalized_ext_condition =
            -1.0;

        if (schur_valid)
        {
            lambda_ext_given_pose =
                lambda_ee -
                lambda_pe.transpose() *
                    lambda_pp_pinv *
                    lambda_pe;

            lambda_ext_given_pose =
                0.5 *
                (lambda_ext_given_pose +
                 lambda_ext_given_pose.transpose());

            // Normalize the six extrinsic coordinates using the current
            // prior uncertainty.  This removes most rad-vs-m scale effects.
            Matrix6d prior_ext_covariance =
                prior_covariance.block<6, 6>(
                    LioStateIndex::EXTRINSIC_ROTATION,
                    LioStateIndex::EXTRINSIC_ROTATION);

            prior_ext_covariance =
                0.5 *
                (prior_ext_covariance +
                 prior_ext_covariance.transpose());

            Eigen::SelfAdjointEigenSolver<Matrix6d>
                ext_covariance_solver(
                    prior_ext_covariance);

            if (ext_covariance_solver.info() ==
                    Eigen::Success &&
                ext_covariance_solver
                    .eigenvalues()
                    .allFinite())
            {
                Vector6d sqrt_covariance_eigenvalues =
                    Vector6d::Zero();

                for (int index = 0;
                     index < 6;
                     ++index)
                {
                    sqrt_covariance_eigenvalues(index) =
                        std::sqrt(
                            std::max(
                                0.0,
                                ext_covariance_solver
                                    .eigenvalues()(index)));
                }

                const Matrix6d sqrt_prior_ext_covariance =
                    ext_covariance_solver.eigenvectors() *
                    sqrt_covariance_eigenvalues.asDiagonal() *
                    ext_covariance_solver
                        .eigenvectors()
                        .transpose();

                Matrix6d normalized_ext_information =
                    sqrt_prior_ext_covariance *
                    lambda_ext_given_pose *
                    sqrt_prior_ext_covariance;

                normalized_ext_information =
                    0.5 *
                    (normalized_ext_information +
                     normalized_ext_information.transpose());

                Eigen::SelfAdjointEigenSolver<Matrix6d>
                    normalized_ext_solver(
                        normalized_ext_information);

                if (normalized_ext_solver.info() ==
                        Eigen::Success &&
                    normalized_ext_solver
                        .eigenvalues()
                        .allFinite())
                {
                    normalized_ext_eigenvalues =
                        normalized_ext_solver
                            .eigenvalues();

                    const double minimum_eigenvalue =
                        normalized_ext_eigenvalues
                            .minCoeff();

                    const double maximum_eigenvalue =
                        normalized_ext_eigenvalues
                            .maxCoeff();

                    if (minimum_eigenvalue >
                            1.0e-12 &&
                        maximum_eigenvalue >
                            0.0)
                    {
                        normalized_ext_condition =
                            maximum_eigenvalue /
                            minimum_eigenvalue;
                    }
                }
            }
        }

        // -----------------------------------------------------------------
        // 3. Which DIRECT LiDAR-observed state group explains the update?
        //
        // Bias / gravity are NOT direct columns of the point-to-plane H.
        // They receive LiDAR correction through covariance coupling.
        //
        // Therefore here we compare only:
        //
        //     pose vs extrinsic
        //
        // in whitened measurement space using Lambda = H^T R^-1 H.
        // -----------------------------------------------------------------
        const StateVector final_delta =
            BoxMinus(
                current_state,
                prior_state);

        const Vector6d pose_delta =
            final_delta.segment<6>(
                LioStateIndex::ROTATION);

        const Vector6d extrinsic_delta =
            final_delta.segment<6>(
                LioStateIndex::EXTRINSIC_ROTATION);

        const double pose_energy =
            std::max(
                0.0,
                pose_delta.dot(
                    lambda_pp *
                    pose_delta));

        const double extrinsic_energy =
            std::max(
                0.0,
                extrinsic_delta.dot(
                    lambda_ee *
                    extrinsic_delta));

        const double pose_ext_cross =
            pose_delta.dot(
                lambda_pe *
                extrinsic_delta);

        const double total_energy =
            std::max(
                0.0,
                pose_energy +
                    2.0 * pose_ext_cross +
                    extrinsic_energy);

        const double correspondence_count =
            static_cast<double>(
                std::max<std::size_t>(
                    1,
                    final_measurement
                        .correspondences));

        const double pose_measurement_rms =
            std::sqrt(
                pose_energy /
                correspondence_count);

        const double extrinsic_measurement_rms =
            std::sqrt(
                extrinsic_energy /
                correspondence_count);

        const double total_measurement_rms =
            std::sqrt(
                total_energy /
                correspondence_count);

        double pose_measurement_cosine =
            0.0;

        double extrinsic_measurement_cosine =
            0.0;

        if (pose_energy > 1.0e-18 &&
            total_energy > 1.0e-18)
        {
            pose_measurement_cosine =
                (pose_energy +
                 pose_ext_cross) /
                std::sqrt(
                    pose_energy *
                    total_energy);
        }

        if (extrinsic_energy > 1.0e-18 &&
            total_energy > 1.0e-18)
        {
            extrinsic_measurement_cosine =
                (extrinsic_energy +
                 pose_ext_cross) /
                std::sqrt(
                    extrinsic_energy *
                    total_energy);
        }

        std::cout
            << "LIO_COUPLING_DIAG"
            << " | t="
            << current_state.timestamp

            << " | z="
            << current_state.P_WI.z()

            << " | vz="
            << current_state.V_WI.z()

            << " | baz="
            << current_state.accel_bias.z()

            << " | g=["
            << current_state.gravity_W.x() << " "
            << current_state.gravity_W.y() << " "
            << current_state.gravity_W.z() << "]"

            << " | tIL=["
            << current_state.P_IL.x() << " "
            << current_state.P_IL.y() << " "
            << current_state.P_IL.z() << "]"

            << " | dx_z="
            << final_delta(kPositionZ)

            << " | dx_vz="
            << final_delta(kVelocityZ)

            << " | dx_baz="
            << final_delta(kAccelBiasZ)

            << " | dx_ext_r=["
            << final_delta(
                   LioStateIndex::EXTRINSIC_ROTATION)
            << " "
            << final_delta(
                   LioStateIndex::EXTRINSIC_ROTATION + 1)
            << " "
            << final_delta(
                   LioStateIndex::EXTRINSIC_ROTATION + 2)
            << "]"

            << " | dx_ext_t=["
            << final_delta(
                   LioStateIndex::EXTRINSIC_POSITION)
            << " "
            << final_delta(
                   LioStateIndex::EXTRINSIC_POSITION + 1)
            << " "
            << final_delta(
                   LioStateIndex::EXTRINSIC_POSITION + 2)
            << "]"

            << " | rho_pre_z_tILz="
            << rho_prior_z_tilz

            << " | rho_post_z_tILz="
            << rho_post_z_tilz

            << " | rho_pre_pitch_RILy="
            << rho_prior_pitch_rily

            << " | rho_post_pitch_RILy="
            << rho_post_pitch_rily

            << " | rho_pre_roll_RILx="
            << rho_prior_roll_rilx

            << " | rho_post_roll_RILx="
            << rho_post_roll_rilx

            << " | rho_pre_z_baz="
            << rho_prior_z_baz

            << " | rho_post_z_baz="
            << rho_post_z_baz

            << " | rho_pre_vz_baz="
            << rho_prior_vz_baz

            << " | rho_post_vz_baz="
            << rho_post_vz_baz

            << " | ext_obs_eig=["
            << normalized_ext_eigenvalues(0) << " "
            << normalized_ext_eigenvalues(1) << " "
            << normalized_ext_eigenvalues(2) << " "
            << normalized_ext_eigenvalues(3) << " "
            << normalized_ext_eigenvalues(4) << " "
            << normalized_ext_eigenvalues(5) << "]"

            << " | ext_obs_condition="
            << normalized_ext_condition

            << " | meas_pose_rms="
            << pose_measurement_rms

            << " | meas_ext_rms="
            << extrinsic_measurement_rms

            << " | meas_total_rms="
            << total_measurement_rms

            << " | pose_cos="
            << pose_measurement_cosine

            << " | ext_cos="
            << extrinsic_measurement_cosine

            << std::endl;
    }
    state_ =
        current_state;

    covariance_ =
        posterior_covariance;

    result.correspondences =
        final_measurement.correspondences;

    result.downweighted_correspondences =
        final_measurement.downweighted_correspondences;

    result.final_rmse =
        final_measurement.rmse;

    result.final_robust_rmse =
        final_measurement.robust_rmse;

    result.position_information_x =
        final_measurement.information(
            LioStateIndex::POSITION,
            LioStateIndex::POSITION);

    result.position_information_y =
        final_measurement.information(
            LioStateIndex::POSITION + 1,
            LioStateIndex::POSITION + 1);

    result.position_information_z =
        final_measurement.information(
            LioStateIndex::POSITION + 2,
            LioStateIndex::POSITION + 2);

    const double position_information_sum =
        result.position_information_x +
        result.position_information_y +
        result.position_information_z;

    if (std::isfinite(position_information_sum) &&
        position_information_sum > 1.0e-12)
    {
        result.position_information_z_ratio =
            result.position_information_z /
            position_information_sum;
    }


    // =====================================================================
    // FR_LIO_ODOM_INFORMATION_V2
    //
    // Export a dynamic 6x6 keyframe odometry information matrix from the
    // final ACTUAL measurement information. Failure here must never reject a
    // valid LIO frame; the mapping/backend bridge will simply use its legacy
    // fallback information for that keyframe.
    // =====================================================================
    result.pose_graph_information_valid =
        BuildLioPoseGraphInformation(
            final_measurement.information,
            current_state,
            config_.estimate_extrinsic,
            result.pose_graph_information);

    result.success =
        true;

    return true;
}


bool Ieskf::InjectCorrectedLidarPose(
    const Eigen::Isometry3d &corrected_T_WL,
    const Eigen::Vector3d &corrected_V_WI)

{
    if (!initialized_ ||
        !StateIsFinite(state_) ||
        
!corrected_T_WL.matrix().allFinite() ||
        !corrected_V_WI.allFinite()
)
    {
        return false;
    }

    Eigen::Quaterniond q_IL =
        state_.Q_IL;

    if (!q_IL.coeffs().allFinite() ||
        q_IL.norm() <= 1.0e-12 ||
        !state_.P_IL.allFinite())
    {
        return false;
    }

    q_IL.normalize();

    // Runtime convention:
    //
    //     p_I = R_IL p_L + P_IL
    //
    // therefore:
    //
    //     T_WL = T_WI * T_IL
    //
    // and the corrected IMU pose is:
    //
    //     T_WI = T_WL * T_IL^{-1}
    Eigen::Isometry3d T_IL =
        Eigen::Isometry3d::Identity();

    T_IL.linear() =
        q_IL.toRotationMatrix();

    T_IL.translation() =
        state_.P_IL;

    const Eigen::Isometry3d corrected_T_WI =
        corrected_T_WL *
        T_IL.inverse();

    if (!corrected_T_WI.matrix().allFinite())
    {
        return false;
    }

    Eigen::Quaterniond corrected_Q_WI(
        corrected_T_WI.rotation());

    if (!corrected_Q_WI.coeffs().allFinite() ||
        corrected_Q_WI.norm() <= 1.0e-12)
    {
        return false;
    }

    corrected_Q_WI.normalize();

    LioState corrected_state =
        state_;

    // Ground V1 owns ONLY pose correction.
    corrected_state.Q_WI =
        corrected_Q_WI;

    
corrected_state.P_WI =
        corrected_T_WI.translation();

    corrected_state.V_WI =
        corrected_V_WI;


    // Intentionally unchanged:
    //
    // corrected_state.B_G
    // corrected_state.B_A
    // corrected_state.Gravity_W
    // corrected_state.Q_IL
    // corrected_state.P_IL
    // covariance_

    if (!StateIsFinite(corrected_state))
    {
        return false;
    }

    state_ =
        corrected_state;

    return true;
}


bool Ieskf::GroundPoseUpdate(
    const Eigen::Vector2d &tilt_correction_rad,
    double z_correction_m,
    double normal_sigma_rad,
    double height_sigma_m)
{
    if (!initialized_ ||
        !StateIsFinite(state_) ||
        !CovarianceIsFinite(covariance_) ||
        !tilt_correction_rad.allFinite() ||
        !std::isfinite(z_correction_m) ||
        !std::isfinite(normal_sigma_rad) ||
        !std::isfinite(height_sigma_m) ||
        normal_sigma_rad <= 0.0 ||
        height_sigma_m <= 0.0)
    {
        return false;
    }

    // =====================================================================
    // Measurement:
    //
    //     z =
    //       [ dtheta_x
    //         dtheta_y
    //         dp_z     ]
    //
    // H has direct support ONLY on:
    //
    //     ROTATION x
    //     ROTATION y
    //     POSITION z
    //
    // Velocity / bias / gravity may still update through the Kalman
    // cross-covariance, which is exactly what we want from a true filter
    // measurement.
    // =====================================================================

    using GroundJacobian =
        Eigen::Matrix<double, 3, STATE_DIM>;

    using GroundVector =
        Eigen::Matrix<double, 3, 1>;

    using GroundCovariance =
        Eigen::Matrix3d;

    GroundJacobian H =
        GroundJacobian::Zero();

    H(
        0,
        LioStateIndex::ROTATION + 0) =
        1.0;

    H(
        1,
        LioStateIndex::ROTATION + 1) =
        1.0;

    H(
        2,
        LioStateIndex::POSITION + 2) =
        1.0;

    GroundVector measurement =
        GroundVector::Zero();

    measurement(0) =
        tilt_correction_rad.x();

    measurement(1) =
        tilt_correction_rad.y();

    measurement(2) =
        z_correction_m;

    GroundCovariance R =
        GroundCovariance::Zero();

    R(0, 0) =
        normal_sigma_rad *
        normal_sigma_rad;

    R(1, 1) =
        normal_sigma_rad *
        normal_sigma_rad;

    R(2, 2) =
        height_sigma_m *
        height_sigma_m;

    const StateMatrix prior_covariance =
        0.5 *
        (covariance_ +
         covariance_.transpose());

    if (!prior_covariance.allFinite())
    {
        return false;
    }

    const GroundCovariance innovation_covariance =
        H *
            prior_covariance *
            H.transpose() +
        R;

    Eigen::LDLT<GroundCovariance>
        innovation_ldlt(
            innovation_covariance);

    if (innovation_ldlt.info() !=
            Eigen::Success ||
        !innovation_ldlt.isPositive())
    {
        return false;
    }

    const Eigen::Matrix<
        double,
        STATE_DIM,
        3>
        kalman_gain =
            prior_covariance *
            H.transpose() *
            innovation_ldlt.solve(
                GroundCovariance::Identity());

    if (!kalman_gain.allFinite())
    {
        return false;
    }

    StateVector increment =
        kalman_gain *
        measurement;

    if (!increment.allFinite())
    {
        return false;
    }

    // The current production UGV configuration keeps the physical
    // LiDAR-IMU extrinsic fixed.  Do not let the Ground pseudo-measurement
    // reopen those degrees of freedom through covariance coupling.
    if (!config_.estimate_extrinsic)
    {
        increment.segment<3>(
                     LioStateIndex::EXTRINSIC_ROTATION)
            .setZero();

        increment.segment<3>(
                     LioStateIndex::EXTRINSIC_POSITION)
            .setZero();
    }

    LioState updated_state =
        state_;

    BoxPlus(
        updated_state,
        increment);

    if (!StateIsFinite(updated_state))
    {
        return false;
    }

    // Joseph covariance update.
    //
    // This is deliberately used instead of
    //
    //     P = (I-KH)P
    //
    // because Ground is a small, repeated pseudo-measurement and numerical
    // symmetry / PSD robustness is important.
    const StateMatrix identity =
        StateMatrix::Identity();

    const StateMatrix I_KH =
        identity -
        kalman_gain * H;

    StateMatrix updated_covariance =
        I_KH *
            prior_covariance *
            I_KH.transpose() +
        kalman_gain *
            R *
            kalman_gain.transpose();

    updated_covariance =
        0.5 *
        (updated_covariance +
         updated_covariance.transpose());

    if (!CovarianceIsFinite(
            updated_covariance) ||
        (updated_covariance
             .diagonal()
             .array() <
         -1.0e-12)
            .any())
    {
        return false;
    }

    state_ =
        updated_state;

    covariance_ =
        updated_covariance;

    return true;
}

        

bool Ieskf::GroundStateUpdate(
    const Eigen::Isometry3d &corrected_T_WL,
    const Eigen::Vector3d &ground_normal_W,
    double normal_sigma_rad,
    double height_sigma_m,
    double normal_velocity_sigma_mps)
{
    if (!initialized_ ||
        !StateIsFinite(state_) ||
        !CovarianceIsFinite(covariance_) ||
        !corrected_T_WL.matrix().allFinite() ||
        !ground_normal_W.allFinite() ||
        ground_normal_W.norm() <= 1.0e-9 ||
        !std::isfinite(normal_sigma_rad) ||
        !std::isfinite(height_sigma_m) ||
        !std::isfinite(normal_velocity_sigma_mps) ||
        normal_sigma_rad <= 0.0 ||
        height_sigma_m <= 0.0 ||
        normal_velocity_sigma_mps <= 0.0)
    {
        return false;
    }

    const Eigen::Vector3d reference_normal_W =
        ground_normal_W.normalized();

    if (!reference_normal_W.allFinite())
    {
        return false;
    }

    // ------------------------------------------------------------
    // Accepted Ground LiDAR pose -> target IMU pose:
    //
    //     T_WI_target = T_WL_target * T_IL^{-1}
    // ------------------------------------------------------------
    if (!state_.Q_IL.coeffs().allFinite() ||
        state_.Q_IL.norm() <= 1.0e-12 ||
        !state_.P_IL.allFinite())
    {
        return false;
    }

    const Eigen::Quaterniond q_IL =
        state_.Q_IL.normalized();

    Eigen::Isometry3d T_IL =
        Eigen::Isometry3d::Identity();

    T_IL.linear() =
        q_IL.toRotationMatrix();

    T_IL.translation() =
        state_.P_IL;

    const Eigen::Isometry3d corrected_T_WI =
        corrected_T_WL *
        T_IL.inverse();

    if (!corrected_T_WI.matrix().allFinite())
    {
        return false;
    }

    Eigen::Quaterniond corrected_Q_WI(
        corrected_T_WI.rotation());

    if (!corrected_Q_WI.coeffs().allFinite() ||
        corrected_Q_WI.norm() <= 1.0e-12)
    {
        return false;
    }

    corrected_Q_WI.normalize();

    // ------------------------------------------------------------
    // Build target state and use BoxMinus so that the Ground
    // orientation residual follows exactly the same right-error
    // convention as the IESKF:
    //
    //     R_target = R_current * Exp(delta_theta)
    //
    //     delta_theta =
    //         Log(R_current^T * R_target)
    // ------------------------------------------------------------
    LioState target_state =
        state_;

    target_state.Q_WI =
        corrected_Q_WI;

    target_state.P_WI =
        corrected_T_WI.translation();

    const StateVector target_delta =
        BoxMinus(
            target_state,
            state_);

    if (!target_delta.allFinite())
    {
        return false;
    }

    // ------------------------------------------------------------
    // Ground V2-A pseudo-measurement:
    //
    //     [ tilt_x
    //       tilt_y
    //       height_z
    //       normal_velocity ]
    //
    // Ground-normal velocity constraint:
    //
    //     n_G^T V_WI = 0
    //
    // Therefore:
    //
    //     n_G^T delta_v = -n_G^T V_WI
    // ------------------------------------------------------------
    using GroundJacobian =
        Eigen::Matrix<double, 4, STATE_DIM>;

    using GroundVector =
        Eigen::Matrix<double, 4, 1>;

    using GroundCovariance =
        Eigen::Matrix4d;

    GroundJacobian H =
        GroundJacobian::Zero();

    H(
        0,
        LioStateIndex::ROTATION + 0) =
        1.0;

    H(
        1,
        LioStateIndex::ROTATION + 1) =
        1.0;

    H(
        2,
        LioStateIndex::POSITION + 2) =
        1.0;

    H.block<1, 3>(
        3,
        LioStateIndex::VELOCITY) =
        reference_normal_W.transpose();

    GroundVector measurement =
        GroundVector::Zero();

    measurement(0) =
        target_delta(
            LioStateIndex::ROTATION + 0);

    measurement(1) =
        target_delta(
            LioStateIndex::ROTATION + 1);

    measurement(2) =
        target_delta(
            LioStateIndex::POSITION + 2);

    const double normal_velocity_before =
        reference_normal_W.dot(
            state_.V_WI);

    measurement(3) =
        -normal_velocity_before;

    if (!measurement.allFinite())
    {
        return false;
    }

    // ------------------------------------------------------------
    // Measurement covariance.
    // ------------------------------------------------------------
    GroundCovariance R =
        GroundCovariance::Zero();

    R(0, 0) =
        normal_sigma_rad *
        normal_sigma_rad;

    R(1, 1) =
        normal_sigma_rad *
        normal_sigma_rad;

    R(2, 2) =
        height_sigma_m *
        height_sigma_m;

    R(3, 3) =
        normal_velocity_sigma_mps *
        normal_velocity_sigma_mps;

    // ------------------------------------------------------------
    // Kalman update.
    // ------------------------------------------------------------
    const StateMatrix prior_covariance =
        0.5 *
        (covariance_ +
         covariance_.transpose());

    if (!prior_covariance.allFinite())
    {
        return false;
    }

    const GroundCovariance innovation_covariance =
        H *
            prior_covariance *
            H.transpose() +
        R;

    Eigen::LDLT<GroundCovariance>
        innovation_ldlt(
            innovation_covariance);

    if (innovation_ldlt.info() !=
            Eigen::Success ||
        !innovation_ldlt.isPositive())
    {
        return false;
    }

    const Eigen::Matrix<
        double,
        STATE_DIM,
        4>
        kalman_gain =
            prior_covariance *
            H.transpose() *
            innovation_ldlt.solve(
                GroundCovariance::Identity());

    if (!kalman_gain.allFinite())
    {
        return false;
    }

    StateVector increment =
        kalman_gain *
        measurement;

    if (!increment.allFinite())
    {
        return false;
    }

    // Fixed physical LiDAR-IMU extrinsic stays fixed.
    if (!config_.estimate_extrinsic)
    {
        increment.segment<3>(
                     LioStateIndex::EXTRINSIC_ROTATION)
            .setZero();

        increment.segment<3>(
                     LioStateIndex::EXTRINSIC_POSITION)
            .setZero();
    }

    LioState updated_state =
        state_;

    BoxPlus(
        updated_state,
        increment);

    if (!StateIsFinite(updated_state))
    {
        return false;
    }

    // ------------------------------------------------------------
    // Joseph covariance update:
    //
    //     P+ =
    //       (I-KH)P(I-KH)^T + KRK^T
    // ------------------------------------------------------------
    const StateMatrix identity =
        StateMatrix::Identity();

    const StateMatrix I_KH =
        identity -
        kalman_gain * H;

    StateMatrix updated_covariance =
        I_KH *
            prior_covariance *
            I_KH.transpose() +
        kalman_gain *
            R *
            kalman_gain.transpose();

    updated_covariance =
        0.5 *
        (updated_covariance +
         updated_covariance.transpose());

    if (!CovarianceIsFinite(
            updated_covariance) ||
        (updated_covariance
             .diagonal()
             .array() <
         -1.0e-12)
            .any())
    {
        return false;
    }

    state_ =
        updated_state;

    covariance_ =
        updated_covariance;

    return true;
}

bool Ieskf::ConfigIsValid(
    
    const IeskfConfig &config) const
{
    if (!std::isfinite(config.max_imu_dt) ||
        config.max_imu_dt <= 0.0)
    {
        return false;
    }

    if (!InitialUncertaintyIsValid(
            config.initial_uncertainty))
    {
        return false;
    }

    if (!ProcessNoiseIsValid(config))
    {
        return false;
    }

    if (!LidarIterationConfigIsValid(config))
    {
        return false;
    }

    return true;
}

bool Ieskf::StateIsFinite(
    const LioState &state)
{
    return std::isfinite(state.timestamp) &&

           state.Q_WI.coeffs().allFinite() &&
           state.Q_WI.norm() > 1.0e-12 &&

           state.P_WI.allFinite() &&
           state.V_WI.allFinite() &&

           state.gyro_bias.allFinite() &&
           state.accel_bias.allFinite() &&

           state.gravity_W.allFinite() &&
           state.gravity_W.norm() > 1.0e-12 &&

           state.Q_IL.coeffs().allFinite() &&
           state.Q_IL.norm() > 1.0e-12 &&

           state.P_IL.allFinite();
}

bool Ieskf::CovarianceIsFinite(
    const StateMatrix &covariance)
{
    return covariance.allFinite();
}
