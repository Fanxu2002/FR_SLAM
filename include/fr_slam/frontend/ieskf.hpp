#pragma once

#include <cstddef>
#include <limits>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pcl/point_cloud.h>

#include "fr_slam/common/point_types.hpp"
#include "fr_slam/frontend/lio_state.hpp"
#include "fr_slam/imu/imu_types.hpp"

class LioMeasurementBuilder;
struct PreparedLidarTarget;

struct IeskfInitialUncertainty
{
        static constexpr double DEG_TO_RAD =
            0.017453292519943295;

        double rotation_std =
            1.0 * DEG_TO_RAD;

        double position_std =
            0.05;

        double velocity_std =
            0.10;

        double gyro_bias_std =
            0.01;

        double accel_bias_std =
            0.10;

        double gravity_std =
            2.0 * DEG_TO_RAD;

        double extrinsic_rotation_std =
            1.0 * DEG_TO_RAD;

        double extrinsic_position_std =
            0.03;
};

struct IeskfConfig
{
        double gyro_noise_std =
            std::numeric_limits<double>::quiet_NaN();

        double accel_noise_std =
            std::numeric_limits<double>::quiet_NaN();

        double gyro_bias_random_walk_std =
            std::numeric_limits<double>::quiet_NaN();

        double accel_bias_random_walk_std =
            std::numeric_limits<double>::quiet_NaN();

        double max_imu_dt =
            0.05;

        IeskfInitialUncertainty initial_uncertainty;

        int max_lidar_iterations =
            6;

        double lidar_convergence_rotation_rad =
            1.0e-4;

        double lidar_convergence_position_m =
            1.0e-3;

        bool estimate_gravity =
            true;

        bool estimate_extrinsic =
            true;
};

struct IeskfLidarUpdateResult
{
        bool success =
            false;

        bool converged =
            false;

        int iterations =
            0;

        std::size_t correspondences =
            0;

        std::size_t downweighted_correspondences =
            0;

        double initial_rmse =
            std::numeric_limits<double>::infinity();

        double final_rmse =
            std::numeric_limits<double>::infinity();

        double initial_robust_rmse =
            std::numeric_limits<double>::infinity();

        double final_robust_rmse =
            std::numeric_limits<double>::infinity();

        double final_rotation_increment =
            std::numeric_limits<double>::infinity();

        double final_position_increment =
            std::numeric_limits<double>::infinity();

        double final_extrinsic_rotation_increment =
            std::numeric_limits<double>::infinity();

        double final_extrinsic_position_increment =
            std::numeric_limits<double>::infinity();

        // Final LiDAR-only translational information.
        // POSITION block diagonal of Lambda_L.
        double position_information_x = 0.0;
        double position_information_y = 0.0;
        double position_information_z = 0.0;
        double position_information_z_ratio = 0.0;
};
class Ieskf
{
public:
        static constexpr int STATE_DIM =
            LioStateIndex::ERROR_STATE_DIM;

        static constexpr int NOISE_DIM =
            LioStateIndex::PROCESS_NOISE_DIM;

        using StateVector =
            Eigen::Matrix<double, STATE_DIM, 1>;

        using StateMatrix =
            Eigen::Matrix<double, STATE_DIM, STATE_DIM>;

        using NoiseMatrix =
            Eigen::Matrix<double, NOISE_DIM, NOISE_DIM>;

        using NoiseJacobian =
            Eigen::Matrix<double, STATE_DIM, NOISE_DIM>;

public:
        Ieskf();

        explicit Ieskf(
            const IeskfConfig &config);

        bool SetConfig(
            const IeskfConfig &config);

        const IeskfConfig &Config() const;

        bool Initialize(
            const LioState &initial_state);

        bool IsInitialized() const;

        const LioState &State() const;

        const StateMatrix &Covariance() const;

        static void BoxPlus(
            LioState &state,
            const StateVector &delta_x);

        static StateVector BoxMinus(
            const LioState &target,
            const LioState &reference);

        bool Propagate(
            const IMU_DATA &imu_k,
            const IMU_DATA &imu_next);

        bool IteratedLidarUpdate(
            const pcl::PointCloud<LIDAR_POINT>::ConstPtr &deskewed_scan_L,
            const PreparedLidarTarget &target,
            const LioMeasurementBuilder &measurement_builder,
            IeskfLidarUpdateResult &result);

private:
        StateMatrix BuildInitialCovariance() const;

        StateMatrix BuildErrorStateJacobian(
            const LioState &linearization_state,
            const Eigen::Vector3d &omega_unbiased,
            const Eigen::Vector3d &accel_unbiased) const;

        NoiseJacobian BuildNoiseJacobian(
            const LioState &linearization_state) const;

        NoiseMatrix BuildContinuousNoiseCovariance() const;

        StateMatrix BuildPriorTransportJacobian(
            const LioState &current_state,
            const LioState &prior_state) const;

        Eigen::Matrix2d BuildGravityTransportJacobian(
            const Eigen::Vector3d &current_gravity,
            const Eigen::Vector3d &prior_gravity) const;

        void ApplyEstimationMask(
            StateMatrix &system_matrix,
            StateVector &right_hand_side) const;

        bool ConfigIsValid(
            const IeskfConfig &config) const;

        static bool StateIsFinite(
            const LioState &state);

        static bool CovarianceIsFinite(
            const StateMatrix &covariance);

private:
        IeskfConfig config_;

        LioState state_;

        StateMatrix covariance_ =
            StateMatrix::Zero();

        bool initialized_ =
            false;
};
