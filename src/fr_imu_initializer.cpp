/*
IMU initialization:
1. Estimate gyroscope bias.
2. Estimate accelerometer bias.
3. Estimate the gravity direction and align it with the world z-axis.
*/
#include "fr_slam/fr_imu_initializer.hpp"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
bool ImuInitializer::Initialize(
    const std::vector<IMU_DATA> &imu_data,
    IMU_STATE &initial_state)
{
        const std::size_t imu_data_count = imu_data.size();

        // At least 100 IMU measurements are required.
        if (imu_data_count < 100)
        {
                return false;
        }

        const double imu_count =
            static_cast<double>(imu_data_count);

        Eigen::Vector3d gyro_mean =
            Eigen::Vector3d::Zero();

        Eigen::Vector3d accele_mean =
            Eigen::Vector3d::Zero();

        for (const IMU_DATA &imu : imu_data)
        {
                gyro_mean += imu.gyro;
                accele_mean += imu.accelerometer;
        }

        gyro_mean /= imu_count;
        accele_mean /= imu_count;

        if (accele_mean.norm() < 1e-6)
        {
                return false;
        }

        // Check whether the IMU is stationary.
        Eigen::Vector3d gyro_varience =
            Eigen::Vector3d::Zero();

        Eigen::Vector3d accele_varience =
            Eigen::Vector3d::Zero();

        for (const IMU_DATA &imu : imu_data)
        {
                const Eigen::Vector3d gyro_diff =
                    imu.gyro - gyro_mean;

                const Eigen::Vector3d accele_diff =
                    imu.accelerometer - accele_mean;

                gyro_varience +=
                    gyro_diff.cwiseProduct(gyro_diff);

                accele_varience +=
                    accele_diff.cwiseProduct(accele_diff);
        }

        gyro_varience /= imu_count;
        accele_varience /= imu_count;

        const Eigen::Vector3d gyro_std =
            gyro_varience.cwiseSqrt();

        const Eigen::Vector3d accele_std =
            accele_varience.cwiseSqrt();

        if (gyro_std.norm() > 0.05)
        {
                return false;
        }

        if (accele_std.norm() > 0.5)
        {
                return false;
        }

        // Estimate gravity-aligned initial orientation.
        constexpr double KGravity = 9.80665;

        if (std::abs(
                accele_mean.norm() - KGravity) > 2.0)
        {
                return false;
        }

        // Specific-force direction measured in IMU frame.
        const Eigen::Vector3d up_direction_I =
            accele_mean.normalized();

        // Define +Z as upward in the world frame.
        const Eigen::Vector3d up_direction_W =
            Eigen::Vector3d::UnitZ();

        const Eigen::Vector3d gravity_W(
            0.0,
            0.0,
            -KGravity);

        // IMU -> World
        Eigen::Quaterniond Q_WI =
            Eigen::Quaterniond::FromTwoVectors(
                up_direction_I,
                up_direction_W);

        Q_WI.normalize();

        // World -> IMU
        const Eigen::Quaterniond Q_IW =
            Q_WI.conjugate();

        const Eigen::Vector3d gravity_I =
            Q_IW * gravity_W;

        // Initial state is defined at the last sample
        // of the initialization window.
        initial_state.timestamp =
            imu_data.back().timestamp;

        // Static gyro: w_true = 0
        // => bg ≈ mean(w_m)
        initial_state.gyro_bias =
            gyro_mean;

        // Static accelerometer:
        // a_m = -R_IW * g_W + ba
        // => ba = a_m + R_IW * g_W
        initial_state.accel_bias =
            accele_mean + gravity_I;

        initial_state.Q_WI =
            Q_WI;

        initial_state.P_WI =
            Eigen::Vector3d::Zero();

        initial_state.V_WI =
            Eigen::Vector3d::Zero();

        initial_state.gravity_W =
            gravity_W;

        return true;
}