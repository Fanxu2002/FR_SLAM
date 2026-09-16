#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cstddef>

struct LioStateIndex
{
        // ERROR state vector index
        static constexpr int ROTATION = 0;            // rotation: 3
        static constexpr int POSITION = 3;            // position: 3
        static constexpr int VELOCITY = 6;            // velocity: 3
        static constexpr int GYRO_BIAS = 9;           // gyro_bias: 3
        static constexpr int ACCEL_BIAS = 12;         // accel_bias: 3
        static constexpr int GRAVITY = 15;            // gravity: 2
        static constexpr int EXTRINSIC_ROTATION = 17; // extrinsic_rotation: 3
        static constexpr int EXTRINSIC_POSITION = 20; // extrinsic_position: 3
        static constexpr int ERROR_STATE_DIM = 23;    // error state dim: 23

        // process noise vector
        static constexpr int PROCESS_NOISE_DIM = 12; // process noise dim: 12
};

struct LioState
{
        double timestamp = 0.0;

        // POSE Matrix (IMU TO WORLD)
        Eigen::Quaterniond Q_WI = Eigen::Quaterniond::Identity();
        Eigen::Vector3d P_WI = Eigen::Vector3d::Zero();

        // IMU velocity IN WORLD FRAME
        Eigen::Vector3d V_WI = Eigen::Vector3d::Zero();

        // IMU BIAS(GYRO BIAS and ACCLE BIAS)
        Eigen::Vector3d gyro_bias = Eigen::Vector3d::Zero();
        Eigen::Vector3d accel_bias = Eigen::Vector3d::Zero();

        // Gravity in the world frame
        Eigen::Vector3d gravity_W = Eigen::Vector3d(0.0, 0.0, -9.80665);

        // Translation Matrix( LIDAR TO IMU)
        Eigen::Quaterniond Q_IL = Eigen::Quaterniond::Identity();
        Eigen::Vector3d P_IL = Eigen::Vector3d::Zero();
};
