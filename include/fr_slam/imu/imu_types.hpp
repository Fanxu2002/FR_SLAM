#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
struct IMU_DATA
{
        double timestamp = 0.0;
        // absolute timestamp, unit: second

        Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
        // unit: rad/s

        Eigen::Vector3d accelerometer = Eigen::Vector3d::Zero();
        // unit: m/s^2
};

struct IMU_STATE
{

        double timestamp = 0.0;
        // unit: second

        //----- POSE -----
        Eigen::Quaterniond Q_WI = Eigen::Quaterniond::Identity();
        // Rotation from IMU to world frame
        Eigen::Vector3d P_WI = Eigen::Vector3d::Zero();
        // Position in world frame;
        // unit: meter

        //----- velocity -----
        Eigen::Vector3d V_WI = Eigen::Vector3d::Zero();
        // unit: m/s

        //----- IMU bias -----
        Eigen::Vector3d gyro_bias = Eigen::Vector3d::Zero();
        // unit: rad/s
        Eigen::Vector3d accel_bias = Eigen::Vector3d::Zero();
        // unit: m/s^2

        // Gravity
        Eigen::Vector3d gravity_W = Eigen::Vector3d(0, 0, -9.80665);

        Eigen::Matrix4d Pose_Matrix() const
        {
                Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
                T.block<3, 3>(0, 0) = Q_WI.normalized().toRotationMatrix();
                T.block<3, 1>(0, 3) = P_WI;
                return T;
        }
};
struct IMU_POSE
{
        double timestamp = 0.0;
        Eigen::Quaterniond Q_WI = Eigen::Quaterniond::Identity();

        Eigen::Vector3d P_WI = Eigen::Vector3d::Zero();

        Eigen::Vector3d V_WI = Eigen::Vector3d::Zero();
};
