#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include "fr_slam/common/fr_lidar_frame.hpp"
#include "fr_slam/imu/fr_imu_types.hpp"

class LidarDeskewer
{
private:
        bool InterpolatePose(
            const std::vector<IMU_POSE> &imu_pose,
            double timestamp,
            IMU_POSE &output_pose) const;

        Eigen::Isometry3d PoseToTransform(const IMU_POSE &pose) const;

        Eigen::Isometry3d T_IL_ = Eigen::Isometry3d::Identity();

public:
        void SetExtrinsic(const Eigen::Quaterniond &Q_IL, const Eigen::Vector3d &P_IL);
        bool Deskew(const LIDAR_FRAME &input_frame,
                    const std::vector<IMU_POSE> &imu_poses,
                    LIDAR_FRAME &output_frame,
                    bool use_translation = false);
};