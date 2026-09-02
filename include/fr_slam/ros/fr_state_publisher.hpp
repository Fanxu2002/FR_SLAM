#pragma once

#include "fr_slam/imu/fr_imu_types.hpp"

#include <rclcpp/rclcpp.hpp>

#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <memory>
#include <string>

class StatePublisher
{
public:
        explicit StatePublisher(
            rclcpp::Node *node);

        void PublishImu(
            const IMU_STATE &state,
            const IMU_DATA &imu_data);

        void PublishLidar(
            double timestamp,
            const Eigen::Matrix4d &T_WL);

private:
        builtin_interfaces::msg::Time ToRosTime(
            double timestamp) const;

        geometry_msgs::msg::PoseStamped MakePoseStamped(
            double timestamp,
            const Eigen::Vector3d &position,
            const Eigen::Quaterniond &orientation,
            const std::string &frame_id) const;

private:
        rclcpp::Node *node_;

        // IMU
        rclcpp::Publisher<
            nav_msgs::msg::Odometry>::SharedPtr imu_odom_pub_;

        rclcpp::Publisher<
            nav_msgs::msg::Path>::SharedPtr imu_path_pub_;

        nav_msgs::msg::Path imu_path_;

        // LiDAR
        rclcpp::Publisher<
            nav_msgs::msg::Odometry>::SharedPtr lidar_odom_pub_;

        rclcpp::Publisher<
            nav_msgs::msg::Path>::SharedPtr lidar_path_pub_;

        nav_msgs::msg::Path lidar_path_;

        const std::string world_frame_ = "world";
        const std::string imu_frame_ = "imu";
        const std::string lidar_frame_ = "lidar";
};