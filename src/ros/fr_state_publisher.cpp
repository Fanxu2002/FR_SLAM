#include "fr_slam/fr_state_publisher.hpp"

#include <cmath>
#include <cstdint>

StatePublisher::StatePublisher(
    rclcpp::Node *node)
    : node_(node)
{
    imu_odom_pub_ =
        node_->create_publisher<
            nav_msgs::msg::Odometry>(
            "/imu_odometry",
            10);

    imu_path_pub_ =
        node_->create_publisher<
            nav_msgs::msg::Path>(
            "/imu_path",
            10);

    lidar_odom_pub_ =
        node_->create_publisher<
            nav_msgs::msg::Odometry>(
            "/lidar_odometry",
            10);

    lidar_path_pub_ =
        node_->create_publisher<
            nav_msgs::msg::Path>(
            "/lidar_path",
            10);

    imu_path_.header.frame_id =
        world_frame_;

    lidar_path_.header.frame_id =
        world_frame_;
}

builtin_interfaces::msg::Time
StatePublisher::ToRosTime(
    double timestamp) const
{
    builtin_interfaces::msg::Time stamp;

    const double sec_double =
        std::floor(timestamp);

    stamp.sec =
        static_cast<std::int32_t>(
            sec_double);

    stamp.nanosec =
        static_cast<std::uint32_t>(
            (timestamp - sec_double) * 1e9);

    return stamp;
}

geometry_msgs::msg::PoseStamped
StatePublisher::MakePoseStamped(
    double timestamp,
    const Eigen::Vector3d &position,
    const Eigen::Quaterniond &orientation,
    const std::string &frame_id) const
{
    geometry_msgs::msg::PoseStamped pose_msg;

    pose_msg.header.stamp =
        ToRosTime(timestamp);

    pose_msg.header.frame_id =
        frame_id;

    pose_msg.pose.position.x =
        position.x();

    pose_msg.pose.position.y =
        position.y();

    pose_msg.pose.position.z =
        position.z();

    pose_msg.pose.orientation.w =
        orientation.w();

    pose_msg.pose.orientation.x =
        orientation.x();

    pose_msg.pose.orientation.y =
        orientation.y();

    pose_msg.pose.orientation.z =
        orientation.z();

    return pose_msg;
}