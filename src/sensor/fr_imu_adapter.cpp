#include "fr_slam/fr_imu_adapter.hpp"
IMU_DATA ImuAdapter::convert(const sensor_msgs::msg::Imu &msg) const
{
    IMU_DATA imu_data;
    imu_data.timestamp =
        static_cast<double>(msg.header.stamp.sec) +
        static_cast<double>(msg.header.stamp.nanosec) * 1e-9;

    constexpr double KGravity = 9.80665;

    imu_data.gyro.x() = msg.angular_velocity.x;
    imu_data.gyro.y() = msg.angular_velocity.y;
    imu_data.gyro.z() = msg.angular_velocity.z;

    imu_data.accelerometer.x() = msg.linear_acceleration.x * KGravity;
    imu_data.accelerometer.y() = msg.linear_acceleration.y * KGravity;
    imu_data.accelerometer.z() = msg.linear_acceleration.z * KGravity;

    const double qx = msg.orientation.x;
    const double qy = msg.orientation.y;
    const double qz = msg.orientation.z;
    const double qw = msg.orientation.w;

    return imu_data;
}
