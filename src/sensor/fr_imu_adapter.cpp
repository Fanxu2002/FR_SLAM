#include "fr_slam/sensor/fr_imu_adapter.hpp"

#include <cmath>
#include <stdexcept>

void ImuAdapter::setAccelerationScale(
    const double acceleration_scale)
{
    if (!std::isfinite(acceleration_scale) ||
        acceleration_scale <= 0.0)
    {
        throw std::invalid_argument(
            "IMU acceleration scale must be finite and positive.");
    }

    acceleration_scale_ = acceleration_scale;
}

double ImuAdapter::accelerationScale() const
{
    return acceleration_scale_;
}

IMU_DATA ImuAdapter::convert(const sensor_msgs::msg::Imu &msg) const
{
    IMU_DATA imu_data;
    imu_data.timestamp =
        static_cast<double>(msg.header.stamp.sec) +
        static_cast<double>(msg.header.stamp.nanosec) * 1e-9;

    imu_data.gyro.x() = msg.angular_velocity.x;
    imu_data.gyro.y() = msg.angular_velocity.y;
    imu_data.gyro.z() = msg.angular_velocity.z;

    imu_data.accelerometer.x() =
        msg.linear_acceleration.x * acceleration_scale_;

    imu_data.accelerometer.y() =
        msg.linear_acceleration.y * acceleration_scale_;

    imu_data.accelerometer.z() =
        msg.linear_acceleration.z * acceleration_scale_;

    return imu_data;
}
