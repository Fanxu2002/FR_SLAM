#pragma once
#include "fr_slam/imu/fr_imu_types.hpp"
#include "sensor_msgs/msg/imu.hpp"
class ImuAdapter
{
public:
        // Scale applied only to sensor_msgs/Imu::linear_acceleration.
        //
        // Livox publishes acceleration in g on the dataset used by FR-SLAM,
        // so its scale is 9.80665.  Fixposition publishes SI m/s^2 and uses
        // a scale of 1.0.  Angular velocity is already rad/s and is never
        // scaled here.
        void setAccelerationScale(double acceleration_scale);

        double accelerationScale() const;

        IMU_DATA convert(const sensor_msgs::msg::Imu &msg) const;

private:
        // Preserve the historical Livox behavior when an old YAML does not
        // yet provide imu_acceleration_scale.
        double acceleration_scale_ = 9.80665;
};
