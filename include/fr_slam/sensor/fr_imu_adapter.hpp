#pragma once
#include "fr_slam/imu/fr_imu_types.hpp"
#include "sensor_msgs/msg/imu.hpp"
class ImuAdapter
{
public:
        IMU_DATA convert(const sensor_msgs::msg::Imu &msg) const;
};
