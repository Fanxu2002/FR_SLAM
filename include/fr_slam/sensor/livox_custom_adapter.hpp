#pragma once

#include "fr_slam/common/lidar_frame.hpp"

#include <livox_ros_driver2/msg/custom_msg.hpp>

#include <cstdint>

class LivoxCustomAdapter
{
public:
    LIDAR_FRAME convert(
        const livox_ros_driver2::msg::CustomMsg &msg) const;

private:
    bool isValidTag(
        std::uint8_t tag) const;
};
