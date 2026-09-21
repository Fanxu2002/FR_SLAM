#include "fr_slam/sensor/livox_custom_adapter.hpp"

#include <algorithm>

#include <cmath>
#include <cstdint>
#include <limits>

LIDAR_FRAME LivoxCustomAdapter::convert(
    const livox_ros_driver2::msg::CustomMsg &msg) const
{
    LIDAR_FRAME lidar_frame;

    lidar_frame.frame_id =
        msg.header.frame_id;

    if (msg.points.empty())
    {
        lidar_frame.has_point_time = false;
        lidar_frame.scan_duration = 0.0;

        return lidar_frame;
    }

    // Livox Driver2 CustomMsg:
    //
    //   timebase    : timestamp of the first point [ns]
    //   offset_time : offset from timebase [ns]
    //
    // FR-SLAM internal convention:
    //
    //   scan_start_time : absolute time [s]
    //   time_offset     : relative point time [s]

    if (msg.timebase > 0)
    {
        lidar_frame.scan_start_time =
            static_cast<double>(msg.timebase) *
            1.0e-9;
    }
    else
    {
        lidar_frame.scan_start_time =
            static_cast<double>(
                msg.header.stamp.sec) +
            static_cast<double>(
                msg.header.stamp.nanosec) *
                1.0e-9;
    }

    std::uint32_t maximum_offset_time = 0;

    lidar_frame.cloud->reserve(
        msg.points.size());

    for (const auto &raw_point : msg.points)
    {
        maximum_offset_time =
            std::max(
                maximum_offset_time,
                raw_point.offset_time);

        if (!isValidTag(raw_point.tag))
        {
            continue;
        }

        if (!std::isfinite(raw_point.x) ||
            !std::isfinite(raw_point.y) ||
            !std::isfinite(raw_point.z))
        {
            continue;
        }

        // M3DGR CustomMsg contains invalid zero placeholders.
        if (raw_point.x == 0.0F &&
            raw_point.y == 0.0F &&
            raw_point.z == 0.0F)
        {
            continue;
        }

        LIDAR_POINT lidar_point;

        lidar_point.x =
            raw_point.x;

        lidar_point.y =
            raw_point.y;

        lidar_point.z =
            raw_point.z;

        lidar_point.intensity =
            static_cast<float>(
                raw_point.reflectivity);

        lidar_point.ring =
            static_cast<std::uint16_t>(
                raw_point.line);

        lidar_point.time_offset =
            static_cast<double>(
                raw_point.offset_time) *
            1.0e-9;

        lidar_frame.cloud->push_back(
            lidar_point);
    }

    lidar_frame.scan_duration =
        static_cast<double>(
            maximum_offset_time) *
        1.0e-9;

    lidar_frame.has_point_time =
        lidar_frame.scan_duration >
        1.0e-6;

    lidar_frame.cloud->width =
        static_cast<std::uint32_t>(
            lidar_frame.cloud->size());

    lidar_frame.cloud->height = 1;

    lidar_frame.cloud->is_dense = true;

    return lidar_frame;
}

bool LivoxCustomAdapter::isValidTag(
    std::uint8_t tag) const
{
    const std::uint8_t other_status =
        (tag >> 4) & 0x03;

    const std::uint8_t rain_status =
        (tag >> 2) & 0x03;

    const std::uint8_t glue_status =
        tag & 0x03;

    if (other_status >= 2)
    {
        return false;
    }

    if (rain_status >= 2)
    {
        return false;
    }

    if (glue_status >= 2)
    {
        return false;
    }

    return true;
}
