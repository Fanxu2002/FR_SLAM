#include "fr_slam/sensor/velodyne_adapter.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>

#include <sensor_msgs/point_cloud2_iterator.hpp>

LIDAR_FRAME Velodyne_Adapter::convert(
    const sensor_msgs::msg::PointCloud2 &msg)
{
    LIDAR_FRAME lidar_frame;

    lidar_frame.frame_id =
        msg.header.frame_id;

    const double header_time =
        static_cast<double>(msg.header.stamp.sec) +
        static_cast<double>(msg.header.stamp.nanosec) * 1.0e-9;

    // M2DGR / Velodyne:
    //
    // msg.header.stamp:
    //     scan start timestamp
    //
    // point.time:
    //     relative time from scan start
    //     unit: second
    //
    // Therefore:
    //
    // absolute point time =
    //     lidar_frame.scan_start_time
    //     + point.time_offset

    lidar_frame.scan_start_time =
        header_time;

    lidar_frame.scan_duration =
        0.0;

    lidar_frame.has_point_time =
        false;

    // ------------------------------------------------------------
    // Check required PointCloud2 fields.
    // ------------------------------------------------------------

    const auto has_field =
        [&msg](const std::string &field_name)
        {
            return std::any_of(
                msg.fields.begin(),
                msg.fields.end(),
                [&field_name](
                    const sensor_msgs::msg::PointField &field)
                {
                    return field.name == field_name;
                });
        };

    if (!has_field("x") ||
        !has_field("y") ||
        !has_field("z") ||
        !has_field("intensity") ||
        !has_field("ring") ||
        !has_field("time"))
    {
        std::cerr
            << "Velodyne PointCloud2 is missing required fields."
            << std::endl;

        return lidar_frame;
    }

    if (msg.width == 0 ||
        msg.height == 0 ||
        msg.data.empty())
    {
        std::cerr
            << "Velodyne cloud is empty."
            << std::endl;

        return lidar_frame;
    }

    // ------------------------------------------------------------
    // M2DGR PointCloud2 layout:
    //
    // x         FLOAT32
    // y         FLOAT32
    // z         FLOAT32
    // intensity FLOAT32
    // ring      UINT16
    // time      FLOAT32
    // ------------------------------------------------------------

    sensor_msgs::PointCloud2ConstIterator<float>
        x_iterator(msg, "x");

    sensor_msgs::PointCloud2ConstIterator<float>
        y_iterator(msg, "y");

    sensor_msgs::PointCloud2ConstIterator<float>
        z_iterator(msg, "z");

    sensor_msgs::PointCloud2ConstIterator<float>
        intensity_iterator(msg, "intensity");

    sensor_msgs::PointCloud2ConstIterator<std::uint16_t>
        ring_iterator(msg, "ring");

    sensor_msgs::PointCloud2ConstIterator<float>
        time_iterator(msg, "time");

    const std::size_t point_count =
        static_cast<std::size_t>(msg.width) *
        static_cast<std::size_t>(msg.height);

    lidar_frame.cloud->reserve(
        point_count);

    double minimum_time_offset =
        std::numeric_limits<double>::max();

    double maximum_time_offset =
        std::numeric_limits<double>::lowest();

    std::size_t valid_time_count =
        0;

    for (;
         x_iterator != x_iterator.end();
         ++x_iterator,
         ++y_iterator,
         ++z_iterator,
         ++intensity_iterator,
         ++ring_iterator,
         ++time_iterator)
    {
        LIDAR_POINT lidar_point{};

        lidar_point.x =
            *x_iterator;

        lidar_point.y =
            *y_iterator;

        lidar_point.z =
            *z_iterator;

        lidar_point.intensity =
            *intensity_iterator;

        lidar_point.ring =
            *ring_iterator;

        const double point_time_offset =
            static_cast<double>(
                *time_iterator);

        if (std::isfinite(point_time_offset))
        {
            lidar_point.time_offset =
                point_time_offset;

            minimum_time_offset =
                std::min(
                    minimum_time_offset,
                    point_time_offset);

            maximum_time_offset =
                std::max(
                    maximum_time_offset,
                    point_time_offset);

            ++valid_time_count;
        }
        else
        {
            lidar_point.time_offset =
                std::numeric_limits<double>::quiet_NaN();
        }

        lidar_frame.cloud->push_back(
            lidar_point);
    }

    // ------------------------------------------------------------
    // Validate per-point timing.
    // ------------------------------------------------------------

    if (valid_time_count >= 2 &&
        maximum_time_offset >
            minimum_time_offset)
    {
        lidar_frame.scan_duration =
            maximum_time_offset -
            minimum_time_offset;

        lidar_frame.has_point_time =
            lidar_frame.scan_duration >
            1.0e-6;
    }

    // M2DGR Velodyne uses per-point time relative to the
    // PointCloud2 header timestamp.  In door_02 the offsets
    // are approximately [-0.1, 0] s, i.e. the header is near
    // the scan end.  Normalize all sensors to the FR-SLAM rule:
    //
    //   scan_start_time = absolute time of the first point
    //   time_offset     = seconds from scan_start_time
    //
    if (lidar_frame.has_point_time)
    {
        lidar_frame.scan_start_time =
            header_time + minimum_time_offset;

        for (LIDAR_POINT &point : lidar_frame.cloud->points)
        {
            if (std::isfinite(point.time_offset))
            {
                point.time_offset -=
                    minimum_time_offset;
            }
            else
            {
                point.time_offset = 0.0;
            }
        }
    }
    else
    {
        lidar_frame.scan_start_time =
            header_time;

        lidar_frame.scan_duration =
            0.0;

        for (LIDAR_POINT &point : lidar_frame.cloud->points)
        {
            point.time_offset = 0.0;
        }
    }

    lidar_frame.cloud->width =
        static_cast<std::uint32_t>(
            lidar_frame.cloud->size());

    lidar_frame.cloud->height =
        1;

    lidar_frame.cloud->is_dense =
        msg.is_dense;

    return lidar_frame;
}
