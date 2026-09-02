#include "fr_slam/fr_hesai_adapter.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>

#include <pcl_conversions/pcl_conversions.h>

LIDAR_FRAME HESAI_Adapter::convert(
    const sensor_msgs::msg::PointCloud2 &msg)
{
    // 1. Convert ROS2 PointCloud2 to original Hesai point cloud
    pcl::PointCloud<HESAI_POINT>::Ptr raw_pointcloud =
        pcl::make_shared<pcl::PointCloud<HESAI_POINT>>();

    pcl::fromROSMsg(msg, *raw_pointcloud);

    // 2. Create unified LIDAR_FRAME
    LIDAR_FRAME lidar_frame;

    lidar_frame.frame_id = msg.header.frame_id;

    // 3. Check whether the cloud is empty
    if (raw_pointcloud->empty())
    {
        std::cerr
            << "Hesai cloud is empty, conversion failed!"
            << std::endl;

        lidar_frame.has_point_time = false;

        return lidar_frame;
    }

    // 4. Find minimum and maximum valid point timestamp
    double min_timestamp =
        std::numeric_limits<double>::max();

    double max_timestamp =
        std::numeric_limits<double>::lowest();

    bool has_valid_timestamp = false;

    for (const HESAI_POINT &point : raw_pointcloud->points)
    {
        if (!std::isfinite(point.timestamp))
        {
            continue;
        }

        has_valid_timestamp = true;

        if (point.timestamp < min_timestamp)
        {
            min_timestamp = point.timestamp;
        }

        if (point.timestamp > max_timestamp)
        {
            max_timestamp = point.timestamp;
        }
    }

    // 5. Get ROS header timestamp
    //    Unified unit: second

    const double header_time =
        static_cast<double>(msg.header.stamp.sec) +
        static_cast<double>(msg.header.stamp.nanosec) * 1e-9;

    // ============================================================
    // 6. Check whether valid per-point timestamps exist
    // ============================================================

    if (has_valid_timestamp)
    {
        lidar_frame.scan_duration =
            max_timestamp - min_timestamp;

        lidar_frame.has_point_time =
            lidar_frame.scan_duration > 1e-6;
    }
    else
    {
        lidar_frame.has_point_time = false;
    }

    // 7. Determine scan start time
    //
    // If valid per-point timestamps exist:
    //     scan_start_time = minimum point timestamp
    //
    // Otherwise:
    //     use ROS message header timestamp

    if (lidar_frame.has_point_time)
    {
        lidar_frame.scan_start_time =
            min_timestamp;
    }
    else
    {
        lidar_frame.scan_start_time =
            header_time;
    }

    // 8. Convert Hesai points to unified LIDAR_POINT
    lidar_frame.cloud->reserve(
        raw_pointcloud->size());

    for (const HESAI_POINT &raw_point : raw_pointcloud->points)
    {
        LIDAR_POINT hesai_point;

        hesai_point.x =
            raw_point.x;

        hesai_point.y =
            raw_point.y;

        hesai_point.z =
            raw_point.z;

        hesai_point.intensity =
            raw_point.intensity;

        hesai_point.ring =
            raw_point.ring;

        // ----------------------------------------------------
        // Unified point time:
        //
        // time_offset =
        //     point absolute time - scan start time
        //
        // Unit: second
        // ----------------------------------------------------

        if (lidar_frame.has_point_time &&
            std::isfinite(raw_point.timestamp))
        {
            hesai_point.time_offset =
                raw_point.timestamp -
                lidar_frame.scan_start_time;
        }
        else
        {
            hesai_point.time_offset = 0.0;
        }

        lidar_frame.cloud->push_back(
            hesai_point);
    }

    // 9. Finish cloud information
    lidar_frame.cloud->width =
        static_cast<std::uint32_t>(
            lidar_frame.cloud->size());

    lidar_frame.cloud->height = 1;

    lidar_frame.cloud->is_dense =
        raw_pointcloud->is_dense;

    return lidar_frame;
}