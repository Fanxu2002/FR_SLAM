#include "fr_slam/sensor/fr_mid360s_adapter.hpp"
#include <pcl_conversions/pcl_conversions.h>
#include <cstdint>
#include <limits>

LIDAR_FRAME Mid360s_Adapter::convert(const sensor_msgs::msg::PointCloud2 &msg)
{
        // 1. create original Livox mid 360 lidar point cloud;
        pcl::PointCloud<LIVOX_POINT>::Ptr Livox_cloud;
        Livox_cloud = pcl::make_shared<pcl::PointCloud<LIVOX_POINT>>();

        // 2. convert sensor_msg::msg::PointCloud2 into Livox mid 360 lidar point cloud;
        pcl::fromROSMsg(msg, *Livox_cloud);

        // 3. create a LidarFrame
        LIDAR_FRAME lidar_frame;
        lidar_frame.frame_id = msg.header.frame_id;

        // 4. chech empty clooud
        if (Livox_cloud->empty())
        {
                std::cerr << "livox cloud is empty, convertion failed!"
                          << std::endl;
                lidar_frame.has_point_time = false;
                return lidar_frame;
        }

        // 5. Find the maximum and the minimum point timestamps
        double min_timestamp = std::numeric_limits<double>::max();
        double max_timestamp = std::numeric_limits<double>::lowest();

        for (const LIVOX_POINT &point : Livox_cloud->points)
        {
                if (min_timestamp > point.timestamp)
                {
                        min_timestamp = point.timestamp;
                }
                if (max_timestamp < point.timestamp)
                {
                        max_timestamp = point.timestamp;
                }
        }

        // 6. use the frame stamp as the scan begining time
        const double scan_start_time =
            static_cast<double>(msg.header.stamp.sec) +
            static_cast<double>(msg.header.stamp.nanosec) * 1e-9;

        lidar_frame.scan_start_time =
            scan_start_time;

        // 7. chech whether this cloud has valid per-point timestamps
        lidar_frame.scan_duration =
            (max_timestamp - min_timestamp) * 1e-9;
        lidar_frame.has_point_time =
            lidar_frame.scan_duration > 1e-6;

        lidar_frame.cloud->reserve(Livox_cloud->size());

        const long double scan_start_ns =
            static_cast<long double>(msg.header.stamp.sec) * 1e9L +
            static_cast<long double>(msg.header.stamp.nanosec);

        for (const LIVOX_POINT &raw_point : Livox_cloud->points)
        {
                if (!isValidTag(raw_point.tag))
                {
                        continue;
                }
                LIDAR_POINT lidar_point;
                lidar_point.x = raw_point.x;
                lidar_point.y = raw_point.y;
                lidar_point.z = raw_point.z;
                lidar_point.intensity = raw_point.intensity;
                // Livox uses "line" , in lidar point we use "ring"
                lidar_point.ring = static_cast<std::uint16_t>(raw_point.line);
                lidar_point.time_offset =
                    static_cast<double>(
                        (static_cast<long double>(raw_point.timestamp) - scan_start_ns) * 1e-9L);
                lidar_frame.cloud->push_back(lidar_point);
        }
        return lidar_frame;
}
bool Mid360s_Adapter::isValidTag(std::uint8_t tag) const
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