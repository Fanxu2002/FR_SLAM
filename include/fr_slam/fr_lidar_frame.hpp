#pragma once
#include "fr_slam/fr_point_types.hpp"
#include <memory>
#include <pcl/point_cloud.h>
#include <string>

struct LIDAR_FRAME
{
        pcl::PointCloud<LIDAR_POINT>::Ptr cloud;

        double scan_start_time = 0.0;
        // reference timestamp of the current scan, in seconds

        double scan_duration = 0.0;

        bool has_point_time = false;
        // Whether valid per-point time offsets are available.

        std::string frame_id;

        LIDAR_FRAME() : cloud(pcl::make_shared<pcl::PointCloud<LIDAR_POINT>>())
        {
        }

        double scanEndTime() const
        {
                return scan_start_time + scan_duration;
        }
};