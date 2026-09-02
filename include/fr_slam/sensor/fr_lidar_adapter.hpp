#pragma once
#include "fr_slam/common/fr_point_types.hpp"
#include "fr_slam/common/fr_lidar_frame.hpp"
#include <pcl/point_cloud.h>
#include <sensor_msgs/msg/point_cloud2.hpp>

class Lidar_Adapt
{
public:
        virtual ~Lidar_Adapt() = default;
        virtual LIDAR_FRAME convert(const sensor_msgs::msg::PointCloud2 &msg) = 0;
};