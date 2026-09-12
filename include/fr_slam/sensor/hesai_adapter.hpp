#pragma once
#include "fr_slam/sensor/lidar_adapter.hpp"

class HESAI_Adapter : public Lidar_Adapt
{
public:
        LIDAR_FRAME convert(const sensor_msgs::msg::PointCloud2 &msg) override;
};