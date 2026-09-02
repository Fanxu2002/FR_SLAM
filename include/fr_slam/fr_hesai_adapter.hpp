#pragma once
#include "fr_slam/fr_lidar_adapter.hpp"

class HESAI_Adapter : public Lidar_Adapt
{
public:
        LIDAR_FRAME convert(const sensor_msgs::msg::PointCloud2 &msg) override;
};