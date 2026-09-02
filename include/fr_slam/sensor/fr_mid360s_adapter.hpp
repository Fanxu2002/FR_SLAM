#pragma once
#include "fr_slam/sensor/fr_lidar_adapter.hpp"

class Mid360s_Adapter : public Lidar_Adapt
{
public:
        LIDAR_FRAME convert(const sensor_msgs::msg::PointCloud2 &msg) override;

private:
        bool isValidTag(std::uint8_t tag) const;
};
