#pragma once
#include "fr_slam/fr_imu_types.hpp"
#include <vector>

class ImuInitializer
{
public:
        bool Initialize(
            const std::vector<IMU_DATA> &imu_data,
            IMU_STATE &initial_state);
};