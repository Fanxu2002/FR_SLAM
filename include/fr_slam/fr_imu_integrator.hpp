#pragma once
#include "fr_slam/fr_imu_types.hpp"
#include "fr_slam/fr_imu_buffer.hpp"
#include "fr_slam/imu_adapter.hpp"
#include <vector>
class ImuIntegrator
{
private:
        bool InterpolateImu(const IMU_DATA &imu_a,
                            const IMU_DATA &imu_b,
                            double timestamp,
                            IMU_DATA &output);

public:
        bool Integrate(const std::vector<IMU_DATA> &imu_data,
                       const IMU_STATE &initial_state,
                       std::vector<IMU_POSE> &imu_pose,
                       IMU_STATE &final_state);

        bool Extract(const std::vector<IMU_DATA> &raw_imu,
                     double start_time,
                     double end_time,
                     std::vector<IMU_DATA> &extract_imu);
};
