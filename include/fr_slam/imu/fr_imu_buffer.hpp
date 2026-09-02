#pragma once
#include <deque>
#include "fr_slam/imu/fr_imu_types.hpp"
#include <mutex>
#include <vector>

class IMUbuffer
{
private:
        std::deque<IMU_DATA> imu_buffer;
        double imu_history_duration_ = 0.5;
        std::mutex mutex_;

public:
        bool Push(const IMU_DATA &imu_data);
        // input a new imu data

        std::size_t imu_deque_size();
        // get the number of imu data in the deque

        bool imu_deque_is_empty();
        // check whether the imu deque is empty or not

        bool GetTimeRange(double start_time,
                          double end_time,
                          std::vector<IMU_DATA> &output);

        void RemoveOldData(double timestamp);
};