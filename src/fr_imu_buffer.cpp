#include "fr_slam/fr_imu_buffer.hpp"

bool IMUbuffer::Push(const IMU_DATA &imu_data)
{
        std::lock_guard<std::mutex> lock(mutex_);
        if (!imu_buffer.empty())
        // check whether the imu deque is empty or not
        {
                if (imu_data.timestamp <= imu_buffer.back().timestamp)
                {
                        return false;
                }
        }

        // Timestamp is valid, push new IMU data.
        imu_buffer.push_back(imu_data);

        return true;
}

std::size_t IMUbuffer::imu_deque_size()
{
        std::lock_guard<std::mutex> lock(mutex_);
        return imu_buffer.size();
}

bool IMUbuffer::imu_deque_is_empty()
{
        std::lock_guard<std::mutex> lock(mutex_);
        return imu_buffer.empty();
}

bool IMUbuffer::GetTimeRange(double start_time,
                             double end_time,
                             std::vector<IMU_DATA> &output)
{
        std::lock_guard<std::mutex> lock(mutex_);
        output.clear();
        if (start_time >= end_time)
        {
                return false;
        }
        if (imu_buffer.empty())
        {
                return false;
        }

        if (imu_buffer.front().timestamp > start_time)
        {
                return false;
        }
        if (imu_buffer.back().timestamp < end_time)
        {
                return false;
        }

        auto start_iter = imu_buffer.begin();

        for (auto iter = imu_buffer.begin(); iter != imu_buffer.end(); ++iter)
        {
                if (iter->timestamp <= start_time)
                {
                        start_iter = iter;
                }
                else
                {
                        break;
                }
        }

        for (auto iter = start_iter; iter != imu_buffer.end(); ++iter)
        {
                output.push_back(*iter);

                if (iter->timestamp >= end_time)
                {
                        return true;
                }
        }
        return false;
}

void IMUbuffer::RemoveOldData(double timestamp)
{
        std::lock_guard<std::mutex> lock(mutex_);

        if (imu_buffer.empty())
        {
                return;
        }

        const double remove_before_time =
            timestamp -
            imu_history_duration_;

        while (imu_buffer.size() > 1 &&
               imu_buffer[1].timestamp <=
                   remove_before_time)
        {
                imu_buffer.pop_front();
        }
}
