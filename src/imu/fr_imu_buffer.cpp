#include "fr_slam/imu/fr_imu_buffer.hpp"

namespace
{
constexpr double kImuTimeEpsilonSec =
    1.0e-6;
}

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

        // Absolute UNIX timestamps are stored as double. Around 1.78e9 s,
        // one representable step is already about 0.238 us. Treat a
        // sub-microsecond boundary mismatch as the same physical boundary.
        if (imu_buffer.front().timestamp >
            start_time + kImuTimeEpsilonSec)
        {
                return false;
        }

        if (imu_buffer.back().timestamp +
                kImuTimeEpsilonSec <
            end_time)
        {
                return false;
        }

        auto start_iter = imu_buffer.begin();

        for (auto iter = imu_buffer.begin();
             iter != imu_buffer.end();
             ++iter)
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

        for (auto iter = start_iter;
             iter != imu_buffer.end();
             ++iter)
        {
                output.push_back(*iter);

                // Prefer a true bracketing sample whenever it exists.
                if (iter->timestamp >= end_time)
                {
                        return true;
                }
        }

        // Only fall back to the tolerance when the newest buffered IMU is
        // microscopically before end_time. ImuIntegrator::Extract() handles
        // this final <=1 us interval by extending the last measurement to the
        // requested endpoint.
        return !output.empty() &&
               output.back().timestamp +
                       kImuTimeEpsilonSec >=
                   end_time;
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
        (void)remove_before_time;

        while (imu_buffer.size() > 1 &&
               imu_buffer[1].timestamp <= timestamp)
        {
                imu_buffer.pop_front();
        }
}
