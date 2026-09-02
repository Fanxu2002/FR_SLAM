#include "fr_slam/fr_imu_integrator.hpp"
#include <Eigen/Geometry>
#include <Eigen/Core>

#include <cmath>

namespace
{
constexpr double kImuTimeEpsilonSec =
    1.0e-6;
}

bool ImuIntegrator::InterpolateImu(const IMU_DATA &imu_a,
                                   const IMU_DATA &imu_b,
                                   double timestamp,
                                   IMU_DATA &output)
{
        if (imu_a.timestamp >= imu_b.timestamp)
        {
                return false;
        }
        if (timestamp < imu_a.timestamp || timestamp > imu_b.timestamp)
        {
                return false;
        }
        const double alpha = (timestamp - imu_a.timestamp) / (imu_b.timestamp - imu_a.timestamp);

        output.timestamp = timestamp;

        output.gyro = (1.0 - alpha) * imu_a.gyro + alpha * imu_b.gyro;

        output.accelerometer = (1.0 - alpha) * imu_a.accelerometer + alpha * imu_b.accelerometer;
        return true;
}

bool ImuIntegrator::Extract(const std::vector<IMU_DATA> &raw_imu,
                            double start_time,
                            double end_time,
                            std::vector<IMU_DATA> &extract_imu)
{
        extract_imu.clear();

        if (raw_imu.size() < 2)
        {
                return false;
        }

        if (start_time >= end_time)
        {
                return false;
        }

        if (raw_imu.front().timestamp >
            start_time + kImuTimeEpsilonSec)
        {
                return false;
        }

        if (raw_imu.back().timestamp +
                kImuTimeEpsilonSec <
            end_time)
        {
                return false;
        }

        bool start_found = false;

        // If the requested start is only a floating-point ULP before the
        // first IMU sample, extend that first measurement back by <=1 us.
        // This avoids rejecting a physically identical timestamp boundary.
        if (start_time < raw_imu.front().timestamp &&
            raw_imu.front().timestamp - start_time <=
                kImuTimeEpsilonSec)
        {
                IMU_DATA start_imu =
                    raw_imu.front();

                start_imu.timestamp =
                    start_time;

                extract_imu.push_back(
                    start_imu);

                start_found = true;
        }
        else
        {
                for (std::size_t t = 0;
                     t + 1 < raw_imu.size();
                     ++t)
                {
                        const IMU_DATA &imu_a = raw_imu[t];
                        const IMU_DATA &imu_b = raw_imu[t + 1];

                        if ((imu_a.timestamp <= start_time) &&
                            (start_time <= imu_b.timestamp))
                        {
                                IMU_DATA start_imu;
                                if (!InterpolateImu(imu_a,
                                                    imu_b,
                                                    start_time,
                                                    start_imu))
                                {
                                        return false;
                                }
                                extract_imu.push_back(start_imu);
                                start_found = true;
                                break;
                        }
                }
        }

        if (!start_found)
        {
                return false;
        }

        for (const auto &imu : raw_imu)
        {
                if (imu.timestamp > start_time &&
                    imu.timestamp < end_time)
                {
                        extract_imu.push_back(imu);
                }
        }

        bool end_found = false;

        // Normal case: interpolate with a real IMU sample on both sides.
        for (std::size_t i = 0;
             i + 1 < raw_imu.size();
             ++i)
        {
                const IMU_DATA &imu_a = raw_imu[i];
                const IMU_DATA &imu_b = raw_imu[i + 1];

                if (imu_a.timestamp <= end_time &&
                    end_time <= imu_b.timestamp)
                {
                        IMU_DATA end_imu;

                        if (!InterpolateImu(
                                imu_a,
                                imu_b,
                                end_time,
                                end_imu))
                        {
                                return false;
                        }

                        extract_imu.push_back(end_imu);

                        end_found = true;

                        break;
                }
        }

        // Floating-point boundary fallback. In the observed failure the gap
        // was 0.238 us, exactly one double-precision step at the epoch-sized
        // timestamp. Holding gyro/accel constant for <=1 us is negligible
        // compared with a ~5 ms IMU period and keeps the deskew trajectory
        // covering the exact LiDAR endpoint.
        if (!end_found &&
            end_time > raw_imu.back().timestamp &&
            end_time - raw_imu.back().timestamp <=
                kImuTimeEpsilonSec)
        {
                IMU_DATA end_imu =
                    raw_imu.back();

                end_imu.timestamp =
                    end_time;

                extract_imu.push_back(
                    end_imu);

                end_found = true;
        }

        if (!end_found)
        {
                return false;
        }

        return true;
}

bool ImuIntegrator::Integrate(const std::vector<IMU_DATA> &imu_data,
                              const IMU_STATE &initial_state,
                              std::vector<IMU_POSE> &imu_poses,
                              IMU_STATE &final_state)
{
        imu_poses.clear();
        if (imu_data.size() < 2)
        {
                return false;
        }
        const double start_time_error = std::abs(imu_data.front().timestamp - initial_state.timestamp);
        if (start_time_error > 1e-6)
        {
                return false;
        }
        IMU_STATE current_state = initial_state;

        IMU_POSE initial_pose;
        initial_pose.Q_WI = current_state.Q_WI;
        initial_pose.P_WI = current_state.P_WI;
        initial_pose.V_WI = current_state.V_WI;
        initial_pose.timestamp = current_state.timestamp;
        imu_poses.push_back(initial_pose);

        for (std::size_t k = 0; k + 1 < imu_data.size(); ++k)
        {
                const IMU_DATA &imu_k = imu_data[k];
                const IMU_DATA &imu_next = imu_data[k + 1];
                const double dt = imu_next.timestamp - imu_k.timestamp;
                if (dt <= 0)
                {
                        return false;
                }

                const Eigen::Vector3d omega_k = imu_k.gyro - current_state.gyro_bias;

                const Eigen::Vector3d omega_next = imu_next.gyro - current_state.gyro_bias;

                const Eigen::Vector3d omega_mid = 0.5 * (omega_k + omega_next);

                const Eigen::Vector3d delta_theta = omega_mid * dt;

                const double delta_angle = delta_theta.norm();

                Eigen::Quaterniond delta_q = Eigen::Quaterniond::Identity();

                if (delta_angle > 1e-12)
                {
                        const Eigen::Vector3d rotation_axis = delta_theta / delta_angle;

                        delta_q =
                            Eigen::Quaterniond(
                                Eigen::AngleAxisd(
                                    delta_angle,
                                    rotation_axis));
                }

                Eigen::Quaterniond Q_next = current_state.Q_WI * delta_q;

                Q_next.normalize();

                const Eigen::Vector3d accel_k_I = imu_k.accelerometer - current_state.accel_bias;

                const Eigen::Vector3d accel_next_I = imu_next.accelerometer - current_state.accel_bias;

                const Eigen::Vector3d accel_k_W = current_state.Q_WI * accel_k_I + current_state.gravity_W;

                const Eigen::Vector3d accel_next_W = Q_next * accel_next_I + current_state.gravity_W;

                const Eigen::Vector3d accel_mid_W = 0.5 * (accel_k_W + accel_next_W);

                const Eigen::Vector3d P_next = current_state.P_WI + current_state.V_WI * dt + 0.5 * accel_mid_W * dt * dt;

                const Eigen::Vector3d V_next = current_state.V_WI + accel_mid_W * dt;

                current_state.timestamp = imu_next.timestamp;

                current_state.Q_WI = Q_next;

                current_state.P_WI = P_next;

                current_state.V_WI = V_next;

                IMU_POSE pose;

                pose.timestamp = current_state.timestamp;

                pose.Q_WI = current_state.Q_WI;

                pose.P_WI = current_state.P_WI;

                pose.V_WI = current_state.V_WI;

                imu_poses.push_back(
                    pose);
        }
        final_state = current_state;
        return true;
}
