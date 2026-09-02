#include "fr_slam/fr_lidar_registration.hpp"
#include "fr_slam/fr_lidar_preprocessor.hpp"
#include "fr_slam/fr_point_types.hpp"
#include "fr_slam/fr_lidar_frame.hpp"

#include "fr_slam/fr_mid360s_adapter.hpp"
#include "fr_slam/fr_imu_adapter.hpp"

#include "fr_slam/fr_imu_buffer.hpp"
#include "fr_slam/fr_imu_initializer.hpp"
#include "fr_slam/fr_imu_integrator.hpp"
#include "fr_slam/fr_imu_types.hpp"

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/executors/multi_threaded_executor.hpp>

#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include <pcl/point_cloud.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

class TestLidarRegistrationOdometry : public rclcpp::Node
{
private:
    // ============================================================
    // A LiDAR frame waiting for enough IMU data.
    // ============================================================

    struct PendingLidarFrame
    {
        LIDAR_FRAME frame;

        builtin_interfaces::msg::Time stamp;

        std::string frame_id;
    };

    enum class ImuBuildStatus
    {
        SUCCESS,

        // IMU has not reached the end of this LiDAR scan yet.
        // Keep the LiDAR frame in the queue and try again later.
        WAIT_FOR_IMU,

        // This LiDAR frame can no longer be processed correctly.
        // Drop only this frame and continue with the next one.
        DROP_FRAME
    };

    // ============================================================
    // ROS callback groups
    // ============================================================

    rclcpp::CallbackGroup::SharedPtr
        imu_callback_group_;

    rclcpp::CallbackGroup::SharedPtr
        lidar_callback_group_;

    // ============================================================
    // ROS
    // ============================================================

    rclcpp::Subscription<
        sensor_msgs::msg::Imu>::SharedPtr imu_sub_;

    rclcpp::Subscription<
        sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;

    rclcpp::Publisher<
        nav_msgs::msg::Odometry>::SharedPtr odom_pub_;

    rclcpp::Publisher<
        nav_msgs::msg::Path>::SharedPtr path_pub_;

    // ============================================================
    // Sensor adapters
    // ============================================================

    ImuAdapter imu_adapter_;

    Mid360s_Adapter lidar_adapter_;

    // ============================================================
    // IMU
    // ============================================================

    IMUbuffer imu_buffer_;

    ImuInitializer imu_initializer_;

    ImuIntegrator imu_integrator_;

    IMU_STATE current_imu_state_;

    std::atomic<bool> imu_initialized_{
        false};

    std::vector<IMU_DATA>
        initialization_imu_;

    std::size_t initialization_sample_count_ =
        200;

    // Keep some old IMU samples around the current integration state.
    double imu_history_duration_ =
        0.5;

    // ============================================================
    // LiDAR queue + processing thread
    //
    // LiDAR callback only queues frames.
    // The worker waits until IMU covers the entire scan.
    // ============================================================

    std::deque<PendingLidarFrame>
        lidar_queue_;

    std::mutex
        lidar_queue_mutex_;

    std::condition_variable
        data_condition_;

    std::thread
        processing_thread_;

    std::atomic<bool> running_{
        true};

    // ============================================================
    // Preprocessing / Deskew
    // ============================================================

    PreProcessor preprocessor_;

    // ============================================================
    // Registration
    // ============================================================

    LidarRegistration registration_;

    PreparedLidarTarget prepared_target_;

    bool has_target_ =
        false;

    // ============================================================
    // Global LiDAR pose
    //
    // T_WL:
    // LiDAR frame -> World frame
    // ============================================================

    Eigen::Isometry3d T_WL_ =
        Eigen::Isometry3d::Identity();

    // ============================================================
    // Path
    // ============================================================

    nav_msgs::msg::Path path_msg_;

    const std::string world_frame_ =
        "world";

private:
    // ============================================================
    // IMU callback
    //
    // IMPORTANT:
    // This callback does NOT perform deskew or registration.
    // It only:
    //
    //   1. converts IMU
    //   2. pushes it into IMUbuffer
    //   3. initializes IMU at startup
    //   4. wakes the LiDAR processing thread
    // ============================================================

    void ImuCallback(
        const sensor_msgs::msg::Imu::SharedPtr msg)
    {
        const IMU_DATA imu_data =
            imu_adapter_.convert(
                *msg);

        if (!imu_buffer_.Push(
                imu_data))
        {
            RCLCPP_WARN(
                this->get_logger(),
                "IMU push failed. timestamp=%.9f",
                imu_data.timestamp);

            return;
        }

        // A newly arrived IMU sample may make the oldest queued
        // LiDAR frame processable now.
        data_condition_.notify_one();

        RCLCPP_INFO_THROTTLE(
            this->get_logger(),
            *this->get_clock(),
            1000,
            "IMU callback running | buffer=%zu",
            imu_buffer_.imu_deque_size());

        if (imu_initialized_.load())
        {
            return;
        }

        initialization_imu_.push_back(
            imu_data);

        if (initialization_imu_.size() % 20 == 0)
        {
            RCLCPP_INFO(
                this->get_logger(),
                "Collecting initialization IMU: %zu / %zu",
                initialization_imu_.size(),
                initialization_sample_count_);
        }

        if (initialization_imu_.size() <
            initialization_sample_count_)
        {
            return;
        }

        if (!imu_initializer_.Initialize(
                initialization_imu_,
                current_imu_state_))
        {
            RCLCPP_WARN(
                this->get_logger(),
                "IMU initialization failed. "
                "Keep the sensor stationary and retry.");

            initialization_imu_.clear();

            return;
        }

        // Current version uses rotation-only deskew.
        current_imu_state_.P_WI =
            Eigen::Vector3d::Zero();

        current_imu_state_.V_WI =
            Eigen::Vector3d::Zero();

        imu_initialized_.store(
            true);

        data_condition_.notify_all();

        RCLCPP_INFO(
            this->get_logger(),
            "======================================");

        RCLCPP_INFO(
            this->get_logger(),
            "IMU INITIALIZATION SUCCESS");

        RCLCPP_INFO(
            this->get_logger(),
            "Initial timestamp: %.9f",
            current_imu_state_.timestamp);

        RCLCPP_INFO(
            this->get_logger(),
            "Gyro bias: %.6f %.6f %.6f",
            current_imu_state_.gyro_bias.x(),
            current_imu_state_.gyro_bias.y(),
            current_imu_state_.gyro_bias.z());

        RCLCPP_INFO(
            this->get_logger(),
            "Accel bias: %.6f %.6f %.6f",
            current_imu_state_.accel_bias.x(),
            current_imu_state_.accel_bias.y(),
            current_imu_state_.accel_bias.z());

        RCLCPP_INFO(
            this->get_logger(),
            "======================================");
    }

    // ============================================================
    // LiDAR callback
    //
    // IMPORTANT:
    // Do not run Deskew / ICP here.
    //
    // Convert the message and put it into lidar_queue_.
    // ============================================================

    void LidarCallback(
        const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        if (!imu_initialized_.load())
        {
            RCLCPP_INFO_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                1000,
                "Waiting for IMU initialization before queuing LiDAR.");

            return;
        }

        const LIDAR_FRAME raw_frame =
            lidar_adapter_.convert(
                *msg);

        if (!raw_frame.cloud ||
            raw_frame.cloud->empty())
        {
            RCLCPP_WARN(
                this->get_logger(),
                "Converted LiDAR cloud is empty.");

            return;
        }

        PendingLidarFrame pending;

        pending.frame =
            raw_frame;

        pending.stamp =
            msg->header.stamp;

        pending.frame_id =
            msg->header.frame_id;

        std::size_t queue_size =
            0;

        {
            std::lock_guard<std::mutex> lock(
                lidar_queue_mutex_);

            lidar_queue_.push_back(
                std::move(pending));

            queue_size =
                lidar_queue_.size();
        }

        data_condition_.notify_one();

        RCLCPP_INFO(
            this->get_logger(),
            "LiDAR queued | scan_start=%.9f points=%zu queue=%zu",
            raw_frame.scan_start_time,
            raw_frame.cloud->size(),
            queue_size);
    }

    // ============================================================
    // Find actual LiDAR point-time range
    // ============================================================

    bool FindPointTimeRange(
        const LIDAR_FRAME &lidar_frame,
        double &min_point_time,
        double &max_point_time) const
    {
        if (!lidar_frame.cloud ||
            lidar_frame.cloud->empty())
        {
            return false;
        }

        min_point_time =
            std::numeric_limits<double>::max();

        max_point_time =
            std::numeric_limits<double>::lowest();

        for (const LIDAR_POINT &point :
             lidar_frame.cloud->points)
        {
            const double point_time =
                lidar_frame.scan_start_time +
                point.time_offset;

            min_point_time =
                std::min(
                    min_point_time,
                    point_time);

            max_point_time =
                std::max(
                    max_point_time,
                    point_time);
        }

        return true;
    }

    // ============================================================
    // Interpolate an IMU_STATE from the integrated IMU trajectory.
    //
    // Biases / gravity are copied from base_state.
    //
    // This is used to keep the persistent IMU state at the current
    // LiDAR scan START, not at scan END.
    //
    // Why?
    //
    // If we store the state at previous scan_end, the next Livox
    // scan_start can be a few hundred microseconds earlier than that
    // end time. Then the next deskew trajectory no longer covers its
    // own scan start.
    //
    // Keeping state at each scan_start avoids that problem.
    // ============================================================

    bool InterpolateStateFromPoses(
        const std::vector<IMU_POSE> &imu_poses,
        double timestamp,
        const IMU_STATE &base_state,
        IMU_STATE &output_state) const
    {
        if (imu_poses.size() < 2)
        {
            return false;
        }

        if (timestamp < imu_poses.front().timestamp ||
            timestamp > imu_poses.back().timestamp)
        {
            return false;
        }

        for (std::size_t i = 0;
             i + 1 < imu_poses.size();
             ++i)
        {
            const IMU_POSE &pose_a =
                imu_poses[i];

            const IMU_POSE &pose_b =
                imu_poses[i + 1];

            if (pose_a.timestamp <= timestamp &&
                timestamp <= pose_b.timestamp)
            {
                const double dt =
                    pose_b.timestamp -
                    pose_a.timestamp;

                if (dt <= 0.0)
                {
                    return false;
                }

                const double alpha =
                    (timestamp - pose_a.timestamp) /
                    dt;

                output_state =
                    base_state;

                output_state.timestamp =
                    timestamp;

                output_state.Q_WI =
                    pose_a.Q_WI.slerp(
                        alpha,
                        pose_b.Q_WI);

                output_state.Q_WI.normalize();

                output_state.P_WI =
                    (1.0 - alpha) * pose_a.P_WI +
                    alpha * pose_b.P_WI;

                output_state.V_WI =
                    (1.0 - alpha) * pose_a.V_WI +
                    alpha * pose_b.V_WI;

                return true;
            }
        }

        return false;
    }

    // ============================================================
    // Build IMU trajectory for one LiDAR frame.
    //
    // The important behavior here is:
    //
    // GetTimeRange() false because IMU has not reached scan_end yet
    //      -> WAIT_FOR_IMU
    //      -> DO NOT drop the LiDAR frame
    //
    // Once a future IMU sample arrives:
    //
    //      GetTimeRange()
    //      -> Extract()
    //      -> exact start/end interpolation
    //      -> Integrate()
    //      -> Deskew
    // ============================================================

    ImuBuildStatus BuildImuPoses(
        const LIDAR_FRAME &lidar_frame,
        std::vector<IMU_POSE> &imu_poses,
        IMU_STATE &state_at_scan_start)
    {
        if (!imu_initialized_.load())
        {
            return ImuBuildStatus::WAIT_FOR_IMU;
        }

        if (!lidar_frame.has_point_time)
        {
            RCLCPP_WARN(
                this->get_logger(),
                "LiDAR frame has no point time.");

            return ImuBuildStatus::DROP_FRAME;
        }

        double min_point_time =
            0.0;

        double max_point_time =
            0.0;

        if (!FindPointTimeRange(
                lidar_frame,
                min_point_time,
                max_point_time))
        {
            return ImuBuildStatus::DROP_FRAME;
        }

        const double required_start_time =
            std::min(
                lidar_frame.scan_start_time,
                min_point_time);

        const double required_end_time =
            std::max(
                lidar_frame.scan_start_time,
                max_point_time);

        // --------------------------------------------------------
        // current_imu_state_ is intentionally kept at the previous
        // successfully processed LiDAR scan START.
        //
        // Therefore it should normally be earlier than this scan.
        //
        // The first LiDAR frame immediately after IMU initialization
        // may have started before the initialization timestamp.
        // That frame cannot be reconstructed backwards, so drop it.
        // --------------------------------------------------------

        constexpr double time_epsilon =
            1e-6;

        if (current_imu_state_.timestamp >
            required_start_time + time_epsilon)
        {
            RCLCPP_WARN(
                this->get_logger(),
                "Dropping stale LiDAR frame | "
                "IMU state=%.9f required_start=%.9f",
                current_imu_state_.timestamp,
                required_start_time);

            return ImuBuildStatus::DROP_FRAME;
        }

        // --------------------------------------------------------
        // 1. Ask the IMU buffer for data that brackets the complete
        //    interval.
        //
        // GetTimeRange() already checks:
        //
        // imu_buffer.front <= start_time
        // imu_buffer.back  >= end_time
        //
        // Therefore if the newest IMU has not passed scan_end yet,
        // this simply returns false.
        //
        // IMPORTANT:
        // false here means WAIT, not DROP.
        // --------------------------------------------------------

        std::vector<IMU_DATA> raw_imu;

        if (!imu_buffer_.GetTimeRange(
                current_imu_state_.timestamp,
                required_end_time,
                raw_imu))
        {
            return ImuBuildStatus::WAIT_FOR_IMU;
        }

        // --------------------------------------------------------
        // 2. Exact interpolation at:
        //
        // current_imu_state_.timestamp
        // required_end_time
        //
        // This uses your existing InterpolateImu() through Extract().
        // --------------------------------------------------------

        std::vector<IMU_DATA> exact_imu;

        if (!imu_integrator_.Extract(
                raw_imu,
                current_imu_state_.timestamp,
                required_end_time,
                exact_imu))
        {
            RCLCPP_WARN(
                this->get_logger(),
                "IMU Extract failed for LiDAR frame.");

            return ImuBuildStatus::DROP_FRAME;
        }

        // --------------------------------------------------------
        // 3. Integrate once from the stored state to scan end.
        //
        // The trajectory can begin before the current LiDAR scan.
        // That is completely fine:
        //
        // LidarDeskewer only requires that the trajectory covers
        // every LiDAR point time.
        // --------------------------------------------------------

        IMU_STATE state_at_scan_end;

        if (!imu_integrator_.Integrate(
                exact_imu,
                current_imu_state_,
                imu_poses,
                state_at_scan_end))
        {
            RCLCPP_WARN(
                this->get_logger(),
                "IMU Integrate failed for LiDAR frame.");

            return ImuBuildStatus::DROP_FRAME;
        }

        if (imu_poses.size() < 2)
        {
            return ImuBuildStatus::DROP_FRAME;
        }

        // --------------------------------------------------------
        // 4. Compute the IMU state exactly at LiDAR scan_start.
        //
        // We will keep THIS as current_imu_state_ after processing.
        // We deliberately do NOT keep state_at_scan_end.
        // --------------------------------------------------------

        if (!InterpolateStateFromPoses(
                imu_poses,
                lidar_frame.scan_start_time,
                current_imu_state_,
                state_at_scan_start))
        {
            RCLCPP_WARN(
                this->get_logger(),
                "Failed to interpolate IMU state at LiDAR scan start.");

            return ImuBuildStatus::DROP_FRAME;
        }

        // Rotation-only deskew:
        state_at_scan_start.P_WI =
            Eigen::Vector3d::Zero();

        state_at_scan_start.V_WI =
            Eigen::Vector3d::Zero();

        RCLCPP_INFO(
            this->get_logger(),
            "IMU ready | raw=%zu exact=%zu poses=%zu | "
            "state=%.9f scan=[%.9f -> %.9f]",
            raw_imu.size(),
            exact_imu.size(),
            imu_poses.size(),
            current_imu_state_.timestamp,
            required_start_time,
            required_end_time);

        return ImuBuildStatus::SUCCESS;
    }

    // ============================================================
    // Publish current LiDAR pose
    // ============================================================

    void PublishPose(
        const builtin_interfaces::msg::Time &stamp,
        const std::string &lidar_frame)
    {
        const Eigen::Vector3d P_WL =
            T_WL_.translation();

        Eigen::Quaterniond Q_WL(
            T_WL_.rotation());

        Q_WL.normalize();

        nav_msgs::msg::Odometry odom_msg;

        odom_msg.header.stamp =
            stamp;

        odom_msg.header.frame_id =
            world_frame_;

        odom_msg.child_frame_id =
            lidar_frame;

        odom_msg.pose.pose.position.x =
            P_WL.x();

        odom_msg.pose.pose.position.y =
            P_WL.y();

        odom_msg.pose.pose.position.z =
            P_WL.z();

        odom_msg.pose.pose.orientation.w =
            Q_WL.w();

        odom_msg.pose.pose.orientation.x =
            Q_WL.x();

        odom_msg.pose.pose.orientation.y =
            Q_WL.y();

        odom_msg.pose.pose.orientation.z =
            Q_WL.z();

        odom_pub_->publish(
            odom_msg);

        geometry_msgs::msg::PoseStamped pose_msg;

        pose_msg.header.stamp =
            stamp;

        pose_msg.header.frame_id =
            world_frame_;

        pose_msg.pose.position.x =
            P_WL.x();

        pose_msg.pose.position.y =
            P_WL.y();

        pose_msg.pose.position.z =
            P_WL.z();

        pose_msg.pose.orientation.w =
            Q_WL.w();

        pose_msg.pose.orientation.x =
            Q_WL.x();

        pose_msg.pose.orientation.y =
            Q_WL.y();

        pose_msg.pose.orientation.z =
            Q_WL.z();

        path_msg_.header.stamp =
            stamp;

        path_msg_.header.frame_id =
            world_frame_;

        path_msg_.poses.push_back(
            pose_msg);

        path_pub_->publish(
            path_msg_);
    }

    // ============================================================
    // Process one LiDAR frame after its IMU trajectory is ready.
    // ============================================================

    void ProcessLidarFrame(
        const PendingLidarFrame &pending,
        const std::vector<IMU_POSE> &imu_poses,
        const IMU_STATE &state_at_scan_start)
    {
        const LIDAR_FRAME &raw_frame =
            pending.frame;

        // ========================================================
        // 1. Deskew + filter + voxel
        //
        // false = rotation-only deskew
        // ========================================================

        const LIDAR_FRAME registration_frame =
            preprocessor_.Process(
                raw_frame,
                imu_poses,
                false);

        if (!registration_frame.cloud ||
            registration_frame.cloud->empty())
        {
            RCLCPP_WARN(
                this->get_logger(),
                "PreProcessor::Process failed.");

            return;
        }

        const pcl::PointCloud<LIDAR_POINT>::Ptr current_cloud =
            registration_frame.cloud;

        RCLCPP_INFO(
            this->get_logger(),
            "Cloud | raw=%zu registration=%zu",
            raw_frame.cloud->size(),
            current_cloud->size());

        // ========================================================
        // 2. Advance persistent IMU state to THIS scan start.
        //
        // Do this after deskew succeeds.
        // ========================================================

        current_imu_state_ =
            state_at_scan_start;

        // Current IMUbuffer::RemoveOldData() removes data before the
        // timestamp passed to it. Passing (state - history) retains
        // the requested amount of history with your current function.
        imu_buffer_.RemoveOldData(
            current_imu_state_.timestamp -
            imu_history_duration_);

        // ========================================================
        // 3. First processed frame
        // ========================================================

        if (!has_target_)
        {
            T_WL_.setIdentity();

            if (!registration_.PrepareTarget(
                    current_cloud,
                    prepared_target_))
            {
                RCLCPP_WARN(
                    this->get_logger(),
                    "Failed to prepare first target.");

                return;
            }

            has_target_ =
                true;

            PublishPose(
                pending.stamp,
                pending.frame_id);

            RCLCPP_INFO(
                this->get_logger(),
                "First DESKEWED LiDAR frame initialized.");

            return;
        }

        // ========================================================
        // 4. Scan-to-scan registration
        // ========================================================

        const Eigen::Isometry3d initial_guess =
            Eigen::Isometry3d::Identity();

        LidarRegistrationResult result;

        const bool success =
            registration_.Align(
                current_cloud,
                prepared_target_,
                initial_guess,
                result);

        if (!success)
        {
            RCLCPP_WARN(
                this->get_logger(),
                "LiDAR registration failed.");

            return;
        }

        const Eigen::Isometry3d T_previous_current =
            result.T_target_source;

        T_WL_ =
            T_WL_ *
            T_previous_current;

        PublishPose(
            pending.stamp,
            pending.frame_id);

        const Eigen::Vector3d position =
            T_WL_.translation();

        RCLCPP_INFO(
            this->get_logger(),
            "LiDAR pose | "
            "x=%.4f y=%.4f z=%.4f | "
            "corr=%zu rmse=%.6f converged=%s",
            position.x(),
            position.y(),
            position.z(),
            result.correspondences,
            result.rmse,
            result.converged ? "true" : "false");

        // ========================================================
        // 5. Current frame becomes next target
        // ========================================================

        PreparedLidarTarget next_target;

        if (!registration_.PrepareTarget(
                current_cloud,
                next_target))
        {
            RCLCPP_WARN(
                this->get_logger(),
                "Failed to prepare next target.");

            return;
        }

        prepared_target_ =
            std::move(
                next_target);
    }

    // ============================================================
    // LiDAR worker
    //
    // This is the main fix.
    //
    // The oldest LiDAR frame stays at queue.front() while the IMU
    // buffer is still too short.
    //
    // No LiDAR frame is discarded merely because its future IMU
    // sample has not arrived yet.
    // ============================================================

    void ProcessingLoop()
    {
        while (running_.load())
        {
            PendingLidarFrame pending;

            // ----------------------------------------------------
            // Wait until at least one LiDAR frame exists.
            // ----------------------------------------------------

            {
                std::unique_lock<std::mutex> lock(
                    lidar_queue_mutex_);

                data_condition_.wait(
                    lock,
                    [this]()
                    {
                        return !running_.load() ||
                               !lidar_queue_.empty();
                    });

                if (!running_.load())
                {
                    break;
                }

                // Copy the oldest frame, but DO NOT pop it yet.
                pending =
                    lidar_queue_.front();
            }

            // ----------------------------------------------------
            // Try to build the IMU trajectory.
            // ----------------------------------------------------

            std::vector<IMU_POSE> imu_poses;

            IMU_STATE state_at_scan_start;

            const ImuBuildStatus status =
                BuildImuPoses(
                    pending.frame,
                    imu_poses,
                    state_at_scan_start);

            // ----------------------------------------------------
            // IMU has not passed this scan's end time yet.
            //
            // Keep the frame at queue.front().
            // Wait for another IMU callback to notify us.
            // ----------------------------------------------------

            if (status ==
                ImuBuildStatus::WAIT_FOR_IMU)
            {
                std::unique_lock<std::mutex> lock(
                    lidar_queue_mutex_);

                data_condition_.wait_for(
                    lock,
                    std::chrono::milliseconds(
                        5));

                continue;
            }

            // ----------------------------------------------------
            // Hard failure / stale frame:
            // remove only this one frame.
            // ----------------------------------------------------

            if (status ==
                ImuBuildStatus::DROP_FRAME)
            {
                std::lock_guard<std::mutex> lock(
                    lidar_queue_mutex_);

                if (!lidar_queue_.empty())
                {
                    lidar_queue_.pop_front();
                }

                continue;
            }

            // ----------------------------------------------------
            // IMU is ready. Process this exact oldest frame.
            // ----------------------------------------------------

            ProcessLidarFrame(
                pending,
                imu_poses,
                state_at_scan_start);

            // ----------------------------------------------------
            // Processing finished. Now remove the frame.
            // ----------------------------------------------------

            std::size_t queue_size =
                0;

            {
                std::lock_guard<std::mutex> lock(
                    lidar_queue_mutex_);

                if (!lidar_queue_.empty())
                {
                    lidar_queue_.pop_front();
                }

                queue_size =
                    lidar_queue_.size();
            }

            RCLCPP_INFO(
                this->get_logger(),
                "LiDAR frame finished | remaining_queue=%zu",
                queue_size);
        }
    }

public:
    explicit TestLidarRegistrationOdometry(
        const std::string &node_name)
        : rclcpp::Node(
              node_name)
    {
        // ========================================================
        // Callback groups
        // ========================================================

        imu_callback_group_ =
            this->create_callback_group(
                rclcpp::CallbackGroupType::MutuallyExclusive);

        lidar_callback_group_ =
            this->create_callback_group(
                rclcpp::CallbackGroupType::MutuallyExclusive);

        // ========================================================
        // IMU subscriber
        // ========================================================

        rclcpp::SubscriptionOptions imu_options;

        imu_options.callback_group =
            imu_callback_group_;

        imu_sub_ =
            this->create_subscription<
                sensor_msgs::msg::Imu>(
                "/livox/imu",
                rclcpp::SensorDataQoS(),
                std::bind(
                    &TestLidarRegistrationOdometry::ImuCallback,
                    this,
                    std::placeholders::_1),
                imu_options);

        // ========================================================
        // LiDAR subscriber
        // ========================================================

        rclcpp::SubscriptionOptions lidar_options;

        lidar_options.callback_group =
            lidar_callback_group_;

        lidar_sub_ =
            this->create_subscription<
                sensor_msgs::msg::PointCloud2>(
                "/livox/lidar",
                rclcpp::SensorDataQoS(),
                std::bind(
                    &TestLidarRegistrationOdometry::LidarCallback,
                    this,
                    std::placeholders::_1),
                lidar_options);

        // ========================================================
        // LiDAR -> IMU extrinsic
        //
        // Keep Identity for the current test.
        // Replace with calibrated T_IL later.
        // ========================================================

        const Eigen::Quaterniond Q_IL =
            Eigen::Quaterniond::Identity();

        const Eigen::Vector3d P_IL =
            Eigen::Vector3d::Zero();

        preprocessor_.SetDeskewExtrinsic(
            Q_IL,
            P_IL);

        // ========================================================
        // Publishers
        // ========================================================

        odom_pub_ =
            this->create_publisher<
                nav_msgs::msg::Odometry>(
                "/lidar_odometry",
                10);

        path_pub_ =
            this->create_publisher<
                nav_msgs::msg::Path>(
                "/lidar_path",
                10);

        path_msg_.header.frame_id =
            world_frame_;

        // ========================================================
        // Start LiDAR processing worker
        // ========================================================

        processing_thread_ =
            std::thread(
                &TestLidarRegistrationOdometry::ProcessingLoop,
                this);

        RCLCPP_INFO(
            this->get_logger(),
            "======================================");

        RCLCPP_INFO(
            this->get_logger(),
            "LiDAR odometry WITH queued deskew started.");

        RCLCPP_INFO(
            this->get_logger(),
            "IMU   : /livox/imu");

        RCLCPP_INFO(
            this->get_logger(),
            "LiDAR : /livox/lidar");

        RCLCPP_INFO(
            this->get_logger(),
            "LiDAR callback -> queue");

        RCLCPP_INFO(
            this->get_logger(),
            "Worker -> wait IMU -> Extract/interpolate -> "
            "Integrate -> Deskew -> Registration");

        RCLCPP_INFO(
            this->get_logger(),
            "Deskew mode: ROTATION ONLY");

        RCLCPP_INFO(
            this->get_logger(),
            "======================================");
    }

    ~TestLidarRegistrationOdometry() override
    {
        running_.store(
            false);

        data_condition_.notify_all();

        if (processing_thread_.joinable())
        {
            processing_thread_.join();
        }
    }
};

int main(
    int argc,
    char *argv[])
{
    rclcpp::init(
        argc,
        argv);

    const std::shared_ptr<
        TestLidarRegistrationOdometry>
        node =
            std::make_shared<
                TestLidarRegistrationOdometry>(
                "test_lidar_registration_odometry");

    // Callbacks are lightweight now, but two threads keep high-rate
    // IMU reception independent from LiDAR message conversion.
    rclcpp::executors::MultiThreadedExecutor executor(
        rclcpp::ExecutorOptions(),
        2);

    executor.add_node(
        node);

    executor.spin();

    rclcpp::shutdown();

    return 0;
}