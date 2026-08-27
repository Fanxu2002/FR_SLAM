#include <rclcpp/rclcpp.hpp>
#include <rclcpp/executors/multi_threaded_executor.hpp>

#include "fr_slam/fr_lidar_frame.hpp"
#include "fr_slam/fr_lidar_preprocessor.hpp"
#include "fr_slam/fr_point_types.hpp"
#include "fr_slam/fr_registration_scan2localmap.hpp"

#include "fr_slam/mid360s_adapter.hpp"
#include "fr_slam/imu_adapter.hpp"

#include "fr_slam/fr_imu_buffer.hpp"
#include "fr_slam/fr_imu_initializer.hpp"
#include "fr_slam/fr_imu_integrator.hpp"
#include "fr_slam/fr_imu_types.hpp"

#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <geometry_msgs/msg/pose_stamped.hpp>

#include <builtin_interfaces/msg/time.hpp>

#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

class lidar_registration_scan2localmap : public rclcpp::Node
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

    // ============================================================
    // Result of building IMU trajectory for one LiDAR frame.
    // ============================================================
    enum class ImuBuildStatus
    {
        SUCCESS,

        // Future IMU has not reached the LiDAR scan end yet.
        WAIT_FOR_IMU,

        // This LiDAR frame can no longer be processed correctly.
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
    // ROS subscribers
    // ============================================================
    rclcpp::Subscription<
        sensor_msgs::msg::Imu>::SharedPtr
        imu_sub_;

    rclcpp::Subscription<
        sensor_msgs::msg::PointCloud2>::SharedPtr
        lidar_sub_;

    // ============================================================
    // ROS publishers
    // ============================================================
    rclcpp::Publisher<
        nav_msgs::msg::Path>::SharedPtr
        path_pub_;

    // Keyframe odometry-only trajectory reconstructed from PoseGraph
    // odometry measurements. This remains the "before loop correction"
    // reference even after G2O overwrites PoseGraph node estimates.
    rclcpp::Publisher<
        nav_msgs::msg::Path>::SharedPtr
        pose_graph_before_path_pub_;

    // Current optimized Keyframe trajectory from PoseGraphNode::T_WK.
    rclcpp::Publisher<
        nav_msgs::msg::Path>::SharedPtr
        optimized_path_pub_;

    rclcpp::Publisher<
        nav_msgs::msg::Odometry>::SharedPtr
        odom_pub_;

    // Real-time pose represented in the corrected backend/map coordinates:
    //
    //     T_map_L = T_map_odom * T_odom_L(raw frontend)
    //
    // Before the first loop correction this is identical to /lidar_odometry.
    rclcpp::Publisher<
        nav_msgs::msg::Odometry>::SharedPtr
        corrected_odom_pub_;

    rclcpp::Publisher<
        sensor_msgs::msg::PointCloud2>::SharedPtr
        local_map_pub_;

    // PoseGraph global-map snapshots.  QoS is transient-local so RViz can be
    // opened after loop closure and still receive the latest heavy map once.
    rclcpp::Publisher<
        sensor_msgs::msg::PointCloud2>::SharedPtr
        raw_keyframe_map_pub_;

    rclcpp::Publisher<
        sensor_msgs::msg::PointCloud2>::SharedPtr
        optimized_map_pub_;

    // Post-PGO local geometric refinement.  Uses the same transient-local QoS
    // as the raw / optimized snapshots so RViz can subscribe after closure.
    rclcpp::Publisher<
        sensor_msgs::msg::PointCloud2>::SharedPtr
        refined_map_pub_;

    // Debug clouds for the latest accepted Post-PGO local refinement window.
    // Use separate RViz PointCloud2 displays (prefer Fixed Color) to compare
    // historical geometry and the same current window before/after refinement.
    rclcpp::Publisher<
        sensor_msgs::msg::PointCloud2>::SharedPtr
        refinement_historical_target_pub_;

    rclcpp::Publisher<
        sensor_msgs::msg::PointCloud2>::SharedPtr
        refinement_current_before_pub_;

    rclcpp::Publisher<
        sensor_msgs::msg::PointCloud2>::SharedPtr
        refinement_current_after_pub_;

    rclcpp::Publisher<
        visualization_msgs::msg::MarkerArray>::SharedPtr
        pose_graph_marker_pub_;

    // ============================================================
    // Sensor adapters
    // ============================================================
    Mid360s_Adapter
        lidar_adapter_;

    ImuAdapter
        imu_adapter_;

    // ============================================================
    // IMU modules
    // ============================================================
    IMUbuffer
        imu_buffer_;

    ImuInitializer
        imu_initializer_;

    ImuIntegrator
        imu_integrator_;

    // ============================================================
    // Persistent IMU state used by the Deskew integration chain.
    //
    // IMPORTANT:
    //
    // This state advances with LiDAR TIME even if LiDAR registration
    // later rejects a frame.
    //
    // It is NOT the accepted SLAM pose.
    // ============================================================
    IMU_STATE
    current_imu_state_;

    std::atomic<bool>
        imu_initialized_{
            false};

    std::vector<IMU_DATA>
        initialization_imu_;

    std::size_t
        initialization_sample_count_ =
            200;

    // Keep a little IMU history because consecutive Livox scans can
    // overlap slightly in time.
    double
        imu_history_duration_ =
            0.5;

    // ============================================================
    // IMU orientation associated with the LAST ACCEPTED LiDAR pose.
    //
    // This is the key state for IMU initial-guess prediction.
    //
    // current_imu_state_:
    //      advances every processed LiDAR scan for Deskew.
    //
    // last_accepted_Q_WI_:
    //      advances ONLY when Scan-to-LocalMap accepts a frame.
    //
    // Therefore, after one or more rejected LiDAR frames, the IMU
    // rotation prediction automatically spans from the last accepted
    // LiDAR frame all the way to the current frame.
    // ============================================================
    Eigen::Quaterniond
        last_accepted_Q_WI_ =
            Eigen::Quaterniond::Identity();

    bool
        has_last_accepted_imu_orientation_ =
            false;

    // ============================================================
    // LiDAR -> IMU extrinsic.
    //
    // Convention used here:
    //
    //      p_I = R_IL * p_L + P_IL
    //
    // Current temporary test values:
    //
    //      R_IL = Identity
    //      P_IL = Zero
    //
    // Replace with calibrated extrinsic later.
    // ============================================================
    Eigen::Quaterniond
        Q_IL_ =
            Eigen::Quaterniond::Identity();

    Eigen::Vector3d
        P_IL_ =
            Eigen::Vector3d::Zero();

    // ============================================================
    // LiDAR queue + worker thread
    //
    // LiDAR callback:
    //      convert -> queue
    //
    // Worker:
    //      wait IMU -> build poses -> Deskew -> registration
    // ============================================================
    std::deque<PendingLidarFrame>
        lidar_queue_;

    std::mutex
        lidar_queue_mutex_;

    std::condition_variable
        data_condition_;

    std::thread
        processing_thread_;

    std::atomic<bool>
        running_{
            true};

    // ============================================================
    // Real-time overload protection.
    //
    // IMPORTANT:
    // The worker keeps queue.front() until processing is finished.
    // Therefore the callback NEVER removes queue.front(). When the
    // queue is full it removes the second-oldest waiting frame instead.
    // This preserves the frame currently owned by the worker.
    // ============================================================
    std::size_t
        max_lidar_queue_size_ =
            3;

    std::atomic<std::size_t>
        dropped_lidar_frames_{
            0};

    std::atomic<std::size_t>
        processed_lidar_frames_{
            0};

    bool
        preprocessor_enable_sor_ =
            false;

    bool
        preprocessor_enable_ror_ =
            false;

    // ============================================================
    // Preprocessing / rotation-only Deskew
    // ============================================================
    PreProcessor
        preprocessor_;

    // ============================================================
    // Scan-to-LocalMap frontend
    // ============================================================
    std::unique_ptr<RegistrationScan2LocalMap>
        scan_to_local_map_;

    // ============================================================
    // Latest ACCEPTED global LiDAR pose.
    //
    //      p_W = T_WL_ * p_L
    // ============================================================
    Eigen::Isometry3d
        T_WL_ =
            Eigen::Isometry3d::Identity();

    // ============================================================
    // Path
    // ============================================================
    nav_msgs::msg::Path
        path_msg_;

    // Last backend-map revision already converted to PointCloud2 and
    // published.  This prevents expensive million-point conversions on every
    // LiDAR frame after a loop closure.
    std::size_t
        last_published_global_map_revision_ =
            0;

    // Used only for concise diagnostics when a new backend correction arrives.
    std::size_t
        last_seen_map_odom_revision_ =
            0;

    const std::string
        world_frame_ =
            "world";

private:
    // ============================================================
    // IMU callback
    //
    // This callback does NOT run Deskew or registration.
    // It only:
    //
    //      1. converts IMU
    //      2. pushes IMU into buffer
    //      3. initializes IMU at startup
    //      4. wakes the LiDAR worker
    // ============================================================
    void ImuCallback(
        const sensor_msgs::msg::Imu::ConstSharedPtr msg)
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

        // New IMU data may make the oldest LiDAR frame processable.
        data_condition_.notify_one();

        RCLCPP_INFO_THROTTLE(
            this->get_logger(),
            *this->get_clock(),
            1000,
            "IMU callback running | buffer=%zu",
            imu_buffer_.imu_deque_size());

        // ========================================================
        // IMU already initialized
        // ========================================================
        if (imu_initialized_.load())
        {
            return;
        }

        // ========================================================
        // Collect stationary initialization samples
        // ========================================================
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

        // ========================================================
        // IMU initialization
        // ========================================================
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

        // ========================================================
        // Rotation-only Deskew
        //
        // We deliberately do NOT trust raw accelerometer double
        // integration for translation prediction yet.
        // ========================================================
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
            "Deskew mode: ROTATION ONLY");

        RCLCPP_INFO(
            this->get_logger(),
            "Initial guess rotation: IMU");

        RCLCPP_INFO(
            this->get_logger(),
            "Initial guess translation: LiDAR constant motion");

        RCLCPP_INFO(
            this->get_logger(),
            "======================================");
    }

    // ============================================================
    // LiDAR callback
    //
    // This callback only converts and queues LiDAR.
    // ============================================================
    void LidarCallback(
        const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
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

        std::size_t dropped_now =
            0;

        {
            std::lock_guard<std::mutex> lock(
                lidar_queue_mutex_);

            // ----------------------------------------------------
            // Bounded queue.
            //
            // queue.front() may already be copied by ProcessingLoop()
            // and still be under processing. Never erase it here.
            // Remove the second-oldest waiting frame instead, keeping
            // the newest measurements available for catch-up.
            // ----------------------------------------------------
            while (lidar_queue_.size() >=
                   max_lidar_queue_size_)
            {
                if (lidar_queue_.size() <= 1)
                {
                    break;
                }

                lidar_queue_.erase(
                    lidar_queue_.begin() + 1);

                ++dropped_now;
            }

            lidar_queue_.push_back(
                std::move(pending));

            queue_size =
                lidar_queue_.size();
        }

        if (dropped_now > 0)
        {
            dropped_lidar_frames_.fetch_add(
                dropped_now);

            RCLCPP_WARN_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                1000,
                "LiDAR worker overloaded | "
                "dropped_waiting=%zu total_dropped=%zu queue=%zu max_queue=%zu",
                dropped_now,
                dropped_lidar_frames_.load(),
                queue_size,
                max_lidar_queue_size_);
        }

        data_condition_.notify_one();

        RCLCPP_INFO_THROTTLE(
            this->get_logger(),
            *this->get_clock(),
            500,
            "LiDAR queued | "
            "scan_start=%.9f points=%zu queue=%zu dropped_total=%zu",
            raw_frame.scan_start_time,
            raw_frame.cloud->size(),
            queue_size,
            dropped_lidar_frames_.load());
    }

    // ============================================================
    // Find actual point-time range of a LiDAR scan.
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
    // Interpolate IMU state from integrated IMU poses.
    //
    // We use this to obtain the IMU orientation exactly at the
    // LiDAR scan start time.
    // ============================================================
    bool InterpolateStateFromPoses(
        const std::vector<IMU_POSE> &imu_poses,
        const double timestamp,
        const IMU_STATE &base_state,
        IMU_STATE &output_state) const
    {
        if (imu_poses.size() < 2)
        {
            return false;
        }

        if (timestamp <
                imu_poses.front().timestamp ||
            timestamp >
                imu_poses.back().timestamp)
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
                    (timestamp -
                     pose_a.timestamp) /
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
                    (1.0 - alpha) *
                        pose_a.P_WI +
                    alpha *
                        pose_b.P_WI;

                output_state.V_WI =
                    (1.0 - alpha) *
                        pose_a.V_WI +
                    alpha *
                        pose_b.V_WI;

                return true;
            }
        }

        return false;
    }

    // ============================================================
    // Build IMU trajectory covering one LiDAR scan.
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

        // ========================================================
        // Persistent IMU state must not already be newer than the
        // interval that this LiDAR scan needs.
        // ========================================================
        constexpr double time_epsilon =
            1.0e-6;

        if (current_imu_state_.timestamp >
            required_start_time +
                time_epsilon)
        {
            RCLCPP_WARN(
                this->get_logger(),
                "Dropping stale LiDAR frame | "
                "IMU state=%.9f required_start=%.9f",
                current_imu_state_.timestamp,
                required_start_time);

            return ImuBuildStatus::DROP_FRAME;
        }

        // ========================================================
        // Wait until IMU covers the entire LiDAR scan.
        // ========================================================
        std::vector<IMU_DATA>
            raw_imu;

        if (!imu_buffer_.GetTimeRange(
                current_imu_state_.timestamp,
                required_end_time,
                raw_imu))
        {
            return ImuBuildStatus::WAIT_FOR_IMU;
        }

        // ========================================================
        // Extract exact start/end IMU samples.
        // ========================================================
        std::vector<IMU_DATA>
            exact_imu;

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

        // ========================================================
        // Integrate IMU trajectory.
        // ========================================================
        IMU_STATE
        state_at_scan_end;

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
            RCLCPP_WARN(
                this->get_logger(),
                "IMU trajectory has too few poses.");

            return ImuBuildStatus::DROP_FRAME;
        }

        // ========================================================
        // Get IMU state exactly at LiDAR scan start.
        // ========================================================
        if (!InterpolateStateFromPoses(
                imu_poses,
                lidar_frame.scan_start_time,
                current_imu_state_,
                state_at_scan_start))
        {
            RCLCPP_WARN(
                this->get_logger(),
                "Failed to interpolate IMU state "
                "at LiDAR scan start.");

            return ImuBuildStatus::DROP_FRAME;
        }

        // ========================================================
        // Rotation-only mode
        // ========================================================
        state_at_scan_start.P_WI =
            Eigen::Vector3d::Zero();

        state_at_scan_start.V_WI =
            Eigen::Vector3d::Zero();

        RCLCPP_INFO(
            this->get_logger(),
            "IMU ready | "
            "raw=%zu exact=%zu poses=%zu | "
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
    // Build relative LiDAR rotation prediction from IMU orientation.
    //
    // Q_WI convention:
    //      IMU -> World
    //
    // Q_IL convention:
    //      LiDAR -> IMU
    //
    // First:
    //
    //      R_Iprev_Icurr
    //          =
    //      R_WI_prev^T * R_WI_current
    //
    // This maps CURRENT IMU -> PREVIOUS ACCEPTED IMU.
    //
    // Convert to LiDAR coordinates:
    //
    //      R_Lprev_Lcurr
    //          =
    //      R_IL^T * R_Iprev_Icurr * R_IL
    // ============================================================
    bool BuildImuRelativeLidarRotation(
        const Eigen::Quaterniond &current_Q_WI,
        Eigen::Quaterniond &Q_Lprev_Lcurr) const
    {
        if (!has_last_accepted_imu_orientation_)
        {
            return false;
        }

        if (!current_Q_WI.coeffs().allFinite() ||
            current_Q_WI.norm() <= 1.0e-12 ||
            !last_accepted_Q_WI_.coeffs().allFinite() ||
            last_accepted_Q_WI_.norm() <= 1.0e-12 ||
            !Q_IL_.coeffs().allFinite() ||
            Q_IL_.norm() <= 1.0e-12)
        {
            return false;
        }

        Eigen::Quaterniond Q_WI_previous =
            last_accepted_Q_WI_;

        Eigen::Quaterniond Q_WI_current =
            current_Q_WI;

        Eigen::Quaterniond Q_IL =
            Q_IL_;

        Q_WI_previous.normalize();
        Q_WI_current.normalize();
        Q_IL.normalize();

        // Current IMU -> previous accepted IMU.
        Eigen::Quaterniond Q_Iprev_Icurr =
            Q_WI_previous.conjugate() *
            Q_WI_current;

        Q_Iprev_Icurr.normalize();

        // Current LiDAR -> previous accepted LiDAR.
        Q_Lprev_Lcurr =
            Q_IL.conjugate() *
            Q_Iprev_Icurr *
            Q_IL;

        Q_Lprev_Lcurr.normalize();

        return Q_Lprev_Lcurr.coeffs().allFinite();
    }

    // ============================================================
    // Publish current accepted LiDAR pose.
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

        // ========================================================
        // Odometry
        // ========================================================
        nav_msgs::msg::Odometry
            odom_msg;

        odom_msg.header.frame_id =
            world_frame_;

        odom_msg.child_frame_id =
            lidar_frame;

        odom_msg.header.stamp =
            stamp;

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

        // ========================================================
        // Path
        // ========================================================
        geometry_msgs::msg::PoseStamped
            pose_msg;

        pose_msg.header.frame_id =
            world_frame_;

        pose_msg.header.stamp =
            stamp;

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
    // Publish the CURRENT accepted LiDAR pose in corrected backend/map
    // coordinates, without touching the live frontend state.
    //
    //     raw:       T_odom_L = T_WL_
    //     corrected: T_map_L  = T_map_odom * T_odom_L
    //
    // There is deliberately no dense corrected ordinary-scan Path here.
    // Historical graph correction is non-rigid along the trajectory, so one
    // single latest T_map_odom should not be retroactively applied to every old
    // ordinary scan.  /optimized_path remains the correct historical backend
    // Keyframe trajectory.
    // ============================================================
    void PublishCorrectedPose(
        const builtin_interfaces::msg::Time &stamp,
        const std::string &lidar_frame)
    {
        if (!scan_to_local_map_ ||
            !corrected_odom_pub_)
        {
            return;
        }

        const Eigen::Isometry3d T_corrected =
            scan_to_local_map_->GetCorrectedPose(
                T_WL_);

        if (!T_corrected.matrix().allFinite())
        {
            return;
        }

        Eigen::Quaterniond Q_corrected(
            T_corrected.rotation());

        if (!Q_corrected.coeffs().allFinite() ||
            Q_corrected.norm() <= 1.0e-12)
        {
            return;
        }

        Q_corrected.normalize();

        nav_msgs::msg::Odometry corrected_odom_msg;

        // Keep the existing world frame label in this bridge-only V2 so the
        // new topic can be overlaid immediately with current RViz settings.
        // The numeric pose is in the corrected backend/map coordinates.
        // A later TF integration step can split this formally into map/odom.
        corrected_odom_msg.header.frame_id =
            world_frame_;

        corrected_odom_msg.child_frame_id =
            lidar_frame;

        corrected_odom_msg.header.stamp =
            stamp;

        corrected_odom_msg.pose.pose.position.x =
            T_corrected.translation().x();

        corrected_odom_msg.pose.pose.position.y =
            T_corrected.translation().y();

        corrected_odom_msg.pose.pose.position.z =
            T_corrected.translation().z();

        corrected_odom_msg.pose.pose.orientation.w =
            Q_corrected.w();

        corrected_odom_msg.pose.pose.orientation.x =
            Q_corrected.x();

        corrected_odom_msg.pose.pose.orientation.y =
            Q_corrected.y();

        corrected_odom_msg.pose.pose.orientation.z =
            Q_corrected.z();

        corrected_odom_pub_->publish(
            corrected_odom_msg);

        const std::size_t map_odom_revision =
            scan_to_local_map_->MapOdomRevision();

        if (map_odom_revision != 0 &&
            map_odom_revision != last_seen_map_odom_revision_)
        {
            const Eigen::Isometry3d T_map_odom =
                scan_to_local_map_->GetMapOdomCorrection();

            RCLCPP_INFO(
                this->get_logger(),
                "Corrected real-time pose bridge active | revision=%zu | "
                "T_map_odom_translation=[%.4f %.4f %.4f] | "
                "current_corrected=[%.4f %.4f %.4f]",
                map_odom_revision,
                T_map_odom.translation().x(),
                T_map_odom.translation().y(),
                T_map_odom.translation().z(),
                T_corrected.translation().x(),
                T_corrected.translation().y(),
                T_corrected.translation().z());

            last_seen_map_odom_revision_ =
                map_odom_revision;
        }
    }

    // ============================================================
    // Convert one SE(3) pose into a PoseStamped for path display.
    // ============================================================
    geometry_msgs::msg::PoseStamped MakePoseStamped(
        const Eigen::Isometry3d &T_WK,
        const builtin_interfaces::msg::Time &stamp) const
    {
        geometry_msgs::msg::PoseStamped pose_msg;

        pose_msg.header.frame_id =
            world_frame_;

        pose_msg.header.stamp =
            stamp;

        pose_msg.pose.position.x =
            T_WK.translation().x();

        pose_msg.pose.position.y =
            T_WK.translation().y();

        pose_msg.pose.position.z =
            T_WK.translation().z();

        Eigen::Quaterniond quaternion(
            T_WK.rotation());

        if (quaternion.coeffs().allFinite() &&
            quaternion.norm() > 1.0e-12)
        {
            quaternion.normalize();
        }
        else
        {
            quaternion =
                Eigen::Quaterniond::Identity();
        }

        pose_msg.pose.orientation.x =
            quaternion.x();

        pose_msg.pose.orientation.y =
            quaternion.y();

        pose_msg.pose.orientation.z =
            quaternion.z();

        pose_msg.pose.orientation.w =
            quaternion.w();

        return pose_msg;
    }

    // ============================================================
    // Reconstruct the Keyframe trajectory BEFORE loop correction.
    //
    // IMPORTANT:
    //
    // PoseGraph node estimates T_WK are overwritten by G2O, so reading
    // node.T_WK after optimization cannot recover the pre-optimization
    // trajectory.
    //
    // However, odometry edge measurements are NOT changed by G2O:
    //
    //     Z_ij = T_Ki_Kj = X_i^-1 * X_j
    //
    // therefore:
    //
    //     X_j = X_i * Z_ij
    //
    // Starting from the fixed root node and integrating ONLY odometry
    // edges reconstructs the odometry-only Keyframe path. Loop edges are
    // deliberately ignored.
    //
    // This gives a fair Keyframe-to-Keyframe comparison:
    //
    //     /pose_graph_before_path : odometry-only
    //     /optimized_path         : current G2O estimates
    // ============================================================
    bool BuildOdometryOnlyKeyframePoses(
        const PoseGraph &graph,
        std::unordered_map<
            std::size_t,
            Eigen::Isometry3d> &poses) const
    {
        poses.clear();

        const std::vector<PoseGraphNode> &nodes =
            graph.GetNodes();

        if (nodes.empty())
        {
            return false;
        }

        const PoseGraphNode *root_node =
            nullptr;

        for (const PoseGraphNode &node :
             nodes)
        {
            if (node.fixed)
            {
                root_node =
                    &node;

                break;
            }
        }

        if (root_node == nullptr)
        {
            root_node =
                &nodes.front();
        }

        if (!root_node->T_WK
                 .matrix()
                 .allFinite())
        {
            return false;
        }

        poses.emplace(
            root_node->id,
            root_node->T_WK);

        // The odometry edges form a chain in normal operation, but the
        // repeated-pass implementation also works if edge insertion order
        // is not strictly from old to new.
        for (std::size_t pass = 0;
             pass < nodes.size();
             ++pass)
        {
            bool made_progress =
                false;

            for (const PoseGraphEdge &edge :
                 graph.GetEdges())
            {
                if (edge.type !=
                    PoseGraphEdgeType::Odometry)
                {
                    continue;
                }

                if (!edge.T_from_to
                         .matrix()
                         .allFinite())
                {
                    continue;
                }

                const auto from_iterator =
                    poses.find(
                        edge.from_id);

                const auto to_iterator =
                    poses.find(
                        edge.to_id);

                // X_to = X_from * Z_from_to
                if (from_iterator != poses.end() &&
                    to_iterator == poses.end())
                {
                    const Eigen::Isometry3d T_W_to =
                        from_iterator->second *
                        edge.T_from_to;

                    if (T_W_to
                            .matrix()
                            .allFinite())
                    {
                        poses.emplace(
                            edge.to_id,
                            T_W_to);

                        made_progress =
                            true;
                    }

                    continue;
                }

                // Reverse propagation, useful if the fixed node is not
                // the first node in edge insertion order:
                //
                // X_from = X_to * Z_from_to^-1
                if (from_iterator == poses.end() &&
                    to_iterator != poses.end())
                {
                    const Eigen::Isometry3d T_W_from =
                        to_iterator->second *
                        edge.T_from_to.inverse();

                    if (T_W_from
                            .matrix()
                            .allFinite())
                    {
                        poses.emplace(
                            edge.from_id,
                            T_W_from);

                        made_progress =
                            true;
                    }
                }
            }

            if (!made_progress)
            {
                break;
            }
        }

        return !poses.empty();
    }

    // ============================================================
    // Publish fair "before vs after" Keyframe paths.
    //
    // /pose_graph_before_path:
    //     odometry-only path reconstructed from odometry edges.
    //
    // /optimized_path:
    //     current PoseGraph node estimates after G2O.
    //
    // Before the first accepted loop these two paths should nearly
    // overlap. After global optimization they separate and directly show
    // how much the graph was corrected.
    // ============================================================
    void PublishPoseGraphPaths(
        const builtin_interfaces::msg::Time &stamp)
    {
        if (!scan_to_local_map_ ||
            !pose_graph_before_path_pub_ ||
            !optimized_path_pub_)
        {
            return;
        }

        const PoseGraph &graph =
            scan_to_local_map_->GetPoseGraph();

        const std::vector<PoseGraphNode> &nodes =
            graph.GetNodes();

        if (nodes.empty())
        {
            return;
        }

        std::unordered_map<
            std::size_t,
            Eigen::Isometry3d>
            before_poses;

        if (!BuildOdometryOnlyKeyframePoses(
                graph,
                before_poses))
        {
            return;
        }

        nav_msgs::msg::Path before_path;
        nav_msgs::msg::Path optimized_path;

        before_path.header.frame_id =
            world_frame_;

        before_path.header.stamp =
            stamp;

        optimized_path.header.frame_id =
            world_frame_;

        optimized_path.header.stamp =
            stamp;

        before_path.poses.reserve(
            nodes.size());

        optimized_path.poses.reserve(
            nodes.size());

        for (const PoseGraphNode &node :
             nodes)
        {
            const auto before_iterator =
                before_poses.find(
                    node.id);

            if (before_iterator !=
                before_poses.end())
            {
                before_path.poses.push_back(
                    MakePoseStamped(
                        before_iterator->second,
                        stamp));
            }

            if (node.T_WK
                    .matrix()
                    .allFinite())
            {
                optimized_path.poses.push_back(
                    MakePoseStamped(
                        node.T_WK,
                        stamp));
            }
        }

        if (!before_path.poses.empty())
        {
            pose_graph_before_path_pub_->publish(
                before_path);
        }

        if (!optimized_path.poses.empty())
        {
            optimized_path_pub_->publish(
                optimized_path);
        }
    }

    void PublishPoseGraph(
        const builtin_interfaces::msg::Time &stamp)
    {
        if (!scan_to_local_map_ ||
            !pose_graph_marker_pub_)
        {
            return;
        }

        const PoseGraph &graph =
            scan_to_local_map_->GetPoseGraph();

        visualization_msgs::msg::MarkerArray
            marker_array;

        // ============================================================
        // 1. Keyframe PoseGraph nodes
        // ============================================================
        visualization_msgs::msg::Marker
            node_marker;

        node_marker.header.frame_id =
            world_frame_;

        node_marker.header.stamp =
            stamp;

        node_marker.ns =
            "pose_graph_keyframe_nodes";

        node_marker.id =
            0;

        node_marker.type =
            visualization_msgs::msg::Marker::SPHERE_LIST;

        node_marker.action =
            visualization_msgs::msg::Marker::ADD;

        node_marker.pose.orientation.w =
            1.0;

        node_marker.scale.x =
            0.16;

        node_marker.scale.y =
            0.16;

        node_marker.scale.z =
            0.16;

        node_marker.color.r =
            0.1f;

        node_marker.color.g =
            0.8f;

        node_marker.color.b =
            1.0f;

        node_marker.color.a =
            1.0f;

        for (const PoseGraphNode &node :
             graph.GetNodes())
        {
            geometry_msgs::msg::Point point;

            point.x =
                node.T_WK.translation().x();

            point.y =
                node.T_WK.translation().y();

            point.z =
                node.T_WK.translation().z();

            node_marker.points.push_back(
                point);
        }

        marker_array.markers.push_back(
            node_marker);

        // ============================================================
        // 2. Odometry edges
        // ============================================================
        visualization_msgs::msg::Marker
            odom_edge_marker;

        odom_edge_marker.header.frame_id =
            world_frame_;

        odom_edge_marker.header.stamp =
            stamp;

        odom_edge_marker.ns =
            "pose_graph_odom_edges";

        odom_edge_marker.id =
            1;

        odom_edge_marker.type =
            visualization_msgs::msg::Marker::LINE_LIST;

        odom_edge_marker.action =
            visualization_msgs::msg::Marker::ADD;

        odom_edge_marker.pose.orientation.w =
            1.0;

        odom_edge_marker.scale.x =
            0.04;

        odom_edge_marker.color.r =
            0.6f;

        odom_edge_marker.color.g =
            0.6f;

        odom_edge_marker.color.b =
            0.6f;

        odom_edge_marker.color.a =
            0.8f;

        // ============================================================
        // 3. Loop edges
        // ============================================================
        visualization_msgs::msg::Marker
            loop_edge_marker;

        loop_edge_marker.header.frame_id =
            world_frame_;

        loop_edge_marker.header.stamp =
            stamp;

        loop_edge_marker.ns =
            "pose_graph_loop_edges";

        loop_edge_marker.id =
            2;

        loop_edge_marker.type =
            visualization_msgs::msg::Marker::LINE_LIST;

        loop_edge_marker.action =
            visualization_msgs::msg::Marker::ADD;

        loop_edge_marker.pose.orientation.w =
            1.0;

        loop_edge_marker.scale.x =
            0.18;

        loop_edge_marker.color.r =
            1.0f;

        loop_edge_marker.color.g =
            0.0f;

        loop_edge_marker.color.b =
            0.0f;

        loop_edge_marker.color.a =
            1.0f;

        // ============================================================
        // 4. Read edges
        // ============================================================
        for (const PoseGraphEdge &edge :
             graph.GetEdges())
        {
            const PoseGraphNode *from_node =
                graph.GetNode(
                    edge.from_id);

            const PoseGraphNode *to_node =
                graph.GetNode(
                    edge.to_id);

            if (from_node == nullptr ||
                to_node == nullptr)
            {
                continue;
            }

            geometry_msgs::msg::Point
                from_point;

            geometry_msgs::msg::Point
                to_point;

            from_point.x =
                from_node->T_WK.translation().x();

            from_point.y =
                from_node->T_WK.translation().y();

            from_point.z =
                from_node->T_WK.translation().z();

            to_point.x =
                to_node->T_WK.translation().x();

            to_point.y =
                to_node->T_WK.translation().y();

            to_point.z =
                to_node->T_WK.translation().z();

            if (edge.type ==
                PoseGraphEdgeType::Odometry)
            {
                odom_edge_marker.points.push_back(
                    from_point);

                odom_edge_marker.points.push_back(
                    to_point);
            }
            else if (edge.type ==
                     PoseGraphEdgeType::Loop)
            {
                loop_edge_marker.points.push_back(
                    from_point);

                loop_edge_marker.points.push_back(
                    to_point);
            }
        }

        marker_array.markers.push_back(
            odom_edge_marker);

        marker_array.markers.push_back(
            loop_edge_marker);

        // ============================================================
        // 5. Optimized Keyframe orientation arrows
        //
        // Draw one arrow every 20 Keyframes.
        //
        // The arrow shows the Keyframe local +X axis expressed in
        // World coordinates:
        //
        //     direction_W = R_WK * [1, 0, 0]^T
        //
        // IMPORTANT:
        // visualization_msgs::msg::Marker has ARROW, but there is no
        // ARROW_LIST type. Therefore each Keyframe orientation is one
        // individual ARROW marker inside the MarkerArray.
        // ============================================================
        constexpr std::size_t orientation_interval =
            20;

        constexpr double orientation_arrow_length =
            0.80;

        for (const PoseGraphNode &node :
             graph.GetNodes())
        {
            if (node.id % orientation_interval != 0)
            {
                continue;
            }

            if (!node.T_WK.matrix().allFinite())
            {
                continue;
            }

            if (node.id >
                static_cast<std::size_t>(
                    std::numeric_limits<int>::max()))
            {
                continue;
            }

            const Eigen::Vector3d arrow_start =
                node.T_WK.translation();

            Eigen::Vector3d arrow_direction =
                node.T_WK.rotation() *
                Eigen::Vector3d::UnitX();

            if (!arrow_direction.allFinite() ||
                arrow_direction.norm() <= 1.0e-12)
            {
                continue;
            }

            arrow_direction.normalize();

            const Eigen::Vector3d arrow_end =
                arrow_start +
                orientation_arrow_length *
                    arrow_direction;

            visualization_msgs::msg::Marker
                orientation_arrow;

            orientation_arrow.header.frame_id =
                world_frame_;

            orientation_arrow.header.stamp =
                stamp;

            orientation_arrow.ns =
                "pose_graph_keyframe_orientation";

            orientation_arrow.id =
                static_cast<int>(
                    node.id);

            orientation_arrow.type =
                visualization_msgs::msg::Marker::ARROW;

            orientation_arrow.action =
                visualization_msgs::msg::Marker::ADD;

            orientation_arrow.pose.orientation.w =
                1.0;

            // For ARROW markers defined by two points:
            // scale.x = shaft diameter
            // scale.y = head diameter
            // scale.z = head length
            orientation_arrow.scale.x =
                0.04;

            orientation_arrow.scale.y =
                0.10;

            orientation_arrow.scale.z =
                0.14;

            orientation_arrow.color.r =
                1.0f;

            orientation_arrow.color.g =
                1.0f;

            orientation_arrow.color.b =
                0.0f;

            orientation_arrow.color.a =
                1.0f;

            geometry_msgs::msg::Point
                start_point;

            start_point.x =
                arrow_start.x();

            start_point.y =
                arrow_start.y();

            start_point.z =
                arrow_start.z();

            geometry_msgs::msg::Point
                end_point;

            end_point.x =
                arrow_end.x();

            end_point.y =
                arrow_end.y();

            end_point.z =
                arrow_end.z();

            orientation_arrow.points.push_back(
                start_point);

            orientation_arrow.points.push_back(
                end_point);

            marker_array.markers.push_back(
                orientation_arrow);
        }

        // ============================================================
        // 6. Before -> After correction vectors
        //
        // One vector every 20 Keyframes:
        //
        //     start = odometry-only Keyframe position
        //     end   = optimized Keyframe position
        //
        // They are nearly invisible before loop closure and become visible
        // after G2O, showing where and how strongly the graph moved.
        // ============================================================
        std::unordered_map<
            std::size_t,
            Eigen::Isometry3d>
            before_poses;

        if (BuildOdometryOnlyKeyframePoses(
                graph,
                before_poses))
        {
            visualization_msgs::msg::Marker
                correction_marker;

            correction_marker.header.frame_id =
                world_frame_;

            correction_marker.header.stamp =
                stamp;

            correction_marker.ns =
                "pose_graph_before_after_corrections";

            correction_marker.id =
                0;

            correction_marker.type =
                visualization_msgs::msg::Marker::LINE_LIST;

            correction_marker.action =
                visualization_msgs::msg::Marker::ADD;

            correction_marker.pose.orientation.w =
                1.0;

            correction_marker.scale.x =
                0.035;

            correction_marker.color.r =
                1.0f;

            correction_marker.color.g =
                0.2f;

            correction_marker.color.b =
                1.0f;

            correction_marker.color.a =
                0.75f;

            constexpr std::size_t correction_interval =
                20;

            for (const PoseGraphNode &node :
                 graph.GetNodes())
            {
                if (node.id % correction_interval != 0)
                {
                    continue;
                }

                const auto before_iterator =
                    before_poses.find(
                        node.id);

                if (before_iterator ==
                    before_poses.end())
                {
                    continue;
                }

                if (!node.T_WK
                         .matrix()
                         .allFinite())
                {
                    continue;
                }

                const Eigen::Vector3d before_position =
                    before_iterator->second.translation();

                const Eigen::Vector3d after_position =
                    node.T_WK.translation();

                geometry_msgs::msg::Point before_point;
                geometry_msgs::msg::Point after_point;

                before_point.x =
                    before_position.x();

                before_point.y =
                    before_position.y();

                before_point.z =
                    before_position.z();

                after_point.x =
                    after_position.x();

                after_point.y =
                    after_position.y();

                after_point.z =
                    after_position.z();

                correction_marker.points.push_back(
                    before_point);

                correction_marker.points.push_back(
                    after_point);
            }

            marker_array.markers.push_back(
                correction_marker);
        }

        pose_graph_marker_pub_->publish(
            marker_array);
    }

    // ============================================================
    // Publish LocalMap.
    //
    // LocalMap points are already in World coordinates.
    // ============================================================
    void PublishLocalMap(
        const builtin_interfaces::msg::Time &stamp)
    {
        if (!scan_to_local_map_)
        {
            return;
        }

        const pcl::PointCloud<LIDAR_POINT>::ConstPtr local_map =
            scan_to_local_map_->GetLocalMap();

        if (!local_map ||
            local_map->empty())
        {
            return;
        }

        sensor_msgs::msg::PointCloud2
            map_msg;

        pcl::toROSMsg(
            *local_map,
            map_msg);

        map_msg.header.stamp =
            stamp;

        map_msg.header.frame_id =
            world_frame_;

        local_map_pub_->publish(
            map_msg);
    }

    // ============================================================
    // Publish raw / optimized global Keyframe maps.
    //
    // RegistrationScan2LocalMap increments GlobalMapRevision() whenever the
    // backend incremental map cache changes: normally on each new Keyframe and
    // again after a PoseGraph correction.  This function still does almost
    // nothing on ordinary non-Keyframe LiDAR scans.
    // ============================================================
    void PublishGlobalMapSnapshots(
        const builtin_interfaces::msg::Time &stamp)
    {
        if (!scan_to_local_map_ ||
            !raw_keyframe_map_pub_ ||
            !optimized_map_pub_)
        {
            return;
        }

        const std::size_t revision =
            scan_to_local_map_->GlobalMapRevision();

        if (revision == 0 ||
            revision == last_published_global_map_revision_)
        {
            return;
        }

        const pcl::PointCloud<LIDAR_POINT>::ConstPtr raw_map =
            scan_to_local_map_->GetRawKeyframeMap();

        const pcl::PointCloud<LIDAR_POINT>::ConstPtr optimized_map =
            scan_to_local_map_->GetOptimizedMap();

        const pcl::PointCloud<LIDAR_POINT>::ConstPtr refined_map =
            scan_to_local_map_->GetRefinedMap();

        const std::size_t refined_revision =
            scan_to_local_map_->RefinedMapRevision();

        const pcl::PointCloud<LIDAR_POINT>::ConstPtr refinement_historical_target =
            scan_to_local_map_->GetRefinementHistoricalTarget();

        const pcl::PointCloud<LIDAR_POINT>::ConstPtr refinement_current_before =
            scan_to_local_map_->GetRefinementCurrentBefore();

        const pcl::PointCloud<LIDAR_POINT>::ConstPtr refinement_current_after =
            scan_to_local_map_->GetRefinementCurrentAfter();

        const std::size_t refinement_debug_revision =
            scan_to_local_map_->RefinementDebugRevision();

        if (!raw_map ||
            !optimized_map ||
            raw_map->empty() ||
            optimized_map->empty())
        {
            return;
        }

        sensor_msgs::msg::PointCloud2 raw_map_msg;
        sensor_msgs::msg::PointCloud2 optimized_map_msg;

        pcl::toROSMsg(
            *raw_map,
            raw_map_msg);

        pcl::toROSMsg(
            *optimized_map,
            optimized_map_msg);

        raw_map_msg.header.stamp =
            stamp;

        raw_map_msg.header.frame_id =
            world_frame_;

        optimized_map_msg.header.stamp =
            stamp;

        optimized_map_msg.header.frame_id =
            world_frame_;

        raw_keyframe_map_pub_->publish(
            raw_map_msg);

        optimized_map_pub_->publish(
            optimized_map_msg);

        std::size_t refined_points = 0;

        // Publish /refined_map only when it was built from THIS exact G2O
        // snapshot.  Never replay a stale refined cloud after a later graph
        // optimization.
        if (refined_map_pub_ &&
            refined_map &&
            !refined_map->empty() &&
            refined_revision == revision)
        {
            sensor_msgs::msg::PointCloud2 refined_map_msg;

            pcl::toROSMsg(
                *refined_map,
                refined_map_msg);

            refined_map_msg.header.stamp =
                stamp;

            refined_map_msg.header.frame_id =
                world_frame_;

            refined_map_pub_->publish(
                refined_map_msg);

            refined_points =
                refined_map->size();
        }

        std::size_t debug_historical_points = 0;
        std::size_t debug_before_points = 0;
        std::size_t debug_after_points = 0;

        if (refinement_debug_revision == revision)
        {
            auto publish_debug_cloud =
                [&stamp, this](
                    const pcl::PointCloud<LIDAR_POINT>::ConstPtr &cloud,
                    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &publisher)
            {
                if (!publisher ||
                    !cloud ||
                    cloud->empty())
                {
                    return;
                }

                sensor_msgs::msg::PointCloud2 msg;

                pcl::toROSMsg(
                    *cloud,
                    msg);

                msg.header.stamp =
                    stamp;

                msg.header.frame_id =
                    world_frame_;

                publisher->publish(
                    msg);
            };

            publish_debug_cloud(
                refinement_historical_target,
                refinement_historical_target_pub_);

            publish_debug_cloud(
                refinement_current_before,
                refinement_current_before_pub_);

            publish_debug_cloud(
                refinement_current_after,
                refinement_current_after_pub_);

            if (refinement_historical_target)
            {
                debug_historical_points =
                    refinement_historical_target->size();
            }

            if (refinement_current_before)
            {
                debug_before_points =
                    refinement_current_before->size();
            }

            if (refinement_current_after)
            {
                debug_after_points =
                    refinement_current_after->size();
            }
        }

        last_published_global_map_revision_ =
            revision;

        RCLCPP_INFO(
            this->get_logger(),
            "Global map snapshots published | revision=%zu raw_points=%zu optimized_points=%zu refined_points=%zu refinement_debug=[hist:%zu before:%zu after:%zu]",
            revision,
            raw_map->size(),
            optimized_map->size(),
            refined_points,
            debug_historical_points,
            debug_before_points,
            debug_after_points);
    }

    // ============================================================
    // Process one LiDAR frame after IMU coverage is ready.
    //
    // Complete frontend:
    //
    //      raw LiDAR
    //          ↓
    //      IMU trajectory
    //          ↓
    //      rotation-only Deskew
    //          ↓
    //      filtering + voxel
    //          ↓
    //      IMU rotation initial guess
    //          ↓
    //      Scan-to-LocalMap
    //          ↓
    //      Quality Gate
    //          ↓
    //      accepted T_WL + LocalMap
    // ============================================================
    void ProcessLidarFrame(
        const PendingLidarFrame &pending,
        const std::vector<IMU_POSE> &imu_poses,
        const IMU_STATE &state_at_scan_start)
    {
        using Clock = std::chrono::steady_clock;

        const Clock::time_point frame_start =
            Clock::now();

        const LIDAR_FRAME &raw_frame =
            pending.frame;

        // ========================================================
        // 1. Rotation-only Deskew + preprocessing + voxel
        // ========================================================
        const Clock::time_point preprocess_start =
            Clock::now();

        const LIDAR_FRAME processed_frame =
            preprocessor_.Process(
                raw_frame,
                imu_poses,
                false);

        const Clock::time_point preprocess_end =
            Clock::now();

        const double preprocess_ms =
            std::chrono::duration<double, std::milli>(
                preprocess_end - preprocess_start)
                .count();

        if (!processed_frame.cloud ||
            processed_frame.cloud->empty())
        {
            RCLCPP_WARN(
                this->get_logger(),
                "PreProcessor::Process failed.");

            return;
        }

        const pcl::PointCloud<LIDAR_POINT>::Ptr current_cloud =
            processed_frame.cloud;

        const PreprocessorTiming &preprocess_timing =
            preprocessor_.GetLastTiming();

        RCLCPP_INFO(
            this->get_logger(),
            "DESKEWED cloud | "
            "raw=%zu basic=%zu voxel=%zu final=%zu",
            raw_frame.cloud->size(),
            preprocess_timing.after_basic_points,
            preprocess_timing.after_voxel_points,
            current_cloud->size());

        // ========================================================
        // 2. Advance persistent IMU state for FUTURE Deskew.
        // ========================================================
        current_imu_state_ =
            state_at_scan_start;

        imu_buffer_.RemoveOldData(
            current_imu_state_.timestamp -
            imu_history_duration_);

        // ========================================================
        // 3. Build IMU rotation prediction for Scan-to-LocalMap.
        // ========================================================
        Eigen::Quaterniond imu_relative_rotation =
            Eigen::Quaterniond::Identity();

        const Eigen::Quaterniond *imu_rotation_ptr =
            nullptr;

        if (BuildImuRelativeLidarRotation(
                state_at_scan_start.Q_WI,
                imu_relative_rotation))
        {
            imu_rotation_ptr =
                &imu_relative_rotation;

            const Eigen::AngleAxisd angle_axis(
                imu_relative_rotation);

            const double relative_rotation_deg =
                std::abs(
                    angle_axis.angle()) *
                180.0 /
                3.14159265358979323846;

            RCLCPP_INFO(
                this->get_logger(),
                "IMU initial guess | "
                "relative_rotation=%.3f deg | "
                "translation_source=LiDAR_constant_motion",
                relative_rotation_deg);
        }
        else
        {
            RCLCPP_INFO(
                this->get_logger(),
                "IMU initial guess unavailable | "
                "fall back to LiDAR constant-motion prediction.");
        }

        if (!scan_to_local_map_)
        {
            RCLCPP_ERROR(
                this->get_logger(),
                "Scan-to-LocalMap module is not initialized.");

            return;
        }

        Eigen::Isometry3d T_WL_current =
            Eigen::Isometry3d::Identity();

        LidarRegistrationResult
            result;

        // ========================================================
        // 4. Scan-to-LocalMap + Keyframe / Loop / PoseGraph work.
        // ========================================================
        const Clock::time_point registration_start =
            Clock::now();

        const bool success =
            scan_to_local_map_->AddFrame(
                current_cloud,
                raw_frame.scan_start_time,
                T_WL_current,
                result,
                imu_rotation_ptr);

        const Clock::time_point registration_end =
            Clock::now();

        const double registration_ms =
            std::chrono::duration<double, std::milli>(
                registration_end - registration_start)
                .count();

        if (!success)
        {
            std::size_t queue_size =
                0;

            {
                std::lock_guard<std::mutex> lock(
                    lidar_queue_mutex_);

                queue_size =
                    lidar_queue_.size();
            }

            const Clock::time_point frame_end =
                Clock::now();

            const double total_ms =
                std::chrono::duration<double, std::milli>(
                    frame_end - frame_start)
                    .count();

            RCLCPP_WARN(
                this->get_logger(),
                "Scan-to-LocalMap frame rejected | "
                "corr=%zu rmse=%.6f converged=%s",
                result.correspondences,
                result.rmse,
                result.converged
                    ? "true"
                    : "false");

            RCLCPP_INFO(
                this->get_logger(),
                "Pipeline timing | "
                "deskew=%.2f basic=%.2f voxel=%.2f sor=%.2f ror=%.2f "
                "preprocess=%.2f registration=%.2f publish=0.00 total=%.2f ms | "
                "queue=%zu dropped=%zu",
                preprocess_timing.deskew_ms,
                preprocess_timing.basic_ms,
                preprocess_timing.voxel_ms,
                preprocess_timing.sor_ms,
                preprocess_timing.ror_ms,
                preprocess_ms,
                registration_ms,
                total_ms,
                queue_size,
                dropped_lidar_frames_.load());

            return;
        }

        // ========================================================
        // 5. Accept current global LiDAR pose.
        // ========================================================
        T_WL_ =
            T_WL_current;

        // ========================================================
        // 6. Update IMU orientation associated with accepted scan.
        // ========================================================
        last_accepted_Q_WI_ =
            state_at_scan_start.Q_WI;

        if (last_accepted_Q_WI_.coeffs().allFinite() &&
            last_accepted_Q_WI_.norm() > 1.0e-12)
        {
            last_accepted_Q_WI_.normalize();

            has_last_accepted_imu_orientation_ =
                true;
        }
        else
        {
            RCLCPP_WARN(
                this->get_logger(),
                "Accepted LiDAR frame has invalid IMU orientation; "
                "IMU prediction will be unavailable next frame.");

            has_last_accepted_imu_orientation_ =
                false;
        }

        const Eigen::Vector3d position =
            T_WL_.translation();

        RCLCPP_INFO(
            this->get_logger(),
            "DESKEWED Scan To LocalMap | "
            "x=%.4f y=%.4f z=%.4f | "
            "corr=%zu rmse=%.6f converged=%s | "
            "keyframes=%zu | "
            "map_frames=%zu map_points=%zu | "
            "rotation_guess=%s",
            position.x(),
            position.y(),
            position.z(),
            result.correspondences,
            result.rmse,
            result.converged
                ? "true"
                : "false",
            scan_to_local_map_->KeyframeCount(),
            scan_to_local_map_->LocalMapFrameCount(),
            scan_to_local_map_->LocalMapPointCount(),
            imu_rotation_ptr != nullptr
                ? "IMU"
                : "LiDAR");

        // ========================================================
        // 7. Publish. Measure it because PointCloud2 conversion and RViz
        // publication can also become a real-time bottleneck.
        // ========================================================
        const Clock::time_point publish_start =
            Clock::now();

        PublishPose(
            pending.stamp,
            pending.frame_id);

        PublishCorrectedPose(
            pending.stamp,
            pending.frame_id);

        PublishLocalMap(
            pending.stamp);

        PublishPoseGraph(
            pending.stamp);

        PublishPoseGraphPaths(
            pending.stamp);

        PublishGlobalMapSnapshots(
            pending.stamp);

        const Clock::time_point publish_end =
            Clock::now();

        const double publish_ms =
            std::chrono::duration<double, std::milli>(
                publish_end - publish_start)
                .count();

        const double total_ms =
            std::chrono::duration<double, std::milli>(
                publish_end - frame_start)
                .count();

        std::size_t queue_size =
            0;

        {
            std::lock_guard<std::mutex> lock(
                lidar_queue_mutex_);

            queue_size =
                lidar_queue_.size();
        }

        RCLCPP_INFO(
            this->get_logger(),
            "Pipeline timing | "
            "deskew=%.2f basic=%.2f voxel=%.2f sor=%.2f ror=%.2f "
            "preprocess=%.2f registration=%.2f publish=%.2f total=%.2f ms | "
            "queue=%zu dropped=%zu",
            preprocess_timing.deskew_ms,
            preprocess_timing.basic_ms,
            preprocess_timing.voxel_ms,
            preprocess_timing.sor_ms,
            preprocess_timing.ror_ms,
            preprocess_ms,
            registration_ms,
            publish_ms,
            total_ms,
            queue_size,
            dropped_lidar_frames_.load());
    }

    // ============================================================
    // LiDAR processing worker.
    //
    // Keep queue.front() until IMU covers the complete scan.
    // ============================================================
    void ProcessingLoop()
    {
        while (running_.load())
        {
            PendingLidarFrame
                pending;

            // ====================================================
            // 1. Wait for LiDAR.
            // ====================================================
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

                // Keep queue.front() in the deque until processing ends.
                // LidarCallback() knows this invariant and never erases front().
                pending =
                    lidar_queue_.front();
            }

            // ====================================================
            // 2. Build IMU trajectory.
            // ====================================================
            std::vector<IMU_POSE>
                imu_poses;

            IMU_STATE
            state_at_scan_start;

            const ImuBuildStatus status =
                BuildImuPoses(
                    pending.frame,
                    imu_poses,
                    state_at_scan_start);

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

            if (status ==
                ImuBuildStatus::DROP_FRAME)
            {
                {
                    std::lock_guard<std::mutex> lock(
                        lidar_queue_mutex_);

                    if (!lidar_queue_.empty())
                    {
                        lidar_queue_.pop_front();
                    }
                }

                dropped_lidar_frames_.fetch_add(
                    1);

                continue;
            }

            // ====================================================
            // 3. Process exact oldest frame.
            // ====================================================
            ProcessLidarFrame(
                pending,
                imu_poses,
                state_at_scan_start);

            processed_lidar_frames_.fetch_add(
                1);

            // ====================================================
            // 4. Processing finished: now pop the protected front.
            // ====================================================
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
                "LiDAR frame finished | "
                "remaining_queue=%zu processed=%zu dropped=%zu",
                queue_size,
                processed_lidar_frames_.load(),
                dropped_lidar_frames_.load());
        }
    }

public:
    // ============================================================
    // Constructor
    // ============================================================
    lidar_registration_scan2localmap()
        : rclcpp::Node(
              "scan2local_map")
    {
        // ========================================================
        // 1. Scan-to-LocalMap configuration
        // ========================================================
        LidarRegistrationConfig
            registration_config;

        LocalMapConfig
            local_map_config;

        local_map_config.max_frames =
            10;

        local_map_config.voxel_leaf_size =
            0.30f;

        // ========================================================
        // 2. Create Scan-to-LocalMap module
        // ========================================================
        scan_to_local_map_ =
            std::make_unique<
                RegistrationScan2LocalMap>(
                registration_config,
                local_map_config);

        // ========================================================
        // 3. Real-time pipeline parameters.
        // ========================================================
        const int configured_max_queue_size =
            this->declare_parameter<int>(
                "max_lidar_queue_size",
                3);

        max_lidar_queue_size_ =
            static_cast<std::size_t>(
                std::max(
                    2,
                    configured_max_queue_size));

        preprocessor_enable_sor_ =
            this->declare_parameter<bool>(
                "preprocessor_enable_sor",
                false);

        preprocessor_enable_ror_ =
            this->declare_parameter<bool>(
                "preprocessor_enable_ror",
                false);

        preprocessor_.SetOutlierFiltersEnabled(
            preprocessor_enable_sor_,
            preprocessor_enable_ror_);

        // ========================================================
        // 4. Deskew extrinsic
        //
        // Q_IL_:
        //      LiDAR -> IMU rotation
        //
        // Current test uses Identity. Replace with calibrated
        // extrinsic later.
        // ========================================================
        preprocessor_.SetDeskewExtrinsic(
            Q_IL_,
            P_IL_);

        // ========================================================
        // 4. Callback groups
        // ========================================================
        imu_callback_group_ =
            this->create_callback_group(
                rclcpp::CallbackGroupType::
                    MutuallyExclusive);

        lidar_callback_group_ =
            this->create_callback_group(
                rclcpp::CallbackGroupType::
                    MutuallyExclusive);

        // ========================================================
        // 5. IMU subscriber
        // ========================================================
        rclcpp::SubscriptionOptions
            imu_options;

        imu_options.callback_group =
            imu_callback_group_;

        imu_sub_ =
            this->create_subscription<
                sensor_msgs::msg::Imu>(
                "/livox/imu",
                rclcpp::SensorDataQoS(),
                std::bind(
                    &lidar_registration_scan2localmap::
                        ImuCallback,
                    this,
                    std::placeholders::_1),
                imu_options);

        // ========================================================
        // 6. LiDAR subscriber
        // ========================================================
        rclcpp::SubscriptionOptions
            lidar_options;

        lidar_options.callback_group =
            lidar_callback_group_;

        lidar_sub_ =
            this->create_subscription<
                sensor_msgs::msg::PointCloud2>(
                "/livox/lidar",
                rclcpp::SensorDataQoS(),
                std::bind(
                    &lidar_registration_scan2localmap::
                        LidarCallback,
                    this,
                    std::placeholders::_1),
                lidar_options);

        // ========================================================
        // 7. Publishers
        // ========================================================
        path_pub_ =
            this->create_publisher<
                nav_msgs::msg::Path>(
                "/lidar_path",
                10);

        pose_graph_before_path_pub_ =
            this->create_publisher<
                nav_msgs::msg::Path>(
                "/pose_graph_before_path",
                10);

        optimized_path_pub_ =
            this->create_publisher<
                nav_msgs::msg::Path>(
                "/optimized_path",
                10);

        odom_pub_ =
            this->create_publisher<
                nav_msgs::msg::Odometry>(
                "/lidar_odometry",
                10);

        corrected_odom_pub_ =
            this->create_publisher<
                nav_msgs::msg::Odometry>(
                "/corrected_odometry",
                10);

        local_map_pub_ =
            this->create_publisher<
                sensor_msgs::msg::PointCloud2>(
                "/local_map",
                10);

        rclcpp::QoS global_map_qos(
            rclcpp::KeepLast(1));

        global_map_qos.reliable();
        global_map_qos.transient_local();

        raw_keyframe_map_pub_ =
            this->create_publisher<
                sensor_msgs::msg::PointCloud2>(
                "/raw_keyframe_map",
                global_map_qos);

        optimized_map_pub_ =
            this->create_publisher<
                sensor_msgs::msg::PointCloud2>(
                "/optimized_map",
                global_map_qos);

        refined_map_pub_ =
            this->create_publisher<
                sensor_msgs::msg::PointCloud2>(
                "/refined_map",
                global_map_qos);

        refinement_historical_target_pub_ =
            this->create_publisher<
                sensor_msgs::msg::PointCloud2>(
                "/refinement_historical_target",
                global_map_qos);

        refinement_current_before_pub_ =
            this->create_publisher<
                sensor_msgs::msg::PointCloud2>(
                "/refinement_current_before",
                global_map_qos);

        refinement_current_after_pub_ =
            this->create_publisher<
                sensor_msgs::msg::PointCloud2>(
                "/refinement_current_after",
                global_map_qos);

        pose_graph_marker_pub_ =
            this->create_publisher<
                visualization_msgs::msg::MarkerArray>(
                "/pose_graph_markers",
                10);

        path_msg_.header.frame_id =
            world_frame_;

        // ========================================================
        // 8. Start worker
        // ========================================================
        processing_thread_ =
            std::thread(
                &lidar_registration_scan2localmap::
                    ProcessingLoop,
                this);

        // ========================================================
        // 9. Startup information
        // ========================================================
        RCLCPP_INFO(
            this->get_logger(),
            "==============================================");

        RCLCPP_INFO(
            this->get_logger(),
            "Scan-to-LocalMap WITH IMU rotation prediction started.");

        RCLCPP_INFO(
            this->get_logger(),
            "IMU   : /livox/imu");

        RCLCPP_INFO(
            this->get_logger(),
            "LiDAR : /livox/lidar");

        RCLCPP_INFO(
            this->get_logger(),
            "Deskew: ROTATION ONLY");

        RCLCPP_INFO(
            this->get_logger(),
            "Initial guess rotation: IMU relative rotation");

        RCLCPP_INFO(
            this->get_logger(),
            "Initial guess translation: previous LiDAR relative motion");

        RCLCPP_INFO(
            this->get_logger(),
            "Pipeline:");

        RCLCPP_INFO(
            this->get_logger(),
            "LiDAR callback -> queue");

        RCLCPP_INFO(
            this->get_logger(),
            "Worker -> wait IMU -> Extract -> Integrate");

        RCLCPP_INFO(
            this->get_logger(),
            "-> rotation-only Deskew -> Basic Filter -> Voxel -> optional SOR/ROR");

        RCLCPP_INFO(
            this->get_logger(),
            "-> IMU rotation prediction -> Scan-to-LocalMap");

        RCLCPP_INFO(
            this->get_logger(),
            "-> Quality Gate -> KeyframeDetector");

        RCLCPP_INFO(
            this->get_logger(),
            "-> KeyframeManager -> LocalMap");

        RCLCPP_INFO(
            this->get_logger(),
            "Odom     : /lidar_odometry");

        RCLCPP_INFO(
            this->get_logger(),
            "CorrOdom : /corrected_odometry");

        RCLCPP_INFO(
            this->get_logger(),
            "Path     : /lidar_path");

        RCLCPP_INFO(
            this->get_logger(),
            "Before   : /pose_graph_before_path");

        RCLCPP_INFO(
            this->get_logger(),
            "Optimized: /optimized_path");

        RCLCPP_INFO(
            this->get_logger(),
            "LocalMap : /local_map");

        RCLCPP_INFO(
            this->get_logger(),
            "RawMap   : /raw_keyframe_map");

        RCLCPP_INFO(
            this->get_logger(),
            "OptMap   : /optimized_map");

        RCLCPP_INFO(
            this->get_logger(),
            "Refined  : /refined_map");

        RCLCPP_INFO(
            this->get_logger(),
            "GlobalMap: incremental backend Keyframe blocks (10 KF/block, 0.30 m block voxel)");

        RCLCPP_INFO(
            this->get_logger(),
            "PoseGraph: Gravity Guard + 2-edge first-loop batch + XY shape guard");

        RCLCPP_INFO(
            this->get_logger(),
            "RefDbg H : /refinement_historical_target");

        RCLCPP_INFO(
            this->get_logger(),
            "RefDbg B : /refinement_current_before");

        RCLCPP_INFO(
            this->get_logger(),
            "RefDbg A : /refinement_current_after");

        RCLCPP_INFO(
            this->get_logger(),
            "Realtime queue max: %zu",
            max_lidar_queue_size_);

        RCLCPP_INFO(
            this->get_logger(),
            "Preprocessor SOR: %s | ROR: %s",
            preprocessor_enable_sor_
                ? "ON"
                : "OFF",
            preprocessor_enable_ror_
                ? "ON"
                : "OFF");

        RCLCPP_INFO(
            this->get_logger(),
            "==============================================");
    }

    // ============================================================
    // Destructor
    // ============================================================
    ~lidar_registration_scan2localmap() override
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

// ============================================================
// main
// ============================================================
int main(
    int argc,
    char *argv[])
{
    rclcpp::init(
        argc,
        argv);

    const std::shared_ptr<
        lidar_registration_scan2localmap>
        node =
            std::make_shared<
                lidar_registration_scan2localmap>();

    // IMU and LiDAR callbacks are separated into different callback
    // groups. Heavy LiDAR processing itself runs in our worker thread.
    rclcpp::executors::MultiThreadedExecutor
        executor(
            rclcpp::ExecutorOptions(),
            2);

    executor.add_node(
        node);

    executor.spin();

    rclcpp::shutdown();

    return 0;
}