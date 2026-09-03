#include <rclcpp/rclcpp.hpp>
#include <rclcpp/executors/multi_threaded_executor.hpp>

#include "fr_slam/common/fr_lidar_frame.hpp"
#include "fr_slam/lidar/fr_lidar_preprocessor.hpp"
#include "fr_slam/common/fr_point_types.hpp"
#include "fr_slam/frontend/fr_lidar_frontend.hpp"
#include "fr_slam/mapping/fr_keyframe.hpp"

#include "fr_slam/sensor/fr_lidar_adapter.hpp"
#include "fr_slam/sensor/fr_mid360s_adapter.hpp"
#include "fr_slam/sensor/fr_hesai_adapter.hpp"
#include "fr_slam/sensor/fr_imu_adapter.hpp"

#include "fr_slam/imu/fr_imu_buffer.hpp"
#include "fr_slam/imu/fr_imu_initializer.hpp"
#include "fr_slam/imu/fr_imu_integrator.hpp"
#include "fr_slam/imu/fr_imu_types.hpp"

#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_srvs/srv/trigger.hpp>

#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <tf2_ros/transform_broadcaster.h>

#include <builtin_interfaces/msg/time.hpp>

#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <ctime>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <stdexcept>
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

        // The persistent IMU state has already advanced beyond this scan.
        DROP_STALE,

        // Point time / interpolation / integration data is invalid.
        DROP_INVALID
    };

    enum class LidarWorkerState : int
    {
        IDLE = 0,
        WAIT_FOR_IMU = 1,
        PROCESSING = 2
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
    // On-demand SLAM output export service.
    //
    // Service:
    //     /save_slam_maps   [std_srvs/srv/Trigger]
    //
    // Maps:
    //     raw_keyframe_map.pcd
    //     optimized_map.pcd
    //     refined_map.pcd
    //
    // Trajectories:
    //     frontend_trajectory.csv
    //     pose_graph_before_trajectory.csv
    //     optimized_trajectory.csv
    // ============================================================
    rclcpp::Service<
        std_srvs::srv::Trigger>::SharedPtr
        save_maps_service_;

    std::string
        save_root_directory_;

    std::mutex
        map_save_mutex_;

    // Protects all persistent Path messages that can also be read by the
    // save service while the LiDAR worker is still running.
    std::mutex
        trajectory_mutex_;

    // ============================================================
    // TF broadcaster
    //
    // Dynamic TF tree:
    //
    //     world(map) -> odom -> current LiDAR frame
    //
    // The realtime frontend remains continuous in odom:
    //
    //     T_odom_L = T_WL_
    //
    // Loop closure changes only the backend bridge:
    //
    //     T_world_odom = T_map_odom
    //
    // Therefore the corrected current pose seen by RViz is:
    //
    //     T_world_L = T_world_odom * T_odom_L
    //
    // without overwriting or jumping the live frontend state.
    // ============================================================
    std::unique_ptr<
        tf2_ros::TransformBroadcaster>
        tf_broadcaster_;

    // ============================================================
    // Sensor adapters
    //
    // Both concrete LiDAR adapters already implement the common
    // Lidar_Adapt interface.  The YAML parameter "lidar_type"
    // selects which adapter is used at runtime.
    //
    // Keep the concrete adapters as normal members and store only a
    // non-owning base pointer.  No dynamic allocation is required.
    // ============================================================
    Mid360s_Adapter
        mid360s_adapter_;

    HESAI_Adapter
        hesai_adapter_;

    Lidar_Adapt *
        lidar_adapter_ =
            nullptr;

    ImuAdapter
        imu_adapter_;

    // ============================================================
    // Sensor configuration loaded from config/fr_slam.yaml
    // ============================================================
    std::string
        lidar_type_ =
            "mid360s";

    std::string
        lidar_topic_ =
            "/livox/lidar";

    std::string
        imu_topic_ =
            "/livox/imu";

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

    // Conversion applied to sensor_msgs/Imu::linear_acceleration.
    // Livox dataset: g -> m/s^2, scale=9.80665.
    // Fixposition: already m/s^2, scale=1.0.
    double
        imu_acceleration_scale_ =
            9.80665;

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

    // ============================================================
    // LiDAR/IMU synchronization diagnostics.
    // ============================================================
    std::atomic<int>
        lidar_worker_state_{
            static_cast<int>(LidarWorkerState::IDLE)};

    std::atomic<double>
        latest_imu_timestamp_{
            -std::numeric_limits<double>::infinity()};

    std::atomic<std::size_t>
        dropped_stale_lidar_frames_{0};

    std::atomic<std::size_t>
        dropped_invalid_lidar_frames_{0};

    std::atomic<std::size_t>
        dropped_queue_while_imu_wait_{0};

    std::atomic<std::size_t>
        dropped_queue_while_processing_{0};

    std::atomic<std::size_t>
        dropped_queue_other_{0};

    std::atomic<std::size_t>
        imu_wait_events_{0};

    std::atomic<std::size_t>
        imu_wait_iterations_{0};

    // Written only by ProcessingLoop(), read after joining the worker.
    double
        max_imu_wait_ms_ = 0.0;

    double
        max_imu_lag_ms_ = 0.0;

    // IMU messages are small and high-rate.  Keep a deeper DDS history so a
    // short executor / rosbag scheduling jitter does not overwrite the few
    // samples needed to cover the oldest LiDAR scan.
    std::size_t
        imu_qos_depth_ = 1000;

    std::string
        preprocessor_sor_mode_ =
            "always";

    std::size_t
        preprocessor_sor_adaptive_max_points_ =
            6000;

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
    // Latest ACCEPTED realtime-front-end LiDAR pose.
    //
    // Historically this variable is named T_WL_, but after introducing the
    // map/odom split its semantic role is:
    //
    //      p_odom = T_WL_ * p_L
    //
    // The backend/global pose is obtained with:
    //
    //      T_world_L = T_world_odom * T_WL_
    // ============================================================
    Eigen::Isometry3d
        T_WL_ =
            Eigen::Isometry3d::Identity();

    // ============================================================
    // Persistent trajectories.
    //
    // path_msg_:
    //     accepted ordinary LiDAR frames in odom.
    //
    // pose_graph_before_path_msg_:
    //     odometry-only Keyframe trajectory reconstructed from PoseGraph.
    //
    // optimized_path_msg_:
    //     current optimized Keyframe trajectory in world.
    // ============================================================
    nav_msgs::msg::Path
        path_msg_;

    nav_msgs::msg::Path
        pose_graph_before_path_msg_;

    nav_msgs::msg::Path
        optimized_path_msg_;

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

    std::string
        world_frame_ =
            "world";

    // Continuous local frame owned by the realtime frontend.
    // Loop closure must never jump this frame; the backend correction is
    // represented by the dynamic world -> odom transform instead.
    std::string
        odom_frame_ =
            "odom";

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

        latest_imu_timestamp_.store(
            imu_data.timestamp,
            std::memory_order_relaxed);

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

        if (lidar_adapter_ == nullptr)
        {
            RCLCPP_ERROR(
                this->get_logger(),
                "LiDAR adapter is not initialized.");

            return;
        }

        const LIDAR_FRAME raw_frame =
            lidar_adapter_->convert(
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

            const LidarWorkerState worker_state =
                static_cast<LidarWorkerState>(
                    lidar_worker_state_.load(
                        std::memory_order_relaxed));

            const char *overflow_reason =
                "OTHER_BACKLOG";

            if (worker_state ==
                LidarWorkerState::WAIT_FOR_IMU)
            {
                dropped_queue_while_imu_wait_.fetch_add(
                    dropped_now);
                overflow_reason =
                    "IMU_WAIT_BACKLOG";
            }
            else if (worker_state ==
                     LidarWorkerState::PROCESSING)
            {
                dropped_queue_while_processing_.fetch_add(
                    dropped_now);
                overflow_reason =
                    "PROCESSING_BACKLOG";
            }
            else
            {
                dropped_queue_other_.fetch_add(
                    dropped_now);
            }

            RCLCPP_WARN_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                1000,
                "LiDAR queue overflow | "
                "reason=%s dropped_waiting=%zu total_dropped=%zu "
                "queue=%zu max_queue=%zu latest_imu=%.9f",
                overflow_reason,
                dropped_now,
                dropped_lidar_frames_.load(),
                queue_size,
                max_lidar_queue_size_,
                latest_imu_timestamp_.load(
                    std::memory_order_relaxed));
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

            return ImuBuildStatus::DROP_INVALID;
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
            return ImuBuildStatus::DROP_INVALID;
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

            return ImuBuildStatus::DROP_STALE;
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
            const double latest_imu =
                latest_imu_timestamp_.load(
                    std::memory_order_relaxed);

            // If IMU time has already passed the scan end but the buffer
            // still cannot return [state_time, scan_end], the missing part is
            // historical. Future IMU samples cannot repair that interval, so
            // waiting forever would only fill the LiDAR queue.
            if (std::isfinite(latest_imu) &&
                latest_imu + time_epsilon >=
                    required_end_time)
            {
                RCLCPP_WARN(
                    this->get_logger(),
                    "IMU coverage gap | "
                    "state=%.9f required_end=%.9f latest_imu=%.9f | "
                    "future IMU cannot repair missing historical coverage",
                    current_imu_state_.timestamp,
                    required_end_time,
                    latest_imu);

                return ImuBuildStatus::DROP_INVALID;
            }

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

            return ImuBuildStatus::DROP_INVALID;
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

            return ImuBuildStatus::DROP_INVALID;
        }

        if (imu_poses.size() < 2)
        {
            RCLCPP_WARN(
                this->get_logger(),
                "IMU trajectory has too few poses.");

            return ImuBuildStatus::DROP_INVALID;
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

            return ImuBuildStatus::DROP_INVALID;
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
    //
    // IMPORTANT FRAME CONVENTION:
    //
    //     T_WL_ is the continuous FRONTEND pose and is therefore treated as
    //     T_odom_L after loop closure is enabled.
    //
    //     T_world_odom comes from the asynchronous backend correction.
    //
    // TF tree:
    //
    //     world -> odom -> lidar
    //
    // This keeps Scan-to-LocalMap continuous while allowing RViz to display
    // the live scan in the optimized global-map frame.
    // ============================================================
    void PublishPose(
        const builtin_interfaces::msg::Time &stamp,
        const std::string &lidar_frame)
    {
        const Eigen::Isometry3d T_odom_L =
            T_WL_;

        const Eigen::Vector3d P_odom_L =
            T_odom_L.translation();

        Eigen::Quaterniond Q_odom_L(
            T_odom_L.rotation());

        if (!Q_odom_L.coeffs().allFinite() ||
            Q_odom_L.norm() <= 1.0e-12)
        {
            Q_odom_L =
                Eigen::Quaterniond::Identity();
        }
        else
        {
            Q_odom_L.normalize();
        }

        Eigen::Isometry3d T_world_odom =
            Eigen::Isometry3d::Identity();

        if (scan_to_local_map_)
        {
            const Eigen::Isometry3d correction =
                scan_to_local_map_->GetMapOdomCorrection();

            if (correction.matrix().allFinite())
            {
                T_world_odom =
                    correction;
            }
        }

        Eigen::Quaterniond Q_world_odom(
            T_world_odom.rotation());

        if (!Q_world_odom.coeffs().allFinite() ||
            Q_world_odom.norm() <= 1.0e-12)
        {
            Q_world_odom =
                Eigen::Quaterniond::Identity();
            T_world_odom.linear() =
                Eigen::Matrix3d::Identity();
        }
        else
        {
            Q_world_odom.normalize();
        }

        // ========================================================
        // Dynamic TF #1: world -> odom
        // ========================================================
        if (tf_broadcaster_)
        {
            geometry_msgs::msg::TransformStamped
                map_to_odom_msg;

            map_to_odom_msg.header.stamp =
                stamp;

            map_to_odom_msg.header.frame_id =
                world_frame_;

            map_to_odom_msg.child_frame_id =
                odom_frame_;

            map_to_odom_msg.transform.translation.x =
                T_world_odom.translation().x();

            map_to_odom_msg.transform.translation.y =
                T_world_odom.translation().y();

            map_to_odom_msg.transform.translation.z =
                T_world_odom.translation().z();

            map_to_odom_msg.transform.rotation.x =
                Q_world_odom.x();

            map_to_odom_msg.transform.rotation.y =
                Q_world_odom.y();

            map_to_odom_msg.transform.rotation.z =
                Q_world_odom.z();

            map_to_odom_msg.transform.rotation.w =
                Q_world_odom.w();

            tf_broadcaster_->sendTransform(
                map_to_odom_msg);

            // ====================================================
            // Dynamic TF #2: odom -> current LiDAR frame
            // ====================================================
            geometry_msgs::msg::TransformStamped
                odom_to_lidar_msg;

            odom_to_lidar_msg.header.stamp =
                stamp;

            odom_to_lidar_msg.header.frame_id =
                odom_frame_;

            odom_to_lidar_msg.child_frame_id =
                lidar_frame;

            odom_to_lidar_msg.transform.translation.x =
                P_odom_L.x();

            odom_to_lidar_msg.transform.translation.y =
                P_odom_L.y();

            odom_to_lidar_msg.transform.translation.z =
                P_odom_L.z();

            odom_to_lidar_msg.transform.rotation.x =
                Q_odom_L.x();

            odom_to_lidar_msg.transform.rotation.y =
                Q_odom_L.y();

            odom_to_lidar_msg.transform.rotation.z =
                Q_odom_L.z();

            odom_to_lidar_msg.transform.rotation.w =
                Q_odom_L.w();

            tf_broadcaster_->sendTransform(
                odom_to_lidar_msg);
        }

        // ========================================================
        // Raw frontend odometry: pose is expressed in odom.
        // ========================================================
        nav_msgs::msg::Odometry
            odom_msg;

        odom_msg.header.frame_id =
            odom_frame_;

        odom_msg.child_frame_id =
            lidar_frame;

        odom_msg.header.stamp =
            stamp;

        odom_msg.pose.pose.position.x =
            P_odom_L.x();

        odom_msg.pose.pose.position.y =
            P_odom_L.y();

        odom_msg.pose.pose.position.z =
            P_odom_L.z();

        odom_msg.pose.pose.orientation.w =
            Q_odom_L.w();

        odom_msg.pose.pose.orientation.x =
            Q_odom_L.x();

        odom_msg.pose.pose.orientation.y =
            Q_odom_L.y();

        odom_msg.pose.pose.orientation.z =
            Q_odom_L.z();

        odom_pub_->publish(
            odom_msg);

        // ========================================================
        // Raw frontend path: also expressed in odom.
        //
        // /optimized_path remains the authoritative historical path after
        // non-rigid PoseGraph correction.
        // ========================================================
        geometry_msgs::msg::PoseStamped
            pose_msg;

        pose_msg.header.frame_id =
            odom_frame_;

        pose_msg.header.stamp =
            stamp;

        pose_msg.pose.position.x =
            P_odom_L.x();

        pose_msg.pose.position.y =
            P_odom_L.y();

        pose_msg.pose.position.z =
            P_odom_L.z();

        pose_msg.pose.orientation.w =
            Q_odom_L.w();

        pose_msg.pose.orientation.x =
            Q_odom_L.x();

        pose_msg.pose.orientation.y =
            Q_odom_L.y();

        pose_msg.pose.orientation.z =
            Q_odom_L.z();

        // Keep the persistent frontend trajectory coherent with the save
        // service.  The same mutex protects the snapshot copy during export.
        {
            std::lock_guard<std::mutex> trajectory_lock(
                trajectory_mutex_);

            path_msg_.header.stamp =
                stamp;

            path_msg_.header.frame_id =
                odom_frame_;

            path_msg_.poses.push_back(
                pose_msg);

            path_pub_->publish(
                path_msg_);
        }
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

        // Corrected pose is explicitly expressed in the backend/global world
        // frame. The same relation is now also represented formally in TF as
        // world -> odom -> lidar.
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

        const PoseGraph graph =
            scan_to_local_map_->GetPoseGraphSnapshot();

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

        // PoseGraph Path messages used to stamp every Keyframe with the time
        // at which the path happened to be published.  That is fine for RViz,
        // but wrong for trajectory export.  Recover each real LiDAR Keyframe
        // timestamp here so the CSV can be used for quantitative evaluation.
        std::unordered_map<
            std::size_t,
            double>
            keyframe_timestamp_by_id;

        const std::vector<Keyframe> &keyframes =
            scan_to_local_map_->GetKeyframes();

        keyframe_timestamp_by_id.reserve(
            keyframes.size());

        for (const Keyframe &keyframe :
             keyframes)
        {
            if (!std::isfinite(keyframe.timestamp))
            {
                continue;
            }

            keyframe_timestamp_by_id.emplace(
                keyframe.id,
                keyframe.timestamp);
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
            builtin_interfaces::msg::Time pose_stamp =
                stamp;

            const auto timestamp_iterator =
                keyframe_timestamp_by_id.find(
                    node.id);

            if (timestamp_iterator !=
                keyframe_timestamp_by_id.end())
            {
                const double timestamp =
                    timestamp_iterator->second;

                const double seconds_floor =
                    std::floor(timestamp);

                std::int64_t nanoseconds =
                    static_cast<std::int64_t>(
                        std::llround(
                            (timestamp - seconds_floor) *
                            1.0e9));

                std::int64_t seconds =
                    static_cast<std::int64_t>(
                        seconds_floor);

                if (nanoseconds >= 1000000000LL)
                {
                    ++seconds;
                    nanoseconds -= 1000000000LL;
                }
                else if (nanoseconds < 0)
                {
                    --seconds;
                    nanoseconds += 1000000000LL;
                }

                if (seconds >=
                        static_cast<std::int64_t>(
                            std::numeric_limits<std::int32_t>::min()) &&
                    seconds <=
                        static_cast<std::int64_t>(
                            std::numeric_limits<std::int32_t>::max()))
                {
                    pose_stamp.sec =
                        static_cast<std::int32_t>(
                            seconds);

                    pose_stamp.nanosec =
                        static_cast<std::uint32_t>(
                            nanoseconds);
                }
            }

            const auto before_iterator =
                before_poses.find(
                    node.id);

            if (before_iterator !=
                before_poses.end())
            {
                before_path.poses.push_back(
                    MakePoseStamped(
                        before_iterator->second,
                        pose_stamp));
            }

            if (node.T_WK
                    .matrix()
                    .allFinite())
            {
                optimized_path.poses.push_back(
                    MakePoseStamped(
                        node.T_WK,
                        pose_stamp));
            }
        }

        // Store the exact paths that were just generated.  The save service
        // only copies these immutable snapshots; it never reads or modifies
        // the live backend graph directly.
        {
            std::lock_guard<std::mutex> trajectory_lock(
                trajectory_mutex_);

            pose_graph_before_path_msg_ =
                before_path;

            optimized_path_msg_ =
                optimized_path;
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

        const PoseGraph graph =
            scan_to_local_map_->GetPoseGraphSnapshot();

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

        // LocalMap is assembled from raw frontend Keyframe poses and therefore
        // lives in the continuous odom frame. RViz will move it into world via
        // the dynamic world -> odom correction.
        map_msg.header.frame_id =
            odom_frame_;

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

        const RegistrationScan2LocalMap::BackendMapSnapshot
            backend_snapshot =
                scan_to_local_map_->GetBackendMapSnapshot();

        const std::size_t revision =
            backend_snapshot.global_revision;

        if (revision == 0 ||
            revision == last_published_global_map_revision_)
        {
            return;
        }

        const pcl::PointCloud<LIDAR_POINT>::ConstPtr raw_map =
            backend_snapshot.raw_map;

        const pcl::PointCloud<LIDAR_POINT>::ConstPtr optimized_map =
            backend_snapshot.optimized_map;

        const pcl::PointCloud<LIDAR_POINT>::ConstPtr refined_map =
            backend_snapshot.refined_map;

        const std::size_t refined_revision =
            backend_snapshot.refined_revision;

        const pcl::PointCloud<LIDAR_POINT>::ConstPtr refinement_historical_target =
            backend_snapshot.refinement_historical_target;

        const pcl::PointCloud<LIDAR_POINT>::ConstPtr refinement_current_before =
            backend_snapshot.refinement_current_before;

        const pcl::PointCloud<LIDAR_POINT>::ConstPtr refinement_current_after =
            backend_snapshot.refinement_current_after;

        const std::size_t refinement_debug_revision =
            backend_snapshot.refinement_debug_revision;

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
        const std::chrono::steady_clock::time_point frame_start =
            std::chrono::steady_clock::now();

        const LIDAR_FRAME &raw_frame =
            pending.frame;

        // ========================================================
        // 1. Rotation-only Deskew + preprocessing + voxel
        // ========================================================
        const std::chrono::steady_clock::time_point preprocess_start =
            std::chrono::steady_clock::now();

        const LIDAR_FRAME processed_frame =
            preprocessor_.Process(
                raw_frame,
                imu_poses,
                false);

        const std::chrono::steady_clock::time_point preprocess_end =
            std::chrono::steady_clock::now();

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
        const std::chrono::steady_clock::time_point registration_start =
            std::chrono::steady_clock::now();

        const bool success =
            scan_to_local_map_->AddFrame(
                current_cloud,
                raw_frame.scan_start_time,
                T_WL_current,
                result,
                imu_rotation_ptr);

        const std::chrono::steady_clock::time_point registration_end =
            std::chrono::steady_clock::now();

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

            const std::chrono::steady_clock::time_point frame_end =
                std::chrono::steady_clock::now();

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
                "sor_run=%s points=[basic:%zu voxel:%zu sor:%zu final:%zu] | "
                "queue=%zu dropped=%zu",
                preprocess_timing.deskew_ms,
                preprocess_timing.basic_ms,
                preprocess_timing.voxel_ms,
                preprocess_timing.sor_ms,
                preprocess_timing.ror_ms,
                preprocess_ms,
                registration_ms,
                total_ms,
                preprocess_timing.sor_executed
                    ? "true"
                    : "false",
                preprocess_timing.after_basic_points,
                preprocess_timing.after_voxel_points,
                preprocess_timing.after_sor_points,
                preprocess_timing.after_ror_points,
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
        const std::chrono::steady_clock::time_point publish_start =
            std::chrono::steady_clock::now();

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

        const std::chrono::steady_clock::time_point publish_end =
            std::chrono::steady_clock::now();

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
            "sor_run=%s points=[basic:%zu voxel:%zu sor:%zu final:%zu] | "
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
            preprocess_timing.sor_executed
                ? "true"
                : "false",
            preprocess_timing.after_basic_points,
            preprocess_timing.after_voxel_points,
            preprocess_timing.after_sor_points,
            preprocess_timing.after_ror_points,
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
        bool waiting_for_imu = false;
        double waiting_scan_start =
            std::numeric_limits<double>::quiet_NaN();

        std::chrono::steady_clock::time_point
            imu_wait_start =
                std::chrono::steady_clock::now();

        while (running_.load())
        {
            PendingLidarFrame pending;

            // ====================================================
            // 1. Wait for LiDAR.
            // ====================================================
            {
                lidar_worker_state_.store(
                    static_cast<int>(LidarWorkerState::IDLE),
                    std::memory_order_relaxed);

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

                pending =
                    lidar_queue_.front();
            }

            // ====================================================
            // 2. Build IMU trajectory.
            // ====================================================
            std::vector<IMU_POSE> imu_poses;
            IMU_STATE state_at_scan_start;

            const ImuBuildStatus status =
                BuildImuPoses(
                    pending.frame,
                    imu_poses,
                    state_at_scan_start);

            if (status ==
                ImuBuildStatus::WAIT_FOR_IMU)
            {
                double min_point_time = 0.0;
                double max_point_time = 0.0;

                const bool has_time_range =
                    FindPointTimeRange(
                        pending.frame,
                        min_point_time,
                        max_point_time);

                const double required_end_time =
                    has_time_range
                        ? std::max(
                              pending.frame.scan_start_time,
                              max_point_time)
                        : pending.frame.scan_start_time;

                const double latest_imu =
                    latest_imu_timestamp_.load(
                        std::memory_order_relaxed);

                const std::chrono::steady_clock::time_point now =
                    std::chrono::steady_clock::now();

                if (!waiting_for_imu ||
                    !std::isfinite(waiting_scan_start) ||
                    std::abs(
                        waiting_scan_start -
                        pending.frame.scan_start_time) > 1.0e-9)
                {
                    waiting_for_imu = true;
                    waiting_scan_start =
                        pending.frame.scan_start_time;
                    imu_wait_start = now;
                    imu_wait_events_.fetch_add(1);
                }

                imu_wait_iterations_.fetch_add(1);

                const double wait_ms =
                    std::chrono::duration<double, std::milli>(
                        now - imu_wait_start)
                        .count();

                double imu_lag_ms =
                    std::numeric_limits<double>::infinity();

                if (std::isfinite(latest_imu) &&
                    std::isfinite(required_end_time))
                {
                    imu_lag_ms =
                        std::max(
                            0.0,
                            (required_end_time - latest_imu) * 1000.0);
                }

                max_imu_wait_ms_ =
                    std::max(
                        max_imu_wait_ms_,
                        wait_ms);

                if (std::isfinite(imu_lag_ms))
                {
                    max_imu_lag_ms_ =
                        std::max(
                            max_imu_lag_ms_,
                            imu_lag_ms);
                }

                lidar_worker_state_.store(
                    static_cast<int>(LidarWorkerState::WAIT_FOR_IMU),
                    std::memory_order_relaxed);

                std::unique_lock<std::mutex> lock(
                    lidar_queue_mutex_);

                const std::size_t queue_size =
                    lidar_queue_.size();

                RCLCPP_WARN_THROTTLE(
                    this->get_logger(),
                    *this->get_clock(),
                    250,
                    "FR_SYNC IMU_WAIT | "
                    "scan_start=%.9f required_end=%.9f latest_imu=%.9f "
                    "imu_lag_ms=%.3f wait_ms=%.3f queue=%zu",
                    pending.frame.scan_start_time,
                    required_end_time,
                    latest_imu,
                    imu_lag_ms,
                    wait_ms,
                    queue_size);

                // Event-driven wait.  The short timeout is only for
                // diagnostics / shutdown; it is no longer a 5 ms busy poll.
                data_condition_.wait_for(
                    lock,
                    std::chrono::milliseconds(20),
                    [this, required_end_time]()
                    {
                        if (!running_.load())
                        {
                            return true;
                        }

                        const double latest =
                            latest_imu_timestamp_.load(
                                std::memory_order_relaxed);

                        return std::isfinite(latest) &&
                               latest + 1.0e-6 >=
                                   required_end_time;
                    });

                continue;
            }

            if (waiting_for_imu)
            {
                const double wait_ms =
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() -
                        imu_wait_start)
                        .count();

                RCLCPP_INFO(
                    this->get_logger(),
                    "FR_SYNC IMU_WAIT_RESOLVED | "
                    "scan_start=%.9f wait_ms=%.3f latest_imu=%.9f",
                    pending.frame.scan_start_time,
                    wait_ms,
                    latest_imu_timestamp_.load(
                        std::memory_order_relaxed));

                waiting_for_imu = false;
                waiting_scan_start =
                    std::numeric_limits<double>::quiet_NaN();
            }

            if (status == ImuBuildStatus::DROP_STALE ||
                status == ImuBuildStatus::DROP_INVALID)
            {
                {
                    std::lock_guard<std::mutex> lock(
                        lidar_queue_mutex_);

                    if (!lidar_queue_.empty())
                    {
                        lidar_queue_.pop_front();
                    }
                }

                dropped_lidar_frames_.fetch_add(1);

                const char *drop_reason =
                    "INVALID_IMU_BUILD";

                if (status == ImuBuildStatus::DROP_STALE)
                {
                    dropped_stale_lidar_frames_.fetch_add(1);
                    drop_reason = "STALE_SCAN";
                }
                else
                {
                    dropped_invalid_lidar_frames_.fetch_add(1);
                }

                RCLCPP_WARN(
                    this->get_logger(),
                    "FR_SYNC LIDAR_DROP | reason=%s "
                    "scan_start=%.9f total_dropped=%zu",
                    drop_reason,
                    pending.frame.scan_start_time,
                    dropped_lidar_frames_.load());

                continue;
            }

            // ====================================================
            // 3. Process exact oldest frame.
            // ====================================================
            lidar_worker_state_.store(
                static_cast<int>(LidarWorkerState::PROCESSING),
                std::memory_order_relaxed);

            ProcessLidarFrame(
                pending,
                imu_poses,
                state_at_scan_start);

            processed_lidar_frames_.fetch_add(1);

            // ====================================================
            // 4. Processing finished: now pop the protected front.
            // ====================================================
            std::size_t queue_size = 0;

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

            lidar_worker_state_.store(
                static_cast<int>(LidarWorkerState::IDLE),
                std::memory_order_relaxed);

            RCLCPP_INFO(
                this->get_logger(),
                "LiDAR frame finished | "
                "remaining_queue=%zu processed=%zu dropped=%zu",
                queue_size,
                processed_lidar_frames_.load(),
                dropped_lidar_frames_.load());
        }

        lidar_worker_state_.store(
            static_cast<int>(LidarWorkerState::IDLE),
            std::memory_order_relaxed);
    }

    // ============================================================
    // Build one unique snapshot directory name.
    //
    // Example:
    //
    //     20260902_183501_327
    //
    // Format:
    //
    //     YYYYMMDD_HHMMSS_mmm
    //
    // Each /save_slam_maps call therefore receives its own directory
    // and never overwrites a previous SLAM export.
    // ============================================================
    std::string BuildSaveSnapshotId() const
    {
        const std::chrono::system_clock::time_point now =
            std::chrono::system_clock::now();

        const std::time_t current_time =
            std::chrono::system_clock::to_time_t(
                now);

        std::tm local_time{};

        localtime_r(
            &current_time,
            &local_time);

        const std::int64_t milliseconds_since_epoch =
            std::chrono::duration_cast<
                std::chrono::milliseconds>(
                now.time_since_epoch())
                .count();

        const int milliseconds =
            static_cast<int>(
                milliseconds_since_epoch %
                1000);

        std::ostringstream stream;

        stream
            << std::put_time(
                   &local_time,
                   "%Y%m%d_%H%M%S")
            << "_"
            << std::setw(3)
            << std::setfill('0')
            << milliseconds;

        return stream.str();
    }

    // ============================================================
    // Save one ROS Path as a CSV trajectory.
    //
    // Format:
    //     timestamp,x,y,z,qx,qy,qz,qw
    //
    // Quaternion order intentionally follows the common TUM / SLAM
    // convention qx qy qz qw.
    // ============================================================
    bool SavePathCsv(
        const nav_msgs::msg::Path &path,
        const std::filesystem::path &file_path) const
    {
        if (path.poses.empty())
        {
            return false;
        }

        std::ofstream output(
            file_path,
            std::ios::out |
                std::ios::trunc);

        if (!output.is_open())
        {
            return false;
        }

        output
            << "timestamp,x,y,z,qx,qy,qz,qw\n";

        output
            << std::setprecision(16);

        std::size_t written_rows = 0;

        for (const geometry_msgs::msg::PoseStamped &pose_stamped :
             path.poses)
        {
            const geometry_msgs::msg::Pose &pose =
                pose_stamped.pose;

            const double timestamp =
                static_cast<double>(
                    pose_stamped.header.stamp.sec) +
                1.0e-9 *
                    static_cast<double>(
                        pose_stamped.header.stamp.nanosec);

            if (!std::isfinite(timestamp) ||
                !std::isfinite(pose.position.x) ||
                !std::isfinite(pose.position.y) ||
                !std::isfinite(pose.position.z) ||
                !std::isfinite(pose.orientation.x) ||
                !std::isfinite(pose.orientation.y) ||
                !std::isfinite(pose.orientation.z) ||
                !std::isfinite(pose.orientation.w))
            {
                continue;
            }

            Eigen::Quaterniond quaternion(
                pose.orientation.w,
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z);

            if (!quaternion.coeffs().allFinite() ||
                quaternion.norm() <= 1.0e-12)
            {
                continue;
            }

            quaternion.normalize();

            output
                << timestamp << ','
                << pose.position.x << ','
                << pose.position.y << ','
                << pose.position.z << ','
                << quaternion.x() << ','
                << quaternion.y() << ','
                << quaternion.z() << ','
                << quaternion.w() << '\n';

            ++written_rows;
        }

        output.flush();

        return output.good() &&
               written_rows > 0;
    }

    // ============================================================
    // SaveRawOptimizedRefinedMaps()
    //
    // This service is intentionally on-demand.  PCD disk I/O can take
    // noticeably longer than one 10 Hz LiDAR period, so it must NOT be part
    // of the realtime LiDAR processing path.  The LiDAR worker and backend
    // remain unchanged; normally call the service after the bag has reached
    // the desired final map state.
    // ============================================================
    void SaveRawOptimizedRefinedMaps(
        const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
        std::shared_ptr<std_srvs::srv::Trigger::Response> response)
    {
        (void)request;

        std::lock_guard<std::mutex> save_lock(
            map_save_mutex_);

        if (!scan_to_local_map_)
        {
            response->success = false;
            response->message =
                "Scan-to-LocalMap backend is not available.";
            return;
        }

        const RegistrationScan2LocalMap::BackendMapSnapshot snapshot =
            scan_to_local_map_->GetBackendMapSnapshot();

        if (!snapshot.raw_map ||
            !snapshot.optimized_map ||
            !snapshot.refined_map)
        {
            response->success = false;
            response->message =
                "One or more backend map snapshots are null.";
            return;
        }

        if (snapshot.raw_map->empty() ||
            snapshot.optimized_map->empty() ||
            snapshot.refined_map->empty())
        {
            response->success = false;
            response->message =
                "One or more backend map snapshots are empty.";
            return;
        }

        // Copy all trajectory messages before disk I/O.  The LiDAR worker is
        // blocked only for the short memory-copy window, not while files are
        // being written to disk.
        nav_msgs::msg::Path frontend_path_snapshot;
        nav_msgs::msg::Path pose_graph_before_path_snapshot;
        nav_msgs::msg::Path optimized_path_snapshot;

        {
            std::lock_guard<std::mutex> trajectory_lock(
                trajectory_mutex_);

            frontend_path_snapshot =
                path_msg_;

            pose_graph_before_path_snapshot =
                pose_graph_before_path_msg_;

            optimized_path_snapshot =
                optimized_path_msg_;
        }

        if (frontend_path_snapshot.poses.empty() ||
            pose_graph_before_path_snapshot.poses.empty() ||
            optimized_path_snapshot.poses.empty())
        {
            response->success = false;
            response->message =
                "One or more trajectory snapshots are empty. "
                "Wait until SLAM has produced Keyframes and try again.";
            return;
        }

        try
        {
            // ========================================================
            // One service call = one immutable SLAM snapshot.
            //
            // Current launch configuration:
            //
            //     pcd_save_directory_
            //         = <output>/maps
            //
            // Therefore:
            //
            //     parent_path()
            //         = <output>
            //
            // New export layout:
            //
            //     <output>/saves/<timestamp>/
            //         maps/
            //         trajectory/
            // ========================================================

            const std::filesystem::path save_root_directory(
                save_root_directory_);

            const std::string snapshot_id =
                BuildSaveSnapshotId();

            const std::filesystem::path snapshot_directory =
                save_root_directory /
                snapshot_id;

            const std::filesystem::path map_directory =
                snapshot_directory /
                "maps";

            const std::filesystem::path trajectory_directory =
                snapshot_directory /
                "trajectory";

            std::filesystem::create_directories(
                map_directory);

            std::filesystem::create_directories(
                trajectory_directory);

            const std::filesystem::path raw_path =
                map_directory / "raw_keyframe_map.pcd";

            const std::filesystem::path optimized_map_path =
                map_directory / "optimized_map.pcd";

            const std::filesystem::path refined_path =
                map_directory / "refined_map.pcd";

            const std::filesystem::path frontend_trajectory_path =
                trajectory_directory / "frontend_trajectory.csv";

            const std::filesystem::path pose_graph_before_trajectory_path =
                trajectory_directory /
                "pose_graph_before_trajectory.csv";

            const std::filesystem::path optimized_trajectory_path =
                trajectory_directory /
                "optimized_trajectory.csv";

            // Binary PCD is intentionally used instead of ASCII: it is much
            // faster to write and substantially smaller for a global map.
            const int raw_result =
                pcl::io::savePCDFileBinary(
                    raw_path.string(),
                    *snapshot.raw_map);

            const int optimized_result =
                pcl::io::savePCDFileBinary(
                    optimized_map_path.string(),
                    *snapshot.optimized_map);

            const int refined_result =
                pcl::io::savePCDFileBinary(
                    refined_path.string(),
                    *snapshot.refined_map);

            const bool frontend_trajectory_ok =
                SavePathCsv(
                    frontend_path_snapshot,
                    frontend_trajectory_path);

            const bool pose_graph_before_trajectory_ok =
                SavePathCsv(
                    pose_graph_before_path_snapshot,
                    pose_graph_before_trajectory_path);

            const bool optimized_trajectory_ok =
                SavePathCsv(
                    optimized_path_snapshot,
                    optimized_trajectory_path);

            if (raw_result != 0 ||
                optimized_result != 0 ||
                refined_result != 0 ||
                !frontend_trajectory_ok ||
                !pose_graph_before_trajectory_ok ||
                !optimized_trajectory_ok)
            {
                response->success = false;
                response->message =
                    "Failed to write one or more SLAM output files.";

                RCLCPP_ERROR(
                    this->get_logger(),
                    "SLAM export failed | "
                    "pcd=[raw:%d optimized:%d refined:%d] | "
                    "trajectory=[frontend:%s before:%s optimized:%s]",
                    raw_result,
                    optimized_result,
                    refined_result,
                    frontend_trajectory_ok ? "ok" : "failed",
                    pose_graph_before_trajectory_ok ? "ok" : "failed",
                    optimized_trajectory_ok ? "ok" : "failed");
                return;
            }

            response->success = true;

            response->message =
                "Saved SLAM snapshot: " +
                snapshot_directory.string();

            RCLCPP_INFO(
                this->get_logger(),
                "SLAM export SUCCESS | "
                "snapshot=%s | "
                "maps=%s | trajectories=%s | "
                "global_revision=%zu refined_revision=%zu | "
                "raw_points=%zu optimized_points=%zu refined_points=%zu | "
                "frontend_poses=%zu before_kf_poses=%zu optimized_kf_poses=%zu",
                snapshot_directory.string().c_str(),
                map_directory.string().c_str(),
                trajectory_directory.string().c_str(),
                snapshot.global_revision,
                snapshot.refined_revision,
                snapshot.raw_map->size(),
                snapshot.optimized_map->size(),
                snapshot.refined_map->size(),
                frontend_path_snapshot.poses.size(),
                pose_graph_before_path_snapshot.poses.size(),
                optimized_path_snapshot.poses.size());
        }
        catch (const std::exception &exception)
        {
            response->success = false;
            response->message =
                std::string("SLAM export exception: ") +
                exception.what();

            RCLCPP_ERROR(
                this->get_logger(),
                "SLAM export exception: %s",
                exception.what());
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
        // 1. Sensor configuration
        //
        // Sensor-specific ROS data is normalized by the adapter layer:
        //
        //   Mid360 PointCloud2 -> Mid360s_Adapter --+
        //                                           +-> LIDAR_FRAME
        //   Hesai PointCloud2  -> HESAI_Adapter ----+
        //
        // Everything after LIDAR_FRAME is sensor-independent.
        // ========================================================
        lidar_type_ =
            this->declare_parameter<std::string>(
                "lidar_type",
                "mid360s");

        std::transform(
            lidar_type_.begin(),
            lidar_type_.end(),
            lidar_type_.begin(),
            [](unsigned char character)
            {
                return static_cast<char>(
                    std::tolower(character));
            });

        lidar_topic_ =
            this->declare_parameter<std::string>(
                "lidar_topic",
                "/livox/lidar");

        imu_topic_ =
            this->declare_parameter<std::string>(
                "imu_topic",
                "/livox/imu");

        imu_acceleration_scale_ =
            this->declare_parameter<double>(
                "imu_acceleration_scale",
                9.80665);

        if (!std::isfinite(
                imu_acceleration_scale_) ||
            imu_acceleration_scale_ <= 0.0)
        {
            RCLCPP_FATAL(
                this->get_logger(),
                "Invalid imu_acceleration_scale=%.9f. "
                "The value must be finite and positive.",
                imu_acceleration_scale_);

            throw std::runtime_error(
                "Invalid imu_acceleration_scale");
        }

        imu_adapter_.setAccelerationScale(
            imu_acceleration_scale_);

        world_frame_ =
            this->declare_parameter<std::string>(
                "world_frame",
                "world");

        odom_frame_ =
            this->declare_parameter<std::string>(
                "odom_frame",
                "odom");

        if (lidar_type_ == "mid360s" ||
            lidar_type_ == "mid360" ||
            lidar_type_ == "livox_mid360")
        {
            lidar_type_ =
                "mid360s";

            lidar_adapter_ =
                &mid360s_adapter_;
        }
        else if (lidar_type_ == "hesai")
        {
            lidar_adapter_ =
                &hesai_adapter_;
        }
        else
        {
            RCLCPP_FATAL(
                this->get_logger(),
                "Unsupported lidar_type='%s'. "
                "Valid values: mid360s, hesai.",
                lidar_type_.c_str());

            throw std::runtime_error(
                "Unsupported lidar_type: " +
                lidar_type_);
        }

        // ========================================================
        // 2. Scan-to-LocalMap configuration
        // ========================================================
        LidarRegistrationConfig
            registration_config;

        LocalMapConfig
            local_map_config;

        const int configured_local_map_max_frames =
            this->declare_parameter<int>(
                "local_map_max_frames",
                10);

        local_map_config.max_frames =
            static_cast<std::size_t>(
                std::max(
                    1,
                    configured_local_map_max_frames));

        local_map_config.voxel_leaf_size =
            static_cast<float>(
                std::max(
                    0.01,
                    this->declare_parameter<double>(
                        "local_map_voxel_leaf_size",
                        0.30)));

        // ========================================================
        // 3. Create Scan-to-LocalMap module
        // ========================================================
        scan_to_local_map_ =
            std::make_unique<
                RegistrationScan2LocalMap>(
                registration_config,
                local_map_config);

        // ========================================================
        // 2.1 SLAM output export configuration.
        //
        // fr_slam.launch.py sets FR_SLAM_OUTPUT_DIR and also passes explicit
        // map / trajectory directories.  These defaults are only fallbacks
        // for running the executable directly.
        // ========================================================
        const char *output_directory_environment =
            std::getenv("FR_SLAM_OUTPUT_DIR");

        const std::filesystem::path default_output_directory =
            output_directory_environment != nullptr
                ? std::filesystem::path(
                      output_directory_environment)
                : std::filesystem::temp_directory_path() /
                      "fr_slam";

        const std::string default_save_root_directory =
            (default_output_directory /
             "saves")
                .string();

        save_root_directory_ =
            this->declare_parameter<std::string>(
                "save_root_directory",
                default_save_root_directory);

        save_maps_service_ =
            this->create_service<std_srvs::srv::Trigger>(
                "/save_slam_maps",
                std::bind(
                    &lidar_registration_scan2localmap::
                        SaveRawOptimizedRefinedMaps,
                    this,
                    std::placeholders::_1,
                    std::placeholders::_2));

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

        const int configured_imu_qos_depth =
            this->declare_parameter<int>(
                "imu_qos_depth",
                1000);

        imu_qos_depth_ =
            static_cast<std::size_t>(
                std::max(
                    50,
                    configured_imu_qos_depth));

        const int configured_initialization_sample_count =
            this->declare_parameter<int>(
                "imu_initialization_sample_count",
                200);

        initialization_sample_count_ =
            static_cast<std::size_t>(
                std::max(
                    20,
                    configured_initialization_sample_count));

        imu_history_duration_ =
            std::max(
                0.10,
                this->declare_parameter<double>(
                    "imu_history_duration",
                    0.50));

        // ========================================================
        // 3.1 LiDAR preprocessing parameters from YAML.
        // ========================================================
        PreprocessorConfig
            preprocessor_config;

        preprocessor_config.range_min =
            this->declare_parameter<double>(
                "preprocessor_range_min",
                1.0);

        preprocessor_config.range_max =
            this->declare_parameter<double>(
                "preprocessor_range_max",
                30.0);

        preprocessor_config.enable_ROI =
            this->declare_parameter<bool>(
                "preprocessor_enable_range_filter",
                true);

        preprocessor_config.enable_passthrough =
            this->declare_parameter<bool>(
                "preprocessor_enable_passthrough",
                true);

        preprocessor_config.ROI_min_x =
            this->declare_parameter<double>(
                "preprocessor_roi_min_x",
                -30.0);

        preprocessor_config.ROI_max_x =
            this->declare_parameter<double>(
                "preprocessor_roi_max_x",
                30.0);

        preprocessor_config.ROI_min_y =
            this->declare_parameter<double>(
                "preprocessor_roi_min_y",
                -15.0);

        preprocessor_config.ROI_max_y =
            this->declare_parameter<double>(
                "preprocessor_roi_max_y",
                15.0);

        preprocessor_config.ROI_min_z =
            this->declare_parameter<double>(
                "preprocessor_roi_min_z",
                -2.0);

        preprocessor_config.ROI_max_z =
            this->declare_parameter<double>(
                "preprocessor_roi_max_z",
                10.0);

        preprocessor_config.enable_cropbox =
            this->declare_parameter<bool>(
                "preprocessor_enable_cropbox",
                true);

        preprocessor_config.cropbox_min_x =
            static_cast<float>(
                this->declare_parameter<double>(
                    "preprocessor_cropbox_min_x",
                    -0.15));

        preprocessor_config.cropbox_max_x =
            static_cast<float>(
                this->declare_parameter<double>(
                    "preprocessor_cropbox_max_x",
                    0.15));

        preprocessor_config.cropbox_min_y =
            static_cast<float>(
                this->declare_parameter<double>(
                    "preprocessor_cropbox_min_y",
                    -0.15));

        preprocessor_config.cropbox_max_y =
            static_cast<float>(
                this->declare_parameter<double>(
                    "preprocessor_cropbox_max_y",
                    0.15));

        preprocessor_config.cropbox_min_z =
            static_cast<float>(
                this->declare_parameter<double>(
                    "preprocessor_cropbox_min_z",
                    -0.15));

        preprocessor_config.cropbox_max_z =
            static_cast<float>(
                this->declare_parameter<double>(
                    "preprocessor_cropbox_max_z",
                    0.15));

        preprocessor_config.enable_voxel =
            this->declare_parameter<bool>(
                "preprocessor_enable_voxel",
                true);

        preprocessor_config.voxel_leaf_size =
            static_cast<float>(
                std::max(
                    0.01,
                    this->declare_parameter<double>(
                        "preprocessor_voxel_leaf_size",
                        0.30)));

        const std::int64_t configured_voxel_min_points =
            this->declare_parameter<std::int64_t>(
                "preprocessor_voxel_min_points",
                1);

        preprocessor_config.voxel_min_points =
            static_cast<unsigned int>(
                std::max<std::int64_t>(
                    1,
                    configured_voxel_min_points));

        const std::int64_t configured_sor_mean_k =
            this->declare_parameter<std::int64_t>(
                "preprocessor_sor_mean_k",
                50);

        preprocessor_config.sor_mean_k =
            static_cast<int>(
                std::max<std::int64_t>(
                    1,
                    configured_sor_mean_k));

        preprocessor_config.sor_stddev_mul_thresh =
            std::max(
                0.01,
                this->declare_parameter<double>(
                    "preprocessor_sor_stddev_mul_thresh",
                    1.0));

        preprocessor_config.ror_RadiusSearch =
            std::max(
                0.01,
                this->declare_parameter<double>(
                    "preprocessor_ror_radius",
                    0.30));

        const std::int64_t configured_ror_min_neighbors =
            this->declare_parameter<std::int64_t>(
                "preprocessor_ror_min_neighbors",
                1);

        preprocessor_config.ror_MinNeighborsInRadius =
            static_cast<int>(
                std::max<std::int64_t>(
                    1,
                    configured_ror_min_neighbors));

        preprocessor_.SetConfig(
            preprocessor_config);

        preprocessor_sor_mode_ =
            this->declare_parameter<std::string>(
                "preprocessor_sor_mode",
                "always");

        std::transform(
            preprocessor_sor_mode_.begin(),
            preprocessor_sor_mode_.end(),
            preprocessor_sor_mode_.begin(),
            [](unsigned char character)
            {
                return static_cast<char>(
                    std::tolower(character));
            });

        PreprocessorSorMode preprocessor_sor_mode =
            PreprocessorSorMode::ALWAYS;

        if (preprocessor_sor_mode_ ==
            "off")
        {
            preprocessor_sor_mode =
                PreprocessorSorMode::OFF;
        }
        else if (preprocessor_sor_mode_ ==
                 "adaptive")
        {
            preprocessor_sor_mode =
                PreprocessorSorMode::ADAPTIVE;
        }
        else if (preprocessor_sor_mode_ !=
                 "always")
        {
            RCLCPP_WARN(
                this->get_logger(),
                "Unknown preprocessor_sor_mode='%s'. "
                "Falling back to 'always'. Valid modes: always, off, adaptive.",
                preprocessor_sor_mode_.c_str());

            preprocessor_sor_mode_ =
                "always";

            preprocessor_sor_mode =
                PreprocessorSorMode::ALWAYS;
        }

        const int configured_sor_adaptive_max_points =
            this->declare_parameter<int>(
                "preprocessor_sor_adaptive_max_points",
                6000);

        preprocessor_sor_adaptive_max_points_ =
            static_cast<std::size_t>(
                std::max(
                    1,
                    configured_sor_adaptive_max_points));

        preprocessor_enable_ror_ =
            this->declare_parameter<bool>(
                "preprocessor_enable_ror",
                false);

        preprocessor_.SetOutlierFilterPolicy(
            preprocessor_sor_mode,
            preprocessor_enable_ror_,
            preprocessor_sor_adaptive_max_points_);

        // ========================================================
        // 4. Deskew extrinsic
        //
        // Q_IL_:
        //      LiDAR -> IMU rotation
        //
        // Current test uses Identity. Replace with calibrated
        // extrinsic later.
        // ========================================================
        const double q_il_x =
            this->declare_parameter<double>(
                "imu_extrinsic_q_il_x",
                0.0);

        const double q_il_y =
            this->declare_parameter<double>(
                "imu_extrinsic_q_il_y",
                0.0);

        const double q_il_z =
            this->declare_parameter<double>(
                "imu_extrinsic_q_il_z",
                0.0);

        const double q_il_w =
            this->declare_parameter<double>(
                "imu_extrinsic_q_il_w",
                1.0);

        const double p_il_x =
            this->declare_parameter<double>(
                "imu_extrinsic_p_il_x",
                0.0);

        const double p_il_y =
            this->declare_parameter<double>(
                "imu_extrinsic_p_il_y",
                0.0);

        const double p_il_z =
            this->declare_parameter<double>(
                "imu_extrinsic_p_il_z",
                0.0);

        Q_IL_ =
            Eigen::Quaterniond(
                q_il_w,
                q_il_x,
                q_il_y,
                q_il_z);

        if (!Q_IL_.coeffs().allFinite() ||
            Q_IL_.norm() <= 1.0e-12)
        {
            throw std::runtime_error(
                "Invalid LiDAR-IMU rotation extrinsic.");
        }

        Q_IL_.normalize();

        P_IL_ =
            Eigen::Vector3d(
                p_il_x,
                p_il_y,
                p_il_z);

        if (!P_IL_.allFinite())
        {
            throw std::runtime_error(
                "Invalid LiDAR-IMU translation extrinsic.");
        }

        RCLCPP_INFO(
            this->get_logger(),
            "LiDAR->IMU extrinsic loaded | "
            "q_IL=[%.8f %.8f %.8f %.8f] | "
            "p_IL=[%.4f %.4f %.4f]",
            Q_IL_.x(),
            Q_IL_.y(),
            Q_IL_.z(),
            Q_IL_.w(),
            P_IL_.x(),
            P_IL_.y(),
            P_IL_.z());

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

        rclcpp::QoS imu_qos{
            rclcpp::KeepLast(
                imu_qos_depth_)};

        imu_qos.best_effort();
        imu_qos.durability_volatile();

        imu_sub_ =
            this->create_subscription<
                sensor_msgs::msg::Imu>(
                imu_topic_,
                imu_qos,
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
                lidar_topic_,
                rclcpp::SensorDataQoS(),
                std::bind(
                    &lidar_registration_scan2localmap::
                        LidarCallback,
                    this,
                    std::placeholders::_1),
                lidar_options);

        // ========================================================
        // 7. TF broadcaster
        // ========================================================
        tf_broadcaster_ =
            std::make_unique<
                tf2_ros::TransformBroadcaster>(
                this);

        // ========================================================
        // 8. Publishers
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
            odom_frame_;

        pose_graph_before_path_msg_.header.frame_id =
            world_frame_;

        optimized_path_msg_.header.frame_id =
            world_frame_;

        // ========================================================
        // 9. Start worker
        // ========================================================
        processing_thread_ =
            std::thread(
                &lidar_registration_scan2localmap::
                    ProcessingLoop,
                this);

        // ========================================================
        // 10. Startup information
        // ========================================================
        RCLCPP_INFO(
            this->get_logger(),
            "==============================================");

        RCLCPP_INFO(
            this->get_logger(),
            "Scan-to-LocalMap WITH IMU rotation prediction started.");

        RCLCPP_INFO(
            this->get_logger(),
            "LiDAR : type=%s topic=%s",
            lidar_type_.c_str(),
            lidar_topic_.c_str());

        RCLCPP_INFO(
            this->get_logger(),
            "IMU   : topic=%s",
            imu_topic_.c_str());

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
            "TF       : %s -> %s -> LiDAR",
            world_frame_.c_str(),
            odom_frame_.c_str());

        RCLCPP_INFO(
            this->get_logger(),
            "Frames   : frontend=[%s] backend=[%s]",
            odom_frame_.c_str(),
            world_frame_.c_str());

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
            "SaveRoot : %s",
            save_root_directory_.c_str());

        RCLCPP_INFO(
            this->get_logger(),
            "SaveCmd  : ros2 service call /save_slam_maps std_srvs/srv/Trigger {}");

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
            "IMU QoS depth: %zu (best_effort)",
            imu_qos_depth_);

        RCLCPP_INFO(
            this->get_logger(),
            "IMU init samples: %zu | history=%.3f s",
            initialization_sample_count_,
            imu_history_duration_);

        RCLCPP_INFO(
            this->get_logger(),
            "IMU acceleration scale: %.8f | output_unit=m/s^2",
            imu_adapter_.accelerationScale());

        RCLCPP_INFO(
            this->get_logger(),
            "Frames: world=%s odom=%s",
            world_frame_.c_str(),
            odom_frame_.c_str());

        RCLCPP_INFO(
            this->get_logger(),
            "Preprocessor SOR mode: %s | adaptive_max_points=%zu | ROR: %s",
            preprocessor_sor_mode_.c_str(),
            preprocessor_sor_adaptive_max_points_,
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

        RCLCPP_INFO(
            this->get_logger(),
            "FR_SYNC SUMMARY | "
            "processed=%zu dropped_total=%zu "
            "drop_stale=%zu drop_invalid=%zu "
            "drop_queue_imu_wait=%zu drop_queue_processing=%zu "
            "drop_queue_other=%zu imu_wait_events=%zu "
            "imu_wait_iterations=%zu max_imu_wait_ms=%.3f "
            "max_imu_lag_ms=%.3f",
            processed_lidar_frames_.load(),
            dropped_lidar_frames_.load(),
            dropped_stale_lidar_frames_.load(),
            dropped_invalid_lidar_frames_.load(),
            dropped_queue_while_imu_wait_.load(),
            dropped_queue_while_processing_.load(),
            dropped_queue_other_.load(),
            imu_wait_events_.load(),
            imu_wait_iterations_.load(),
            max_imu_wait_ms_,
            max_imu_lag_ms_);
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
