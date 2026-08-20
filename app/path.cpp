#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <builtin_interfaces/msg/time.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <sophus/so3.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <fstream>
#include <iomanip>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <string>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>

#include "fr_slam/mid360s_adapter.hpp"
#include "fr_slam/fr_lidar_preprocessor.hpp"
#include "fr_slam/fr_lidar_registration.hpp"
#include "fr_slam/fr_lidar_frame.hpp"
#include "fr_slam/fr_point_types.hpp"
#include "fr_slam/fr_local_map.hpp"

namespace
{

    LidarRegistrationConfig MakeRegistrationConfig()
    {
        LidarRegistrationConfig config;

        // Real LiDAR scan-to-scan test.
        config.max_iterations = 5;
        config.knn = 5;

        config.max_correspondence_distance = 1.0;
        config.max_plane_fit_error = 0.15;
        config.max_point_to_plane_distance = 0.5;

        config.min_correspondences = 50;

        // Convergence here means only that the GN update is already small.
        // 1.5e-3 rad ~= 0.086 deg.
        config.rotation_convergence_threshold = 1.5e-3;

        // 3 mm.
        config.translation_convergence_threshold = 3.0e-3;

        return config;
    }

    LocalMapConfig MakeLocalMapConfig()
    {
        LocalMapConfig config;

        // First scan-to-local-map version:
        // keep about one second of accepted LiDAR data at 10 Hz.
        config.max_frames = 10;

        // Keep local-map density close to registration cloud density.
        config.voxel_leaf_size = 0.30f;

        return config;
    }

    // Keep an Eigen Isometry rotation exactly on SO(3).
    // This is useful at module boundaries because a rotation matrix can pick up
    // tiny numerical non-orthogonality after repeated floating-point operations.
    Eigen::Isometry3d NormalizeIsometryRotation(
        const Eigen::Isometry3d &T)
    {
        Eigen::Isometry3d normalized = T;

        Eigen::Quaterniond q(T.rotation());

        if (!q.coeffs().allFinite() ||
            q.norm() < 1.0e-12)
        {
            return Eigen::Isometry3d::Identity();
        }

        q.normalize();
        normalized.linear() = q.toRotationMatrix();

        return normalized;
    }

    double RotationAngleDegrees(
        const Eigen::Matrix3d &R)
    {
        Eigen::Quaterniond q(R);

        if (!q.coeffs().allFinite() ||
            q.norm() < 1.0e-12)
        {
            return 0.0;
        }

        q.normalize();

        // q and -q represent the same rotation.  Using |w| gives the principal
        // rotation angle in [0, pi].
        double w = std::abs(q.w());
        w = std::clamp(w, 0.0, 1.0);

        constexpr double kRadToDeg =
            57.2957795130823208768;

        return 2.0 * std::acos(w) * kRadToDeg;
    }

} // namespace

class TestLidarOnlyOdometry : public rclcpp::Node
{
public:
    TestLidarOnlyOdometry()
        : Node("test_lidar_only_odometry"),
          registration_(MakeRegistrationConfig()),
          local_map_(MakeLocalMapConfig())
    {
        rclcpp::QoS lidar_qos(
            rclcpp::KeepLast(100));

        lidar_qos.reliable();
        lidar_qos.durability_volatile();

        lidar_sub_ =
            this->create_subscription<
                sensor_msgs::msg::PointCloud2>(
                "/livox/lidar",
                lidar_qos,
                std::bind(
                    &TestLidarOnlyOdometry::LidarCallback,
                    this,
                    std::placeholders::_1));

        // ============================================================
        // Odometry / path publishers
        // ============================================================

        odom_pub_ =
            this->create_publisher<nav_msgs::msg::Odometry>(
                "/fr_slam/lidar_odometry",
                rclcpp::QoS(10));

        path_pub_ =
            this->create_publisher<nav_msgs::msg::Path>(
                "/fr_slam/lidar_path",
                rclcpp::QoS(10));

        world_cloud_pub_ =
            this->create_publisher<sensor_msgs::msg::PointCloud2>(
                "/fr_slam/world_cloud",
                rclcpp::QoS(5));

        global_map_pub_ =
            this->create_publisher<sensor_msgs::msg::PointCloud2>(
                "/fr_slam/global_map",
                rclcpp::QoS(2).transient_local());

        tf_broadcaster_ =
            std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        global_map_ =
            pcl::make_shared<pcl::PointCloud<pcl::PointXYZI>>();

        path_msg_.header.frame_id =
            world_frame_id_;

        trajectory_file_.open(
            trajectory_file_path_,
            std::ios::out | std::ios::trunc);

        if (trajectory_file_.is_open())
        {
            trajectory_file_
                << "timestamp,x,y,z,qx,qy,qz,qw\n";

            trajectory_file_
                << std::fixed
                << std::setprecision(9);

            RCLCPP_INFO(
                this->get_logger(),
                "Trajectory CSV: %s",
                trajectory_file_path_.c_str());
        }
        else
        {
            RCLCPP_WARN(
                this->get_logger(),
                "Failed to open trajectory CSV: %s",
                trajectory_file_path_.c_str());
        }

        T_WL_ =
            Eigen::Isometry3d::Identity();

        last_relative_transform_ =
            Eigen::Isometry3d::Identity();

        RCLCPP_INFO(
            this->get_logger(),
            "==============================================");

        RCLCPP_INFO(
            this->get_logger(),
            "Pure LiDAR odometry started.");

        RCLCPP_INFO(
            this->get_logger(),
            "No IMU. No deskew.");

        RCLCPP_INFO(
            this->get_logger(),
            "Pipeline: ROS callback -> queue -> worker -> "
            "Preprocess -> Voxel -> Predict -> Scan-to-Local-Map -> Pose -> Update Local Map");

        RCLCPP_INFO(
            this->get_logger(),
            "Initial guess: T_WL(previous) * previous relative motion.");

        RCLCPP_INFO(
            this->get_logger(),
            "Accepted RMSE threshold: %.3f m",
            max_accepted_rmse_);

        RCLCPP_INFO(
            this->get_logger(),
            "Accepted minimum correspondences: %zu",
            min_accepted_correspondences_);

        RCLCPP_INFO(
            this->get_logger(),
            "RViz outputs: /fr_slam/lidar_path, /fr_slam/lidar_odometry, "
            "/fr_slam/world_cloud, /fr_slam/global_map, TF world->LiDAR");

        RCLCPP_INFO(
            this->get_logger(),
            "==============================================");

        processing_thread_ =
            std::thread(
                &TestLidarOnlyOdometry::ProcessingLoop,
                this);
    }

    ~TestLidarOnlyOdometry() override
    {
        stop_processing_.store(true);
        lidar_queue_cv_.notify_all();

        if (processing_thread_.joinable())
        {
            processing_thread_.join();
        }

        if (trajectory_file_.is_open())
        {
            trajectory_file_.flush();
            trajectory_file_.close();
        }

        std::size_t queue_size = 0;

        {
            std::lock_guard<std::mutex> lock(
                lidar_queue_mutex_);

            queue_size =
                lidar_queue_.size();
        }

        RCLCPP_INFO(
            this->get_logger(),
            "Final frame statistics | "
            "received=%zu dequeued=%zu accepted=%zu rejected=%zu queue=%zu",
            received_frames_.load(),
            dequeued_frames_.load(),
            accepted_frames_.load(),
            rejected_frames_.load(),
            queue_size);
    }

private:
    // ============================================================
    // ROS callback: producer only
    // ============================================================

    void LidarCallback(
        const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        ++received_frames_;

        std::size_t queue_size = 0;

        {
            std::lock_guard<std::mutex> lock(
                lidar_queue_mutex_);

            // Do not intentionally discard old scans.
            lidar_queue_.push_back(msg);

            queue_size =
                lidar_queue_.size();
        }

        lidar_queue_cv_.notify_one();

        if (queue_size >
            backlog_warning_threshold_)
        {
            RCLCPP_WARN(
                this->get_logger(),
                "LiDAR processing backlog | "
                "received=%zu dequeued=%zu accepted=%zu rejected=%zu queue=%zu",
                received_frames_.load(),
                dequeued_frames_.load(),
                accepted_frames_.load(),
                rejected_frames_.load(),
                queue_size);
        }
    }

    // ============================================================
    // FIFO consumer loop
    // ============================================================

    void ProcessingLoop()
    {
        while (true)
        {
            sensor_msgs::msg::PointCloud2::SharedPtr msg;

            {
                std::unique_lock<std::mutex> lock(
                    lidar_queue_mutex_);

                lidar_queue_cv_.wait(
                    lock,
                    [this]()
                    {
                        return stop_processing_.load() ||
                               !lidar_queue_.empty();
                    });

                if (stop_processing_.load() &&
                    lidar_queue_.empty())
                {
                    break;
                }

                msg =
                    lidar_queue_.front();

                lidar_queue_.pop_front();
            }

            ++dequeued_frames_;

            try
            {
                ProcessFrame(msg);
            }
            catch (const std::exception &e)
            {
                ++rejected_frames_;
                RCLCPP_ERROR(
                    this->get_logger(),
                    "Unhandled exception while processing LiDAR frame: %s",
                    e.what());
            }
            catch (...)
            {
                ++rejected_frames_;
                RCLCPP_ERROR(
                    this->get_logger(),
                    "Unknown exception while processing LiDAR frame.");
            }

            if (dequeued_frames_.load() %
                    statistics_period_frames_ ==
                0)
            {
                PrintFrameStatistics();
            }
        }
    }

    void PrintFrameStatistics()
    {
        std::size_t queue_size = 0;

        {
            std::lock_guard<std::mutex> lock(
                lidar_queue_mutex_);

            queue_size =
                lidar_queue_.size();
        }

        RCLCPP_INFO(
            this->get_logger(),
            "Frame statistics | "
            "received=%zu dequeued=%zu accepted=%zu rejected=%zu queue=%zu",
            received_frames_.load(),
            dequeued_frames_.load(),
            accepted_frames_.load(),
            rejected_frames_.load(),
            queue_size);
    }

    void PrintTargetPreparation(
        const PreparedLidarTarget &target,
        const char *label)
    {
        const double detailed_plane_ms =
            target.plane_centroid_ms +
            target.plane_covariance_ms +
            target.plane_eigen_ms +
            target.plane_validation_ms;

        double plane_other_ms =
            target.plane_pca_ms -
            detailed_plane_ms;

        if (plane_other_ms < 0.0)
        {
            plane_other_ms = 0.0;
        }

        RCLCPP_INFO(
            this->get_logger(),
            "%s target preparation | points=%zu valid=%zu invalid=%zu | "
            "xyz=%.3f ms kdtree=%.3f ms plane_knn=%.3f ms "
            "plane_fit=%.3f ms total=%.3f ms",
            label,
            target.cloud
                ? target.cloud->size()
                : 0,
            target.valid_planes,
            target.invalid_planes,
            target.xyz_copy_ms,
            target.kdtree_build_ms,
            target.plane_knn_ms,
            target.plane_pca_ms,
            target.prepare_total_ms);

        RCLCPP_INFO(
            this->get_logger(),
            "%s plane-fit detail | centroid=%.3f ms covariance=%.3f ms "
            "eigen=%.3f ms validation=%.3f ms other=%.3f ms",
            label,
            target.plane_centroid_ms,
            target.plane_covariance_ms,
            target.plane_eigen_ms,
            target.plane_validation_ms,
            plane_other_ms);
    }

    // ============================================================
    // Actual LiDAR front-end processing
    // ============================================================

    void ProcessFrame(
        const sensor_msgs::msg::PointCloud2::SharedPtr &msg)
    {
        const auto pipeline_begin =
            std::chrono::steady_clock::now();

        if (!msg->header.frame_id.empty())
        {
            lidar_frame_id_ = msg->header.frame_id;
        }

        // ====================================================
        // 1. ROS PointCloud2 -> LIDAR_FRAME
        // ====================================================

        const LIDAR_FRAME raw_frame =
            lidar_adapter_.convert(*msg);

        if (!raw_frame.cloud ||
            raw_frame.cloud->empty())
        {
            ++rejected_frames_;

            RCLCPP_WARN(
                this->get_logger(),
                "Raw LiDAR frame is empty.");

            return;
        }

        // ====================================================
        // 2. Basic preprocessing
        // ====================================================

        LIDAR_FRAME clean_frame =
            preprocessor_.preprocess(
                raw_frame);

        if (!clean_frame.cloud ||
            clean_frame.cloud->empty())
        {
            ++rejected_frames_;

            RCLCPP_WARN(
                this->get_logger(),
                "Cloud empty after preprocessing.");

            return;
        }

        // ====================================================
        // 3. Voxel downsample for registration/local map
        // ====================================================

        LIDAR_FRAME registration_frame =
            preprocessor_.VoxelGrid(
                clean_frame);

        if (!registration_frame.cloud ||
            registration_frame.cloud->empty())
        {
            ++rejected_frames_;

            RCLCPP_WARN(
                this->get_logger(),
                "Cloud empty after VoxelGrid.");

            return;
        }

        RCLCPP_INFO(
            this->get_logger(),
            "Cloud | raw=%zu clean=%zu voxel=%zu",
            raw_frame.cloud->size(),
            clean_frame.cloud->size(),
            registration_frame.cloud->size());

        // ====================================================
        // 4. First frame
        //
        // World frame is initialized at the first LiDAR pose.
        // The first registration cloud is transformed with Identity
        // and inserted into the local map.
        // ====================================================

        if (!has_local_map_target_)
        {
            T_WL_.setIdentity();
            last_relative_transform_.setIdentity();

            const bool map_ok =
                local_map_.AddFrame(
                    registration_frame.cloud,
                    T_WL_);

            if (!map_ok ||
                !local_map_.GetMap() ||
                local_map_.GetMap()->empty())
            {
                ++rejected_frames_;

                RCLCPP_WARN(
                    this->get_logger(),
                    "Failed to initialize local map.");

                return;
            }

            auto first_target =
                std::make_shared<
                    PreparedLidarTarget>();

            const bool prepare_ok =
                registration_.PrepareTarget(
                    local_map_.GetMap(),
                    *first_target);

            if (!prepare_ok)
            {
                ++rejected_frames_;

                RCLCPP_WARN(
                    this->get_logger(),
                    "Failed to prepare first local-map target.");

                return;
            }

            local_map_target_ =
                first_target;

            previous_scan_time_ =
                registration_frame.scan_start_time;

            has_local_map_target_ =
                true;

            ++accepted_frames_;

            PrintTargetPreparation(
                *local_map_target_,
                "First local map");

            RCLCPP_INFO(
                this->get_logger(),
                "First LiDAR frame initialized | scan_start=%.9f | "
                "T_WL0=Identity | local_map_frames=%zu local_map_points=%zu",
                previous_scan_time_,
                local_map_.FrameCount(),
                local_map_.PointCount());

            PublishOdometryPathAndTrajectory(
                previous_scan_time_);

            PublishCloudsAndUpdateMap(
                clean_frame.cloud,
                previous_scan_time_);

            return;
        }

        // ====================================================
        // 5. LiDAR timestamp gap
        // ====================================================

        const double current_scan_time =
            registration_frame.scan_start_time;

        const double scan_dt =
            current_scan_time -
            previous_scan_time_;

        RCLCPP_INFO(
            this->get_logger(),
            "LiDAR scan time | previous=%.9f current=%.9f dt=%.6f s",
            previous_scan_time_,
            current_scan_time,
            scan_dt);

        if (scan_dt <= 0.0)
        {
            ++rejected_frames_;

            RCLCPP_WARN(
                this->get_logger(),
                "Invalid LiDAR scan dt: %.9f",
                scan_dt);

            return;
        }

        // ====================================================
        // 6. Constant-motion prediction in WORLD coordinates
        //
        // last_relative_transform_ =
        //     T_WL(k-2)^-1 * T_WL(k-1)
        //
        // predicted current pose:
        //     T_WL_guess = T_WL(k-1) * last_relative_transform_
        // ====================================================

        const Eigen::Isometry3d initial_guess =
            T_WL_ *
            last_relative_transform_;

        // ====================================================
        // 7. Scan-to-local-map registration
        //
        // source: current scan in LiDAR frame
        // target: local map in WORLD frame
        //
        // Therefore Align() directly estimates:
        //     result.T_target_source = T_WL(current)
        // ====================================================

        LidarRegistrationResult result;

        const auto solve_begin =
            std::chrono::steady_clock::now();

        const bool success =
            registration_.Align(
                registration_frame.cloud,
                *local_map_target_,
                initial_guess,
                result);

        const auto solve_end =
            std::chrono::steady_clock::now();

        const double registration_time_ms =
            std::chrono::duration<double, std::milli>(
                solve_end - solve_begin)
                .count();

        if (!success ||
            !result.success)
        {
            ++rejected_frames_;

            RCLCPP_WARN(
                this->get_logger(),
                "Scan-to-local-map registration failed | "
                "scan_dt=%.6f s | align=%.3f ms",
                scan_dt,
                registration_time_ms);

            return;
        }

        // ====================================================
        // 8. Quality gate
        // ====================================================

        if (!result.converged)
        {
            RCLCPP_WARN(
                this->get_logger(),
                "Registration reached max iterations; "
                "checking result quality | iter=%d corr=%zu rmse=%.6f",
                result.iterations,
                result.correspondences,
                result.rmse);
        }

        const bool rmse_good =
            std::isfinite(result.rmse) &&
            result.rmse <=
                max_accepted_rmse_;

        const bool correspondence_good =
            result.correspondences >=
            min_accepted_correspondences_;

        const bool transform_good =
            result.T_target_source
                .matrix()
                .allFinite();

        if (!(rmse_good &&
              correspondence_good &&
              transform_good))
        {
            ++rejected_frames_;

            RCLCPP_WARN(
                this->get_logger(),
                "Registration rejected by quality gate | "
                "corr=%zu (min=%zu) | rmse=%.6f (max=%.6f) | "
                "finite_transform=%s",
                result.correspondences,
                min_accepted_correspondences_,
                result.rmse,
                max_accepted_rmse_,
                transform_good
                    ? "true"
                    : "false");

            return;
        }

        // ====================================================
        // 9. IMPORTANT coordinate-frame change from scan-to-scan
        //
        // Old scan-to-scan:
        //     T_WL(k) = T_WL(k-1) * T_L(k-1)_L(k)
        //
        // New scan-to-local-map:
        //     target is already WORLD,
        //     so result is directly T_WL(k).
        // ====================================================

        const Eigen::Isometry3d previous_pose =
            T_WL_;

        // Project the accepted pose rotation back onto SO(3).  The GN update
        // already uses SO(3), so this should be only a tiny numerical cleanup.
        // It also prevents strict downstream rotation checks from aborting the
        // process because of floating-point orthogonality error.
        const Eigen::Isometry3d candidate_pose =
            NormalizeIsometryRotation(
                result.T_target_source);

        const Eigen::Isometry3d relative_transform =
            NormalizeIsometryRotation(
                previous_pose.inverse() *
                candidate_pose);

        // ====================================================
        // 10. Update local map using the accepted candidate pose
        // ====================================================

        const auto map_update_begin =
            std::chrono::steady_clock::now();

        const bool map_ok =
            local_map_.AddFrame(
                registration_frame.cloud,
                candidate_pose);

        if (!map_ok ||
            !local_map_.GetMap() ||
            local_map_.GetMap()->empty())
        {
            ++rejected_frames_;

            RCLCPP_WARN(
                this->get_logger(),
                "Failed to update local map.");

            return;
        }

        auto new_local_map_target =
            std::make_shared<
                PreparedLidarTarget>();

        const bool prepare_ok =
            registration_.PrepareTarget(
                local_map_.GetMap(),
                *new_local_map_target);

        const auto map_update_end =
            std::chrono::steady_clock::now();

        const double map_update_time_ms =
            std::chrono::duration<double, std::milli>(
                map_update_end -
                map_update_begin)
                .count();

        if (!prepare_ok)
        {
            ++rejected_frames_;

            RCLCPP_WARN(
                this->get_logger(),
                "Failed to prepare updated local-map target.");

            return;
        }

        // ====================================================
        // 11. Commit accepted state
        // ====================================================

        T_WL_ =
            candidate_pose;

        last_relative_transform_ =
            relative_transform;

        local_map_target_ =
            new_local_map_target;

        previous_scan_time_ =
            current_scan_time;

        ++accepted_frames_;

        PublishOdometryPathAndTrajectory(
            current_scan_time);

        PublishCloudsAndUpdateMap(
            clean_frame.cloud,
            current_scan_time);

        // ====================================================
        // 12. Diagnostics
        // ====================================================

        const auto pipeline_end =
            std::chrono::steady_clock::now();

        const double pipeline_time_ms =
            std::chrono::duration<double, std::milli>(
                pipeline_end - pipeline_begin)
                .count();

        PrintTargetPreparation(
            *local_map_target_,
            "Local map");

        RCLCPP_INFO(
            this->get_logger(),
            "Local map | frames=%zu points=%zu | update+prepare=%.3f ms",
            local_map_.FrameCount(),
            local_map_.PointCount(),
            map_update_time_ms);

        PrintPose(
            result,
            relative_transform,
            scan_dt,
            registration_time_ms,
            map_update_time_ms,
            pipeline_time_ms);
    }

    builtin_interfaces::msg::Time ToRosStamp(
        const double scan_time) const
    {
        const double sec_floor =
            std::floor(scan_time);

        int64_t sec =
            static_cast<int64_t>(sec_floor);

        int64_t nanosec =
            static_cast<int64_t>(
                std::llround(
                    (scan_time - sec_floor) *
                    1.0e9));

        if (nanosec >= 1000000000LL)
        {
            ++sec;
            nanosec -= 1000000000LL;
        }
        else if (nanosec < 0)
        {
            --sec;
            nanosec += 1000000000LL;
        }

        builtin_interfaces::msg::Time stamp;
        stamp.sec = static_cast<int32_t>(sec);
        stamp.nanosec = static_cast<uint32_t>(nanosec);
        return stamp;
    }

    void PublishTf(
        const double scan_time)
    {
        if (!tf_broadcaster_)
        {
            return;
        }

        geometry_msgs::msg::TransformStamped transform;

        transform.header.stamp =
            ToRosStamp(scan_time);

        transform.header.frame_id =
            world_frame_id_;

        transform.child_frame_id =
            lidar_frame_id_;

        const Eigen::Vector3d translation =
            T_WL_.translation();

        Eigen::Quaterniond orientation(
            T_WL_.rotation());

        orientation.normalize();

        transform.transform.translation.x =
            translation.x();

        transform.transform.translation.y =
            translation.y();

        transform.transform.translation.z =
            translation.z();

        transform.transform.rotation.x =
            orientation.x();

        transform.transform.rotation.y =
            orientation.y();

        transform.transform.rotation.z =
            orientation.z();

        transform.transform.rotation.w =
            orientation.w();

        tf_broadcaster_->sendTransform(
            transform);
    }

    void PublishCloudsAndUpdateMap(
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &cloud_lidar,
        const double scan_time)
    {
        if (!cloud_lidar ||
            cloud_lidar->empty())
        {
            return;
        }

        pcl::PointCloud<pcl::PointXYZI>::Ptr world_cloud =
            pcl::make_shared<pcl::PointCloud<pcl::PointXYZI>>();

        world_cloud->reserve(
            cloud_lidar->size());

        for (const auto &point_lidar : cloud_lidar->points)
        {
            const Eigen::Vector3d p_lidar(
                static_cast<double>(point_lidar.x),
                static_cast<double>(point_lidar.y),
                static_cast<double>(point_lidar.z));

            const Eigen::Vector3d p_world =
                T_WL_ * p_lidar;

            pcl::PointXYZI point_world;
            point_world.x = static_cast<float>(p_world.x());
            point_world.y = static_cast<float>(p_world.y());
            point_world.z = static_cast<float>(p_world.z());
            point_world.intensity =
                static_cast<float>(point_lidar.intensity);

            world_cloud->push_back(
                point_world);
        }

        world_cloud->width =
            static_cast<std::uint32_t>(world_cloud->size());

        world_cloud->height = 1;
        world_cloud->is_dense = false;

        // ------------------------------------------------------------
        // 1. Publish current scan already transformed into world frame.
        // This does not depend on RViz finding a historical TF at the
        // exact PointCloud2 timestamp.
        // ------------------------------------------------------------

        sensor_msgs::msg::PointCloud2 world_cloud_msg;

        pcl::toROSMsg(
            *world_cloud,
            world_cloud_msg);

        world_cloud_msg.header.frame_id =
            world_frame_id_;

        world_cloud_msg.header.stamp =
            ToRosStamp(scan_time);

        world_cloud_pub_->publish(
            world_cloud_msg);

        // ------------------------------------------------------------
        // 2. Accumulate a simple global visualization map.
        // This is still pure scan-to-scan odometry.  The map is NOT
        // used by registration; it is only for visualization/debugging.
        // ------------------------------------------------------------

        *global_map_ +=
            *world_cloud;

        const std::size_t accepted =
            accepted_frames_.load();

        if (accepted > 0 &&
            accepted % map_filter_period_frames_ == 0)
        {
            pcl::VoxelGrid<pcl::PointXYZI> voxel_filter;

            voxel_filter.setInputCloud(
                global_map_);

            voxel_filter.setLeafSize(
                global_map_voxel_leaf_size_,
                global_map_voxel_leaf_size_,
                global_map_voxel_leaf_size_);

            pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_map =
                pcl::make_shared<pcl::PointCloud<pcl::PointXYZI>>();

            voxel_filter.filter(
                *filtered_map);

            global_map_ =
                filtered_map;
        }

        if (accepted == 1 ||
            accepted % map_publish_period_frames_ == 0)
        {
            sensor_msgs::msg::PointCloud2 map_msg;

            pcl::toROSMsg(
                *global_map_,
                map_msg);

            map_msg.header.frame_id =
                world_frame_id_;

            map_msg.header.stamp =
                ToRosStamp(scan_time);

            global_map_pub_->publish(
                map_msg);
        }
    }

    void PublishOdometryPathAndTrajectory(
        const double scan_time)
    {
        const Eigen::Vector3d position =
            T_WL_.translation();

        Eigen::Quaterniond orientation(
            T_WL_.rotation());

        orientation.normalize();

        geometry_msgs::msg::PoseStamped pose_msg;

        pose_msg.header.frame_id =
            world_frame_id_;

        pose_msg.header.stamp =
            ToRosStamp(scan_time);

        pose_msg.pose.position.x =
            position.x();

        pose_msg.pose.position.y =
            position.y();

        pose_msg.pose.position.z =
            position.z();

        pose_msg.pose.orientation.x =
            orientation.x();

        pose_msg.pose.orientation.y =
            orientation.y();

        pose_msg.pose.orientation.z =
            orientation.z();

        pose_msg.pose.orientation.w =
            orientation.w();

        // ------------------------------------------------------------
        // 1. nav_msgs/Odometry
        // ------------------------------------------------------------

        nav_msgs::msg::Odometry odom_msg;

        odom_msg.header =
            pose_msg.header;

        odom_msg.child_frame_id =
            lidar_frame_id_;

        odom_msg.pose.pose =
            pose_msg.pose;

        odom_pub_->publish(
            odom_msg);

        PublishTf(
            scan_time);

        // ------------------------------------------------------------
        // 2. nav_msgs/Path
        // ------------------------------------------------------------

        path_msg_.header =
            pose_msg.header;

        path_msg_.poses.push_back(
            pose_msg);

        path_pub_->publish(
            path_msg_);

        // ------------------------------------------------------------
        // 3. CSV trajectory
        // timestamp,x,y,z,qx,qy,qz,qw
        // ------------------------------------------------------------

        if (trajectory_file_.is_open())
        {
            trajectory_file_
                << scan_time << ','
                << position.x() << ','
                << position.y() << ','
                << position.z() << ','
                << orientation.x() << ','
                << orientation.y() << ','
                << orientation.z() << ','
                << orientation.w() << '\n';
        }
    }

    void PrintPose(
        const LidarRegistrationResult &result,
        const Eigen::Isometry3d &relative_transform,
        const double scan_dt,
        const double registration_time_ms,
        const double map_update_time_ms,
        const double pipeline_time_ms)
    {
        const Eigen::Vector3d relative_t =
            relative_transform.translation();

        // Do not construct Sophus::SO3 directly from a raw Matrix3d here.
        // Some Sophus versions abort when the matrix is even slightly outside
        // SO(3).  For diagnostics we only need the principal rotation angle,
        // so a normalized Eigen quaternion is both sufficient and robust.
        const double relative_rotation_angle_deg =
            RotationAngleDegrees(
                relative_transform.rotation());

        const Eigen::Vector3d global_p =
            T_WL_.translation();

        const double global_rotation_angle_deg =
            RotationAngleDegrees(
                T_WL_.rotation());

        RCLCPP_INFO(
            this->get_logger(),
            "Timing | scan_dt=%.6f s | align=%.3f ms | "
            "local_map_update=%.3f ms | pipeline=%.3f ms",
            scan_dt,
            registration_time_ms,
            map_update_time_ms,
            pipeline_time_ms);

        RCLCPP_INFO(
            this->get_logger(),
            "Registration | iter=%d corr=%zu rmse=%.6f converged=%s",
            result.iterations,
            result.correspondences,
            result.rmse,
            result.converged
                ? "true"
                : "false");

        RCLCPP_INFO(
            this->get_logger(),
            "Relative motion | dx=%.4f dy=%.4f dz=%.4f | "
            "|dt|=%.4f m | dR=%.4f deg",
            relative_t.x(),
            relative_t.y(),
            relative_t.z(),
            relative_t.norm(),
            relative_rotation_angle_deg);

        RCLCPP_INFO(
            this->get_logger(),
            "Global LiDAR pose | x=%.4f y=%.4f z=%.4f | "
            "|P|=%.4f m | rotation=%.4f deg",
            global_p.x(),
            global_p.y(),
            global_p.z(),
            global_p.norm(),
            global_rotation_angle_deg);
    }

private:
    // ============================================================
    // ROS
    // ============================================================

    rclcpp::Subscription<
        sensor_msgs::msg::PointCloud2>::SharedPtr
        lidar_sub_;

    rclcpp::Publisher<
        nav_msgs::msg::Odometry>::SharedPtr
        odom_pub_;

    rclcpp::Publisher<
        nav_msgs::msg::Path>::SharedPtr
        path_pub_;

    rclcpp::Publisher<
        sensor_msgs::msg::PointCloud2>::SharedPtr
        world_cloud_pub_;

    rclcpp::Publisher<
        sensor_msgs::msg::PointCloud2>::SharedPtr
        global_map_pub_;

    std::unique_ptr<tf2_ros::TransformBroadcaster>
        tf_broadcaster_;

    nav_msgs::msg::Path
        path_msg_;

    const std::string
        world_frame_id_ = "world";

    std::string
        lidar_frame_id_ = "livox_frame";

    const std::string
        trajectory_file_path_ =
            "/tmp/fr_slam_lidar_trajectory.csv";

    std::ofstream
        trajectory_file_;

    pcl::PointCloud<pcl::PointXYZI>::Ptr
        global_map_;

    const float
        global_map_voxel_leaf_size_ = 0.15f;

    const std::size_t
        map_filter_period_frames_ = 10;

    const std::size_t
        map_publish_period_frames_ = 5;

    // ============================================================
    // LiDAR modules
    // ============================================================

    Mid360s_Adapter
        lidar_adapter_;

    PreProcessor
        preprocessor_;

    LidarRegistration
        registration_;

    LocalMap
        local_map_;

    // ============================================================
    // Scan-to-local-map state
    // ============================================================

    std::shared_ptr<PreparedLidarTarget>
        local_map_target_;

    bool
        has_local_map_target_ = false;

    double
        previous_scan_time_ = 0.0;

    // LiDAR -> World
    Eigen::Isometry3d
        T_WL_ =
            Eigen::Isometry3d::Identity();

    Eigen::Isometry3d
        last_relative_transform_ =
            Eigen::Isometry3d::Identity();

    // ============================================================
    // Registration quality gate
    // ============================================================

    double
        max_accepted_rmse_ = 0.25;

    std::size_t
        min_accepted_correspondences_ = 1000;

    // ============================================================
    // Producer-consumer queue
    // ============================================================

    std::deque<
        sensor_msgs::msg::PointCloud2::SharedPtr>
        lidar_queue_;

    std::mutex
        lidar_queue_mutex_;

    std::condition_variable
        lidar_queue_cv_;

    std::thread
        processing_thread_;

    std::atomic<bool>
        stop_processing_{false};

    // ============================================================
    // Diagnostics
    // ============================================================

    std::atomic<std::size_t>
        received_frames_{0};

    std::atomic<std::size_t>
        dequeued_frames_{0};

    std::atomic<std::size_t>
        accepted_frames_{0};

    std::atomic<std::size_t>
        rejected_frames_{0};

    const std::size_t
        backlog_warning_threshold_ = 3;

    const std::size_t
        statistics_period_frames_ = 20;
};

int main(
    int argc,
    char **argv)
{
    rclcpp::init(
        argc,
        argv);

    rclcpp::spin(
        std::make_shared<
            TestLidarOnlyOdometry>());

    rclcpp::shutdown();

    return 0;
}