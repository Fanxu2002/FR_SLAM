#include "fr_slam/fr_imu_types.hpp"
#include "fr_slam/imu_adapter.hpp"
#include "fr_slam/fr_imu_initializer.hpp"
#include "fr_slam/fr_imu_integrator.hpp"

#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/imu.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include <functional>
#include <memory>
#include <vector>
#include <string>
#include <cstdint>

class TestImuIntegrator : public rclcpp::Node
{
private:
    rclcpp::Subscription<
        sensor_msgs::msg::Imu>::SharedPtr imu_sub_;

    rclcpp::Publisher<
        nav_msgs::msg::Odometry>::SharedPtr odom_pub_;

    rclcpp::Publisher<
        nav_msgs::msg::Path>::SharedPtr path_pub_;

    ImuAdapter imu_adapter_;
    ImuInitializer imu_initializer_;
    ImuIntegrator imu_integrator_;

    // --------------------------------------------------------
    // Initialization
    // --------------------------------------------------------
    static constexpr std::size_t kInitializationImuNum = 100;

    std::vector<IMU_DATA> initialization_imu_data_;

    bool initialized_ = false;

    // --------------------------------------------------------
    // IMU propagation
    // --------------------------------------------------------
    IMU_STATE current_state_;

    IMU_DATA previous_imu_;

    std::size_t integration_count_ = 0;

    // --------------------------------------------------------
    // Path
    // --------------------------------------------------------
    nav_msgs::msg::Path path_msg_;

    const std::string world_frame_ = "world";
    const std::string imu_frame_ = "imu";

private:
    builtin_interfaces::msg::Time ToRosTime(
        const double timestamp) const
    {
        const std::int64_t nanoseconds =
            static_cast<std::int64_t>(
                timestamp * 1e9);

        return rclcpp::Time(
            nanoseconds);
    }

    void PublishState(
        const IMU_STATE &state,
        const IMU_DATA &current_imu)
    {
        const builtin_interfaces::msg::Time stamp =
            ToRosTime(state.timestamp);

        // ====================================================
        // 1. Publish Odometry
        // ====================================================
        nav_msgs::msg::Odometry odom_msg;

        odom_msg.header.stamp = stamp;
        odom_msg.header.frame_id = world_frame_;

        odom_msg.child_frame_id = imu_frame_;

        // Position: P_WI
        odom_msg.pose.pose.position.x =
            state.P_WI.x();

        odom_msg.pose.pose.position.y =
            state.P_WI.y();

        odom_msg.pose.pose.position.z =
            state.P_WI.z();

        // Orientation: Q_WI
        odom_msg.pose.pose.orientation.w =
            state.Q_WI.w();

        odom_msg.pose.pose.orientation.x =
            state.Q_WI.x();

        odom_msg.pose.pose.orientation.y =
            state.Q_WI.y();

        odom_msg.pose.pose.orientation.z =
            state.Q_WI.z();

        // ----------------------------------------------------
        // nav_msgs/Odometry twist is expressed in child frame.
        //
        // We store V_WI in the World frame,
        // so convert it back into the IMU frame.
        // ----------------------------------------------------
        const Eigen::Vector3d velocity_I =
            state.Q_WI.conjugate() *
            state.V_WI;

        odom_msg.twist.twist.linear.x =
            velocity_I.x();

        odom_msg.twist.twist.linear.y =
            velocity_I.y();

        odom_msg.twist.twist.linear.z =
            velocity_I.z();

        // Gyroscope is already expressed in the IMU frame.
        const Eigen::Vector3d omega_I =
            current_imu.gyro -
            state.gyro_bias;

        odom_msg.twist.twist.angular.x =
            omega_I.x();

        odom_msg.twist.twist.angular.y =
            omega_I.y();

        odom_msg.twist.twist.angular.z =
            omega_I.z();

        odom_pub_->publish(
            odom_msg);

        // ====================================================
        // 2. Add current pose to Path
        // ====================================================
        geometry_msgs::msg::PoseStamped pose_msg;

        pose_msg.header.stamp = stamp;
        pose_msg.header.frame_id = world_frame_;

        pose_msg.pose.position.x =
            state.P_WI.x();

        pose_msg.pose.position.y =
            state.P_WI.y();

        pose_msg.pose.position.z =
            state.P_WI.z();

        pose_msg.pose.orientation.w =
            state.Q_WI.w();

        pose_msg.pose.orientation.x =
            state.Q_WI.x();

        pose_msg.pose.orientation.y =
            state.Q_WI.y();

        pose_msg.pose.orientation.z =
            state.Q_WI.z();

        path_msg_.header.stamp = stamp;
        path_msg_.header.frame_id =
            world_frame_;

        path_msg_.poses.push_back(
            pose_msg);

        path_pub_->publish(
            path_msg_);
    }

    void callback(
        const sensor_msgs::msg::Imu::SharedPtr msg)
    {
        // ====================================================
        // 1. Convert ROS IMU
        // ====================================================
        const IMU_DATA current_imu =
            imu_adapter_.convert(*msg);

        // ====================================================
        // 2. Initialization
        // ====================================================
        if (!initialized_)
        {
            initialization_imu_data_.push_back(
                current_imu);

            RCLCPP_INFO(
                this->get_logger(),
                "Collecting IMU: %zu / %zu",
                initialization_imu_data_.size(),
                kInitializationImuNum);

            if (initialization_imu_data_.size() <
                kInitializationImuNum)
            {
                return;
            }

            const bool success =
                imu_initializer_.Initialize(
                    initialization_imu_data_,
                    current_state_);

            if (!success)
            {
                RCLCPP_WARN(
                    this->get_logger(),
                    "IMU initialization failed!");

                initialization_imu_data_.clear();

                return;
            }

            // current_state_ is defined at the timestamp
            // of the last initialization IMU sample.
            previous_imu_ =
                initialization_imu_data_.back();

            initialized_ = true;

            RCLCPP_INFO(
                this->get_logger(),
                "==============================================");

            RCLCPP_INFO(
                this->get_logger(),
                "IMU initialization success.");

            RCLCPP_INFO(
                this->get_logger(),
                "timestamp = %.9f",
                current_state_.timestamp);

            RCLCPP_INFO(
                this->get_logger(),
                "P_WI = [%.6f %.6f %.6f]",
                current_state_.P_WI.x(),
                current_state_.P_WI.y(),
                current_state_.P_WI.z());

            RCLCPP_INFO(
                this->get_logger(),
                "V_WI = [%.6f %.6f %.6f]",
                current_state_.V_WI.x(),
                current_state_.V_WI.y(),
                current_state_.V_WI.z());

            RCLCPP_INFO(
                this->get_logger(),
                "Q_WI = [w=%.6f x=%.6f y=%.6f z=%.6f]",
                current_state_.Q_WI.w(),
                current_state_.Q_WI.x(),
                current_state_.Q_WI.y(),
                current_state_.Q_WI.z());

            RCLCPP_INFO(
                this->get_logger(),
                "gyro_bias = [%.6f %.6f %.6f]",
                current_state_.gyro_bias.x(),
                current_state_.gyro_bias.y(),
                current_state_.gyro_bias.z());

            RCLCPP_INFO(
                this->get_logger(),
                "accel_bias = [%.6f %.6f %.6f]",
                current_state_.accel_bias.x(),
                current_state_.accel_bias.y(),
                current_state_.accel_bias.z());

            RCLCPP_INFO(
                this->get_logger(),
                "==============================================");

            // Publish the initial state as the first path point.
            PublishState(
                current_state_,
                previous_imu_);

            return;
        }

        // ====================================================
        // 3. Build integration interval
        // ====================================================
        std::vector<IMU_DATA> imu_interval;

        imu_interval.reserve(2);

        imu_interval.push_back(
            previous_imu_);

        imu_interval.push_back(
            current_imu);

        // ====================================================
        // 4. Integrate
        // ====================================================
        std::vector<IMU_POSE> imu_poses;

        IMU_STATE final_state;

        const bool success =
            imu_integrator_.Integrate(
                imu_interval,
                current_state_,
                imu_poses,
                final_state);

        if (!success)
        {
            RCLCPP_WARN(
                this->get_logger(),
                "IMU integration failed!");

            return;
        }

        // ====================================================
        // 5. Update propagated state
        // ====================================================
        current_state_ =
            final_state;

        previous_imu_ =
            current_imu;

        ++integration_count_;

        // ====================================================
        // 6. Publish Odometry + Path
        // ====================================================
        PublishState(
            current_state_,
            current_imu);

        // Print every 100 integrations.
        if (integration_count_ % 100 == 0)
        {
            RCLCPP_INFO(
                this->get_logger(),
                "IMU Integration %zu | "
                "P=[%.4f %.4f %.4f] m | "
                "V=[%.4f %.4f %.4f] m/s",
                integration_count_,
                current_state_.P_WI.x(),
                current_state_.P_WI.y(),
                current_state_.P_WI.z(),
                current_state_.V_WI.x(),
                current_state_.V_WI.y(),
                current_state_.V_WI.z());
        }
    }

public:
    explicit TestImuIntegrator(
        const std::string &node_name)
        : rclcpp::Node(node_name)
    {
        imu_sub_ =
            this->create_subscription<
                sensor_msgs::msg::Imu>(
                "/livox/imu",
                rclcpp::SensorDataQoS(),
                std::bind(
                    &TestImuIntegrator::callback,
                    this,
                    std::placeholders::_1));

        odom_pub_ =
            this->create_publisher<
                nav_msgs::msg::Odometry>(
                "/imu_odometry",
                10);

        path_pub_ =
            this->create_publisher<
                nav_msgs::msg::Path>(
                "/imu_path",
                10);

        path_msg_.header.frame_id =
            world_frame_;

        RCLCPP_INFO(
            this->get_logger(),
            "IMU integrator test started.");

        RCLCPP_INFO(
            this->get_logger(),
            "Odometry topic: /imu_odometry");

        RCLCPP_INFO(
            this->get_logger(),
            "Path topic: /imu_path");
    }
};

int main(
    int argc,
    char *argv[])
{
    rclcpp::init(
        argc,
        argv);

    auto node =
        std::make_shared<
            TestImuIntegrator>(
            "test_imu_integrator");

    rclcpp::spin(
        node);

    rclcpp::shutdown();

    return 0;
}