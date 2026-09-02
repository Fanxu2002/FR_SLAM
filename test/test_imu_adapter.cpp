#include "fr_slam/fr_imu_adapter.hpp"
#include "fr_slam/fr_imu_buffer.hpp"
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include <functional>
#include <memory>

class TestImuAdapter : public rclcpp::Node
{
private:
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_;

    ImuAdapter imu_adapter_;

    IMUbuffer imu_buffer_;

    double previous_time_ = -1.0;

    void callback(
        const sensor_msgs::msg::Imu::SharedPtr msg)
    {
        RCLCPP_INFO(
            this->get_logger(),
            "callback entered");

        // 1. Convert ROS IMU message to unified IMU_DATA
        const IMU_DATA imu_data =
            imu_adapter_.convert(*msg);

        RCLCPP_INFO(
            this->get_logger(),
            "convert finished");

        const bool push_imu_successfully = imu_buffer_.Push(imu_data);

        RCLCPP_INFO(
            this->get_logger(),
            "push finished");

        if (!push_imu_successfully)
        {
            RCLCPP_WARN(this->get_logger(), "Invilid IMU timestamp!");
            return;
        }

        RCLCPP_INFO(this->get_logger(), "timestamp: %.9f, buffer size %zu", imu_data.timestamp, imu_buffer_.imu_deque_size());

        // 2. Calculate time interval
        double dt = 0.0;

        if (previous_time_ > 0.0)
        {
            dt =
                imu_data.timestamp -
                previous_time_;
        }

        previous_time_ =
            imu_data.timestamp;

        // 3. Calculate gyro / acceleration magnitude
        const double gyro_norm =
            imu_data.gyro.norm();

        const double acc_norm =
            imu_data.accelerometer.norm();

        // 4. Print results
        RCLCPP_INFO(
            this->get_logger(),
            "================ IMU Adapter Test ================");

        RCLCPP_INFO(
            this->get_logger(),
            "Timestamp: %.9f s",
            imu_data.timestamp);

        RCLCPP_INFO(
            this->get_logger(),
            "dt: %.9f s",
            dt);

        RCLCPP_INFO(
            this->get_logger(),
            "Angular velocity: x=%.6f y=%.6f z=%.6f rad/s",
            imu_data.gyro.x(),
            imu_data.gyro.y(),
            imu_data.gyro.z());

        RCLCPP_INFO(
            this->get_logger(),
            "Gyro norm: %.6f rad/s",
            gyro_norm);

        RCLCPP_INFO(
            this->get_logger(),
            "Linear acceleration: x=%.6f y=%.6f z=%.6f m/s^2",
            imu_data.accelerometer.x(),
            imu_data.accelerometer.y(),
            imu_data.accelerometer.z());

        RCLCPP_INFO(
            this->get_logger(),
            "Acceleration norm: %.6f m/s^2",
            acc_norm);
    }

public:
    explicit TestImuAdapter(
        const std::string &node_name)
        : rclcpp::Node(node_name)
    {
        sub_ =
            this->create_subscription<
                sensor_msgs::msg::Imu>(
                "/livox/imu",
                rclcpp::SensorDataQoS(),
                std::bind(
                    &TestImuAdapter::callback,
                    this,
                    std::placeholders::_1));
    }
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);

    rclcpp::spin(
        std::make_shared<TestImuAdapter>(
            "test_imu_adapter"));

    rclcpp::shutdown();

    return 0;
}