#include "fr_slam/fr_imu_buffer.hpp"
#include "fr_slam/fr_imu_types.hpp"
#include "fr_slam/fr_imu_adapter.hpp"
#include "fr_slam/fr_imu_initializer.hpp"
#include <rclcpp/rclcpp.hpp>
#include <vector>
#include <functional>
#include <memory>
#include <sensor_msgs/msg/imu.hpp>

class test_imu_initializer : public rclcpp::Node
{
private:
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr _sub;

    ImuAdapter imu_adapter_;
    IMUbuffer imu_buffer_;
    ImuInitializer imu_initializer_;

    IMU_STATE original_state_;

    static constexpr std::size_t initialization_IMU_data_num = 100;

    std::vector<IMU_DATA> imu_data_initialize_;

    bool initialized_ = false;

    void callback(
        const sensor_msgs::msg::Imu::SharedPtr msg)
    {
        // Initialization has already been completed.
        if (initialized_)
        {
            return;
        }

        // 1. Convert ROS IMU message.
        const IMU_DATA converted_imu_data =
            imu_adapter_.convert(*msg);

        // 2. Push into IMU buffer.
        if (!imu_buffer_.Push(converted_imu_data))
        {
            RCLCPP_WARN(
                this->get_logger(),
                "Invalid IMU timestamp!");
            return;
        }

        RCLCPP_INFO(
            this->get_logger(),
            "IMU buffer size: %zu",
            imu_buffer_.imu_deque_size());

        // 3. Collect initialization IMU measurements.
        imu_data_initialize_.push_back(
            converted_imu_data);

        RCLCPP_INFO(
            this->get_logger(),
            "Initialization IMU data: %zu / %zu",
            imu_data_initialize_.size(),
            initialization_IMU_data_num);

        if (imu_data_initialize_.size() <
            initialization_IMU_data_num)
        {
            return;
        }

        // 4. Initialize IMU state.
        const bool success =
            imu_initializer_.Initialize(
                imu_data_initialize_,
                original_state_);

        if (!success)
        {
            RCLCPP_WARN(
                this->get_logger(),
                "IMU initialization failed!");

            imu_data_initialize_.clear();

            return;
        }

        initialized_ = true;

        // 5. Print initial state.
        RCLCPP_INFO(
            this->get_logger(),
            "========== IMU Initialization Success ==========");

        RCLCPP_INFO(
            this->get_logger(),
            "timestamp = %.9f",
            original_state_.timestamp);

        RCLCPP_INFO(
            this->get_logger(),
            "Q_WI = [w=%.6f x=%.6f y=%.6f z=%.6f]",
            original_state_.Q_WI.w(),
            original_state_.Q_WI.x(),
            original_state_.Q_WI.y(),
            original_state_.Q_WI.z());

        RCLCPP_INFO(
            this->get_logger(),
            "gyro_bias = [%.6f %.6f %.6f]",
            original_state_.gyro_bias.x(),
            original_state_.gyro_bias.y(),
            original_state_.gyro_bias.z());

        RCLCPP_INFO(
            this->get_logger(),
            "accel_bias = [%.6f %.6f %.6f]",
            original_state_.accel_bias.x(),
            original_state_.accel_bias.y(),
            original_state_.accel_bias.z());

        RCLCPP_INFO(
            this->get_logger(),
            "gravity_W = [%.6f %.6f %.6f]",
            original_state_.gravity_W.x(),
            original_state_.gravity_W.y(),
            original_state_.gravity_W.z());

        RCLCPP_INFO(
            this->get_logger(),
            "P_WI = [%.6f %.6f %.6f]",
            original_state_.P_WI.x(),
            original_state_.P_WI.y(),
            original_state_.P_WI.z());

        RCLCPP_INFO(
            this->get_logger(),
            "V_WI = [%.6f %.6f %.6f]",
            original_state_.V_WI.x(),
            original_state_.V_WI.y(),
            original_state_.V_WI.z());
    }

public:
    test_imu_initializer(
        const std::string &name_node)
        : rclcpp::Node(name_node)
    {
        _sub =
            this->create_subscription<
                sensor_msgs::msg::Imu>(
                "/livox/imu",
                rclcpp::SensorDataQoS(),
                std::bind(
                    &test_imu_initializer::callback,
                    this,
                    std::placeholders::_1));
    }
};
int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    std::shared_ptr<test_imu_initializer> initial_node;
    initial_node = std::make_shared<test_imu_initializer>("test_imu_initializer");
    rclcpp::spin(initial_node);
}