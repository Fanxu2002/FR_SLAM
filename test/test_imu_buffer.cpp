#include "fr_slam/fr_imu_buffer.hpp"
#include "fr_slam/imu_adapter.hpp"
#include <rclcpp/rclcpp.hpp>
#include "sensor_msgs/msg/imu.hpp"
class test_imu_buffer : public rclcpp::Node
{
private:
        rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr _sub;
        IMUbuffer imu_buffer;

        void callback(const sensor_msgs::msg::Imu &raw_imu_data)
        {
                ImuAdapter imu_adapter;

                IMU_DATA imu_data = imu_adapter.convert(raw_imu_data);
                RCLCPP_INFO(this->get_logger(), "IMU DATA convertion successfully!");

                imu_buffer.Push(imu_data);
                RCLCPP_INFO(this->get_logger(), "IMU DATA enter the deque successfully!");
                RCLCPP_INFO(this->get_logger(), "THERE are %zu IMU DATA entering the deque", imu_buffer.imu_deque_size());
        }

public:
        test_imu_buffer(const std::string node_name) : rclcpp::Node(node_name)
        {
                _sub = this->create_subscription<sensor_msgs::msg::Imu>("/livox/imu", rclcpp::SensorDataQoS(), std::bind(&test_imu_buffer::callback, this, std::placeholders::_1));
        }
};

int main(int argc, char *argv[])
{
        rclcpp::init(argc, argv);
        std::shared_ptr<test_imu_buffer> test_imu_buffer_node;
        test_imu_buffer_node = std::make_shared<test_imu_buffer>("test_imu_buffer");
        rclcpp::spin(test_imu_buffer_node);
}