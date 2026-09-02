#include <rclcpp/rclcpp.hpp>
#include "fr_slam/fr_hesai_adapter.hpp"
#include <limits>
#include <memory>
#include "sensor_msgs/msg/point_cloud2.hpp"
#include <pcl_conversions/pcl_conversions.h>

class TEST_HESAI_ADAPTER : public rclcpp::Node
{
private:
        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr _sub;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr _pub;
        void callbback(const sensor_msgs::msg::PointCloud2 &data)
        {
                HESAI_Adapter hesai_convert;
                LIDAR_FRAME lidar_frame = hesai_convert.convert(data);
                if (lidar_frame.cloud->empty())
                {
                        RCLCPP_WARN(this->get_logger(), "Converted cloud is empty!");
                        return;
                }
                double min_time_offset = std::numeric_limits<double>::max();
                double max_time_offset = std::numeric_limits<double>::lowest();
                std::uint16_t min_ring_id = std::numeric_limits<std::uint16_t>::max();
                std::uint16_t max_ring_id = std::numeric_limits<std::uint16_t>::lowest();

                for (const LIDAR_POINT &point : lidar_frame.cloud->points)
                {
                        if (min_time_offset > point.time_offset)
                        {
                                min_time_offset = point.time_offset;
                        }
                        if (max_time_offset < point.time_offset)
                        {
                                max_time_offset = point.time_offset;
                        }
                        if (min_ring_id > point.ring)
                        {
                                min_ring_id = point.ring;
                        }
                        if (max_ring_id < point.ring)
                        {
                                max_ring_id = point.ring;
                        }
                }

                RCLCPP_INFO(this->get_logger(), "================ HESAI Adapter Test ================");
                RCLCPP_INFO(this->get_logger(), "ROS raw cloud size: %u", data.width * data.height);
                RCLCPP_INFO(this->get_logger(), "unified cloud size: %u", lidar_frame.cloud->height * lidar_frame.cloud->width);
                RCLCPP_INFO(this->get_logger(), "Frame ID: %s", lidar_frame.frame_id.c_str());
                RCLCPP_INFO(this->get_logger(), "Scan start time:%.9f s", lidar_frame.scan_start_time);
                RCLCPP_INFO(this->get_logger(), "Has point time: %s", lidar_frame.has_point_time ? "true" : "false");
                RCLCPP_INFO(this->get_logger(), "Time offset: %.9f ~ %.9f s", min_time_offset, max_time_offset);
                RCLCPP_INFO(this->get_logger(), "Scan duration: %.9f s", max_time_offset - min_time_offset);
                RCLCPP_INFO(this->get_logger(), "Ring Range: %u ~ %u ", min_ring_id, max_ring_id);
                for (int i = 0; i < 5; i++)
                {
                        const LIDAR_POINT &first_point = lidar_frame.cloud->points[i];
                        RCLCPP_INFO(
                            this->get_logger(),
                            "First point: "
                            "x=%.3f y=%.3f z=%.3f "
                            "intensity=%.3f ring=%u time=%.9f",
                            first_point.x,
                            first_point.y,
                            first_point.z,
                            first_point.intensity,
                            first_point.ring,
                            first_point.time_offset);
                }

                sensor_msgs::msg::PointCloud2 _pub_data;
                pcl::toROSMsg(*lidar_frame.cloud, _pub_data);
                _pub_data.header.frame_id = lidar_frame.frame_id;
                _pub_data.header.stamp = data.header.stamp;
                _pub->publish(_pub_data);
        }

public:
        TEST_HESAI_ADAPTER(const std::string node_name) : rclcpp::Node(node_name)
        {
                _sub = this->create_subscription<sensor_msgs::msg::PointCloud2>("/lidar_points", rclcpp::SensorDataQoS(), std::bind(&TEST_HESAI_ADAPTER::callbback, this, std::placeholders::_1));
                _pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("/test_hesai_point", rclcpp::SensorDataQoS());
        }
};
int main(int argc, char *argv[])
{
        rclcpp::init(argc, argv);
        std::shared_ptr<TEST_HESAI_ADAPTER> hesai_adapter_node;
        hesai_adapter_node = std::make_shared<TEST_HESAI_ADAPTER>("hesai_adapter_node");
        rclcpp::spin(hesai_adapter_node);
}