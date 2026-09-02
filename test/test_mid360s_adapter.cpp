#include "rclcpp/rclcpp.hpp"
#include <fr_slam/fr_mid360s_adapter.hpp>
#include "sensor_msgs/msg/point_cloud2.hpp"
#include <memory>
#include <limits>

class TEST_mid360s_Adapter : public rclcpp::Node
{
private:
        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr _sub;
        void callback(const sensor_msgs::msg::PointCloud2 data)
        {
                Mid360s_Adapter mid_360s_adapter;
                LIDAR_FRAME frame = mid_360s_adapter.convert(data);
                if (frame.cloud->empty())
                {
                        RCLCPP_WARN(this->get_logger(), "Converted cloud is empty!");
                        return;
                }
                double min_time_offset = std::numeric_limits<double>::max();
                double max_time_offset = std::numeric_limits<double>::lowest();

                std::uint16_t min_ring_id = std::numeric_limits<std::uint16_t>::max();
                std::uint16_t max_ring_id = std::numeric_limits<std::uint16_t>::lowest();

                for (const LIDAR_POINT &point : frame.cloud->points)
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
                RCLCPP_INFO(this->get_logger(), "================ MID360S Adapter Test ================");
                RCLCPP_INFO(this->get_logger(), "ROS raw cloud size: %u", data.width * data.height);
                RCLCPP_INFO(this->get_logger(), "unified cloud size: %u", frame.cloud->height * frame.cloud->width);
                RCLCPP_INFO(this->get_logger(), "Frame ID: %s", frame.frame_id.c_str());
                RCLCPP_INFO(this->get_logger(), "Scan start time:%.9f s", frame.scan_start_time);
                RCLCPP_INFO(this->get_logger(), "Has point time: %s", frame.has_point_time ? "true" : "false");
                RCLCPP_INFO(this->get_logger(), "Time offset: %.9f ~ %.9f s", min_time_offset, max_time_offset);
                RCLCPP_INFO(this->get_logger(), "Scan duration: %.9f s", max_time_offset - min_time_offset);
                RCLCPP_INFO(this->get_logger(), "Ring Range: %u ~ %u ", min_ring_id, max_ring_id);
                const LIDAR_POINT &first_point = frame.cloud->points[0];
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

public:
        TEST_mid360s_Adapter(std::string node_name) : rclcpp::Node(node_name)
        {
                _sub = this->create_subscription<sensor_msgs::msg::PointCloud2>("/livox/lidar", rclcpp::SensorDataQoS(), std::bind(&TEST_mid360s_Adapter::callback, this, std::placeholders::_1));
        };
};

int main(int argc, char *argv[])
{
        rclcpp::init(argc, argv);
        std::shared_ptr<TEST_mid360s_Adapter> mid360s_adapter_node;
        mid360s_adapter_node = std::make_shared<TEST_mid360s_Adapter>("mid360s_adapter_node");
        rclcpp::spin(mid360s_adapter_node);
        return 0;
}