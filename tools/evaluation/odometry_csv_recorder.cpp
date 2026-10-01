#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

class OdometryCsvRecorder final : public rclcpp::Node
{
public:
    OdometryCsvRecorder()
        : Node("odometry_csv_recorder")
    {
        topic_ = declare_parameter<std::string>(
            "topic", "/Odometry");

        output_path_ = declare_parameter<std::string>(
            "output", "/tmp/odometry_trajectory.csv");

        output_.open(
            output_path_,
            std::ios::out | std::ios::trunc);

        if (!output_.is_open())
        {
            throw std::runtime_error(
                "Cannot open output: " + output_path_);
        }

        output_ << "timestamp,x,y,z,qx,qy,qz,qw\n";
        output_ << std::setprecision(16);

        subscription_ =
            create_subscription<nav_msgs::msg::Odometry>(
                topic_,
                rclcpp::QoS(100).reliable(),
                [this](const nav_msgs::msg::Odometry::SharedPtr msg)
                {
                    const double t =
                        static_cast<double>(msg->header.stamp.sec) +
                        static_cast<double>(msg->header.stamp.nanosec) * 1e-9;

                    const auto& p = msg->pose.pose.position;
                    const auto& q = msg->pose.pose.orientation;

                    output_
                        << t << ","
                        << p.x << ","
                        << p.y << ","
                        << p.z << ","
                        << q.x << ","
                        << q.y << ","
                        << q.z << ","
                        << q.w << "\n";

                    ++count_;

                    if (count_ % 100 == 0)
                    {
                        output_.flush();

                        RCLCPP_INFO(
                            get_logger(),
                            "Recorded %zu poses",
                            count_);
                    }
                });

        RCLCPP_INFO(
            get_logger(),
            "Odometry CSV recorder started | topic=%s | output=%s",
            topic_.c_str(),
            output_path_.c_str());
    }

    ~OdometryCsvRecorder() override
    {
        if (output_.is_open())
        {
            output_.flush();
            output_.close();
        }
    }

private:
    std::string topic_;
    std::string output_path_;
    std::ofstream output_;
    std::size_t count_{0};

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr
        subscription_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    try
    {
        rclcpp::spin(
            std::make_shared<OdometryCsvRecorder>());
    }
    catch (const std::exception& e)
    {
        std::cerr
            << "odometry_csv_recorder ERROR: "
            << e.what() << "\n";

        rclcpp::shutdown();
        return 1;
    }

    rclcpp::shutdown();
    return 0;
}
