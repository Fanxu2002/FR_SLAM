#include <rclcpp/rclcpp.hpp>

#include "fr_slam/fr_ground_segmenter.hpp"
#include "fr_slam/fr_lidar_frame.hpp"
#include "fr_slam/fr_point_types.hpp"
#include "fr_slam/fr_mid360s_adapter.hpp"

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <sstream>
#include <string>

class GroundSegmentationDebugNode : public rclcpp::Node
{
private:
    rclcpp::Subscription<
        sensor_msgs::msg::PointCloud2>::SharedPtr
        lidar_sub_;

    rclcpp::Publisher<
        sensor_msgs::msg::PointCloud2>::SharedPtr
        ground_pub_;

    rclcpp::Publisher<
        sensor_msgs::msg::PointCloud2>::SharedPtr
        nonground_pub_;

    rclcpp::Publisher<
        sensor_msgs::msg::PointCloud2>::SharedPtr
        support_ground_pub_;

    rclcpp::Publisher<
        sensor_msgs::msg::PointCloud2>::SharedPtr
        support_constraint_pub_;

    rclcpp::Publisher<
        sensor_msgs::msg::PointCloud2>::SharedPtr
        analysis_cloud_pub_;

    rclcpp::Publisher<
        visualization_msgs::msg::Marker>::SharedPtr
        local_ground_normal_pub_;

    rclcpp::Publisher<
        visualization_msgs::msg::Marker>::SharedPtr
        support_ground_normal_pub_;

    rclcpp::Publisher<
        visualization_msgs::msg::Marker>::SharedPtr
        support_constraint_normal_pub_;

    Mid360s_Adapter
        lidar_adapter_;

    fr_slam::GroundSegmenter
        ground_segmenter_;

    double input_voxel_leaf_m_ =
        0.15;

private:
    pcl::PointCloud<pcl::PointXYZ>::Ptr ConvertToXYZ(
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &cloud) const
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr xyz_cloud(
            new pcl::PointCloud<pcl::PointXYZ>);

        if (!cloud)
        {
            return xyz_cloud;
        }

        xyz_cloud->reserve(
            cloud->size());

        for (const LIDAR_POINT &point :
             cloud->points)
        {
            if (!std::isfinite(point.x) ||
                !std::isfinite(point.y) ||
                !std::isfinite(point.z))
            {
                continue;
            }

            pcl::PointXYZ xyz;

            xyz.x = point.x;
            xyz.y = point.y;
            xyz.z = point.z;

            xyz_cloud->push_back(
                xyz);
        }

        xyz_cloud->width =
            static_cast<std::uint32_t>(
                xyz_cloud->size());

        xyz_cloud->height = 1;
        xyz_cloud->is_dense = true;

        return xyz_cloud;
    }


    pcl::PointCloud<pcl::PointXYZ>::Ptr VoxelFilter(
        const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud) const
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(
            new pcl::PointCloud<pcl::PointXYZ>);

        if (!cloud ||
            cloud->empty())
        {
            return filtered;
        }

        if (!(input_voxel_leaf_m_ > 0.0))
        {
            *filtered =
                *cloud;

            return filtered;
        }

        pcl::VoxelGrid<pcl::PointXYZ> voxel;

        voxel.setInputCloud(
            cloud);

        const float leaf =
            static_cast<float>(
                input_voxel_leaf_m_);

        voxel.setLeafSize(
            leaf,
            leaf,
            leaf);

        voxel.filter(
            *filtered);

        return filtered;
    }


    void PublishCloud(
        const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud,
        const std_msgs::msg::Header &header,
        const rclcpp::Publisher<
            sensor_msgs::msg::PointCloud2>::SharedPtr &publisher)
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

        msg.header =
            header;

        publisher->publish(
            msg);
    }


    void PublishGroundNormal(
        const fr_slam::GroundSegmentationResult &result,
        const std_msgs::msg::Header &header)
    {
        if (!local_ground_normal_pub_)
        {
            return;
        }

        visualization_msgs::msg::Marker marker;

        marker.header =
            header;

        marker.ns =
            "ground_v3_local_normal";

        marker.id = 0;

        marker.type =
            visualization_msgs::msg::Marker::ARROW;

        marker.action =
            visualization_msgs::msg::Marker::ADD;

        marker.pose.orientation.w =
            1.0;

        marker.scale.x = 0.04;
        marker.scale.y = 0.10;
        marker.scale.z = 0.14;

        marker.color.r = 1.0F;
        marker.color.g = 1.0F;
        marker.color.b = 0.0F;
        marker.color.a = 1.0F;

        if (!result.local_plane_valid ||
            !result.local_ground_normal_L.allFinite())
        {
            marker.action =
                visualization_msgs::msg::Marker::DELETE;

            local_ground_normal_pub_->publish(
                marker);

            return;
        }

        Eigen::Vector3d normal =
            result.local_ground_normal_L;

        const double normal_norm =
            normal.norm();

        if (!std::isfinite(normal_norm) ||
            normal_norm < 1.0e-9)
        {
            marker.action =
                visualization_msgs::msg::Marker::DELETE;

            local_ground_normal_pub_->publish(
                marker);

            return;
        }

        normal /=
            normal_norm;

        geometry_msgs::msg::Point start;
        geometry_msgs::msg::Point end;

        start.x = 0.0;
        start.y = 0.0;
        start.z = 0.0;

        constexpr double kArrowLengthM =
            1.5;

        end.x =
            kArrowLengthM *
            normal.x();

        end.y =
            kArrowLengthM *
            normal.y();

        end.z =
            kArrowLengthM *
            normal.z();

        marker.points.push_back(
            start);

        marker.points.push_back(
            end);

        local_ground_normal_pub_->publish(
            marker);
    }


    void PublishSupportGroundNormal(
        const fr_slam::GroundSegmentationResult &result,
        const std_msgs::msg::Header &header)
    {
        if (!support_ground_normal_pub_)
        {
            return;
        }

        visualization_msgs::msg::Marker marker;

        marker.header =
            header;

        marker.ns =
            "ground_v40_support_normal";

        marker.id = 0;

        marker.type =
            visualization_msgs::msg::Marker::ARROW;

        marker.action =
            visualization_msgs::msg::Marker::ADD;

        marker.pose.orientation.w =
            1.0;

        marker.scale.x = 0.055;
        marker.scale.y = 0.14;
        marker.scale.z = 0.18;

        // Cyan support-normal arrow. RViz can still override point-cloud
        // colors independently.
        marker.color.r = 0.0F;
        marker.color.g = 1.0F;
        marker.color.b = 1.0F;
        marker.color.a = 1.0F;

        if (!result.support_plane_valid ||
            !result.support_ground_normal_L.allFinite())
        {
            marker.action =
                visualization_msgs::msg::Marker::DELETE;

            support_ground_normal_pub_->publish(
                marker);

            return;
        }

        Eigen::Vector3d normal =
            result.support_ground_normal_L;

        const double normal_norm =
            normal.norm();

        if (!std::isfinite(normal_norm) ||
            normal_norm < 1.0e-9)
        {
            marker.action =
                visualization_msgs::msg::Marker::DELETE;

            support_ground_normal_pub_->publish(
                marker);

            return;
        }

        normal /=
            normal_norm;

        geometry_msgs::msg::Point start;
        geometry_msgs::msg::Point end;

        start.x = 0.0;
        start.y = 0.0;
        start.z = 0.0;

        constexpr double kArrowLengthM =
            1.8;

        end.x =
            kArrowLengthM *
            normal.x();

        end.y =
            kArrowLengthM *
            normal.y();

        end.z =
            kArrowLengthM *
            normal.z();

        marker.points.push_back(
            start);

        marker.points.push_back(
            end);

        support_ground_normal_pub_->publish(
            marker);
    }


    std::string ConstraintRejectMaskToString(
        std::uint32_t mask) const
    {
        if (mask ==
            fr_slam::SUPPORT_CONSTRAINT_REJECT_NONE)
        {
            return "NONE";
        }

        std::ostringstream stream;
        bool first = true;

        const auto append_reason =
            [&](
                std::uint32_t bit,
                const char *name)
            {
                if ((mask & bit) == 0U)
                {
                    return;
                }

                if (!first)
                {
                    stream << "+";
                }

                stream << name;
                first = false;
            };

        append_reason(
            fr_slam::SUPPORT_CONSTRAINT_REJECT_NO_SUPPORT,
            "NO_SUPPORT");
        append_reason(
            fr_slam::SUPPORT_CONSTRAINT_REJECT_BOOTSTRAP,
            "BOOTSTRAP");
        append_reason(
            fr_slam::SUPPORT_CONSTRAINT_REJECT_LOW_SCORE,
            "LOW_SCORE");
        append_reason(
            fr_slam::SUPPORT_CONSTRAINT_REJECT_LOW_POINTS,
            "LOW_POINTS");
        append_reason(
            fr_slam::SUPPORT_CONSTRAINT_REJECT_LOW_CELLS,
            "LOW_CELLS");
        append_reason(
            fr_slam::SUPPORT_CONSTRAINT_REJECT_LOW_CENTER,
            "LOW_CENTER");
        append_reason(
            fr_slam::SUPPORT_CONSTRAINT_REJECT_LOW_INLIER,
            "LOW_INLIER");
        append_reason(
            fr_slam::SUPPORT_CONSTRAINT_REJECT_HIGH_RMSE,
            "HIGH_RMSE");
        append_reason(
            fr_slam::SUPPORT_CONSTRAINT_REJECT_TEMPORAL,
            "TEMPORAL");
        append_reason(
            fr_slam::SUPPORT_CONSTRAINT_REJECT_ANCHOR,
            "ANCHOR");
        append_reason(
            fr_slam::SUPPORT_CONSTRAINT_REJECT_TRUSTED_JUMP,
            "TRUSTED_JUMP");
        append_reason(
            fr_slam::SUPPORT_CONSTRAINT_REJECT_LOW_CONFIDENCE,
            "LOW_CONF");

        return stream.str();
    }


    void PublishSupportConstraintCloud(
        const fr_slam::GroundSegmentationResult &result,
        const std_msgs::msg::Header &header)
    {
        if (!support_constraint_pub_)
        {
            return;
        }

        pcl::PointCloud<pcl::PointXYZ> cloud;

        if (result.support_constraint_valid &&
            result.support_ground_cloud)
        {
            cloud =
                *result.support_ground_cloud;
        }

        sensor_msgs::msg::PointCloud2 msg;

        pcl::toROSMsg(
            cloud,
            msg);

        msg.header =
            header;

        support_constraint_pub_->publish(
            msg);
    }


    void PublishSupportConstraintNormal(
        const fr_slam::GroundSegmentationResult &result,
        const std_msgs::msg::Header &header)
    {
        if (!support_constraint_normal_pub_)
        {
            return;
        }

        visualization_msgs::msg::Marker marker;

        marker.header =
            header;

        marker.ns =
            "ground_v40_constraint_normal";

        marker.id = 0;
        marker.type =
            visualization_msgs::msg::Marker::ARROW;
        marker.action =
            visualization_msgs::msg::Marker::ADD;
        marker.pose.orientation.w = 1.0;

        marker.scale.x = 0.070;
        marker.scale.y = 0.17;
        marker.scale.z = 0.22;

        // Magenta = trusted enough for future ICP use.
        marker.color.r = 1.0F;
        marker.color.g = 0.0F;
        marker.color.b = 1.0F;
        marker.color.a = 1.0F;

        if (!result.support_constraint_valid ||
            !result.support_ground_normal_L.allFinite())
        {
            marker.action =
                visualization_msgs::msg::Marker::DELETE;

            support_constraint_normal_pub_->publish(
                marker);

            return;
        }

        Eigen::Vector3d normal =
            result.support_ground_normal_L;

        const double normal_norm =
            normal.norm();

        if (!std::isfinite(normal_norm) ||
            normal_norm < 1.0e-9)
        {
            marker.action =
                visualization_msgs::msg::Marker::DELETE;

            support_constraint_normal_pub_->publish(
                marker);

            return;
        }

        normal /=
            normal_norm;

        geometry_msgs::msg::Point start;
        geometry_msgs::msg::Point end;

        constexpr double kArrowLengthM =
            2.0;

        end.x =
            kArrowLengthM * normal.x();
        end.y =
            kArrowLengthM * normal.y();
        end.z =
            kArrowLengthM * normal.z();

        marker.points.push_back(
            start);
        marker.points.push_back(
            end);

        support_constraint_normal_pub_->publish(
            marker);
    }


    void LidarCallback(
        const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
    {
        if (!msg)
        {
            return;
        }

        const LIDAR_FRAME frame =
            lidar_adapter_.convert(
                *msg);

        if (!frame.cloud ||
            frame.cloud->empty())
        {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                1000,
                "Ground V4.0: converted LiDAR cloud is empty.");

            return;
        }

        const pcl::PointCloud<pcl::PointXYZ>::Ptr xyz_cloud =
            ConvertToXYZ(
                frame.cloud);

        const pcl::PointCloud<pcl::PointXYZ>::Ptr analysis_cloud =
            VoxelFilter(
                xyz_cloud);

        if (!analysis_cloud ||
            analysis_cloud->empty())
        {
            return;
        }

        const fr_slam::GroundSegmentationResult result =
            ground_segmenter_.Segment(
                analysis_cloud);

        PublishCloud(
            analysis_cloud,
            msg->header,
            analysis_cloud_pub_);

        PublishCloud(
            result.ground_cloud,
            msg->header,
            ground_pub_);

        PublishCloud(
            result.nonground_cloud,
            msg->header,
            nonground_pub_);

        PublishCloud(
            result.support_ground_cloud,
            msg->header,
            support_ground_pub_);

        PublishGroundNormal(
            result,
            msg->header);

        PublishSupportGroundNormal(
            result,
            msg->header);

        PublishSupportConstraintCloud(
            result,
            msg->header);

        PublishSupportConstraintNormal(
            result,
            msg->header);

        const std::string constraint_reasons =
            ConstraintRejectMaskToString(
                result.support_constraint_rejection_mask);

        RCLCPP_INFO_THROTTLE(
            this->get_logger(),
            *this->get_clock(),
            500,
            "GROUND_V40 | success=%s | raw=%zu analysis=%zu | ground_cells=%zu predicted=%zu fallback=%zu ground_pts=%zu | support=%s corridor=%zu comps=%zu small=%zu nocenter=%zu fitrej=%zu gaplinks=%zu candidates=%zu selected=%zu comp=%zu center=%zu score=%.3f support_cells=%zu support_pts=%zu | support_normal=[%.4f %.4f %.4f] tilt=%.3f deg distance=%.3f m fit=%zu/%zu ratio=%.3f rmse=%.4f | temporal=%s dnormal=%.3f ddistance=%.3f recovery=%s pending=%zu reinit=%s | constraint=%s conf=%.3f reason=%s trusted_dnormal=%.3f trusted_ddistance=%.3f | anchor=%s d=%.3f sigma=%.4f tol=%.3f err=%.3f hist=%zu sample=%s | general=%s general_distance=%.3f",
            result.success ? "true" : "false",
            frame.cloud->size(),
            analysis_cloud->size(),
            result.ground_cells,
            result.predicted_ground_cells,
            result.fallback_ground_cells,
            result.ground_points,
            result.support_plane_valid ? "VALID" : "INVALID",
            result.support_corridor_cells,
            result.support_component_count_total,
            result.support_component_rejected_small,
            result.support_component_rejected_no_center,
            result.support_component_rejected_fit,
            result.support_component_gap_links,
            result.support_candidate_count,
            result.support_selected_candidate_index,
            result.support_selected_component_cells,
            result.support_selected_center_strip_cells,
            result.support_selected_score,
            result.support_ground_cells,
            result.support_ground_points,
            result.support_ground_normal_L.x(),
            result.support_ground_normal_L.y(),
            result.support_ground_normal_L.z(),
            result.support_ground_tilt_deg,
            result.support_ground_distance_m,
            result.support_plane_inliers,
            result.support_plane_cells,
            result.support_plane_inlier_ratio,
            result.support_plane_rmse_m,
            result.support_temporal_gate_passed ? "PASS" : "FAIL",
            result.support_normal_change_deg,
            result.support_distance_change_m,
            result.support_recovery_pending ? "PENDING" : "NONE",
            result.support_recovery_pending_count,
            result.support_reinitialized ? "YES" : "NO",
            result.support_constraint_valid ? "VALID" : "INVALID",
            result.support_constraint_confidence,
            constraint_reasons.c_str(),
            result.support_trusted_normal_change_deg,
            result.support_trusted_distance_change_m,
            result.support_clearance_anchor_valid ? "READY" : "BOOTSTRAP",
            result.support_clearance_anchor_m,
            result.support_clearance_anchor_sigma_m,
            result.support_clearance_anchor_tolerance_m,
            result.support_clearance_error_m,
            result.support_clearance_history_samples,
            result.support_clearance_sample_accepted ? "ADD" : "NO",
            result.local_plane_valid ? "VALID" : "INVALID",
            result.local_ground_distance_m);

    }

public:
    GroundSegmentationDebugNode()
        : rclcpp::Node(
              "ground_segmentation_debug")
    {
        input_voxel_leaf_m_ =
            this->declare_parameter<double>(
                "input_voxel_leaf_m",
                0.15);

        fr_slam::GroundSegmentationConfig config;

        config.minimum_range_m =
            this->declare_parameter<double>(
                "minimum_range_m",
                0.8);

        config.maximum_range_m =
            this->declare_parameter<double>(
                "maximum_range_m",
                15.0);

        config.grid_size_m =
            this->declare_parameter<double>(
                "grid_size_m",
                0.40);

        config.minimum_points_per_cell =
            static_cast<std::size_t>(
                std::max<std::int64_t>(
                    1,
                    this->declare_parameter<std::int64_t>(
                        "minimum_points_per_cell",
                        2)));

        config.low_surface_fraction =
            this->declare_parameter<double>(
                "low_surface_fraction",
                0.30);

        config.seed_minimum_range_m =
            this->declare_parameter<double>(
                "seed_minimum_range_m",
                0.8);

        config.seed_maximum_range_m =
            this->declare_parameter<double>(
                "seed_maximum_range_m",
                4.0);

        config.minimum_seed_component_cells =
            static_cast<std::size_t>(
                std::max<std::int64_t>(
                    1,
                    this->declare_parameter<std::int64_t>(
                        "minimum_seed_component_cells",
                        6)));

        config.seed_angular_sector_count =
            static_cast<std::size_t>(
                std::max<std::int64_t>(
                    1,
                    this->declare_parameter<std::int64_t>(
                        "seed_angular_sector_count",
                        16)));

        config.minimum_seed_angular_sectors =
            static_cast<std::size_t>(
                std::max<std::int64_t>(
                    1,
                    this->declare_parameter<std::int64_t>(
                        "minimum_seed_angular_sectors",
                        4)));

        config.maximum_cell_low_roughness_m =
            this->declare_parameter<double>(
                "maximum_cell_low_roughness_m",
                0.10);

        config.prediction_neighbor_radius_cells =
            static_cast<int>(
                std::max<std::int64_t>(
                    1,
                    this->declare_parameter<std::int64_t>(
                        "prediction_neighbor_radius_cells",
                        2)));

        config.minimum_prediction_support_cells =
            static_cast<std::size_t>(
                std::max<std::int64_t>(
                    3,
                    this->declare_parameter<std::int64_t>(
                        "minimum_prediction_support_cells",
                        4)));

        config.maximum_predicted_surface_slope_deg =
            this->declare_parameter<double>(
                "maximum_predicted_surface_slope_deg",
                40.0);

        config.surface_prediction_base_tolerance_m =
            this->declare_parameter<double>(
                "surface_prediction_base_tolerance_m",
                0.07);

        config.surface_prediction_roughness_scale =
            this->declare_parameter<double>(
                "surface_prediction_roughness_scale",
                2.5);

        config.surface_prediction_rmse_scale =
            this->declare_parameter<double>(
                "surface_prediction_rmse_scale",
                2.0);

        config.maximum_surface_prediction_tolerance_m =
            this->declare_parameter<double>(
                "maximum_surface_prediction_tolerance_m",
                0.20);

        config.maximum_local_slope_deg =
            this->declare_parameter<double>(
                "maximum_local_slope_deg",
                30.0);

        config.maximum_neighbor_height_jump_m =
            this->declare_parameter<double>(
                "maximum_neighbor_height_jump_m",
                0.40);

        config.point_below_surface_tolerance_m =
            this->declare_parameter<double>(
                "point_below_surface_tolerance_m",
                0.06);

        config.point_height_base_threshold_m =
            this->declare_parameter<double>(
                "point_height_base_threshold_m",
                0.06);

        config.point_height_roughness_scale =
            this->declare_parameter<double>(
                "point_height_roughness_scale",
                3.0);

        config.point_height_minimum_threshold_m =
            this->declare_parameter<double>(
                "point_height_minimum_threshold_m",
                0.05);

        config.point_height_maximum_threshold_m =
            this->declare_parameter<double>(
                "point_height_maximum_threshold_m",
                0.16);

        config.local_plane_minimum_range_m =
            this->declare_parameter<double>(
                "local_plane_minimum_range_m",
                0.8);

        config.local_plane_maximum_range_m =
            this->declare_parameter<double>(
                "local_plane_maximum_range_m",
                4.0);

        config.minimum_local_plane_cells =
            static_cast<std::size_t>(
                std::max<std::int64_t>(
                    3,
                    this->declare_parameter<std::int64_t>(
                        "minimum_local_plane_cells",
                        12)));

        config.local_plane_base_residual_m =
            this->declare_parameter<double>(
                "local_plane_base_residual_m",
                0.05);

        config.local_plane_mad_scale =
            this->declare_parameter<double>(
                "local_plane_mad_scale",
                3.0);

        config.local_plane_maximum_residual_m =
            this->declare_parameter<double>(
                "local_plane_maximum_residual_m",
                0.18);

        config.minimum_local_plane_inlier_ratio =
            this->declare_parameter<double>(
                "minimum_local_plane_inlier_ratio",
                0.55);

        // ========================================================
        // Ground V3.3 gap-tolerant multi-hypothesis support-surface selector.
        // ========================================================
        config.support_corridor_rear_m =
            this->declare_parameter<double>(
                "support_corridor_rear_m",
                1.5);

        config.support_corridor_forward_m =
            this->declare_parameter<double>(
                "support_corridor_forward_m",
                4.0);

        config.support_corridor_half_width_m =
            this->declare_parameter<double>(
                "support_corridor_half_width_m",
                1.10);

        config.support_maximum_range_m =
            this->declare_parameter<double>(
                "support_maximum_range_m",
                5.0);

        config.support_primary_half_width_m =
            this->declare_parameter<double>(
                "support_primary_half_width_m",
                0.60);

        config.support_component_local_model_radius_cells =
            static_cast<int>(
                std::max<std::int64_t>(
                    1,
                    this->declare_parameter<std::int64_t>(
                        "support_component_local_model_radius_cells",
                        2)));

        config.minimum_support_component_local_cells =
            static_cast<std::size_t>(
                std::max<std::int64_t>(
                    3,
                    this->declare_parameter<std::int64_t>(
                        "minimum_support_component_local_cells",
                        4)));

        config.support_component_maximum_normal_change_deg =
            this->declare_parameter<double>(
                "support_component_maximum_normal_change_deg",
                6.0);

        config.support_component_maximum_model_residual_m =
            this->declare_parameter<double>(
                "support_component_maximum_model_residual_m",
                0.08);

        config.support_component_fallback_maximum_slope_deg =
            this->declare_parameter<double>(
                "support_component_fallback_maximum_slope_deg",
                18.0);

        config.support_component_fallback_maximum_height_jump_m =
            this->declare_parameter<double>(
                "support_component_fallback_maximum_height_jump_m",
                0.20);

        config.support_component_gap_radius_cells =
            static_cast<int>(
                std::clamp<std::int64_t>(
                    this->declare_parameter<std::int64_t>(
                        "support_component_gap_radius_cells",
                        2),
                    1,
                    2));

        config.support_component_gap_normal_scale =
            this->declare_parameter<double>(
                "support_component_gap_normal_scale",
                0.75);

        config.support_component_gap_residual_scale =
            this->declare_parameter<double>(
                "support_component_gap_residual_scale",
                0.75);

        config.support_component_gap_fallback_slope_scale =
            this->declare_parameter<double>(
                "support_component_gap_fallback_slope_scale",
                0.75);

        config.support_component_gap_height_jump_scale =
            this->declare_parameter<double>(
                "support_component_gap_height_jump_scale",
                0.75);

        config.minimum_support_candidate_cells =
            static_cast<std::size_t>(
                std::max<std::int64_t>(
                    3,
                    this->declare_parameter<std::int64_t>(
                        "minimum_support_candidate_cells",
                        4)));

        config.minimum_support_center_strip_cells =
            static_cast<std::size_t>(
                std::max<std::int64_t>(
                    1,
                    this->declare_parameter<std::int64_t>(
                        "minimum_support_center_strip_cells",
                        2)));

        config.support_center_gaussian_sigma_m =
            this->declare_parameter<double>(
                "support_center_gaussian_sigma_m",
                0.45);

        config.support_score_center_strip_weight =
            this->declare_parameter<double>(
                "support_score_center_strip_weight",
                4.0);

        config.support_score_center_proximity_weight =
            this->declare_parameter<double>(
                "support_score_center_proximity_weight",
                2.0);

        config.support_score_area_weight =
            this->declare_parameter<double>(
                "support_score_area_weight",
                1.0);

        config.support_score_fit_weight =
            this->declare_parameter<double>(
                "support_score_fit_weight",
                1.5);

        config.support_score_rmse_weight =
            this->declare_parameter<double>(
                "support_score_rmse_weight",
                1.0);

        config.support_score_temporal_normal_weight =
            this->declare_parameter<double>(
                "support_score_temporal_normal_weight",
                3.0);

        config.support_score_temporal_distance_weight =
            this->declare_parameter<double>(
                "support_score_temporal_distance_weight",
                3.0);

        config.support_score_area_saturation_cells =
            static_cast<std::size_t>(
                std::max<std::int64_t>(
                    1,
                    this->declare_parameter<std::int64_t>(
                        "support_score_area_saturation_cells",
                        25)));

        config.support_score_temporal_normal_sigma_deg =
            this->declare_parameter<double>(
                "support_score_temporal_normal_sigma_deg",
                4.0);

        config.support_score_temporal_distance_sigma_m =
            this->declare_parameter<double>(
                "support_score_temporal_distance_sigma_m",
                0.06);

        config.support_recovery_required_consecutive_frames =
            static_cast<std::size_t>(
                std::max<std::int64_t>(
                    1,
                    this->declare_parameter<std::int64_t>(
                        "support_recovery_required_consecutive_frames",
                        6)));

        config.support_recovery_maximum_candidate_normal_change_deg =
            this->declare_parameter<double>(
                "support_recovery_maximum_candidate_normal_change_deg",
                2.5);

        config.support_recovery_maximum_candidate_distance_change_m =
            this->declare_parameter<double>(
                "support_recovery_maximum_candidate_distance_change_m",
                0.05);

        config.support_recovery_minimum_center_strip_cells =
            static_cast<std::size_t>(
                std::max<std::int64_t>(
                    1,
                    this->declare_parameter<std::int64_t>(
                        "support_recovery_minimum_center_strip_cells",
                        4)));

        config.support_recovery_minimum_inlier_ratio =
            this->declare_parameter<double>(
                "support_recovery_minimum_inlier_ratio",
                0.85);

        config.support_recovery_maximum_rmse_m =
            this->declare_parameter<double>(
                "support_recovery_maximum_rmse_m",
                0.05);

        config.minimum_support_corridor_cells =
            static_cast<std::size_t>(
                std::max<std::int64_t>(
                    3,
                    this->declare_parameter<std::int64_t>(
                        "minimum_support_corridor_cells",
                        8)));

        config.minimum_support_plane_cells =
            static_cast<std::size_t>(
                std::max<std::int64_t>(
                    3,
                    this->declare_parameter<std::int64_t>(
                        "minimum_support_plane_cells",
                        10)));

        config.support_plane_base_residual_m =
            this->declare_parameter<double>(
                "support_plane_base_residual_m",
                0.04);

        config.support_plane_mad_scale =
            this->declare_parameter<double>(
                "support_plane_mad_scale",
                3.0);

        config.support_plane_maximum_residual_m =
            this->declare_parameter<double>(
                "support_plane_maximum_residual_m",
                0.12);

        config.minimum_support_plane_inlier_ratio =
            this->declare_parameter<double>(
                "minimum_support_plane_inlier_ratio",
                0.60);

        config.support_cell_base_residual_m =
            this->declare_parameter<double>(
                "support_cell_base_residual_m",
                0.06);

        config.support_cell_roughness_scale =
            this->declare_parameter<double>(
                "support_cell_roughness_scale",
                2.0);

        config.support_cell_maximum_residual_m =
            this->declare_parameter<double>(
                "support_cell_maximum_residual_m",
                0.14);

        config.maximum_support_normal_change_deg =
            this->declare_parameter<double>(
                "maximum_support_normal_change_deg",
                8.0);

        config.maximum_support_distance_change_m =
            this->declare_parameter<double>(
                "maximum_support_distance_change_m",
                0.12);

        config.support_constraint_minimum_points =
            static_cast<std::size_t>(
                std::max<std::int64_t>(
                    1,
                    this->declare_parameter<std::int64_t>(
                        "support_constraint_minimum_points",
                        120)));

        config.support_constraint_minimum_cells =
            static_cast<std::size_t>(
                std::max<std::int64_t>(
                    1,
                    this->declare_parameter<std::int64_t>(
                        "support_constraint_minimum_cells",
                        35)));

        config.support_constraint_minimum_center_strip_cells =
            static_cast<std::size_t>(
                std::max<std::int64_t>(
                    1,
                    this->declare_parameter<std::int64_t>(
                        "support_constraint_minimum_center_strip_cells",
                        3)));

        config.support_constraint_minimum_inlier_ratio =
            this->declare_parameter<double>(
                "support_constraint_minimum_inlier_ratio",
                0.90);

        config.support_constraint_maximum_rmse_m =
            this->declare_parameter<double>(
                "support_constraint_maximum_rmse_m",
                0.030);

        config.support_constraint_minimum_selected_score =
            this->declare_parameter<double>(
                "support_constraint_minimum_selected_score",
                8.5);

        config.support_constraint_enter_confidence =
            this->declare_parameter<double>(
                "support_constraint_enter_confidence",
                0.72);

        config.support_constraint_keep_confidence =
            this->declare_parameter<double>(
                "support_constraint_keep_confidence",
                0.62);

        config.support_constraint_maximum_trusted_normal_change_deg =
            this->declare_parameter<double>(
                "support_constraint_maximum_trusted_normal_change_deg",
                8.0);

        config.support_constraint_maximum_trusted_distance_change_m =
            this->declare_parameter<double>(
                "support_constraint_maximum_trusted_distance_change_m",
                0.10);

        config.support_clearance_bootstrap_samples =
            static_cast<std::size_t>(
                std::max<std::int64_t>(
                    1,
                    this->declare_parameter<std::int64_t>(
                        "support_clearance_bootstrap_samples",
                        8)));

        config.support_clearance_history_size =
            static_cast<std::size_t>(
                std::max<std::int64_t>(
                    1,
                    this->declare_parameter<std::int64_t>(
                        "support_clearance_history_size",
                        31)));

        config.support_clearance_bootstrap_minimum_points =
            static_cast<std::size_t>(
                std::max<std::int64_t>(
                    1,
                    this->declare_parameter<std::int64_t>(
                        "support_clearance_bootstrap_minimum_points",
                        150)));

        config.support_clearance_bootstrap_minimum_cells =
            static_cast<std::size_t>(
                std::max<std::int64_t>(
                    1,
                    this->declare_parameter<std::int64_t>(
                        "support_clearance_bootstrap_minimum_cells",
                        40)));

        config.support_clearance_bootstrap_minimum_center_strip_cells =
            static_cast<std::size_t>(
                std::max<std::int64_t>(
                    1,
                    this->declare_parameter<std::int64_t>(
                        "support_clearance_bootstrap_minimum_center_strip_cells",
                        4)));

        config.support_clearance_bootstrap_minimum_inlier_ratio =
            this->declare_parameter<double>(
                "support_clearance_bootstrap_minimum_inlier_ratio",
                0.92);

        config.support_clearance_bootstrap_maximum_rmse_m =
            this->declare_parameter<double>(
                "support_clearance_bootstrap_maximum_rmse_m",
                0.030);

        config.support_clearance_bootstrap_minimum_selected_score =
            this->declare_parameter<double>(
                "support_clearance_bootstrap_minimum_selected_score",
                7.5);

        config.support_clearance_bootstrap_provisional_gate_samples =
            static_cast<std::size_t>(
                std::max<std::int64_t>(
                    1,
                    this->declare_parameter<std::int64_t>(
                        "support_clearance_bootstrap_provisional_gate_samples",
                        3)));

        config.support_clearance_bootstrap_maximum_deviation_m =
            this->declare_parameter<double>(
                "support_clearance_bootstrap_maximum_deviation_m",
                0.07);

        config.support_clearance_anchor_base_tolerance_m =
            this->declare_parameter<double>(
                "support_clearance_anchor_base_tolerance_m",
                0.060);

        config.support_clearance_anchor_mad_scale =
            this->declare_parameter<double>(
                "support_clearance_anchor_mad_scale",
                3.5);

        config.support_clearance_anchor_maximum_tolerance_m =
            this->declare_parameter<double>(
                "support_clearance_anchor_maximum_tolerance_m",
                0.090);

        config.support_clearance_history_update_maximum_error_m =
            this->declare_parameter<double>(
                "support_clearance_history_update_maximum_error_m",
                0.050);

        ground_segmenter_.SetConfig(
            config);

        ground_pub_ =
            this->create_publisher<
                sensor_msgs::msg::PointCloud2>(
                "/ground_cloud",
                rclcpp::SensorDataQoS());

        nonground_pub_ =
            this->create_publisher<
                sensor_msgs::msg::PointCloud2>(
                "/nonground_cloud",
                rclcpp::SensorDataQoS());

        support_ground_pub_ =
            this->create_publisher<
                sensor_msgs::msg::PointCloud2>(
                "/support_ground_cloud",
                rclcpp::SensorDataQoS());

        support_constraint_pub_ =
            this->create_publisher<
                sensor_msgs::msg::PointCloud2>(
                "/support_constraint_cloud",
                rclcpp::SensorDataQoS());

        analysis_cloud_pub_ =
            this->create_publisher<
                sensor_msgs::msg::PointCloud2>(
                "/ground_analysis_cloud",
                rclcpp::SensorDataQoS());

        local_ground_normal_pub_ =
            this->create_publisher<
                visualization_msgs::msg::Marker>(
                "/ground_local_normal",
                10);

        support_ground_normal_pub_ =
            this->create_publisher<
                visualization_msgs::msg::Marker>(
                "/support_ground_normal",
                10);

        support_constraint_normal_pub_ =
            this->create_publisher<
                visualization_msgs::msg::Marker>(
                "/support_constraint_normal",
                10);

        lidar_sub_ =
            this->create_subscription<
                sensor_msgs::msg::PointCloud2>(
                "/livox/lidar",
                rclcpp::SensorDataQoS(),
                std::bind(
                    &GroundSegmentationDebugNode::LidarCallback,
                    this,
                    std::placeholders::_1));

        RCLCPP_INFO(
            this->get_logger(),
            "Ground V4.0 trusted support-constraint node started | input=/livox/lidar | ground=/ground_cloud | support=/support_ground_cloud | trusted=/support_constraint_cloud | support_normal=/support_ground_normal | trusted_normal=/support_constraint_normal | voxel=%.3f m | grid=%.3f m | corridor=[rear %.2f forward %.2f half_width %.2f primary %.2f] m | gap_radius=%d min_candidate=%zu | anchor_bootstrap=%zu history=%zu",
            input_voxel_leaf_m_,
            config.grid_size_m,
            config.support_corridor_rear_m,
            config.support_corridor_forward_m,
            config.support_corridor_half_width_m,
            config.support_primary_half_width_m,
            config.support_component_gap_radius_cells,
            config.minimum_support_candidate_cells,
            config.support_clearance_bootstrap_samples,
            config.support_clearance_history_size);
    }
};


int main(
    int argc,
    char **argv)
{
    rclcpp::init(
        argc,
        argv);

    const std::shared_ptr<GroundSegmentationDebugNode> node =
        std::make_shared<GroundSegmentationDebugNode>();

    rclcpp::spin(
        node);

    rclcpp::shutdown();

    return 0;
}
