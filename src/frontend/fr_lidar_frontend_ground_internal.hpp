#pragma once

#include <cstddef>
#include <cstdlib>
#include <string>

#include <Eigen/Geometry>
#include <pcl/point_cloud.h>

#include "fr_slam/frontend/fr_lidar_frontend.hpp"
#include "fr_slam/frontend/fr_ground_segmenter.hpp"

// Source-private bridge used only to physically split the original monolithic
// fr_lidar_frontend.cpp.  No public FR-SLAM API or runtime behavior is changed.
struct GroundIcpRuntime
{
    explicit GroundIcpRuntime(
        const LidarRegistrationConfig &config)
        : registration_config(config)
    {
        const char *enable_env =
            std::getenv("FR_SLAM_GROUND_ICP_ENABLE");

        if (enable_env != nullptr)
        {
            const std::string value(enable_env);

            enabled =
                !(value == "0" ||
                  value == "false" ||
                  value == "FALSE" ||
                  value == "off" ||
                  value == "OFF");
        }
    }

    fr_slam::GroundSegmenter segmenter;
    LidarRegistrationConfig registration_config;

    bool enabled = true;
    double analysis_voxel_leaf_m = 0.15;
    double base_weight = 4.0;
    std::size_t maximum_support_points = 160;
    std::size_t minimum_ground_correspondences = 30;
    double target_normal_compatibility_deg = 20.0;
    double ground_maximum_residual_m = 0.15;
    double ground_huber_delta_m = 0.05;
    int maximum_refinement_iterations = 2;
    double maximum_total_rotation_correction_deg = 0.50;
    double maximum_total_translation_correction_m = 0.030;
    double maximum_general_rmse_ratio = 1.03;
    double maximum_general_rmse_absolute_increase_m = 0.002;
    double minimum_general_correspondence_ratio = 0.85;
};

enum class GroundJointIcpStatus
{
    NotEligible,
    Success,
    Failed
};

GroundIcpRuntime *RegisterGroundIcpRuntime(
    const RegistrationScan2LocalMap *owner,
    const LidarRegistrationConfig &registration_config);

void ResetGroundIcpRuntime(
    const RegistrationScan2LocalMap *owner);

void RemoveGroundIcpRuntime(
    const RegistrationScan2LocalMap *owner);

fr_slam::GroundSegmentationResult SegmentFrontendGround(
    const RegistrationScan2LocalMap *owner,
    const pcl::PointCloud<LIDAR_POINT>::ConstPtr &cloud_lidar,
    const char *input_source);

GroundJointIcpStatus RunTrustedGroundJointIcpV12(
    const RegistrationScan2LocalMap *owner,
    const pcl::PointCloud<LIDAR_POINT>::ConstPtr &source,
    const PreparedLidarTarget &target,
    const fr_slam::GroundSegmentationResult &ground_result,
    const Eigen::Isometry3d &initial_guess,
    LidarRegistrationResult &result);
