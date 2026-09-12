#pragma once

#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <pcl/point_cloud.h>

#include "fr_slam/frontend/lo_frontend.hpp"
#include "fr_slam/frontend/ground_segmenter.hpp"

// Source-private runtime for Ground segmentation, the frozen world reference
// plane, and per-frame Ground refinement diagnostics.
struct GroundIcpRuntime
{
    explicit GroundIcpRuntime(
        const LidarRegistrationConfig &registration,
        const GroundConstraintConfig &ground_constraint)
        : registration_config(registration),
          constraint_config(ground_constraint),
          segmenter(ground_constraint.segmentation),
          enabled(ground_constraint.enabled)
    {
        const char *enable_env =
            std::getenv("FR_SLAM_GROUND_ICP_ENABLE");

        if (enable_env != nullptr)
        {
            const std::string value(enable_env);

            enabled =
                enabled &&
                !(value == "0" ||
                  value == "false" ||
                  value == "FALSE" ||
                  value == "off" ||
                  value == "OFF");
        }

        enabled =
            enabled &&
            constraint_config.mode != "off" &&
            constraint_config.mode != "disabled";
    }

    LidarRegistrationConfig registration_config;
    GroundConstraintConfig constraint_config;
    fr_slam::GroundSegmenter segmenter;

    bool enabled = true;

    bool reference_plane_valid = false;
    Eigen::Vector3d reference_normal_W = Eigen::Vector3d::UnitZ();
    double reference_plane_d_W = 0.0;
    std::vector<Eigen::Vector3d> anchor_normal_samples_W;
    std::vector<double> anchor_plane_d_samples_W;
    double last_anchor_sample_timestamp =
        std::numeric_limits<double>::quiet_NaN();

    std::string last_input_source = "UNKNOWN";
    std::size_t last_raw_input_points = 0;
    std::size_t last_analysis_input_points = 0;

    std::ofstream diagnostics_stream;
    std::size_t diagnostics_rows_since_flush = 0;
};

enum class GroundJointIcpStatus
{
    NotEligible,
    Success,
    Failed
};

GroundIcpRuntime *RegisterGroundIcpRuntime(
    const RegistrationScan2LocalMap *owner,
    const LidarRegistrationConfig &registration_config,
    const GroundConstraintConfig &ground_constraint_config);

void ResetGroundIcpRuntime(
    const RegistrationScan2LocalMap *owner);

void RemoveGroundIcpRuntime(
    const RegistrationScan2LocalMap *owner);

fr_slam::GroundSegmentationResult SegmentFrontendGround(
    const RegistrationScan2LocalMap *owner,
    const pcl::PointCloud<LIDAR_POINT>::ConstPtr &cloud_lidar,
    const char *input_source);

GroundJointIcpStatus RunTrustedGroundPlaneRefinementV14(
    const RegistrationScan2LocalMap *owner,
    const pcl::PointCloud<LIDAR_POINT>::ConstPtr &source,
    const PreparedLidarTarget &target,
    const fr_slam::GroundSegmentationResult &ground_result,
    const LidarRegistrationResult &general_result,
    double frontend_maximum_accepted_rmse,
    std::size_t frontend_minimum_accepted_correspondences,
    double timestamp,
    const char *stage,
    LidarRegistrationResult &result);
