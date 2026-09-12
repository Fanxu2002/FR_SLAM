#include "fr_slam/frontend/lo_frontend.hpp"
#include "fr_slam/frontend/ground_segmenter.hpp"
#include "fr_slam/frontend/ground_input_bridge.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <cstdint>
#include <cstddef>
#include <exception>
#include <utility>
#include <vector>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>
#include <pcl/search/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <Eigen/Eigenvalues>

#include <sophus/so3.hpp>

#include <rclcpp/rclcpp.hpp>
#include "lo_frontend_ground_internal.hpp"

    struct GroundIcpLinearization
    {
        Eigen::Matrix<double, 6, 6> H =
            Eigen::Matrix<double, 6, 6>::Zero();

        Eigen::Matrix<double, 6, 1> b =
            Eigen::Matrix<double, 6, 1>::Zero();

        std::size_t general_correspondences = 0;
        std::size_t ground_correspondences = 0;

        std::size_t general_downweighted_correspondences = 0;
        double general_min_robust_weight = 1.0;

        double general_raw_squared_error_sum = 0.0;
        double general_weighted_squared_error_sum = 0.0;
        double general_weight_sum = 0.0;

        double ground_raw_squared_error_sum = 0.0;
        double ground_weighted_squared_error_sum = 0.0;
        double ground_weight_sum = 0.0;

        double ground_height_residual_m =
            std::numeric_limits<double>::quiet_NaN();

        double ground_normal_residual_deg =
            std::numeric_limits<double>::quiet_NaN();

        double ground_quality_weight = 0.0;

        std::vector<double> correspondence_ranges;

        double GeneralRmse() const
        {
            if (general_correspondences == 0)
            {
                return std::numeric_limits<double>::infinity();
            }

            return std::sqrt(
                general_raw_squared_error_sum /
                static_cast<double>(general_correspondences));
        }

        double GroundRmse() const
        {
            if (ground_correspondences == 0)
            {
                return std::numeric_limits<double>::infinity();
            }

            return std::sqrt(
                ground_raw_squared_error_sum /
                static_cast<double>(ground_correspondences));
        }

        double CombinedWeightedMeanSquaredError() const
        {
            const double total_weight =
                general_weight_sum +
                ground_weight_sum;

            if (!std::isfinite(total_weight) ||
                total_weight <= 1.0e-12)
            {
                return std::numeric_limits<double>::infinity();
            }

            return (general_weighted_squared_error_sum +
                    ground_weighted_squared_error_sum) /
                   total_weight;
        }
    };

    struct GroundDiagnosticRecord
    {
        double timestamp =
            std::numeric_limits<double>::quiet_NaN();

        std::string stage = "UNKNOWN";
        std::string status = "NOT_ELIGIBLE";
        std::string reason = "UNKNOWN";

        double quality_weight = 0.0;
        double height_residual_m =
            std::numeric_limits<double>::quiet_NaN();
        double normal_residual_deg =
            std::numeric_limits<double>::quiet_NaN();

        std::size_t general_correspondences_before = 0;
        double general_rmse_before =
            std::numeric_limits<double>::infinity();
        std::size_t general_correspondences_after = 0;
        double general_rmse_after =
            std::numeric_limits<double>::infinity();

        Eigen::Vector3d correction_translation_W =
            Eigen::Vector3d::Zero();
        double correction_rotation_deg = 0.0;
        int refinement_iterations = 0;
    };

    constexpr double kGroundPi =
        3.14159265358979323846;

    double GroundRadiansToDegrees(double radians)
    {
        return radians * 180.0 / kGroundPi;
    }

    double GroundDegreesToRadians(double degrees)
    {
        return degrees * kGroundPi / 180.0;
    }

    double GroundMedian(std::vector<double> values)
    {
        if (values.empty())
        {
            return std::numeric_limits<double>::quiet_NaN();
        }

        const std::size_t middle_index =
            values.size() / 2U;

        std::nth_element(
            values.begin(),
            values.begin() +
                static_cast<std::ptrdiff_t>(middle_index),
            values.end());

        double median = values[middle_index];

        if ((values.size() % 2U) == 0U)
        {
            const auto lower_middle =
                std::max_element(
                    values.begin(),
                    values.begin() +
                        static_cast<std::ptrdiff_t>(middle_index));

            if (lower_middle !=
                values.begin() +
                    static_cast<std::ptrdiff_t>(middle_index))
            {
                median = 0.5 *
                         (median + *lower_middle);
            }
        }

        return median;
    }

    void InitializeGroundDiagnostics(
        GroundIcpRuntime &runtime)
    {
        if (!runtime.constraint_config.diagnostics_enabled)
        {
            return;
        }

        const char *output_directory_environment =
            std::getenv("FR_SLAM_OUTPUT_DIR");

        if (output_directory_environment == nullptr ||
            std::string(output_directory_environment).empty())
        {
            return;
        }

        try
        {
            const std::filesystem::path diagnostics_directory =
                std::filesystem::path(output_directory_environment) /
                "diagnostics" /
                "frontend";

            std::filesystem::create_directories(
                diagnostics_directory);

            runtime.diagnostics_stream.open(
                diagnostics_directory /
                    "ground_frontend_diagnostics.csv",
                std::ios::out |
                    std::ios::trunc);

            if (runtime.diagnostics_stream.is_open())
            {
                runtime.diagnostics_stream
                    << "timestamp,stage,input_source,status,reason,"
                    << "raw_input_points,analysis_input_points,"
                    << "constraint_valid,rejection_mask,confidence,"
                    << "anchor_ready,anchor_samples,"
                    << "reference_nx,reference_ny,reference_nz,reference_d,"
                    << "support_points,support_cells,current_nx,current_ny,"
                    << "current_nz,current_plane_d,current_distance,current_rmse,"
                    << "clearance_anchor,clearance_error,quality_weight,"
                    << "height_residual,normal_residual_deg,"
                    << "general_corr_before,general_rmse_before,"
                    << "general_corr_after,general_rmse_after,"
                    << "delta_x,delta_y,delta_z,delta_rotation_deg,iterations\n";
            }
        }
        catch (const std::exception &exception)
        {
            std::cerr
                << "GROUND_ICP_V14 | diagnostics=DISABLED"
                << " | reason=" << exception.what()
                << std::endl;
        }
    }

    void WriteGroundDiagnostic(
        GroundIcpRuntime &runtime,
        const fr_slam::GroundSegmentationResult &ground_result,
        const GroundDiagnosticRecord &record)
    {
        if (!runtime.diagnostics_stream.is_open())
        {
            return;
        }

        const Eigen::Vector3d current_normal =
            ground_result.support_ground_normal_L;

        runtime.diagnostics_stream
            << std::setprecision(12)
            << record.timestamp << ','
            << record.stage << ','
            << runtime.last_input_source << ','
            << record.status << ','
            << record.reason << ','
            << runtime.last_raw_input_points << ','
            << runtime.last_analysis_input_points << ','
            << (ground_result.support_constraint_valid ? 1 : 0) << ','
            << ground_result.support_constraint_rejection_mask << ','
            << ground_result.support_constraint_confidence << ','
            << (runtime.reference_plane_valid ? 1 : 0) << ','
            << runtime.anchor_plane_d_samples_W.size() << ','
            << runtime.reference_normal_W.x() << ','
            << runtime.reference_normal_W.y() << ','
            << runtime.reference_normal_W.z() << ','
            << runtime.reference_plane_d_W << ','
            << ground_result.support_ground_points << ','
            << ground_result.support_ground_cells << ','
            << current_normal.x() << ','
            << current_normal.y() << ','
            << current_normal.z() << ','
            << ground_result.support_ground_plane_d << ','
            << ground_result.support_ground_distance_m << ','
            << ground_result.support_plane_rmse_m << ','
            << ground_result.support_clearance_anchor_m << ','
            << ground_result.support_clearance_error_m << ','
            << record.quality_weight << ','
            << record.height_residual_m << ','
            << record.normal_residual_deg << ','
            << record.general_correspondences_before << ','
            << record.general_rmse_before << ','
            << record.general_correspondences_after << ','
            << record.general_rmse_after << ','
            << record.correction_translation_W.x() << ','
            << record.correction_translation_W.y() << ','
            << record.correction_translation_W.z() << ','
            << record.correction_rotation_deg << ','
            << record.refinement_iterations
            << '\n';

        ++runtime.diagnostics_rows_since_flush;

        if (runtime.diagnostics_rows_since_flush >=
            std::max<std::size_t>(
                1,
                runtime.constraint_config.diagnostics_flush_interval))
        {
            runtime.diagnostics_stream.flush();
            runtime.diagnostics_rows_since_flush = 0;
        }
    }

    std::mutex &GroundIcpRuntimeMutex()
    {
        static std::mutex mutex;
        return mutex;
    }

    std::unordered_map<
        const RegistrationScan2LocalMap *,
        std::unique_ptr<GroundIcpRuntime>> &
    GroundIcpRuntimeMap()
    {
        static std::unordered_map<
            const RegistrationScan2LocalMap *,
            std::unique_ptr<GroundIcpRuntime>>
            runtime_map;

        return runtime_map;
    }

    GroundIcpRuntime *RegisterGroundIcpRuntime(
        const RegistrationScan2LocalMap *owner,
        const LidarRegistrationConfig &registration_config,
        const GroundConstraintConfig &ground_constraint_config)
    {
        if (owner == nullptr)
        {
            return nullptr;
        }

        std::lock_guard<std::mutex> lock(
            GroundIcpRuntimeMutex());

        std::unique_ptr<GroundIcpRuntime> runtime =
            std::make_unique<GroundIcpRuntime>(
                registration_config,
                ground_constraint_config);

        InitializeGroundDiagnostics(
            *runtime);

        GroundIcpRuntime *runtime_pointer =
            runtime.get();

        GroundIcpRuntimeMap()[owner] =
            std::move(runtime);

        return runtime_pointer;
    }

    GroundIcpRuntime *GetGroundIcpRuntime(
        const RegistrationScan2LocalMap *owner)
    {
        std::lock_guard<std::mutex> lock(
            GroundIcpRuntimeMutex());

        const auto iterator =
            GroundIcpRuntimeMap().find(owner);

        if (iterator == GroundIcpRuntimeMap().end())
        {
            return nullptr;
        }

        return iterator->second.get();
    }

    void ResetGroundIcpRuntime(
        const RegistrationScan2LocalMap *owner)
    {
        GroundIcpRuntime *runtime =
            GetGroundIcpRuntime(owner);

        if (runtime != nullptr)
        {
            runtime->segmenter.Reset();
            runtime->reference_plane_valid = false;
            runtime->reference_normal_W =
                Eigen::Vector3d::UnitZ();
            runtime->reference_plane_d_W = 0.0;
            runtime->anchor_normal_samples_W.clear();
            runtime->anchor_plane_d_samples_W.clear();
            runtime->last_anchor_sample_timestamp =
                std::numeric_limits<double>::quiet_NaN();
        }
    }

    void RemoveGroundIcpRuntime(
        const RegistrationScan2LocalMap *owner)
    {
        std::lock_guard<std::mutex> lock(
            GroundIcpRuntimeMutex());

        GroundIcpRuntimeMap().erase(owner);
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr
    ConvertToGroundAnalysisCloud(
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &cloud_lidar)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(
            new pcl::PointCloud<pcl::PointXYZ>);

        if (!cloud_lidar)
        {
            return cloud_xyz;
        }

        cloud_xyz->reserve(
            cloud_lidar->size());

        for (const LIDAR_POINT &point :
             cloud_lidar->points)
        {
            if (!std::isfinite(point.x) ||
                !std::isfinite(point.y) ||
                !std::isfinite(point.z))
            {
                continue;
            }

            pcl::PointXYZ point_xyz;
            point_xyz.x = point.x;
            point_xyz.y = point.y;
            point_xyz.z = point.z;

            cloud_xyz->push_back(
                point_xyz);
        }

        cloud_xyz->width =
            static_cast<std::uint32_t>(
                cloud_xyz->size());

        cloud_xyz->height = 1;
        cloud_xyz->is_dense = true;

        return cloud_xyz;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr
    VoxelizeGroundAnalysisCloud(
        const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud,
        double leaf_size_m)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(
            new pcl::PointCloud<pcl::PointXYZ>);

        if (!cloud ||
            cloud->empty())
        {
            return filtered;
        }

        if (!std::isfinite(leaf_size_m) ||
            leaf_size_m <= 0.0)
        {
            *filtered = *cloud;
            return filtered;
        }

        pcl::VoxelGrid<pcl::PointXYZ> voxel;
        voxel.setInputCloud(cloud);

        const float leaf =
            static_cast<float>(leaf_size_m);

        voxel.setLeafSize(
            leaf,
            leaf,
            leaf);

        voxel.filter(*filtered);

        return filtered;
    }

    fr_slam::GroundSegmentationResult
    SegmentFrontendGround(
        const RegistrationScan2LocalMap *owner,
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &cloud_lidar,
        const char *input_source)
    {
        fr_slam::GroundSegmentationResult result;

        GroundIcpRuntime *runtime =
            GetGroundIcpRuntime(owner);

        if (runtime == nullptr ||
            !runtime->enabled)
        {
            return result;
        }

        const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz =
            ConvertToGroundAnalysisCloud(
                cloud_lidar);

        const pcl::PointCloud<pcl::PointXYZ>::Ptr analysis_cloud =
            VoxelizeGroundAnalysisCloud(
                cloud_xyz,
                runtime->constraint_config.analysis_voxel_leaf_m);

        runtime->last_input_source =
            input_source != nullptr
                ? input_source
                : "UNKNOWN";

        runtime->last_raw_input_points =
            cloud_xyz->size();

        runtime->last_analysis_input_points =
            analysis_cloud->size();

        result =
            runtime->segmenter.Segment(
                analysis_cloud);

        return result;
    }

    double GroundIcpAnchorFactor(
        const fr_slam::GroundSegmentationResult &ground_result)
    {
        if (!ground_result.support_clearance_anchor_valid ||
            !std::isfinite(
                ground_result.support_clearance_error_m) ||
            !std::isfinite(
                ground_result.support_clearance_anchor_tolerance_m) ||
            ground_result.support_clearance_anchor_tolerance_m <= 0.0)
        {
            return 1.0;
        }

        // A trusted frame sitting near the outer hard anchor gate should not
        // suddenly obtain the same optimizer strength as a frame centered on
        // the learned clearance distribution.
        const double sigma_m =
            std::max(
                0.025,
                0.5 *
                    ground_result
                        .support_clearance_anchor_tolerance_m);

        const double normalized_error =
            ground_result.support_clearance_error_m /
            sigma_m;

        return std::exp(
            -0.5 *
            normalized_error *
            normalized_error);
    }

    double ComputeGroundIcpWeight(
        const GroundIcpRuntime &runtime,
        const fr_slam::GroundSegmentationResult &ground_result)
    {
        const double confidence =
            std::clamp(
                ground_result.support_constraint_confidence,
                0.0,
                1.0);

        const double anchor_factor =
            GroundIcpAnchorFactor(
                ground_result);

        const double maximum_weight =
            std::max(
                0.0,
                runtime.constraint_config
                    .plane_information_scale);

        return std::clamp(
            maximum_weight *
                confidence *
                confidence *
                anchor_factor,
            0.0,
            maximum_weight);
    }

    bool BuildGroundIcpLinearization(
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &source,
        const PreparedLidarTarget &target,
        const Eigen::Isometry3d &T_target_source,
        const fr_slam::GroundSegmentationResult &ground_result,
        const GroundIcpRuntime &runtime,
        double ground_information_weight,
        GroundIcpLinearization &linearization)
    {
        linearization =
            GroundIcpLinearization();

        if (!source ||
            source->empty() ||
            !target.ready ||
            !target.cloud ||
            target.cloud->empty() ||
            !target.kdtree ||
            target.planes.size() != target.cloud->size() ||
            !T_target_source.matrix().allFinite())
        {
            return false;
        }

        const LidarRegistrationConfig &config =
            runtime.registration_config;

        if (config.knn <= 0)
        {
            return false;
        }

        const double maximum_correspondence_distance_squared =
            config.max_correspondence_distance *
            config.max_correspondence_distance;

        std::vector<int> neighbor_indices(
            static_cast<std::size_t>(config.knn));

        std::vector<float> neighbor_squared_distances(
            static_cast<std::size_t>(config.knn));

        const Eigen::Vector3d sensor_origin_target =
            T_target_source.translation();

        const bool use_sensor_centered =
            config.enable_sensor_centered_perturbation;

        const bool use_hessian_scale_normalization =
            config.enable_sensor_centered_perturbation &&
            config.enable_hessian_scale_normalization;

        if (use_hessian_scale_normalization)
        {
            linearization.correspondence_ranges.reserve(
                source->size());
        }

        // --------------------------------------------------------------------
        // A. Existing all-scene point-to-plane objective.
        // --------------------------------------------------------------------
        for (const LIDAR_POINT &source_point :
             source->points)
        {
            const Eigen::Vector3d p_source(
                static_cast<double>(source_point.x),
                static_cast<double>(source_point.y),
                static_cast<double>(source_point.z));

            if (!p_source.allFinite())
            {
                continue;
            }

            const Eigen::Vector3d p_target =
                T_target_source *
                p_source;

            LIDAR_POINT query_point{};
            query_point.x =
                static_cast<float>(p_target.x());
            query_point.y =
                static_cast<float>(p_target.y());
            query_point.z =
                static_cast<float>(p_target.z());

            const int found =
                target.kdtree->nearestKSearch(
                    query_point,
                    config.knn,
                    neighbor_indices,
                    neighbor_squared_distances);

            if (found <= 0)
            {
                continue;
            }

            int plane_index = -1;

            for (int j = 0;
                 j < found;
                 ++j)
            {
                const double neighbor_distance_squared =
                    static_cast<double>(
                        neighbor_squared_distances[static_cast<std::size_t>(j)]);

                if (neighbor_distance_squared >
                    maximum_correspondence_distance_squared)
                {
                    break;
                }

                const int candidate_index =
                    neighbor_indices[static_cast<std::size_t>(j)];

                if (candidate_index < 0)
                {
                    continue;
                }

                const std::size_t candidate =
                    static_cast<std::size_t>(
                        candidate_index);

                if (candidate >= target.planes.size() ||
                    target.planes[candidate].state !=
                        TargetPlane::State::Valid)
                {
                    continue;
                }

                plane_index =
                    candidate_index;

                break;
            }

            if (plane_index < 0)
            {
                continue;
            }

            const TargetPlane &plane =
                target.planes[static_cast<std::size_t>(
                    plane_index)];

            const double residual =
                plane.normal.dot(
                    p_target -
                    plane.point);

            if (!std::isfinite(residual) ||
                std::abs(residual) >
                    config.max_point_to_plane_distance)
            {
                continue;
            }

            Eigen::Matrix<double, 1, 6> J;

            if (use_sensor_centered)
            {
                const Eigen::Vector3d lever_arm_target =
                    p_target -
                    sensor_origin_target;

                J.block<1, 3>(0, 0) =
                    lever_arm_target.cross(
                                        plane.normal)
                        .transpose();

                if (use_hessian_scale_normalization)
                {
                    const double lever_arm_range =
                        lever_arm_target.norm();

                    if (std::isfinite(lever_arm_range) &&
                        lever_arm_range > 1.0e-9)
                    {
                        linearization
                            .correspondence_ranges
                            .push_back(
                                lever_arm_range);
                    }
                }
            }
            else
            {
                J.block<1, 3>(0, 0) =
                    p_target.cross(
                                plane.normal)
                        .transpose();
            }

            J.block<1, 3>(0, 3) =
                plane.normal.transpose();

            double robust_weight =
                1.0;

            const double absolute_residual =
                std::abs(residual);

            if (config.enable_huber_loss &&
                absolute_residual >
                    config.huber_delta)
            {
                robust_weight =
                    config.huber_delta /
                    absolute_residual;

                ++linearization
                      .general_downweighted_correspondences;
            }

            linearization.general_min_robust_weight =
                std::min(
                    linearization.general_min_robust_weight,
                    robust_weight);

            linearization.H.noalias() +=
                robust_weight *
                J.transpose() *
                J;

            linearization.b.noalias() +=
                robust_weight *
                J.transpose() *
                residual;

            linearization.general_raw_squared_error_sum +=
                residual *
                residual;

            linearization.general_weighted_squared_error_sum +=
                robust_weight *
                residual *
                residual;

            linearization.general_weight_sum +=
                robust_weight;

            ++linearization.general_correspondences;
        }

        // --------------------------------------------------------------------
        // B. Ground V1.4 frozen world-plane constraint.
        //
        // Plane in the LiDAR frame:
        //
        //     n_L^T p_L + d_L = 0
        //
        // Under p_W = R_WL p_L + t_WL, the same plane in World is:
        //
        //     n_W = R_WL n_L
        //     d_W = d_L - n_W^T t_WL
        //
        // A frozen reference (n_ref, d_ref) therefore gives independent
        // roll/pitch and height residuals.  The Ground Jacobians explicitly
        // leave yaw/x/y at zero; those remain owned by all-scene ICP.
        // --------------------------------------------------------------------
        if (!runtime.reference_plane_valid ||
            !ground_result.support_constraint_valid ||
            !ground_result.support_plane_valid ||
            ground_information_weight <= 1.0e-9)
        {
            return linearization.general_correspondences > 0;
        }

        Eigen::Vector3d support_normal_source =
            ground_result.support_ground_normal_L;

        const double support_normal_source_norm =
            support_normal_source.norm();

        if (!support_normal_source.allFinite() ||
            !std::isfinite(support_normal_source_norm) ||
            support_normal_source_norm <= 1.0e-12 ||
            !std::isfinite(ground_result.support_ground_plane_d))
        {
            return linearization.general_correspondences > 0;
        }

        double support_plane_d_source =
            ground_result.support_ground_plane_d /
            support_normal_source_norm;

        support_normal_source /=
            support_normal_source_norm;

        Eigen::Vector3d support_normal_world =
            T_target_source.rotation() *
            support_normal_source;

        if (support_normal_world.dot(
                runtime.reference_normal_W) < 0.0)
        {
            support_normal_world =
                -support_normal_world;
            support_plane_d_source =
                -support_plane_d_source;
        }

        const double normal_cosine =
            std::clamp(
                runtime.reference_normal_W.dot(
                    support_normal_world),
                -1.0,
                1.0);

        const double normal_residual_rad =
            std::acos(normal_cosine);

        const double normal_residual_deg =
            GroundRadiansToDegrees(
                normal_residual_rad);

        const double height_residual_m =
            runtime.reference_normal_W.dot(
                T_target_source.translation()) +
            runtime.reference_plane_d_W -
            support_plane_d_source;

        linearization.ground_height_residual_m =
            height_residual_m;
        linearization.ground_normal_residual_deg =
            normal_residual_deg;
        linearization.ground_quality_weight =
            ground_information_weight;

        if (!std::isfinite(height_residual_m) ||
            !std::isfinite(normal_residual_deg) ||
            std::abs(height_residual_m) >
                runtime.constraint_config.maximum_height_residual_m ||
            normal_residual_deg >
                runtime.constraint_config.maximum_normal_residual_deg)
        {
            return linearization.general_correspondences > 0;
        }

        const double height_sigma_m =
            std::max(
                1.0e-4,
                runtime.constraint_config.height_sigma_m);

        const double normal_sigma_rad =
            std::max(
                1.0e-5,
                GroundDegreesToRadians(
                    runtime.constraint_config.normal_sigma_deg));

        double height_robust_weight = 1.0;
        const double absolute_height_residual_m =
            std::abs(height_residual_m);

        if (absolute_height_residual_m >
            runtime.constraint_config.height_huber_delta_m)
        {
            height_robust_weight =
                runtime.constraint_config.height_huber_delta_m /
                absolute_height_residual_m;
        }

        double normal_robust_weight = 1.0;
        const double normal_huber_delta_rad =
            GroundDegreesToRadians(
                runtime.constraint_config.normal_huber_delta_deg);

        if (normal_residual_rad >
            normal_huber_delta_rad)
        {
            normal_robust_weight =
                normal_huber_delta_rad /
                normal_residual_rad;
        }

        Eigen::Vector3d tangent_helper =
            std::abs(runtime.reference_normal_W.z()) < 0.90
                ? Eigen::Vector3d::UnitZ()
                : Eigen::Vector3d::UnitX();

        Eigen::Vector3d tangent_one =
            tangent_helper -
            runtime.reference_normal_W *
                runtime.reference_normal_W.dot(
                    tangent_helper);

        const double tangent_one_norm =
            tangent_one.norm();

        if (!std::isfinite(tangent_one_norm) ||
            tangent_one_norm <= 1.0e-12)
        {
            return linearization.general_correspondences > 0;
        }

        tangent_one /= tangent_one_norm;

        const Eigen::Vector3d tangent_two =
            runtime.reference_normal_W.cross(
                tangent_one);

        const double normal_information =
            ground_information_weight *
            normal_robust_weight /
            (normal_sigma_rad * normal_sigma_rad);

        const Eigen::Vector3d tangents[2] =
        {
            tangent_one,
            tangent_two
        };

        for (const Eigen::Vector3d &tangent : tangents)
        {
            const double residual =
                tangent.dot(
                    support_normal_world);

            Eigen::Matrix<double, 1, 6> J =
                Eigen::Matrix<double, 1, 6>::Zero();

            J.block<1, 3>(0, 0) =
                support_normal_world.cross(
                    tangent).transpose();

            J(0, 2) = 0.0;

            linearization.H.noalias() +=
                normal_information *
                J.transpose() *
                J;

            linearization.b.noalias() +=
                normal_information *
                J.transpose() *
                residual;

            linearization.ground_raw_squared_error_sum +=
                residual * residual;
            linearization.ground_weighted_squared_error_sum +=
                normal_information * residual * residual;
            linearization.ground_weight_sum +=
                normal_information;
            ++linearization.ground_correspondences;
        }

        Eigen::Matrix<double, 1, 6> height_J =
            Eigen::Matrix<double, 1, 6>::Zero();

        height_J(0, 5) =
            runtime.reference_normal_W.z();

        const double height_information =
            ground_information_weight *
            height_robust_weight /
            (height_sigma_m * height_sigma_m);

        linearization.H.noalias() +=
            height_information *
            height_J.transpose() *
            height_J;

        linearization.b.noalias() +=
            height_information *
            height_J.transpose() *
            height_residual_m;

        linearization.ground_raw_squared_error_sum +=
            height_residual_m *
            height_residual_m;
        linearization.ground_weighted_squared_error_sum +=
            height_information *
            height_residual_m *
            height_residual_m;
        linearization.ground_weight_sum +=
            height_information;
        ++linearization.ground_correspondences;

        return linearization.general_correspondences > 0;
    }

    bool ComputeGroundIcpParameterUnscale(
        const GroundIcpLinearization &linearization,
        const LidarRegistrationConfig &config,
        Eigen::Matrix<double, 6, 6> &parameter_unscale,
        double &characteristic_length)
    {
        parameter_unscale =
            Eigen::Matrix<double, 6, 6>::Identity();

        characteristic_length =
            1.0;

        const bool use_hessian_scale_normalization =
            config.enable_sensor_centered_perturbation &&
            config.enable_hessian_scale_normalization;

        if (!use_hessian_scale_normalization)
        {
            return true;
        }

        if (linearization.correspondence_ranges.empty())
        {
            return false;
        }

        std::vector<double> ranges =
            linearization.correspondence_ranges;

        const std::size_t range_count =
            ranges.size();

        std::vector<double>::iterator middle =
            ranges.begin() +
            static_cast<std::ptrdiff_t>(
                range_count / 2);

        std::nth_element(
            ranges.begin(),
            middle,
            ranges.end());

        double median_range =
            *middle;

        if ((range_count % 2U) == 0U)
        {
            const std::vector<double>::iterator lower_middle =
                std::max_element(
                    ranges.begin(),
                    middle);

            if (lower_middle != middle)
            {
                median_range =
                    0.5 *
                    (median_range +
                     *lower_middle);
            }
        }

        if (!std::isfinite(median_range) ||
            median_range <= 0.0)
        {
            return false;
        }

        characteristic_length =
            std::clamp(
                median_range,
                config.hessian_scale_min_range,
                config.hessian_scale_max_range);

        const double inverse_length =
            1.0 /
            characteristic_length;

        parameter_unscale(0, 0) =
            inverse_length;

        parameter_unscale(1, 1) =
            inverse_length;

        parameter_unscale(2, 2) =
            inverse_length;

        return true;
    }

    bool SolveGroundObservableStep(
        const GroundIcpLinearization &linearization,
        const LidarRegistrationConfig &config,
        Eigen::Matrix<double, 6, 1> &dx)
    {
        dx = Eigen::Matrix<double, 6, 1>::Zero();

        Eigen::Matrix<double, 6, 6> parameter_unscale;
        double characteristic_length = 1.0;

        if (!ComputeGroundIcpParameterUnscale(
                linearization,
                config,
                parameter_unscale,
                characteristic_length))
        {
            return false;
        }

        (void)characteristic_length;

        const Eigen::Matrix<double, 6, 6> H_analysis =
            parameter_unscale.transpose() *
            linearization.H *
            parameter_unscale;

        const Eigen::Matrix<double, 6, 1> b_analysis =
            parameter_unscale.transpose() *
            linearization.b;

        // Ground V1.4 is a strict [roll, pitch, z] post-refinement.  Solving
        // this principal block (instead of solving 6-DoF and zeroing entries
        // afterwards) prevents Hessian cross-coupling from moving x/y/yaw.
        const int observable_indices[3] = {0, 1, 5};

        Eigen::Matrix3d H_observable =
            Eigen::Matrix3d::Zero();

        Eigen::Vector3d b_observable =
            Eigen::Vector3d::Zero();

        for (int row = 0; row < 3; ++row)
        {
            b_observable(row) =
                b_analysis(observable_indices[row]);

            for (int column = 0; column < 3; ++column)
            {
                H_observable(row, column) =
                    H_analysis(
                        observable_indices[row],
                        observable_indices[column]);
            }
        }

        H_observable =
            0.5 *
            (H_observable +
             H_observable.transpose());

        if (!H_observable.allFinite() ||
            !b_observable.allFinite())
        {
            return false;
        }

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>
            eigen_solver(H_observable);

        if (eigen_solver.info() != Eigen::Success ||
            !eigen_solver.eigenvalues().allFinite() ||
            !eigen_solver.eigenvectors().allFinite())
        {
            return false;
        }

        const double lambda_max =
            eigen_solver.eigenvalues()(2);

        if (!std::isfinite(lambda_max) ||
            lambda_max <=
                config.degeneracy_absolute_eigenvalue_threshold)
        {
            return false;
        }

        const double eigenvalue_floor =
            std::max(
                config.degeneracy_absolute_eigenvalue_threshold,
                1.0e-8 * lambda_max);

        const Eigen::Vector3d gradient_eigen =
            eigen_solver.eigenvectors().transpose() *
            b_observable;

        Eigen::Vector3d delta_eigen =
            Eigen::Vector3d::Zero();

        int usable_directions = 0;

        for (int index = 0; index < 3; ++index)
        {
            const double eigenvalue =
                eigen_solver.eigenvalues()(index);

            if (!std::isfinite(eigenvalue) ||
                eigenvalue <= eigenvalue_floor)
            {
                continue;
            }

            delta_eigen(index) =
                -gradient_eigen(index) /
                eigenvalue;

            ++usable_directions;
        }

        if (usable_directions <= 0)
        {
            return false;
        }

        const Eigen::Vector3d delta_observable_analysis =
            eigen_solver.eigenvectors() *
            delta_eigen;

        Eigen::Matrix<double, 6, 1> delta_analysis =
            Eigen::Matrix<double, 6, 1>::Zero();

        for (int index = 0; index < 3; ++index)
        {
            delta_analysis(observable_indices[index]) =
                delta_observable_analysis(index);
        }

        dx = parameter_unscale *
             delta_analysis;

        return dx.allFinite();
    }

    Eigen::Isometry3d ApplyGroundObservableIncrement(
        const Eigen::Isometry3d &T_target_source,
        const Eigen::Matrix<double, 6, 1> &dx)
    {
        Eigen::Isometry3d updated_pose =
            T_target_source;

        const Eigen::Vector3d delta_rotation =
            dx.head<3>();

        updated_pose.linear() =
            Sophus::SO3d::exp(
                delta_rotation)
                .matrix() *
            updated_pose.rotation();

        updated_pose.translation() +=
            dx.tail<3>();

        return updated_pose;
    }

    bool UpdateGroundIcpRelativeCovariance(
        const GroundIcpLinearization &linearization,
        const LidarRegistrationConfig &config,
        LidarRegistrationResult &result)
    {
        const bool use_hessian_scale_normalization =
            config.enable_sensor_centered_perturbation &&
            config.enable_hessian_scale_normalization;

        if (!use_hessian_scale_normalization)
        {
            return false;
        }

        Eigen::Matrix<double, 6, 6> parameter_unscale;
        double characteristic_length = 1.0;

        if (!ComputeGroundIcpParameterUnscale(
                linearization,
                config,
                parameter_unscale,
                characteristic_length))
        {
            return false;
        }

        const Eigen::Matrix<double, 6, 6> H_analysis =
            parameter_unscale.transpose() *
            linearization.H *
            parameter_unscale;

        Eigen::SelfAdjointEigenSolver<
            Eigen::Matrix<double, 6, 6>>
            eigen_solver(
                H_analysis);

        if (eigen_solver.info() !=
                Eigen::Success ||
            !eigen_solver.eigenvalues().allFinite() ||
            !eigen_solver.eigenvectors().allFinite())
        {
            return false;
        }

        const Eigen::Matrix<double, 6, 1> eigenvalues =
            eigen_solver.eigenvalues();

        const double lambda_max =
            eigenvalues(5);

        if (!std::isfinite(lambda_max) ||
            lambda_max <=
                config.degeneracy_absolute_eigenvalue_threshold)
        {
            return false;
        }

        const Eigen::Matrix<double, 6, 1> relative_eigenvalues =
            eigenvalues /
            lambda_max;

        Eigen::Matrix<double, 6, 6> relative_covariance =
            Eigen::Matrix<double, 6, 6>::Zero();

        for (int i = 0;
             i < 6;
             ++i)
        {
            if (!std::isfinite(
                    relative_eigenvalues(i)))
            {
                return false;
            }

            const double relative_information =
                std::clamp(
                    relative_eigenvalues(i),
                    0.01,
                    1.0);

            const Eigen::Matrix<double, 6, 1> direction =
                eigen_solver.eigenvectors().col(i);

            relative_covariance.noalias() +=
                (1.0 /
                 relative_information) *
                direction *
                direction.transpose();
        }

        relative_covariance =
            0.5 *
            (relative_covariance +
             relative_covariance.transpose());

        if (!relative_covariance.allFinite())
        {
            return false;
        }

        result.hessian_relative_covariance =
            relative_covariance;

        result.hessian_relative_covariance_valid =
            true;

        return true;
    }

    bool GroundPlaneInWorld(
        const fr_slam::GroundSegmentationResult &ground_result,
        const Eigen::Isometry3d &T_WL,
        Eigen::Vector3d &normal_W,
        double &plane_d_W)
    {
        if (!ground_result.support_plane_valid ||
            !T_WL.matrix().allFinite() ||
            !ground_result.support_ground_normal_L.allFinite() ||
            !std::isfinite(ground_result.support_ground_plane_d))
        {
            return false;
        }

        Eigen::Vector3d normal_L =
            ground_result.support_ground_normal_L;

        const double normal_norm =
            normal_L.norm();

        if (!std::isfinite(normal_norm) ||
            normal_norm <= 1.0e-12)
        {
            return false;
        }

        const double plane_d_L =
            ground_result.support_ground_plane_d /
            normal_norm;

        normal_L /= normal_norm;

        normal_W =
            T_WL.rotation() *
            normal_L;

        plane_d_W =
            plane_d_L -
            normal_W.dot(
                T_WL.translation());

        // GroundSegmenter is designed for +Z-up sensors.  Still normalize the
        // sign here so all bootstrap samples share one plane convention.
        if (normal_W.z() < 0.0)
        {
            normal_W = -normal_W;
            plane_d_W = -plane_d_W;
        }

        return normal_W.allFinite() &&
               std::isfinite(plane_d_W);
    }

    bool BootstrapGroundReference(
        GroundIcpRuntime &runtime,
        const fr_slam::GroundSegmentationResult &ground_result,
        const Eigen::Isometry3d &T_WL,
        double timestamp,
        std::string &reason)
    {
        if (runtime.reference_plane_valid)
        {
            reason = "ANCHOR_ALREADY_READY";
            return true;
        }

        Eigen::Vector3d sample_normal_W;
        double sample_plane_d_W = 0.0;

        if (!GroundPlaneInWorld(
                ground_result,
                T_WL,
                sample_normal_W,
                sample_plane_d_W))
        {
            reason = "INVALID_WORLD_PLANE";
            return false;
        }

        if (std::isfinite(timestamp) &&
            std::isfinite(
                runtime.last_anchor_sample_timestamp) &&
            std::abs(
                timestamp -
                runtime.last_anchor_sample_timestamp) <= 1.0e-9)
        {
            reason = "ANCHOR_SAMPLE_ALREADY_USED";
            return false;
        }

        runtime.last_anchor_sample_timestamp =
            timestamp;

        runtime.anchor_normal_samples_W.push_back(
            sample_normal_W);

        runtime.anchor_plane_d_samples_W.push_back(
            sample_plane_d_W);

        const std::size_t required_samples =
            std::max<std::size_t>(
                1,
                runtime.constraint_config.anchor_bootstrap_frames);

        if (runtime.anchor_plane_d_samples_W.size() <
            required_samples)
        {
            reason = "ANCHOR_BOOTSTRAP";
            return false;
        }

        Eigen::Vector3d reference_normal_W =
            Eigen::Vector3d::Zero();

        const Eigen::Vector3d sign_reference =
            runtime.anchor_normal_samples_W.front();

        for (Eigen::Vector3d normal_W :
             runtime.anchor_normal_samples_W)
        {
            if (normal_W.dot(sign_reference) < 0.0)
            {
                normal_W = -normal_W;
            }

            reference_normal_W += normal_W;
        }

        const double reference_normal_norm =
            reference_normal_W.norm();

        const double reference_plane_d_W =
            GroundMedian(
                runtime.anchor_plane_d_samples_W);

        if (!std::isfinite(reference_normal_norm) ||
            reference_normal_norm <= 1.0e-12 ||
            !std::isfinite(reference_plane_d_W))
        {
            runtime.anchor_normal_samples_W.clear();
            runtime.anchor_plane_d_samples_W.clear();
            reason = "ANCHOR_INVALID_RESET";
            return false;
        }

        reference_normal_W /=
            reference_normal_norm;

        double maximum_normal_spread_deg = 0.0;
        double maximum_plane_d_spread_m = 0.0;

        for (std::size_t index = 0;
             index < runtime.anchor_plane_d_samples_W.size();
             ++index)
        {
            const double cosine =
                std::clamp(
                    reference_normal_W.dot(
                        runtime.anchor_normal_samples_W[index]),
                    -1.0,
                    1.0);

            maximum_normal_spread_deg =
                std::max(
                    maximum_normal_spread_deg,
                    GroundRadiansToDegrees(
                        std::acos(cosine)));

            maximum_plane_d_spread_m =
                std::max(
                    maximum_plane_d_spread_m,
                    std::abs(
                        runtime.anchor_plane_d_samples_W[index] -
                        reference_plane_d_W));
        }

        if (maximum_normal_spread_deg >
                runtime.constraint_config
                    .anchor_maximum_normal_spread_deg ||
            maximum_plane_d_spread_m >
                runtime.constraint_config
                    .anchor_maximum_plane_d_spread_m)
        {
            // Restart from the newest strong observation.  This prevents a
            // moving/rough startup interval from defining the permanent plane.
            runtime.anchor_normal_samples_W.assign(
                1,
                sample_normal_W);
            runtime.anchor_plane_d_samples_W.assign(
                1,
                sample_plane_d_W);
            reason = "ANCHOR_UNSTABLE_RESET";
            return false;
        }

        runtime.reference_normal_W =
            reference_normal_W;
        runtime.reference_plane_d_W =
            reference_plane_d_W;
        runtime.reference_plane_valid =
            true;

        reason = "ANCHOR_READY";


        return true;
    }

    bool EvaluateGroundReferenceResidual(
        const GroundIcpRuntime &runtime,
        const fr_slam::GroundSegmentationResult &ground_result,
        const Eigen::Isometry3d &T_WL,
        double &height_residual_m,
        double &normal_residual_deg)
    {
        if (!runtime.reference_plane_valid ||
            !ground_result.support_plane_valid ||
            !T_WL.matrix().allFinite() ||
            !ground_result.support_ground_normal_L.allFinite() ||
            !std::isfinite(
                ground_result.support_ground_plane_d))
        {
            return false;
        }

        Eigen::Vector3d normal_L =
            ground_result.support_ground_normal_L;

        const double normal_norm =
            normal_L.norm();

        if (!std::isfinite(normal_norm) ||
            normal_norm <= 1.0e-12)
        {
            return false;
        }

        double plane_d_L =
            ground_result.support_ground_plane_d /
            normal_norm;

        normal_L /= normal_norm;

        Eigen::Vector3d normal_W =
            T_WL.rotation() *
            normal_L;

        if (normal_W.dot(
                runtime.reference_normal_W) < 0.0)
        {
            normal_W = -normal_W;
            plane_d_L = -plane_d_L;
        }

        const double cosine =
            std::clamp(
                runtime.reference_normal_W.dot(
                    normal_W),
                -1.0,
                1.0);

        normal_residual_deg =
            GroundRadiansToDegrees(
                std::acos(cosine));

        height_residual_m =
            runtime.reference_normal_W.dot(
                T_WL.translation()) +
            runtime.reference_plane_d_W -
            plane_d_L;

        return std::isfinite(height_residual_m) &&
               std::isfinite(normal_residual_deg);
    }

    void SetResultFromGroundLinearization(
        const GroundIcpLinearization &linearization,
        const Eigen::Isometry3d &T_target_source,
        int total_iterations,
        LidarRegistrationResult &result)
    {
        const double robust_rmse =
            linearization.general_weight_sum > 1.0e-12
                ? std::sqrt(
                      linearization
                          .general_weighted_squared_error_sum /
                      linearization.general_weight_sum)
                : std::numeric_limits<double>::infinity();

        const double downweighted_ratio =
            linearization.general_correspondences > 0
                ? static_cast<double>(
                      linearization
                          .general_downweighted_correspondences) /
                      static_cast<double>(
                          linearization.general_correspondences)
                : 0.0;

        result.success = true;
        result.converged = true;
        result.iterations = total_iterations;
        result.correspondences =
            linearization.general_correspondences;
        result.rmse =
            linearization.GeneralRmse();
        result.robust_downweighted_correspondences =
            linearization.general_downweighted_correspondences;
        result.robust_downweighted_ratio =
            downweighted_ratio;
        result.robust_effective_weight_sum =
            linearization.general_weight_sum;
        result.robust_min_weight =
            linearization.general_min_robust_weight;
        result.robust_rmse =
            robust_rmse;
        result.T_target_source =
            T_target_source;
    }

    // ========================================================================
    // GROUND_ICP_V14_FLAT_WORLD_PLANE_ANCHOR
    //
    // The ordinary all-scene ICP result is always computed first.  Ground then
    // performs at most a few small [roll, pitch, z] iterations against a
    // frozen, startup-learned world plane.  Every failure or safety-gate
    // rejection returns the untouched ordinary ICP result to the caller.
    // ========================================================================
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
        LidarRegistrationResult &result)
    {
        // Rollback is the default for every path below.
        result = general_result;

        GroundIcpRuntime *runtime =
            GetGroundIcpRuntime(owner);

        if (runtime == nullptr)
        {
            return GroundJointIcpStatus::NotEligible;
        }

        GroundDiagnosticRecord diagnostic;
        diagnostic.timestamp = timestamp;
        diagnostic.stage =
            stage != nullptr
                ? stage
                : "UNKNOWN";
        diagnostic.general_correspondences_before =
            general_result.correspondences;
        diagnostic.general_rmse_before =
            general_result.rmse;
        diagnostic.general_correspondences_after =
            general_result.correspondences;
        diagnostic.general_rmse_after =
            general_result.rmse;

        const auto finish =
            [&](GroundJointIcpStatus status,
                const char *status_text,
                const std::string &reason)
            {
                diagnostic.status = status_text;
                diagnostic.reason = reason;

                WriteGroundDiagnostic(
                    *runtime,
                    ground_result,
                    diagnostic);

                if (status == GroundJointIcpStatus::Failed)
                {
                }

                return status;
            };

        const GroundConstraintConfig &ground_config =
            runtime->constraint_config;

        if (!runtime->enabled ||
            !ground_config.enabled ||
            ground_config.mode == "off" ||
            ground_config.mode == "disabled")
        {
            return finish(
                GroundJointIcpStatus::NotEligible,
                "NOT_ELIGIBLE",
                "GROUND_DISABLED");
        }

        if (!general_result.success ||
            !general_result.T_target_source.matrix().allFinite())
        {
            return finish(
                GroundJointIcpStatus::Failed,
                "FAILED",
                "GENERAL_RESULT_INVALID");
        }

        if (!ground_result.support_plane_valid)
        {
            return finish(
                GroundJointIcpStatus::NotEligible,
                "NOT_ELIGIBLE",
                "SUPPORT_PLANE_INVALID");
        }

        if (!runtime->reference_plane_valid)
        {
            // During Ground V4 clearance bootstrap, a geometrically strong
            // sample is exposed through support_clearance_sample_accepted even
            // though the final trusted-constraint flag is not ready yet.
            const bool strong_anchor_sample =
                ground_result.support_constraint_valid ||
                ground_result.support_clearance_sample_accepted;

            if (!strong_anchor_sample)
            {
                return finish(
                    GroundJointIcpStatus::NotEligible,
                    "NOT_ELIGIBLE",
                    "ANCHOR_SAMPLE_NOT_TRUSTED");
            }

            std::string anchor_reason;

            if (!BootstrapGroundReference(
                    *runtime,
                    ground_result,
                    general_result.T_target_source,
                    timestamp,
                    anchor_reason))
            {
                return finish(
                    GroundJointIcpStatus::NotEligible,
                    "NOT_ELIGIBLE",
                    anchor_reason);
            }
        }

        if (!ground_result.support_constraint_valid)
        {
            return finish(
                GroundJointIcpStatus::NotEligible,
                "NOT_ELIGIBLE",
                "SUPPORT_CONSTRAINT_NOT_TRUSTED");
        }

        if (!source ||
            source->empty() ||
            !target.ready ||
            !target.cloud ||
            target.cloud->empty() ||
            !target.kdtree ||
            target.planes.size() != target.cloud->size())
        {
            return finish(
                GroundJointIcpStatus::Failed,
                "FAILED",
                "SOURCE_OR_TARGET_INVALID");
        }

        const double ground_information_weight =
            ComputeGroundIcpWeight(
                *runtime,
                ground_result);

        diagnostic.quality_weight =
            ground_information_weight;

        if (!std::isfinite(ground_information_weight) ||
            ground_information_weight <= 1.0e-3)
        {
            return finish(
                GroundJointIcpStatus::NotEligible,
                "NOT_ELIGIBLE",
                "GROUND_WEIGHT_TOO_LOW");
        }

        if (!EvaluateGroundReferenceResidual(
                *runtime,
                ground_result,
                general_result.T_target_source,
                diagnostic.height_residual_m,
                diagnostic.normal_residual_deg))
        {
            return finish(
                GroundJointIcpStatus::NotEligible,
                "NOT_ELIGIBLE",
                "GROUND_RESIDUAL_INVALID");
        }

        if (std::abs(diagnostic.height_residual_m) >
            ground_config.maximum_height_residual_m)
        {
            return finish(
                GroundJointIcpStatus::NotEligible,
                "NOT_ELIGIBLE",
                "HEIGHT_RESIDUAL_GATE");
        }

        if (diagnostic.normal_residual_deg >
            ground_config.maximum_normal_residual_deg)
        {
            return finish(
                GroundJointIcpStatus::NotEligible,
                "NOT_ELIGIBLE",
                "NORMAL_RESIDUAL_GATE");
        }

        const LidarRegistrationConfig &registration_config =
            runtime->registration_config;

        GroundIcpLinearization baseline_linearization;

        if (!BuildGroundIcpLinearization(
                source,
                target,
                general_result.T_target_source,
                ground_result,
                *runtime,
                ground_information_weight,
                baseline_linearization))
        {
            return finish(
                GroundJointIcpStatus::Failed,
                "FAILED",
                "BASELINE_LINEARIZATION_FAILED");
        }

        diagnostic.general_correspondences_before =
            baseline_linearization.general_correspondences;
        diagnostic.general_rmse_before =
            baseline_linearization.GeneralRmse();

        if (baseline_linearization.general_correspondences <
                registration_config.min_correspondences ||
            baseline_linearization.ground_correspondences != 3U ||
            !std::isfinite(
                diagnostic.general_rmse_before))
        {
            return finish(
                GroundJointIcpStatus::Failed,
                "FAILED",
                "BASELINE_INFORMATION_INSUFFICIENT");
        }

        const Eigen::Isometry3d baseline_pose =
            general_result.T_target_source;

        Eigen::Isometry3d current_pose =
            baseline_pose;

        bool completed_iteration = false;
        bool correction_was_limited = false;
        int refinement_iterations = 0;

        const int maximum_refinement_iterations =
            std::max(
                0,
                ground_config.maximum_refinement_iterations);

        for (int iteration = 0;
             iteration < maximum_refinement_iterations;
             ++iteration)
        {
            GroundIcpLinearization linearization;

            // Iteration zero is exactly the baseline pose already linearized
            // above.  Reuse it to avoid one full nearest-neighbor pass in the
            // realtime LiDAR thread.
            bool linearization_ready = false;

            if (iteration == 0)
            {
                linearization =
                    baseline_linearization;
                linearization_ready = true;
            }
            else
            {
                linearization_ready =
                    BuildGroundIcpLinearization(
                        source,
                        target,
                        current_pose,
                        ground_result,
                        *runtime,
                        ground_information_weight,
                        linearization);
            }

            if (!linearization_ready)
            {
                return finish(
                    GroundJointIcpStatus::Failed,
                    "FAILED",
                    "REFINEMENT_LINEARIZATION_FAILED");
            }

            if (linearization.general_correspondences <
                    registration_config.min_correspondences ||
                linearization.ground_correspondences != 3U)
            {
                return finish(
                    GroundJointIcpStatus::Failed,
                    "FAILED",
                    "REFINEMENT_INFORMATION_INSUFFICIENT");
            }

            Eigen::Matrix<double, 6, 1> dx =
                Eigen::Matrix<double, 6, 1>::Zero();

            if (!SolveGroundObservableStep(
                    linearization,
                    registration_config,
                    dx))
            {
                return finish(
                    GroundJointIcpStatus::Failed,
                    "FAILED",
                    "REFINEMENT_SOLVE_FAILED");
            }

            const double translation_already_used_m =
                (current_pose.translation() -
                 baseline_pose.translation())
                    .norm();

            const double translation_budget_m =
                std::max(
                    0.0,
                    ground_config
                        .maximum_total_translation_correction_m -
                    translation_already_used_m);

            const double proposed_translation_m =
                dx.tail<3>().norm();

            if (proposed_translation_m >
                    translation_budget_m &&
                proposed_translation_m > 1.0e-12)
            {
                dx.tail<3>() *=
                    translation_budget_m /
                    proposed_translation_m;
                correction_was_limited = true;
            }

            const Eigen::AngleAxisd rotation_already_applied(
                current_pose.rotation() *
                baseline_pose.rotation().transpose());

            const double rotation_budget_rad =
                std::max(
                    0.0,
                    GroundDegreesToRadians(
                        ground_config
                            .maximum_total_rotation_correction_deg) -
                    std::abs(
                        rotation_already_applied.angle()));

            const double proposed_rotation_rad =
                dx.head<3>().norm();

            if (proposed_rotation_rad >
                    rotation_budget_rad &&
                proposed_rotation_rad > 1.0e-12)
            {
                dx.head<3>() *=
                    rotation_budget_rad /
                    proposed_rotation_rad;
                correction_was_limited = true;
            }

            const Eigen::Isometry3d trial_pose =
                ApplyGroundObservableIncrement(
                    current_pose,
                    dx);

            if (!trial_pose.matrix().allFinite())
            {
                return finish(
                    GroundJointIcpStatus::Failed,
                    "FAILED",
                    "REFINEMENT_POSE_INVALID");
            }

            current_pose = trial_pose;
            completed_iteration = true;
            refinement_iterations = iteration + 1;

            diagnostic.correction_translation_W =
                current_pose.translation() -
                baseline_pose.translation();

            const Eigen::AngleAxisd correction_rotation(
                current_pose.rotation() *
                baseline_pose.rotation().transpose());

            diagnostic.correction_rotation_deg =
                GroundRadiansToDegrees(
                    std::abs(correction_rotation.angle()));
            diagnostic.refinement_iterations =
                refinement_iterations;

            if (diagnostic.correction_translation_W.norm() >
                    ground_config
                        .maximum_total_translation_correction_m +
                        1.0e-9 ||
                diagnostic.correction_rotation_deg >
                    ground_config
                        .maximum_total_rotation_correction_deg +
                        1.0e-9)
            {
                return finish(
                    GroundJointIcpStatus::Failed,
                    "FAILED",
                    "CORRECTION_SAFETY_GATE");
            }

            if (dx.head<3>().norm() <
                    registration_config
                        .rotation_convergence_threshold &&
                dx.tail<3>().norm() <
                    registration_config
                        .translation_convergence_threshold)
            {
                break;
            }
        }

        if (!completed_iteration)
        {
            return finish(
                GroundJointIcpStatus::NotEligible,
                "NOT_ELIGIBLE",
                "REFINEMENT_DISABLED");
        }

        GroundIcpLinearization final_linearization;

        if (!BuildGroundIcpLinearization(
                source,
                target,
                current_pose,
                ground_result,
                *runtime,
                ground_information_weight,
                final_linearization))
        {
            return finish(
                GroundJointIcpStatus::Failed,
                "FAILED",
                "FINAL_LINEARIZATION_FAILED");
        }

        diagnostic.general_correspondences_after =
            final_linearization.general_correspondences;
        diagnostic.general_rmse_after =
            final_linearization.GeneralRmse();

        EvaluateGroundReferenceResidual(
            *runtime,
            ground_result,
            current_pose,
            diagnostic.height_residual_m,
            diagnostic.normal_residual_deg);

        if (final_linearization.general_correspondences <
                registration_config.min_correspondences ||
            final_linearization.ground_correspondences != 3U ||
            !std::isfinite(
                diagnostic.general_rmse_after))
        {
            return finish(
                GroundJointIcpStatus::Failed,
                "FAILED",
                "FINAL_INFORMATION_INSUFFICIENT");
        }

        const double maximum_allowed_rmse =
            std::max(
                diagnostic.general_rmse_before *
                    ground_config.maximum_general_rmse_ratio,
                diagnostic.general_rmse_before +
                    ground_config
                        .maximum_general_rmse_absolute_increase_m);

        if (diagnostic.general_rmse_after >
            maximum_allowed_rmse)
        {
            return finish(
                GroundJointIcpStatus::Failed,
                "FAILED",
                "GENERAL_RMSE_SAFETY_GATE");
        }

        const double minimum_allowed_correspondences =
            ground_config.minimum_general_correspondence_ratio *
            static_cast<double>(
                diagnostic.general_correspondences_before);

        if (static_cast<double>(
                diagnostic.general_correspondences_after) <
            minimum_allowed_correspondences)
        {
            return finish(
                GroundJointIcpStatus::Failed,
                "FAILED",
                "GENERAL_CORRESPONDENCE_SAFETY_GATE");
        }

        if (diagnostic.general_rmse_after >
                frontend_maximum_accepted_rmse ||
            diagnostic.general_correspondences_after <
                frontend_minimum_accepted_correspondences)
        {
            return finish(
                GroundJointIcpStatus::Failed,
                "FAILED",
                "FRONTEND_QUALITY_SAFETY_GATE");
        }

        result = general_result;

        SetResultFromGroundLinearization(
            final_linearization,
            current_pose,
            general_result.iterations +
                refinement_iterations,
            result);

        UpdateGroundIcpRelativeCovariance(
            final_linearization,
            registration_config,
            result);

        return finish(
            GroundJointIcpStatus::Success,
            "SUCCESS",
            correction_was_limited
                ? "APPLIED_TRUST_REGION_LIMITED"
                : "APPLIED");
    }
