#include "fr_slam/frontend/fr_lidar_frontend.hpp"
#include "fr_slam/frontend/fr_ground_segmenter.hpp"
#include "fr_slam/frontend/fr_ground_input_bridge.hpp"

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
#include "fr_lidar_frontend_ground_internal.hpp"

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
        const LidarRegistrationConfig &registration_config)
    {
        if (owner == nullptr)
        {
            return nullptr;
        }

        std::lock_guard<std::mutex> lock(
            GroundIcpRuntimeMutex());

        std::unique_ptr<GroundIcpRuntime> runtime =
            std::make_unique<GroundIcpRuntime>(
                registration_config);

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
        (void)input_source;

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
                runtime->analysis_voxel_leaf_m);

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

        return std::clamp(
            runtime.base_weight *
                confidence *
                confidence *
                anchor_factor,
            0.0,
            runtime.base_weight);
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
        // B. Trusted Ground V4.0 extra point-to-plane information.
        //
        // Important: this is NOT residual(distance_to_anchor).  Ground points
        // are matched against ACTUAL LocalMap planes.  The source support normal
        // is used only to prevent an accidental support-point -> wall match.
        // --------------------------------------------------------------------
        if (!ground_result.support_constraint_valid ||
            !ground_result.support_plane_valid ||
            !ground_result.support_ground_cloud ||
            ground_result.support_ground_cloud->empty() ||
            ground_information_weight <= 1.0e-9)
        {
            return linearization.general_correspondences > 0;
        }

        Eigen::Vector3d support_normal_source =
            ground_result.support_ground_normal_L;

        const double support_normal_norm =
            support_normal_source.norm();

        if (!support_normal_source.allFinite() ||
            !std::isfinite(support_normal_norm) ||
            support_normal_norm <= 1.0e-12)
        {
            return linearization.general_correspondences > 0;
        }

        support_normal_source /=
            support_normal_norm;

        Eigen::Vector3d support_normal_target =
            T_target_source.rotation() *
            support_normal_source;

        const double support_normal_target_norm =
            support_normal_target.norm();

        if (!support_normal_target.allFinite() ||
            !std::isfinite(support_normal_target_norm) ||
            support_normal_target_norm <= 1.0e-12)
        {
            return linearization.general_correspondences > 0;
        }

        support_normal_target /=
            support_normal_target_norm;

        constexpr double kPi =
            3.14159265358979323846;

        const double normal_cosine_threshold =
            std::cos(
                runtime.target_normal_compatibility_deg *
                kPi /
                180.0);

        const std::size_t support_size =
            ground_result.support_ground_cloud->size();

        const std::size_t stride =
            std::max<std::size_t>(
                1,
                (support_size +
                 runtime.maximum_support_points -
                 1) /
                    std::max<std::size_t>(
                        1,
                        runtime.maximum_support_points));

        for (std::size_t point_index = 0;
             point_index < support_size;
             point_index += stride)
        {
            const pcl::PointXYZ &source_point =
                ground_result
                    .support_ground_cloud
                    ->points[point_index];

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

                const TargetPlane &candidate_plane =
                    target.planes[candidate];

                const double normal_alignment =
                    std::abs(
                        candidate_plane.normal.dot(
                            support_normal_target));

                if (!std::isfinite(normal_alignment) ||
                    normal_alignment <
                        normal_cosine_threshold)
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
                    runtime.ground_maximum_residual_m)
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

            // Ground observable-subspace projection.
            // Parameter order is [rx ry rz tx ty tz].
            // Ground is allowed to add direct information ONLY to
            // roll / pitch / z.  General ICP remains full 6-DoF.
            J(0, 2) = 0.0; // yaw
            J(0, 3) = 0.0; // x
            J(0, 4) = 0.0; // y

            double ground_robust_weight =
                1.0;

            const double absolute_residual =
                std::abs(residual);

            if (absolute_residual >
                runtime.ground_huber_delta_m)
            {
                ground_robust_weight =
                    runtime.ground_huber_delta_m /
                    absolute_residual;
            }

            const double final_weight =
                ground_information_weight *
                ground_robust_weight;

            linearization.H.noalias() +=
                final_weight *
                J.transpose() *
                J;

            linearization.b.noalias() +=
                final_weight *
                J.transpose() *
                residual;

            linearization.ground_raw_squared_error_sum +=
                residual *
                residual;

            linearization.ground_weighted_squared_error_sum +=
                final_weight *
                residual *
                residual;

            linearization.ground_weight_sum +=
                final_weight;

            ++linearization.ground_correspondences;
        }

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

    bool SolveGroundIcpStep(
        const GroundIcpLinearization &linearization,
        const LidarRegistrationConfig &config,
        Eigen::Matrix<double, 6, 1> &dx,
        Eigen::Matrix<double, 6, 6> *final_H_analysis = nullptr,
        Eigen::Matrix<double, 6, 6> *final_parameter_unscale = nullptr)
    {
        dx =
            Eigen::Matrix<double, 6, 1>::Zero();

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

        if (!H_analysis.allFinite() ||
            !b_analysis.allFinite())
        {
            return false;
        }

        Eigen::SelfAdjointEigenSolver<
            Eigen::Matrix<double, 6, 6>>
            eigen_solver(
                H_analysis);

        if (eigen_solver.info() !=
            Eigen::Success)
        {
            return false;
        }

        const Eigen::Matrix<double, 6, 1> eigenvalues =
            eigen_solver.eigenvalues();

        const Eigen::Matrix<double, 6, 6> eigenvectors =
            eigen_solver.eigenvectors();

        if (!eigenvalues.allFinite() ||
            !eigenvectors.allFinite())
        {
            return false;
        }

        const double lambda_max =
            eigenvalues(5);

        if (!std::isfinite(lambda_max) ||
            lambda_max <=
                config.degeneracy_absolute_eigenvalue_threshold)
        {
            return false;
        }

        const bool use_hessian_scale_normalization =
            config.enable_sensor_centered_perturbation &&
            config.enable_hessian_scale_normalization;

        const double hard_relative_threshold =
            use_hessian_scale_normalization
                ? 0.01
                : config.degeneracy_relative_eigenvalue_threshold;

        const Eigen::Matrix<double, 6, 1> relative_eigenvalues =
            eigenvalues /
            lambda_max;

        const Eigen::Matrix<double, 6, 1> gradient_eigen =
            eigenvectors.transpose() *
            b_analysis;

        Eigen::Matrix<double, 6, 1> delta_eigen =
            Eigen::Matrix<double, 6, 1>::Zero();

        int usable_directions = 0;

        for (int i = 0;
             i < 6;
             ++i)
        {
            const double lambda =
                eigenvalues(i);

            const double relative_lambda =
                relative_eigenvalues(i);

            const bool strong_degenerate =
                !std::isfinite(lambda) ||
                lambda <=
                    config.degeneracy_absolute_eigenvalue_threshold ||
                !std::isfinite(relative_lambda) ||
                relative_lambda <
                    hard_relative_threshold;

            if (strong_degenerate)
            {
                continue;
            }

            delta_eigen(i) =
                -gradient_eigen(i) /
                lambda;

            ++usable_directions;
        }

        if (usable_directions <= 0)
        {
            return false;
        }

        const Eigen::Matrix<double, 6, 1> delta_analysis =
            eigenvectors *
            delta_eigen;

        dx =
            parameter_unscale *
            delta_analysis;

        if (!dx.allFinite())
        {
            return false;
        }

        if (final_H_analysis != nullptr)
        {
            *final_H_analysis =
                H_analysis;
        }

        if (final_parameter_unscale != nullptr)
        {
            *final_parameter_unscale =
                parameter_unscale;
        }

        return true;
    }

    Eigen::Isometry3d ApplyGroundIcpIncrement(
        const Eigen::Isometry3d &T_target_source,
        const Eigen::Matrix<double, 6, 1> &dx,
        const LidarRegistrationConfig &config)
    {
        Eigen::Isometry3d updated_pose =
            T_target_source;

        const Eigen::Vector3d delta_rotation =
            dx.head<3>();

        const Eigen::Vector3d delta_translation =
            dx.tail<3>();

        const Eigen::Matrix3d delta_R =
            Sophus::SO3d::exp(
                delta_rotation)
                .matrix();

        if (config.enable_sensor_centered_perturbation)
        {
            updated_pose.linear() =
                delta_R *
                updated_pose.rotation();

            updated_pose.translation() +=
                delta_translation;
        }
        else
        {
            Eigen::Isometry3d delta_T =
                Eigen::Isometry3d::Identity();

            delta_T.linear() =
                delta_R;

            delta_T.translation() =
                delta_translation;

            updated_pose =
                delta_T *
                updated_pose;
        }

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

    // ========================================================================
    // GROUND_ICP_V13_OBSERVABLE_SUBSPACE_JOINT
    //
    // Final frontend Ground design used in this branch:
    //
    //   E(T) = E_general_point_to_plane(T)
    //        + Q_g * E_ground_point_to_local_ground(T)
    //
    // Ground V4 supplies the trusted support points and frame quality Q_g.
    // The historical reference is NOT one frozen world plane.  Each trusted
    // support point is matched to a nearby, normal-compatible plane already
    // present in the current Prepared LocalMap target.
    //
    // Most importantly, the Ground Jacobian is explicitly projected onto the
    // physically observable ground subspace:
    //
    //       [rx, ry, rz, tx, ty, tz]
    //        ^   ^                ^
    //      roll pitch             z
    //
    // rz / tx / ty are set to zero for Ground residuals.  General ICP remains
    // full 6-DoF and continues to estimate x/y/yaw from all scene geometry.
    //
    // Therefore BOTH objectives enter the SAME Gauss-Newton system:
    //
    //   H_total = H_general + H_ground
    //   b_total = b_general + b_ground
    //   H_total * dx = -b_total
    //
    // This removes the V1.2B global frozen-height assumption while keeping the
    // desired quality-weighted roll/pitch/z Ground information.
    // ========================================================================

    GroundJointIcpStatus RunTrustedGroundJointIcpV12(
        const RegistrationScan2LocalMap *owner,
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &source,
        const PreparedLidarTarget &target,
        const fr_slam::GroundSegmentationResult &ground_result,
        const Eigen::Isometry3d &initial_guess,
        LidarRegistrationResult &result)
    {
        result =
            LidarRegistrationResult();

        result.T_target_source =
            initial_guess;

        GroundIcpRuntime *runtime =
            GetGroundIcpRuntime(owner);

        if (runtime == nullptr ||
            !runtime->enabled)
        {
            return GroundJointIcpStatus::NotEligible;
        }

        if (!ground_result.support_constraint_valid ||
            !ground_result.support_plane_valid ||
            !ground_result.support_ground_cloud ||
            ground_result.support_ground_cloud->empty())
        {

            return GroundJointIcpStatus::NotEligible;
        }

        const double ground_information_weight =
            ComputeGroundIcpWeight(
                *runtime,
                ground_result);

        if (!std::isfinite(ground_information_weight) ||
            ground_information_weight <= 1.0e-3)
        {

            return GroundJointIcpStatus::NotEligible;
        }

        const LidarRegistrationConfig &config =
            runtime->registration_config;

        result.robust_kernel_enabled =
            config.enable_huber_loss;

        result.robust_kernel_delta =
            config.huber_delta;

        Eigen::Isometry3d current_pose =
            initial_guess;

        bool completed_iteration =
            false;

        for (int iteration = 0;
             iteration < config.max_iterations;
             ++iteration)
        {
            GroundIcpLinearization linearization;

            // General Point-to-Plane + quality-weighted Ground residuals are
            // assembled together here.  The Ground block inside
            // BuildGroundIcpLinearization() has its Jacobian projected to
            // [roll, pitch, z] only.
            if (!BuildGroundIcpLinearization(
                    source,
                    target,
                    current_pose,
                    ground_result,
                    *runtime,
                    ground_information_weight,
                    linearization))
            {
                std::cout
                    << "GROUND_ICP_V13"
                    << " | stage=JOINT_ITER"
                    << " | iteration=" << iteration
                    << " | action=FALLBACK_GENERAL"
                    << " | reason=LINEARIZATION_FAILED"
                    << std::endl;

                return GroundJointIcpStatus::Failed;
            }

            if (linearization.general_correspondences <
                config.min_correspondences)
            {
                std::cout
                    << "GROUND_ICP_V13"
                    << " | stage=JOINT_ITER"
                    << " | iteration=" << iteration
                    << " | action=FALLBACK_GENERAL"
                    << " | reason=LOW_GENERAL_CORR"
                    << " | general_corr="
                    << linearization.general_correspondences
                    << " | min="
                    << config.min_correspondences
                    << std::endl;

                return GroundJointIcpStatus::Failed;
            }

            if (linearization.ground_correspondences <
                runtime->minimum_ground_correspondences)
            {
                std::cout
                    << "GROUND_ICP_V13"
                    << " | stage=JOINT_ITER"
                    << " | iteration=" << iteration
                    << " | action=FALLBACK_GENERAL"
                    << " | reason=LOW_GROUND_CORR"
                    << " | ground_corr="
                    << linearization.ground_correspondences
                    << " | min="
                    << runtime->minimum_ground_correspondences
                    << std::endl;

                return iteration == 0
                           ? GroundJointIcpStatus::NotEligible
                           : GroundJointIcpStatus::Failed;
            }

            Eigen::Matrix<double, 6, 1> dx =
                Eigen::Matrix<double, 6, 1>::Zero();

            if (!SolveGroundIcpStep(
                    linearization,
                    config,
                    dx))
            {
                std::cout
                    << "GROUND_ICP_V13"
                    << " | stage=JOINT_ITER"
                    << " | iteration=" << iteration
                    << " | action=FALLBACK_GENERAL"
                    << " | reason=SOLVE_FAILED"
                    << std::endl;

                return GroundJointIcpStatus::Failed;
            }

            const Eigen::Vector3d delta_rotation =
                dx.head<3>();

            const Eigen::Vector3d delta_translation =
                dx.tail<3>();

            const double dR =
                delta_rotation.norm();

            const double dT =
                delta_translation.norm();

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

            const Eigen::Isometry3d trial_pose =
                ApplyGroundIcpIncrement(
                    current_pose,
                    dx,
                    config);

            if (!trial_pose.matrix().allFinite())
            {
                return GroundJointIcpStatus::Failed;
            }

            current_pose =
                trial_pose;

            completed_iteration =
                true;

            result.success =
                true;

            result.converged =
                false;

            result.iterations =
                iteration + 1;

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
                current_pose;

            if (dR <
                    config.rotation_convergence_threshold &&
                dT <
                    config.translation_convergence_threshold)
            {
                result.converged =
                    true;
                break;
            }
        }

        if (!completed_iteration ||
            !result.success)
        {
            return GroundJointIcpStatus::Failed;
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
            return GroundJointIcpStatus::Failed;
        }

        if (final_linearization.general_correspondences <
                config.min_correspondences ||
            final_linearization.ground_correspondences <
                runtime->minimum_ground_correspondences)
        {
            return GroundJointIcpStatus::Failed;
        }

        const double final_robust_rmse =
            final_linearization.general_weight_sum > 1.0e-12
                ? std::sqrt(
                      final_linearization
                          .general_weighted_squared_error_sum /
                      final_linearization.general_weight_sum)
                : std::numeric_limits<double>::infinity();

        const double final_downweighted_ratio =
            final_linearization.general_correspondences > 0
                ? static_cast<double>(
                      final_linearization
                          .general_downweighted_correspondences) /
                      static_cast<double>(
                          final_linearization.general_correspondences)
                : 0.0;

        result.correspondences =
            final_linearization.general_correspondences;

        result.rmse =
            final_linearization.GeneralRmse();

        result.robust_downweighted_correspondences =
            final_linearization.general_downweighted_correspondences;

        result.robust_downweighted_ratio =
            final_downweighted_ratio;

        result.robust_effective_weight_sum =
            final_linearization.general_weight_sum;

        result.robust_min_weight =
            final_linearization.general_min_robust_weight;

        result.robust_rmse =
            final_robust_rmse;

        result.T_target_source =
            current_pose;

        UpdateGroundIcpRelativeCovariance(
            final_linearization,
            config,
            result);

        return GroundJointIcpStatus::Success;
    }
