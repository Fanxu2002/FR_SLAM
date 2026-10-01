#include "fr_slam/loop/loop_retrieval_sampler.hpp"
#include "fr_slam/loop/btc_adapter.hpp"
#include "fr_slam/loop/scan_context_window_shadow.hpp"
#include "fr_slam/loop/scan_context_shadow.hpp"
#include "fr_slam/frontend/lo_frontend.hpp"
#include "fr_slam/frontend/ground_segmenter.hpp"
#include "fr_slam/frontend/ground_input_bridge.hpp"
#include "fr_slam/loop/loop_decision_debug.hpp"
#include "fr_slam/loop/btc_descriptor.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <cstdint>
#include <cstddef>
#include <exception>
#include <array>
#include <future>
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
#include <unordered_set>

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
namespace
{
    const rclcpp::Logger kTimingLogger =
        rclcpp::get_logger("scan2local_map.timing");

    std::filesystem::path FrSlamOutputDirectory()
    {
        const char *configured_directory =
            std::getenv("FR_SLAM_OUTPUT_DIR");

        if (configured_directory != nullptr &&
            configured_directory[0] != '\0')
        {
            return std::filesystem::path(
                configured_directory);
        }

        const char *home_directory =
            std::getenv("HOME");

        if (home_directory != nullptr &&
            home_directory[0] != '\0')
        {
            return std::filesystem::path(
                       home_directory) /
                   "ros2_ws" /
                   "src" /
                   "fr_slam" /
                   "output";
        }

        return std::filesystem::path(
            "/tmp/fr_slam_output");
    }

    std::filesystem::path FrontendLoopDirectory()
    {
        return FrSlamOutputDirectory() /
               "loop";
    }

    Eigen::Vector3d FrontendRotationToRpy(
        const Eigen::Matrix3d &R)
    {
        const double sy =
            std::sqrt(
                R(0, 0) * R(0, 0) +
                R(1, 0) * R(1, 0));

        const bool singular =
            sy < 1.0e-8;

        double roll = 0.0;
        double pitch = 0.0;
        double yaw = 0.0;

        if (!singular)
        {
            roll =
                std::atan2(
                    R(2, 1),
                    R(2, 2));

            pitch =
                std::atan2(
                    -R(2, 0),
                    sy);

            yaw =
                std::atan2(
                    R(1, 0),
                    R(0, 0));
        }
        else
        {
            roll =
                std::atan2(
                    -R(1, 2),
                    R(1, 1));

            pitch =
                std::atan2(
                    -R(2, 0),
                    sy);

            yaw = 0.0;
        }

        return Eigen::Vector3d(
            roll,
            pitch,
            yaw);
    }

    double ElapsedMilliseconds(
        const std::chrono::steady_clock::time_point &start,
        const std::chrono::steady_clock::time_point &end)
    {
        return std::chrono::duration<double, std::milli>(
                   end - start)
            .count();
    }

    struct LoopTimingDiagnostics
    {
        std::chrono::steady_clock::time_point start =
            std::chrono::steady_clock::now();

        std::size_t current_keyframe_id = 0;
        std::size_t current_submap_id = 0;
        std::size_t candidates = 0;
        std::size_t verifier_prescore_calls = 0;
        std::size_t verifier_calls = 0;
        std::size_t pose_graph_optimize_calls = 0;

        double btc_retrieval_ms = 0.0;
        double verifier_prescore_ms = 0.0;
        double verifier_ms = 0.0;
        double pose_graph_optimize_ms = 0.0;
        double map_odom_ms = 0.0;
        double global_map_rebuild_ms = 0.0;
        double refinement_ms = 0.0;

        bool geometry_accepted = false;
        bool loop_edge_accepted = false;
        bool optimization_accepted = false;
    };

    class LoopTimingReporter
    {
    public:
        explicit LoopTimingReporter(
            LoopTimingDiagnostics &diagnostics)
            : diagnostics_(diagnostics)
        {
        }

        ~LoopTimingReporter()
        {
            const double total_ms =
                ElapsedMilliseconds(
                    diagnostics_.start,
                    std::chrono::steady_clock::now());

        }

    private:
        LoopTimingDiagnostics &diagnostics_;
    };

    double RelativeRotationDeg(
        const Eigen::Isometry3d &T_A,
        const Eigen::Isometry3d &T_B)
    {
        if (!T_A.matrix().allFinite() ||
            !T_B.matrix().allFinite())
        {
            return std::numeric_limits<double>::quiet_NaN();
        }

        const Eigen::Matrix3d R_AB =
            T_A.rotation().transpose() *
            T_B.rotation();

        Eigen::Quaterniond q_AB(R_AB);

        if (!q_AB.coeffs().allFinite() ||
            q_AB.norm() < 1.0e-12)
        {
            return std::numeric_limits<double>::quiet_NaN();
        }

        q_AB.normalize();

        // q and -q represent the same rotation.
        const double w =
            std::clamp(
                std::abs(q_AB.w()),
                0.0,
                1.0);

        return 2.0 *
               std::acos(w) *
               180.0 /
               M_PI;
    }

    // ============================================================================
    // ConvertToXYZ()
    //
    // Convert the project's LIDAR_POINT cloud into pcl::PointXYZ.
    //
    // Why:
    //     The main point-to-plane registration can keep using the project's custom
    //     point type, but this recovery experiment deliberately uses the standard
    //     PCL PointXYZ type for coarse point-to-point ICP.
    //
    // This also avoids unnecessary template / registration complications around
    // custom point types inside PCL's ICP implementation.
    // ============================================================================
    pcl::PointCloud<pcl::PointXYZ>::Ptr ConvertToXYZ(
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &cloud)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr xyz_cloud(
            new pcl::PointCloud<pcl::PointXYZ>);

        if (!cloud)
        {
            return xyz_cloud;
        }

        xyz_cloud->reserve(cloud->size());

        for (const LIDAR_POINT &point : cloud->points)
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

            xyz_cloud->push_back(xyz);
        }

        xyz_cloud->width =
            static_cast<std::uint32_t>(xyz_cloud->size());

        xyz_cloud->height = 1;
        xyz_cloud->is_dense = true;

        return xyz_cloud;
    }

    // ========================================================================
    // Loop Shadow Point-to-Plane -> Full 6x6 information.
    //
    // This production helper deliberately lives in this .cpp so the public
    // LoopVerifier / RegistrationScan2LocalMap headers do not need new ABI.
    // P2P ICP still owns the loop pose.  Only an edge that has survived the
    // existing loop gates uses this shadow geometry to build its information.
    //
    // Output convention:
    //     order = [tx ty tz rx ry rz]
    //     frame = current/source LiDAR (g2o to-node frame)
    // ========================================================================
    constexpr int kLoopInformationPlaneKnn = 5;
    constexpr double kLoopInformationMaxPlaneFitError = 0.15;
    constexpr double kLoopInformationMinimumScaleRange = 1.0;
    constexpr double kLoopInformationMaximumScaleRange = 50.0;
    constexpr double kLoopInformationRelativeEigenvalueFloor = 0.01;
    constexpr double kLoopInformationMinimumDirectionalConfidence = 0.01;
    constexpr std::size_t kLoopInformationMinimumCorrespondences = 50;

    pcl::PointCloud<pcl::PointXYZ>::Ptr VoxelFilterLoopInformation(
        const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud,
        double leaf_size)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(
            new pcl::PointCloud<pcl::PointXYZ>);

        if (!cloud ||
            cloud->empty() ||
            !std::isfinite(leaf_size) ||
            leaf_size <= 0.0)
        {
            return filtered;
        }

        pcl::VoxelGrid<pcl::PointXYZ> voxel;
        voxel.setInputCloud(cloud);

        const float leaf =
            static_cast<float>(leaf_size);

        voxel.setLeafSize(
            leaf,
            leaf,
            leaf);

        voxel.filter(*filtered);

        return filtered;
    }

    // BTC-only V30: legacy reverse-segment visibility helpers removed.

    bool FitLoopInformationPlane(
        const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &target,
        const std::vector<int> &neighbor_indices,
        Eigen::Vector3d &plane_point,
        Eigen::Vector3d &plane_normal)
    {
        if (!target ||
            neighbor_indices.size() < 3)
        {
            return false;
        }

        Eigen::Vector3d centroid =
            Eigen::Vector3d::Zero();

        for (const int index : neighbor_indices)
        {
            if (index < 0 ||
                static_cast<std::size_t>(index) >= target->size())
            {
                return false;
            }

            const pcl::PointXYZ &point =
                target->points[static_cast<std::size_t>(index)];

            centroid +=
                Eigen::Vector3d(
                    static_cast<double>(point.x),
                    static_cast<double>(point.y),
                    static_cast<double>(point.z));
        }

        centroid /=
            static_cast<double>(
                neighbor_indices.size());

        Eigen::Matrix3d covariance =
            Eigen::Matrix3d::Zero();

        for (const int index : neighbor_indices)
        {
            const pcl::PointXYZ &point =
                target->points[static_cast<std::size_t>(index)];

            const Eigen::Vector3d p_target(
                static_cast<double>(point.x),
                static_cast<double>(point.y),
                static_cast<double>(point.z));

            const Eigen::Vector3d delta =
                p_target - centroid;

            covariance.noalias() +=
                delta * delta.transpose();
        }

        covariance /=
            static_cast<double>(
                neighbor_indices.size());

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>
            eigen_solver(
                covariance,
                Eigen::ComputeEigenvectors);

        if (eigen_solver.info() != Eigen::Success)
        {
            return false;
        }

        Eigen::Vector3d normal =
            eigen_solver.eigenvectors().col(0);

        const double normal_norm =
            normal.norm();

        if (!std::isfinite(normal_norm) ||
            normal_norm < 1.0e-12)
        {
            return false;
        }

        normal /= normal_norm;

        for (const int index : neighbor_indices)
        {
            const pcl::PointXYZ &point =
                target->points[static_cast<std::size_t>(index)];

            const Eigen::Vector3d p_target(
                static_cast<double>(point.x),
                static_cast<double>(point.y),
                static_cast<double>(point.z));

            const double distance =
                std::abs(
                    normal.dot(
                        p_target - centroid));

            if (!std::isfinite(distance) ||
                distance > kLoopInformationMaxPlaneFitError)
            {
                return false;
            }
        }

        plane_point = centroid;
        plane_normal = normal;

        return true;
    }

    double LoopInformationMedian(
        std::vector<double> values)
    {
        if (values.empty())
        {
            return std::numeric_limits<double>::quiet_NaN();
        }

        const std::size_t middle =
            values.size() / 2;

        std::nth_element(
            values.begin(),
            values.begin() +
                static_cast<std::ptrdiff_t>(middle),
            values.end());

        double median =
            values[middle];

        if (values.size() % 2 == 0)
        {
            const double lower =
                *std::max_element(
                    values.begin(),
                    values.begin() +
                        static_cast<std::ptrdiff_t>(middle));

            median =
                0.5 *
                (lower + median);
        }

        return median;
    }

    bool BuildLoopShadowInformationFull6x6(
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &source_current,
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &target_historical,
        const Eigen::Isometry3d &T_target_source,
        const LoopVerifierConfig &verifier_config,
        Eigen::Matrix<double, 6, 6> &information,
        std::size_t &shadow_correspondences,
        double &median_range,
        double &minimum_relative_eigenvalue)
    {
        information =
            Eigen::Matrix<double, 6, 6>::Identity();

        shadow_correspondences = 0;
        median_range =
            std::numeric_limits<double>::quiet_NaN();
        minimum_relative_eigenvalue =
            std::numeric_limits<double>::quiet_NaN();

        if (!source_current ||
            !target_historical ||
            source_current->empty() ||
            target_historical->empty() ||
            !T_target_source.matrix().allFinite() ||
            !std::isfinite(verifier_config.voxel_leaf_size) ||
            verifier_config.voxel_leaf_size <= 0.0 ||
            !std::isfinite(verifier_config.verification_inlier_distance) ||
            verifier_config.verification_inlier_distance <= 0.0)
        {
            return false;
        }

        const pcl::PointCloud<pcl::PointXYZ>::Ptr source_xyz =
            ConvertToXYZ(source_current);

        const pcl::PointCloud<pcl::PointXYZ>::Ptr target_xyz =
            ConvertToXYZ(target_historical);

        const pcl::PointCloud<pcl::PointXYZ>::Ptr source_filtered =
            VoxelFilterLoopInformation(
                source_xyz,
                verifier_config.voxel_leaf_size);

        const pcl::PointCloud<pcl::PointXYZ>::Ptr target_filtered =
            VoxelFilterLoopInformation(
                target_xyz,
                verifier_config.voxel_leaf_size);

        if (!source_filtered ||
            !target_filtered ||
            source_filtered->size() < verifier_config.min_cloud_points ||
            target_filtered->size() < verifier_config.min_cloud_points)
        {
            return false;
        }

        pcl::search::KdTree<pcl::PointXYZ>::Ptr target_kdtree(
            new pcl::search::KdTree<pcl::PointXYZ>());

        target_kdtree->setInputCloud(
            target_filtered);

        Eigen::Matrix<double, 6, 6> H_raw =
            Eigen::Matrix<double, 6, 6>::Zero();

        std::vector<double> correspondence_ranges;
        correspondence_ranges.reserve(
            source_filtered->size());

        std::vector<int> neighbor_indices(
            static_cast<std::size_t>(
                kLoopInformationPlaneKnn));

        std::vector<float> neighbor_squared_distances(
            static_cast<std::size_t>(
                kLoopInformationPlaneKnn));

        const double inlier_distance =
            verifier_config.verification_inlier_distance;

        const double maximum_squared_distance =
            inlier_distance *
            inlier_distance;

        const Eigen::Vector3d sensor_origin_target =
            T_target_source.translation();

        for (const pcl::PointXYZ &source_point :
             source_filtered->points)
        {
            const Eigen::Vector3d p_source(
                static_cast<double>(source_point.x),
                static_cast<double>(source_point.y),
                static_cast<double>(source_point.z));

            const Eigen::Vector3d p_target =
                T_target_source *
                p_source;

            if (!p_target.allFinite())
            {
                continue;
            }

            pcl::PointXYZ query;
            query.x =
                static_cast<float>(p_target.x());
            query.y =
                static_cast<float>(p_target.y());
            query.z =
                static_cast<float>(p_target.z());

            const int found =
                target_kdtree->nearestKSearch(
                    query,
                    kLoopInformationPlaneKnn,
                    neighbor_indices,
                    neighbor_squared_distances);

            if (found < kLoopInformationPlaneKnn)
            {
                continue;
            }

            const double nearest_squared_distance =
                static_cast<double>(
                    neighbor_squared_distances[0]);

            if (!std::isfinite(nearest_squared_distance) ||
                nearest_squared_distance > maximum_squared_distance)
            {
                continue;
            }

            Eigen::Vector3d plane_point;
            Eigen::Vector3d plane_normal;

            if (!FitLoopInformationPlane(
                    target_filtered,
                    neighbor_indices,
                    plane_point,
                    plane_normal))
            {
                continue;
            }

            const double residual =
                plane_normal.dot(
                    p_target -
                    plane_point);

            if (!std::isfinite(residual) ||
                std::abs(residual) > inlier_distance)
            {
                continue;
            }

            const Eigen::Vector3d lever_arm_target =
                p_target -
                sensor_origin_target;

            if (!lever_arm_target.allFinite())
            {
                continue;
            }

            Eigen::Matrix<double, 1, 6> J =
                Eigen::Matrix<double, 1, 6>::Zero();

            J.block<1, 3>(0, 0) =
                lever_arm_target.cross(
                                    plane_normal)
                    .transpose();

            J.block<1, 3>(0, 3) =
                plane_normal.transpose();

            H_raw.noalias() +=
                J.transpose() *
                J;

            const double range =
                lever_arm_target.norm();

            if (std::isfinite(range) &&
                range > 1.0e-9)
            {
                correspondence_ranges.push_back(
                    range);
            }

            ++shadow_correspondences;
        }

        if (shadow_correspondences <
                kLoopInformationMinimumCorrespondences ||
            correspondence_ranges.empty() ||
            !H_raw.allFinite())
        {
            return false;
        }

        median_range =
            LoopInformationMedian(
                correspondence_ranges);

        if (!std::isfinite(median_range) ||
            median_range <= 0.0)
        {
            return false;
        }

        const double scale_L =
            std::clamp(
                median_range,
                kLoopInformationMinimumScaleRange,
                kLoopInformationMaximumScaleRange);

        Eigen::Matrix<double, 6, 6> parameter_unscale =
            Eigen::Matrix<double, 6, 6>::Identity();

        const double inverse_scale =
            1.0 /
            scale_L;

        parameter_unscale(0, 0) = inverse_scale;
        parameter_unscale(1, 1) = inverse_scale;
        parameter_unscale(2, 2) = inverse_scale;

        Eigen::Matrix<double, 6, 6> H_analysis =
            parameter_unscale.transpose() *
            H_raw *
            parameter_unscale;

        H_analysis =
            0.5 *
            (H_analysis +
             H_analysis.transpose());

        if (!H_analysis.allFinite())
        {
            return false;
        }

        Eigen::SelfAdjointEigenSolver<
            Eigen::Matrix<double, 6, 6>>
            hessian_solver(
                H_analysis);

        if (hessian_solver.info() != Eigen::Success ||
            !hessian_solver.eigenvalues().allFinite())
        {
            return false;
        }

        const Eigen::Matrix<double, 6, 1> eigenvalues =
            hessian_solver.eigenvalues();

        const double lambda_max =
            eigenvalues(5);

        if (!std::isfinite(lambda_max) ||
            lambda_max <= 1.0e-12)
        {
            return false;
        }

        const Eigen::Matrix<double, 6, 1> relative_eigenvalues =
            eigenvalues /
            lambda_max;

        if (!relative_eigenvalues.allFinite())
        {
            return false;
        }

        minimum_relative_eigenvalue =
            relative_eigenvalues.minCoeff();

        Eigen::Matrix<double, 6, 6> inverse_relative_eigenvalues =
            Eigen::Matrix<double, 6, 6>::Zero();

        for (int i = 0;
             i < 6;
             ++i)
        {
            const double safe_relative =
                std::clamp(
                    relative_eigenvalues(i),
                    kLoopInformationRelativeEigenvalueFloor,
                    1.0);

            inverse_relative_eigenvalues(i, i) =
                1.0 /
                safe_relative;
        }

        Eigen::Matrix<double, 6, 6> covariance_target_rt =
            hessian_solver.eigenvectors() *
            inverse_relative_eigenvalues *
            hessian_solver.eigenvectors().transpose();

        covariance_target_rt =
            0.5 *
            (covariance_target_rt +
             covariance_target_rt.transpose());

        if (!covariance_target_rt.allFinite())
        {
            return false;
        }

        const Eigen::Matrix3d R_source_target =
            T_target_source.rotation().transpose();

        Eigen::Matrix<double, 6, 6> target_to_source =
            Eigen::Matrix<double, 6, 6>::Zero();

        target_to_source.block<3, 3>(0, 0) =
            R_source_target;

        target_to_source.block<3, 3>(3, 3) =
            R_source_target;

        Eigen::Matrix<double, 6, 6> covariance_source_rt =
            target_to_source *
            covariance_target_rt *
            target_to_source.transpose();

        Eigen::Matrix<double, 6, 6> rt_to_tr =
            Eigen::Matrix<double, 6, 6>::Zero();

        rt_to_tr.block<3, 3>(0, 3) =
            Eigen::Matrix3d::Identity();

        rt_to_tr.block<3, 3>(3, 0) =
            Eigen::Matrix3d::Identity();

        Eigen::Matrix<double, 6, 6> covariance_tr =
            rt_to_tr *
            covariance_source_rt *
            rt_to_tr.transpose();

        covariance_tr =
            0.5 *
            (covariance_tr +
             covariance_tr.transpose());

        if (!covariance_tr.allFinite())
        {
            return false;
        }

        Eigen::Matrix<double, 6, 1> confidence_tr =
            Eigen::Matrix<double, 6, 1>::Ones();

        for (int i = 0;
             i < 6;
             ++i)
        {
            const double variance =
                covariance_tr(i, i);

            if (!std::isfinite(variance) ||
                variance <= 0.0)
            {
                return false;
            }

            confidence_tr(i) =
                std::clamp(
                    1.0 / variance,
                    kLoopInformationMinimumDirectionalConfidence,
                    1.0);
        }

        const double maximum_confidence =
            confidence_tr.maxCoeff();

        if (!std::isfinite(maximum_confidence) ||
            maximum_confidence <= 0.0)
        {
            return false;
        }

        confidence_tr /=
            maximum_confidence;

        for (int i = 0;
             i < 6;
             ++i)
        {
            confidence_tr(i) =
                std::clamp(
                    confidence_tr(i),
                    kLoopInformationMinimumDirectionalConfidence,
                    1.0);
        }

        Eigen::SelfAdjointEigenSolver<
            Eigen::Matrix<double, 6, 6>>
            covariance_solver(
                covariance_tr);

        if (covariance_solver.info() != Eigen::Success ||
            !covariance_solver.eigenvalues().allFinite() ||
            covariance_solver.eigenvalues().minCoeff() <= 1.0e-12)
        {
            return false;
        }

        Eigen::Matrix<double, 6, 6> inverse_covariance_eigenvalues =
            Eigen::Matrix<double, 6, 6>::Zero();

        for (int i = 0;
             i < 6;
             ++i)
        {
            inverse_covariance_eigenvalues(i, i) =
                1.0 /
                covariance_solver.eigenvalues()(i);
        }

        Eigen::Matrix<double, 6, 6> precision_shape =
            covariance_solver.eigenvectors() *
            inverse_covariance_eigenvalues *
            covariance_solver.eigenvectors().transpose();

        precision_shape =
            0.5 *
            (precision_shape +
             precision_shape.transpose());

        if (!precision_shape.allFinite())
        {
            return false;
        }

        Eigen::Matrix<double, 6, 6> standardized_precision =
            Eigen::Matrix<double, 6, 6>::Zero();

        for (int i = 0;
             i < 6;
             ++i)
        {
            if (!std::isfinite(precision_shape(i, i)) ||
                precision_shape(i, i) <= 0.0)
            {
                return false;
            }

            for (int j = 0;
                 j < 6;
                 ++j)
            {
                if (!std::isfinite(precision_shape(j, j)) ||
                    precision_shape(j, j) <= 0.0)
                {
                    return false;
                }

                const double denominator =
                    std::sqrt(
                        precision_shape(i, i) *
                        precision_shape(j, j));

                if (!std::isfinite(denominator) ||
                    denominator <= 0.0)
                {
                    return false;
                }

                standardized_precision(i, j) =
                    precision_shape(i, j) /
                    denominator;
            }
        }

        standardized_precision =
            0.5 *
            (standardized_precision +
             standardized_precision.transpose());

        Eigen::Matrix<double, 6, 6> confidence_scale =
            Eigen::Matrix<double, 6, 6>::Zero();

        for (int i = 0;
             i < 6;
             ++i)
        {
            confidence_scale(i, i) =
                std::sqrt(
                    confidence_tr(i));
        }

        information =
            confidence_scale *
            standardized_precision *
            confidence_scale;

        information =
            0.5 *
            (information +
             information.transpose());

        if (!information.allFinite())
        {
            information =
                Eigen::Matrix<double, 6, 6>::Identity();
            return false;
        }

        Eigen::SelfAdjointEigenSolver<
            Eigen::Matrix<double, 6, 6>>
            information_solver(
                information,
                Eigen::EigenvaluesOnly);

        if (information_solver.info() != Eigen::Success ||
            !information_solver.eigenvalues().allFinite() ||
            information_solver.eigenvalues().minCoeff() <= 1.0e-9)
        {
            information =
                Eigen::Matrix<double, 6, 6>::Identity();
            return false;
        }

        return true;
    }

    // ========================================================================
    // Loop Full 6x6 information helpers.
    //
    // Input/output convention:
    //     order = [tx ty tz rx ry rz]
    //     frame = current/to-node LiDAR frame
    //
    // The matrix is a relative information SHAPE, not a physical covariance.
    // Global translation/rotation calibration (currently 1:30) remains in the
    // PoseGraphOptimizer and is applied there by congruence scaling.
    // ========================================================================
    void ComputeLoopInformationStats(
        const Eigen::Matrix<double, 6, 6> &information,
        double &maximum_absolute_off_diagonal,
        double &maximum_translation_rotation_coupling)
    {
        maximum_absolute_off_diagonal = 0.0;
        maximum_translation_rotation_coupling = 0.0;

        for (int i = 0;
             i < 6;
             ++i)
        {
            for (int j = 0;
                 j < 6;
                 ++j)
            {
                if (i == j)
                {
                    continue;
                }

                const double absolute_value =
                    std::abs(
                        information(i, j));

                maximum_absolute_off_diagonal =
                    std::max(
                        maximum_absolute_off_diagonal,
                        absolute_value);

                const bool translation_rotation_pair =
                    (i < 3 && j >= 3) ||
                    (i >= 3 && j < 3);

                if (translation_rotation_pair)
                {
                    maximum_translation_rotation_coupling =
                        std::max(
                            maximum_translation_rotation_coupling,
                            absolute_value);
                }
            }
        }
    }
} // namespace

RegistrationScan2LocalMap::LoopIcpDebugSnapshot
RegistrationScan2LocalMap::GetLoopIcpDebugSnapshot() const
{
    std::lock_guard<std::mutex> lock(
        backend_output_mutex_);

    LoopIcpDebugSnapshot snapshot;

    snapshot.historical_target_world =
        backend_loop_icp_historical_target_snapshot_;

    snapshot.initial_aligned_world =
        backend_loop_icp_initial_aligned_snapshot_;

    snapshot.final_aligned_world =
        backend_loop_icp_final_aligned_snapshot_;

    snapshot.revision =
        backend_loop_icp_debug_revision_snapshot_;

    snapshot.current_keyframe_id =
        backend_loop_icp_debug_current_kf_snapshot_;

    snapshot.historical_keyframe_id =
        backend_loop_icp_debug_historical_kf_snapshot_;

    snapshot.initial_guess_name =
        backend_loop_icp_debug_guess_name_snapshot_;

    snapshot.correction_translation =
        backend_loop_icp_debug_correction_translation_snapshot_;

    snapshot.correction_rotation_deg =
        backend_loop_icp_debug_correction_rotation_snapshot_;

    return snapshot;
}

bool RegistrationScan2LocalMap::BuildCandidateCenteredHistoricalTarget(
    std::size_t historical_keyframe_id,
    pcl::PointCloud<LIDAR_POINT>::Ptr &target_K,
    std::vector<std::size_t> *included_keyframe_ids) const
{
    target_K =
        pcl::make_shared<
            pcl::PointCloud<LIDAR_POINT>>();

    if (included_keyframe_ids != nullptr)
    {
        included_keyframe_ids->clear();
    }

    const Keyframe *anchor =
        FindBackendKeyframeById(
            historical_keyframe_id);

    if (anchor == nullptr ||
        !anchor->cloud ||
        anchor->cloud->empty() ||
        !anchor->T_WL.matrix().allFinite())
    {
        return false;
    }

    std::vector<const Keyframe *> window_keyframes;
    window_keyframes.reserve(
        2 * online_loop_candidate_target_half_window_ + 1);

    std::size_t estimated_points = 0;

    for (const Keyframe &keyframe :
         backend_keyframes_)
    {
        if (!keyframe.cloud ||
            keyframe.cloud->empty() ||
            !keyframe.T_WL.matrix().allFinite())
        {
            continue;
        }

        const std::size_t id_gap =
            keyframe.id >= historical_keyframe_id
                ? keyframe.id - historical_keyframe_id
                : historical_keyframe_id - keyframe.id;

        if (id_gap >
            online_loop_candidate_target_half_window_)
        {
            continue;
        }

        window_keyframes.push_back(
            &keyframe);

        estimated_points +=
            keyframe.cloud->size();
    }

    if (window_keyframes.size() <
        online_loop_candidate_target_min_keyframes_)
    {
        return false;
    }

    pcl::PointCloud<LIDAR_POINT>::Ptr accumulated_K =
        pcl::make_shared<
            pcl::PointCloud<LIDAR_POINT>>();

    accumulated_K->reserve(
        estimated_points);

    const Eigen::Isometry3d T_K_W =
        anchor->T_WL.inverse();

    if (!T_K_W.matrix().allFinite())
    {
        return false;
    }

    for (const Keyframe *keyframe :
         window_keyframes)
    {
        if (keyframe == nullptr)
        {
            continue;
        }

        const Eigen::Isometry3d T_K_N =
            T_K_W *
            keyframe->T_WL;

        if (!T_K_N.matrix().allFinite())
        {
            continue;
        }

        pcl::PointCloud<LIDAR_POINT>
            cloud_K;

        pcl::transformPointCloud(
            *keyframe->cloud,
            cloud_K,
            T_K_N.matrix().cast<float>());

        *accumulated_K +=
            cloud_K;

        if (included_keyframe_ids != nullptr)
        {
            included_keyframe_ids->push_back(
                keyframe->id);
        }
    }

    if (accumulated_K->empty())
    {
        return false;
    }

    pcl::VoxelGrid<LIDAR_POINT>
        voxel;

    voxel.setLeafSize(
        static_cast<float>(
            online_loop_candidate_target_voxel_leaf_size_),
        static_cast<float>(
            online_loop_candidate_target_voxel_leaf_size_),
        static_cast<float>(
            online_loop_candidate_target_voxel_leaf_size_));

    voxel.setInputCloud(
        accumulated_K);

    voxel.filter(
        *target_K);

    if (!target_K ||
        target_K->empty())
    {
        return false;
    }

    return true;
}

// ============================================================================
// AddKeyframeToPoseGraph()
//
// V6 backend bridge:
//
//     New Keyframe KF_i
//           |
//           +--> PoseGraph Vertex i, X_i = T_WK_i
//           |
//           +--> sequential Keyframe odometry edge from KF_(i-1)
//
// Measurement convention:
//
//     Z_(i-1,i)
//       = T_K(i-1)_Ki
//       = T_WK(i-1)^-1 * T_WKi
//
// Submap finishing is deliberately NOT involved here.
// ============================================================================

const RegistrationScan2LocalMap::BackendSubmapSnapshot *
RegistrationScan2LocalMap::FindBackendSubmapById(
    std::size_t submap_id) const
{
    for (const BackendSubmapSnapshot &submap :
         backend_finished_submaps_)
    {
        if (submap.id == submap_id)
        {
            return &submap;
        }
    }

    return nullptr;
}

const Keyframe *
RegistrationScan2LocalMap::FindBackendKeyframeById(
    std::size_t keyframe_id) const
{
    for (const Keyframe &keyframe :
         backend_keyframes_)
    {
        if (keyframe.id == keyframe_id)
        {
            return &keyframe;
        }
    }

    return nullptr;
}

// ============================================================================
// FindBestFinishedSubmapForKeyframe()
//
// The backend never reads frontend SubmapManager storage.  It searches only
// immutable finished-submap snapshots transferred by BackendKeyframeJob.
// ============================================================================
const RegistrationScan2LocalMap::BackendSubmapSnapshot *
RegistrationScan2LocalMap::FindBestFinishedSubmapForKeyframe(
    std::size_t keyframe_id) const
{
    const BackendSubmapSnapshot *best = nullptr;

    std::size_t best_center_distance =
        std::numeric_limits<std::size_t>::max();

    for (const BackendSubmapSnapshot &submap :
         backend_finished_submaps_)
    {
        if (!submap.cloud_S ||
            submap.cloud_S->empty() ||
            !submap.T_WS.matrix().allFinite())
        {
            continue;
        }

        for (std::size_t index = 0;
             index < submap.keyframe_ids.size();
             ++index)
        {
            if (submap.keyframe_ids[index] != keyframe_id)
            {
                continue;
            }

            const std::size_t center =
                submap.keyframe_ids.size() / 2;

            const std::size_t center_distance =
                index > center
                    ? index - center
                    : center - index;

            if (best == nullptr ||
                center_distance < best_center_distance)
            {
                best = &submap;
                best_center_distance = center_distance;
            }

            break;
        }
    }

    return best;
}

// ============================================================================
// DetectAndVerifyLoopFromKeyframe()
//
// Candidate retrieval:
//     Current finished Submap -> Official BTC database
//
// Geometry verification:
//     Current Keyframe cloud (frame Lcurrent)
//              ->
//     Candidate-centered historical FINISHED Submap cloud_S (frame H)
//
// LoopVerifier returns:
//     T_H_Lcurrent
//
// Historical Submap H is ONLY an ICP target. It is not a graph vertex.
// If K is the historical candidate Keyframe:
//
//     T_H_K = T_WH^-1 * T_WK
//
// therefore the actual Keyframe-PoseGraph loop measurement is:
//
//     T_K_Lcurrent = T_H_K^-1 * T_H_Lcurrent
//
// and the final edge is:
//
//     historical KF K  --------  current KF L
// ============================================================================


std::vector<std::size_t>
RegistrationScan2LocalMap::GetLoopRetrievalSampleKeyframeIdsSnapshot() const
{
    std::lock_guard<std::mutex> lock(
        backend_output_mutex_);

    return
        backend_loop_retrieval_sample_keyframe_ids_snapshot_;
}

void RegistrationScan2LocalMap::DetectAndVerifyLoopFromKeyframe(
    const Keyframe &current_keyframe,
    std::size_t current_submap_id)
{
    LoopTimingDiagnostics loop_timing;
    loop_timing.current_keyframe_id = current_keyframe.id;
    loop_timing.current_submap_id = current_submap_id;

    LoopTimingReporter loop_timing_reporter(loop_timing);

    // Production runtime: experimental loop-closure shadow pipelines are disabled.
    // Keep the implementations for offline/test A/B experiments only.
    constexpr bool kEnableExperimentalLoopShadows = false;

    // ====================================================================
    // FR_LOOP_VERIFICATION_TRACE_V1
    //
    // Diagnostic only.
    // MUST NOT change retrieval / geometry / consistency / PGO decisions.
    // ====================================================================
    struct LoopVerificationTraceRow
    {
        std::size_t current_kf = 0;
        std::size_t historical_kf = 0;

        double retrieval_score =
            std::numeric_limits<double>::quiet_NaN();

        std::string stage;

        bool prescore_valid = false;
        double prescore_overlap =
            std::numeric_limits<double>::quiet_NaN();
        double prescore_rmse =
            std::numeric_limits<double>::quiet_NaN();

        bool verify_ok = false;
        bool geometry_accepted = false;

        double overlap =
            std::numeric_limits<double>::quiet_NaN();
        double rmse =
            std::numeric_limits<double>::quiet_NaN();
        double correction_translation =
            std::numeric_limits<double>::quiet_NaN();
        double correction_rotation_deg =
            std::numeric_limits<double>::quiet_NaN();

        bool gate_overlap_pass = false;
        bool gate_rmse_pass = false;
        bool gate_translation_pass = false;
        bool gate_rotation_pass = false;

        bool consistency_valid = false;

        bool temporal_available = false;
        std::size_t temporal_support = 0;
        bool temporal_consistent = false;

        bool cycle_available = false;
        double cycle_translation_error =
            std::numeric_limits<double>::quiet_NaN();
        double cycle_rotation_error_deg =
            std::numeric_limits<double>::quiet_NaN();
        bool cycle_consistent = false;

        bool consistency_accepted = false;
        bool pending_seed = false;

        std::string decision;
    };

    const auto append_loop_verification_trace =
        [&](const LoopVerificationTraceRow &row)
        {
            try
            {
                const std::filesystem::path directory =
                    FrontendLoopDirectory();

                std::filesystem::create_directories(
                    directory);

                const std::filesystem::path csv_path =
                    directory /
                    "loop_verification_trace.csv";

                const bool csv_exists =
                    std::filesystem::exists(
                        csv_path);

                std::ofstream file(
                    csv_path,
                    std::ios::out |
                        std::ios::app);

                if (!file.is_open())
                {
                    return;
                }

                file
                    << std::fixed
                    << std::setprecision(9);

                if (!csv_exists)
                {
                    file
                        << "current_kf,"
                        << "historical_kf,"
                        << "retrieval_score,"
                        << "stage,"
                        << "prescore_valid,"
                        << "prescore_overlap,"
                        << "prescore_rmse,"
                        << "verify_ok,"
                        << "geometry_accepted,"
                        << "overlap,"
                        << "rmse,"
                        << "correction_translation_m,"
                        << "correction_rotation_deg,"
                        << "gate_overlap_pass,"
                        << "gate_rmse_pass,"
                        << "gate_translation_pass,"
                        << "gate_rotation_pass,"
                        << "consistency_valid,"
                        << "temporal_available,"
                        << "temporal_support,"
                        << "temporal_consistent,"
                        << "cycle_available,"
                        << "cycle_translation_error_m,"
                        << "cycle_rotation_error_deg,"
                        << "cycle_consistent,"
                        << "consistency_accepted,"
                        << "pending_seed,"
                        << "decision\n";
                }

                file
                    << row.current_kf << ","
                    << row.historical_kf << ","
                    << row.retrieval_score << ","
                    << row.stage << ","
                    << (row.prescore_valid ? 1 : 0) << ","
                    << row.prescore_overlap << ","
                    << row.prescore_rmse << ","
                    << (row.verify_ok ? 1 : 0) << ","
                    << (row.geometry_accepted ? 1 : 0) << ","
                    << row.overlap << ","
                    << row.rmse << ","
                    << row.correction_translation << ","
                    << row.correction_rotation_deg << ","
                    << (row.gate_overlap_pass ? 1 : 0) << ","
                    << (row.gate_rmse_pass ? 1 : 0) << ","
                    << (row.gate_translation_pass ? 1 : 0) << ","
                    << (row.gate_rotation_pass ? 1 : 0) << ","
                    << (row.consistency_valid ? 1 : 0) << ","
                    << (row.temporal_available ? 1 : 0) << ","
                    << row.temporal_support << ","
                    << (row.temporal_consistent ? 1 : 0) << ","
                    << (row.cycle_available ? 1 : 0) << ","
                    << row.cycle_translation_error << ","
                    << row.cycle_rotation_error_deg << ","
                    << (row.cycle_consistent ? 1 : 0) << ","
                    << (row.consistency_accepted ? 1 : 0) << ","
                    << (row.pending_seed ? 1 : 0) << ","
                    << row.decision
                    << "\n";

                file.flush();
            }
            catch (const std::exception &)
            {
                // Diagnostics must never affect SLAM.
            }
        };

    const LoopDetectorConfig &loop_config =
        loop_detector_.GetConfig();

    const bool use_sc_single_retrieval =
        loop_runtime_config_.retrieval_mode ==
        LoopRetrievalMode::ScanContextSingle;

    const bool use_sc_window_retrieval =
        loop_runtime_config_.retrieval_mode ==
        LoopRetrievalMode::ScanContextWindow;

    const bool use_btc_retrieval =
        loop_runtime_config_.retrieval_mode ==
        LoopRetrievalMode::Btc;

    const bool use_window_cloud_retrieval =
        use_sc_window_retrieval ||
        use_btc_retrieval;

    const char *formal_retrieval_source_name =
        use_sc_single_retrieval
            ? "SC_SINGLE"
            : (use_sc_window_retrieval
                   ? "SC_WINDOW"
                   : (use_btc_retrieval
                          ? "BTC"
                          : "UNKNOWN"));

    enum class FormalLoopRetrievalSource
    {
        ScanContextSingle,
        ScanContextWindow,
        Btc
    };

    struct FormalLoopCandidate
    {
        FormalLoopRetrievalSource source =
            FormalLoopRetrievalSource::ScanContextSingle;

        std::size_t current_id = 0;
        std::size_t candidate_id = 0;
        std::size_t retrieval_rank = 0;

        double pose_distance =
            std::numeric_limits<double>::quiet_NaN();

        double time_separation_sec =
            std::numeric_limits<double>::quiet_NaN();

        double retrieval_score =
            std::numeric_limits<double>::quiet_NaN();

        std::size_t matched_descriptor_pairs = 0;

        double scan_context_distance =
            std::numeric_limits<double>::quiet_NaN();

        double scan_context_similarity =
            std::numeric_limits<double>::quiet_NaN();

        std::size_t compared_sectors = 0;

        bool has_yaw_hint = false;
        double yaw_hint_deg = 0.0;

        bool has_base_initial_guess = false;

        Eigen::Isometry3d base_initial_guess =
            Eigen::Isometry3d::Identity();
    };

    std::vector<FormalLoopCandidate>
        formal_loop_candidates;

    if (!loop_config.enabled)
    {
        return;
    }

    if (!current_keyframe.cloud ||
        current_keyframe.cloud->empty() ||
        !current_keyframe.T_WL.matrix().allFinite())
    {
        return;
    }

    // ====================================================================
    // BTC-only global retrieval V31.1.
    //
    // IMPORTANT:
    //   - This BTC window is independent from the FR-SLAM frontend/backend
    //     Submap definition.
    //   - Query window: trailing up-to-7 Keyframes, evaluated on EVERY new KF.
    //   - Database window: full 7-KF windows inserted every 4 KFs.
    //   - Source: Keyframe::btc_cloud ONLY (dense pre-frontend-voxel cloud).
    //   - No extra FR-SLAM VoxelGrid / SOR / ROR is applied here.
    //   - BTC itself still performs its own internal voxel/plane processing.
    //
    // Full database windows:
    //     [0..6], [4..10], [8..14], ...
    //
    // Query windows:
    //     KF306 -> [300..306]
    //     KF307 -> [301..307]
    //     KF308 -> [302..308]
    //
    // Window frame C/H is the pose of the center Keyframe.
    // ====================================================================
    const std::size_t btc_window_size =
        std::max<std::size_t>(
            1,
            loop_runtime_config_.btc_window_size);

    const std::size_t btc_window_stride =
        std::min(
            btc_window_size,
            std::max<std::size_t>(
                1,
                loop_runtime_config_.btc_window_stride));

    const std::size_t btc_min_valid_dense_keyframes =
        std::min(
            btc_window_size,
            std::max<std::size_t>(
                1,
                loop_runtime_config_.btc_min_valid_dense_keyframes));

    std::vector<LoopCandidate> btc_candidates;
    std::unordered_map<std::size_t, Eigen::Isometry3d>
        btc_initial_guess_by_kf;

    static fr_slam_btc::OfficialBtcAdapter official_btc_adapter(
        loop_runtime_config_.btc_config_profile,
        loop_runtime_config_.btc_skip_near_num,
        loop_runtime_config_.btc_proj_plane_num,
        btc_window_size,
        btc_window_stride);
    static bool btc_csv_initialized = false;

// ====================================================================
// SC-WINDOW SHADOW
//
// Uses the SAME accumulated 7-KF cloud as BTC.
//
// IMPORTANT:
//     Retrieval diagnostics only.
//     SC-WINDOW does NOT create loop edges.
//     SC-WINDOW does NOT modify PGO.
// ====================================================================
static std::unique_ptr<ScanContextWindowShadow>
    sc_window_shadow;

if (!sc_window_shadow)
{
    ScanContextWindowConfig sc_config;

    sc_config.min_keyframe_id_separation =
        loop_config.min_keyframe_id_separation;

    sc_config.min_time_separation_sec =
        loop_config.min_time_separation_sec;

    // Historical Scan Context retrieval threshold.
    sc_config.max_scan_context_distance =
        0.40;

    // SC-WINDOW tuning V1:
    // Old single-KF default was 0.50.
    // For the 7-KF accumulated window this over-penalized coverage
    // differences and reversed some otherwise-correct raw-cosine rankings.
    sc_config.scan_context.coverage_penalty_weight =
        0.30;

    // Frontend pose is diagnostic only.
    sc_config.use_pose_distance_gate =
        false;

    sc_config.max_candidate_distance =
        loop_config.max_candidate_distance;

    sc_config.max_candidates =
        loop_config.max_candidates;

    sc_window_shadow =
        std::make_unique<ScanContextWindowShadow>(
            sc_config);
}

    // ================================================================
    // ================================================================
    // SC-WINDOW FORMAL V1 bridge state
    // Per-call only; SC-WINDOW remains the retrieval source.
    // ================================================================
    std::vector<ScanContextShadowCandidate>
        sc_window_formal_candidates;

    Eigen::Isometry3d
        sc_window_formal_T_C_L =
            Eigen::Isometry3d::Identity();

    bool sc_window_formal_bridge_valid =
        false;

    // FR-SLAM V31.11
    //
    // First geometry-valid but temporally unconfirmed loop
    // creates a short-lived PENDING track.
    //
    // PENDING only relaxes BTC rough candidate admission locally.
    // It never bypasses:
    //     candidate_verify
    //     BTC plane verification
    //     FR geometry
    //     temporal consistency
    //     cycle consistency
    //     PGO
    // ================================================================
    static bool
        btc_pending_rescue_active_v31_11 = false;

    static std::size_t
        btc_pending_current_kf_v31_11 = 0;

    static std::size_t
        btc_pending_historical_kf_v31_11 = 0;

    const LoopConsistencyConfig &
        btc_consistency_config_v31_11 =
            loop_consistency_checker_.GetConfig();

    if (btc_pending_rescue_active_v31_11 &&
        current_keyframe.id >
            btc_pending_current_kf_v31_11)
    {
        const std::size_t pending_current_gap =
            current_keyframe.id -
            btc_pending_current_kf_v31_11;

        if (pending_current_gap >
            btc_consistency_config_v31_11
                .temporal_max_current_gap)
        {

            official_btc_adapter
                .ClearPendingRescue();

            btc_pending_rescue_active_v31_11 =
                false;
        }
    }

    // ====================================================================
    // LOOP RETRIEVAL SAMPLER V1
    //
    // Mapping Keyframe:
    //     existing 0.5 m OR 5 deg rule -- UNCHANGED.
    //
    // Loop Retrieval Sample:
    //     accumulated Mapping-KF path >= 0.50 m
    //     OR
    //     rotation from last Loop Sample >= 10 deg.
    //
    // SC and BTC consume exactly the same Loop Sample sequence.
    // ====================================================================
    static LoopRetrievalSampler
        loop_retrieval_sampler(
            0.50,
            10.0);

    static std::vector<LoopRetrievalSample>
        loop_retrieval_samples;

    LoopRetrievalSample
        current_loop_sample;

    LoopRetrievalSamplingDecision
        loop_sampling_decision;

    const bool loop_sample_accepted =
        loop_retrieval_sampler.ProcessMappingKeyframe(
            current_keyframe.id,
            current_keyframe.timestamp,
            current_keyframe.T_WL,
            current_loop_sample,
            loop_sampling_decision);

    if (!loop_sampling_decision.valid)
    {
        return;
    }

    if (loop_sample_accepted)
    {
        loop_retrieval_samples.push_back(
            current_loop_sample);

        {
            std::lock_guard<std::mutex> lock(
                backend_output_mutex_);

            backend_loop_retrieval_sample_keyframe_ids_snapshot_
                .push_back(
                    current_loop_sample.mapping_keyframe_id);
        }
    }
    // ====================================================================
    // LOOP RETRIEVAL SAMPLER V1 CSV
    //
    // Diagnostic only.
    // One row per Mapping Keyframe.
    //
    // This allows offline validation of:
    //   - Mapping KF -> Loop Sample reduction
    //   - accumulated path spacing
    //   - rotation triggers
    //   - physical scale of the 7-sample BTC/SC window
    // ====================================================================
    try
    {
        const std::filesystem::path sampler_directory =
            FrontendLoopDirectory();

        std::filesystem::create_directories(
            sampler_directory);

        const std::filesystem::path sampler_csv_path =
            sampler_directory /
            "loop_retrieval_samples.csv";

        const bool sampler_csv_exists =
            std::filesystem::exists(
                sampler_csv_path);

        std::ofstream sampler_csv(
            sampler_csv_path,
            std::ios::app);

        if (sampler_csv.is_open())
        {
            sampler_csv
                << std::fixed
                << std::setprecision(9);

            if (!sampler_csv_exists)
            {
                sampler_csv
                    << "mapping_keyframe_id,"
                    << "timestamp,"
                    << "accepted,"
                    << "sample_id,"
                    << "accumulated_path_m,"
                    << "rotation_from_last_sample_deg,"
                    << "trigger_translation,"
                    << "trigger_rotation,"
                    << "first_sample,"
                    << "total_samples"
                    << '\n';
            }

            sampler_csv
                << current_keyframe.id << ','
                << current_keyframe.timestamp << ','
                << (loop_sample_accepted ? 1 : 0)
                << ',';

            if (loop_sample_accepted)
            {
                sampler_csv
                    << current_loop_sample.sample_id;
            }
            else
            {
                sampler_csv
                    << -1;
            }

            sampler_csv
                << ','
                << loop_sampling_decision
                       .accumulated_path_distance_m
                << ','
                << loop_sampling_decision
                       .rotation_from_last_sample_deg
                << ','
                << (loop_sampling_decision
                            .trigger_translation
                        ? 1
                        : 0)
                << ','
                << (loop_sampling_decision
                            .trigger_rotation
                        ? 1
                        : 0)
                << ','
                << (loop_sampling_decision
                            .first_sample
                        ? 1
                        : 0)
                << ','
                << loop_retrieval_samples.size()
                << '\n';
        }
    }
    catch (const std::exception &)
    {
        // Diagnostics must never affect SLAM.
    }

    // Only accepted Loop Samples trigger SC/BTC retrieval.
    if (!loop_sample_accepted)
    {
        return;
    }

    // ====================================================================
    // SC-SINGLE SHADOW V1
    //
    // IMPORTANT:
    //
    // This detector consumes ONLY accepted Loop Retrieval Samples.
    //
    // It uses ONE original Mapping-Keyframe cloud per descriptor:
    //
    //     Loop Sample -> Mapping KF -> Keyframe::cloud
    //
    // It is completely independent from:
    //
    //     SC-WINDOW
    //     BTC
    //     LoopVerifier
    //     PoseGraph
    //     PGO
    //
    // Diagnostic only. It NEVER creates loop edges.
    // ====================================================================
    static std::unique_ptr<ScanContextShadowDetector>
        sc_single_shadow;

    if (!sc_single_shadow)
    {
        ScanContextShadowConfig
            sc_single_config;

        // Keep the same historical-separation policy as the
        // existing loop frontend for the first experiment.
        sc_single_config.min_keyframe_id_separation =
            loop_config.min_keyframe_id_separation;

        sc_single_config.min_time_separation_sec =
            loop_config.min_time_separation_sec;

        sc_single_config.max_scan_context_distance =
            0.45;

        // Frontend pose must NOT decide retrieval.
        sc_single_config.use_pose_distance_gate =
            false;

        sc_single_config.max_candidate_distance =
            loop_config.max_candidate_distance;

        sc_single_config.max_candidates =
            loop_config.max_candidates;

        // ============================================================
        // SC POSE SUPPLEMENT CONFIG V1
        //
        // Normal SC Top-K remains unchanged.
        //
        // Add at most ONE frontend-pose-nearest historical candidate
        // for TrackRecovery only.
        // ============================================================
        sc_single_config.enable_pose_supplement =
            true;

        sc_single_config
            .pose_supplement_min_keyframe_id_separation =
                100U;

        sc_single_config
            .pose_supplement_max_distance =
                2.0;

        sc_single_shadow =
            std::make_unique<ScanContextShadowDetector>(
                sc_single_config);
    }

    ScanContextShadowDiagnostics
        sc_single_diagnostics;

    std::vector<ScanContextShadowCandidate>
        sc_single_candidates;

    const bool sc_single_added =
        use_sc_single_retrieval
            ? sc_single_shadow->AddKeyframe(
                  current_keyframe)
            : false;

    if (use_sc_single_retrieval &&
        sc_single_added)
    {
        sc_single_candidates =
            sc_single_shadow->Detect(
                current_keyframe.id,
                &sc_single_diagnostics);

        formal_loop_candidates.clear();
        formal_loop_candidates.reserve(
            sc_single_candidates.size());

        for (std::size_t rank_index = 0;
             rank_index < sc_single_candidates.size();
             ++rank_index)
        {
            const ScanContextShadowCandidate &candidate =
                sc_single_candidates[rank_index];

            FormalLoopCandidate formal_candidate;

            formal_candidate.source =
                FormalLoopRetrievalSource::ScanContextSingle;

            formal_candidate.current_id =
                candidate.current_id;

            formal_candidate.candidate_id =
                candidate.candidate_id;

            formal_candidate.retrieval_rank =
                rank_index;

            formal_candidate.pose_distance =
                candidate.pose_distance;

            formal_candidate.time_separation_sec =
                candidate.time_separation_sec;

            formal_candidate.scan_context_distance =
                candidate.scan_context_distance;

            formal_candidate.scan_context_similarity =
                candidate.scan_context_similarity;

            formal_candidate.compared_sectors =
                candidate.compared_sectors;

            formal_candidate.has_yaw_hint =
                std::isfinite(candidate.yaw_shift_deg);

            formal_candidate.yaw_hint_deg =
                candidate.yaw_shift_deg;

            formal_candidate.has_base_initial_guess =
                true;

            formal_candidate.base_initial_guess =
                Eigen::Isometry3d::Identity();

            formal_loop_candidates.push_back(
                formal_candidate);
        }
    }

    // --------------------------------------------------------------------
    if (kEnableExperimentalLoopShadows &&
        use_sc_single_retrieval &&
        sc_single_added)
    {
    // SC-SINGLE CSV
    //
    // One row per accepted Loop Retrieval Sample.
    // --------------------------------------------------------------------
    try
    {
        const std::filesystem::path
            sc_single_directory =
                FrontendLoopDirectory();

        std::filesystem::create_directories(
            sc_single_directory);

        const std::filesystem::path
            sc_single_csv_path =
                sc_single_directory /
                "scan_context_single_shadow.csv";

        const bool sc_single_csv_exists =
            std::filesystem::exists(
                sc_single_csv_path);

        std::ofstream sc_single_csv(
            sc_single_csv_path,
            std::ios::app);

        if (sc_single_csv.is_open())
        {
            sc_single_csv
                << std::fixed
                << std::setprecision(9);

            if (!sc_single_csv_exists)
            {
                sc_single_csv
                    << "sample_id,"
                    << "mapping_keyframe_id,"
                    << "timestamp,"
                    << "database_entries,"
                    << "eligible_entries,"
                    << "valid_matches,"
                    << "accepted_candidates,"
                    << "best_raw_kf,"
                    << "best_raw_distance,"
                    << "best_raw_similarity,"
                    << "best_raw_yaw_deg,"
                    << "accepted_kf,"
                    << "accepted_distance,"
                    << "accepted_similarity,"
                    << "accepted_yaw_deg"
                    << '\n';
            }

            sc_single_csv
                << current_loop_sample.sample_id << ','
                << current_keyframe.id << ','
                << current_keyframe.timestamp << ','
                << sc_single_diagnostics.database_entries
                << ','
                << sc_single_diagnostics.separation_eligible
                << ','
                << sc_single_diagnostics.valid_matches
                << ','
                << sc_single_diagnostics.accepted_candidates
                << ',';

            // ----------------------------------------------------
            // Best RAW candidate.
            // Recorded even when distance > 0.40.
            // ----------------------------------------------------
            if (sc_single_diagnostics.has_best_match)
            {
                const ScanContextShadowCandidate &
                    best_raw =
                        sc_single_diagnostics.best_match;

                sc_single_csv
                    << best_raw.candidate_id << ','
                    << best_raw.scan_context_distance << ','
                    << best_raw.scan_context_similarity << ','
                    << best_raw.yaw_shift_deg << ',';
            }
            else
            {
                sc_single_csv
                    << "-1,nan,nan,nan,";
            }

            // ----------------------------------------------------
            // Best threshold-accepted candidate.
            // ----------------------------------------------------
            if (!sc_single_candidates.empty())
            {
                const ScanContextShadowCandidate &
                    best =
                        sc_single_candidates.front();

                sc_single_csv
                    << best.candidate_id << ','
                    << best.scan_context_distance << ','
                    << best.scan_context_similarity << ','
                    << best.yaw_shift_deg;
            }
            else
            {
                sc_single_csv
                    << "-1,nan,nan,nan";
            }

            sc_single_csv << '\n';
        }
    }
    catch (const std::exception &)
    {
        // Shadow diagnostics must never affect SLAM.
    }



    // ====================================================================
    // SC-SINGLE TOP-K LONG-FORM CSV V1
    //
    // One row = one threshold-accepted SC-SINGLE candidate.
    //
    // Used for:
    //
    //     Recall@1
    //     Recall@5
    //     Recall@10
    //
    // IMPORTANT:
    //     Diagnostic only.
    //     Does NOT enter LoopVerifier / PoseGraph / PGO.
    // ====================================================================
    try
    {
        const std::filesystem::path
            sc_single_topk_directory =
                FrontendLoopDirectory();

        std::filesystem::create_directories(
            sc_single_topk_directory);

        const std::filesystem::path
            sc_single_topk_path =
                sc_single_topk_directory /
                "scan_context_single_topk.csv";

        const bool sc_single_topk_exists =
            std::filesystem::exists(
                sc_single_topk_path);

        std::ofstream sc_single_topk_csv(
            sc_single_topk_path,
            std::ios::app);

        if (sc_single_topk_csv.is_open())
        {
            sc_single_topk_csv
                << std::fixed
                << std::setprecision(9);

            if (!sc_single_topk_exists)
            {
                sc_single_topk_csv
                    << "sample_id,"
                    << "mapping_keyframe_id,"
                    << "timestamp,"
                    << "database_entries,"
                    << "eligible_entries,"
                    << "valid_matches,"
                    << "accepted_candidates,"
                    << "rank,"
                    << "historical_keyframe_id,"
                    << "sc_distance,"
                    << "sc_similarity,"
                    << "raw_cosine_similarity,"
                    << "sector_coverage_ratio,"
                    << "cell_coverage_ratio,"
                    << "compared_sectors,"
                    << "yaw_shift_deg,"
                    << "time_separation_sec,"
                    << "pose_distance"
                    << '\n';
            }

            for (std::size_t rank_index = 0;
                 rank_index <
                     sc_single_candidates.size();
                 ++rank_index)
            {
                const ScanContextShadowCandidate &
                    candidate =
                        sc_single_candidates[
                            rank_index];

                sc_single_topk_csv
                    << current_loop_sample.sample_id << ','
                    << current_keyframe.id << ','
                    << current_keyframe.timestamp << ','
                    << sc_single_diagnostics.database_entries << ','
                    << sc_single_diagnostics.separation_eligible << ','
                    << sc_single_diagnostics.valid_matches << ','
                    << sc_single_diagnostics.accepted_candidates << ','
                    << (rank_index + 1) << ','
                    << candidate.candidate_id << ','
                    << candidate.scan_context_distance << ','
                    << candidate.scan_context_similarity << ','
                    << candidate.raw_cosine_similarity << ','
                    << candidate.sector_coverage_ratio << ','
                    << candidate.cell_coverage_ratio << ','
                    << candidate.compared_sectors << ','
                    << candidate.yaw_shift_deg << ','
                    << candidate.time_separation_sec << ','
                    << candidate.pose_distance
                    << '\n';
            }
        }
    }
    catch (const std::exception &)
    {
        // Diagnostic output must NEVER affect SLAM.
    }


    // ====================================================================
    // ====================================================================

    // ====================================================================
    // SC-SINGLE VERIFIER SHADOW V1
    //
    // Retrieval:
    //     SC-SINGLE accepted Top-K candidates.
    //
    // Geometry:
    //     current Keyframe::cloud
    //         ->
    //     candidate-centered historical LocalMap in frame K.
    //
    // Initial hypotheses in candidate frame K:
    //
    //     IDENTITY
    //     SC_YAW_POSITIVE
    //     SC_YAW_NEGATIVE
    //
    // IMPORTANT:
    //     Diagnostic only.
    //
    //     Does NOT:
    //       - modify SC retrieval
    //       - modify BTC retrieval
    //       - enter temporal consistency
    //       - enter cycle consistency
    //       - create PoseGraph edges
    //       - run PGO
    //
    // A completely separate LoopVerifier instance is used so this shadow
    // experiment does not alter the production BTC verifier cache.
    // ====================================================================

    static LoopVerifier
        sc_single_shadow_verifier(
            loop_verifier_.GetConfig());

    try
    {
        const std::filesystem::path
            sc_verify_directory =
                FrontendLoopDirectory();

        std::filesystem::create_directories(
            sc_verify_directory);

        const std::filesystem::path
            sc_verify_csv_path =
                sc_verify_directory /
                "scan_context_single_verifier_shadow.csv";

        const bool sc_verify_csv_exists =
            std::filesystem::exists(
                sc_verify_csv_path);

        std::ofstream sc_verify_csv(
            sc_verify_csv_path,
            std::ios::app);

        if (false && sc_verify_csv.is_open())
        {
            sc_verify_csv
                << std::fixed
                << std::setprecision(9);

            if (!sc_verify_csv_exists)
            {
                sc_verify_csv
                    << "sample_id,"
                    << "current_kf,"
                    << "rank,"
                    << "historical_kf,"
                    << "sc_distance,"
                    << "sc_similarity,"
                    << "sc_yaw_deg,"
                    << "target_built,"
                    << "target_keyframes,"
                    << "target_points,"
                    << "best_prescore_guess,"
                    << "prescore_valid,"
                    << "prescore_overlap,"
                    << "prescore_rmse,"
                    << "selected_verify_guess,"
                    << "verify_ran,"
                    << "verify_ok,"
                    << "success,"
                    << "converged,"
                    << "accepted_geometry,"
                    << "source_points,"
                    << "verifier_target_points,"
                    << "inliers,"
                    << "overlap,"
                    << "rmse,"
                    << "correction_translation,"
                    << "correction_rotation_deg,"
                    << "decision"
                    << '\n';
            }

            for (std::size_t sc_rank = 0;
                 sc_rank < sc_single_candidates.size();
                 ++sc_rank)
            {
                const ScanContextShadowCandidate &
                    sc_candidate =
                        sc_single_candidates[sc_rank];

                pcl::PointCloud<LIDAR_POINT>::Ptr
                    sc_target_K;

                std::vector<std::size_t>
                    sc_target_keyframes;

                const bool sc_target_built =
                    BuildCandidateCenteredHistoricalTarget(
                        sc_candidate.candidate_id,
                        sc_target_K,
                        &sc_target_keyframes);

                const double sc_nan =
                    std::numeric_limits<double>::
                        quiet_NaN();

                const std::size_t sc_target_points =
                    (sc_target_K != nullptr)
                        ? sc_target_K->size()
                        : 0;

                if (!sc_target_built ||
                    !sc_target_K ||
                    sc_target_K->empty())
                {
                    sc_verify_csv
                        << current_loop_sample.sample_id << ','
                        << current_keyframe.id << ','
                        << (sc_rank + 1) << ','
                        << sc_candidate.candidate_id << ','
                        << sc_candidate.scan_context_distance << ','
                        << sc_candidate.scan_context_similarity << ','
                        << sc_candidate.yaw_shift_deg << ','
                        << 0 << ','
                        << sc_target_keyframes.size() << ','
                        << sc_target_points << ','
                        << "NONE,"
                        << 0 << ','
                        << sc_nan << ','
                        << sc_nan << ','
                        << "NONE,"
                        << 0 << ','
                        << 0 << ','
                        << 0 << ','
                        << 0 << ','
                        << 0 << ','
                        << current_keyframe.cloud->size() << ','
                        << 0 << ','
                        << 0 << ','
                        << sc_nan << ','
                        << sc_nan << ','
                        << sc_nan << ','
                        << sc_nan << ','
                        << "TARGET_REJECT"
                        << '\n';

                    continue;
                }

                // ------------------------------------------------------------
                // Candidate-centered target is expressed in historical
                // candidate frame K.
                //
                // Therefore:
                //
                //   historical candidate pose itself = Identity in K.
                //
                // Scan Context provides only a coarse relative yaw.
                // Test both signs because SC shift sign is intentionally not
                // hard-coded as a final SE(3) convention.
                // ------------------------------------------------------------
                struct ScShadowInitialGuess
                {
                    const char *name = "NONE";

                    Eigen::Isometry3d transform =
                        Eigen::Isometry3d::Identity();

                    LoopVerifierInitialGuessScore
                        prescore;

                    bool score_ok = false;
                };

                std::vector<ScShadowInitialGuess>
                    sc_guesses;

                sc_guesses.reserve(3);

                ScShadowInitialGuess
                    identity_guess;

                identity_guess.name =
                    "IDENTITY";

                identity_guess.transform =
                    Eigen::Isometry3d::Identity();

                sc_guesses.push_back(
                    identity_guess);

                if (std::isfinite(
                        sc_candidate.yaw_shift_deg))
                {
                    constexpr double
                        kDegToRad =
                            3.14159265358979323846 /
                            180.0;

                    const double sc_yaw_rad =
                        sc_candidate.yaw_shift_deg *
                        kDegToRad;

                    ScShadowInitialGuess
                        positive_guess;

                    positive_guess.name =
                        "SC_YAW_POSITIVE";

                    positive_guess.transform =
                        Eigen::Isometry3d::Identity();

                    positive_guess.transform.linear() =
                        Eigen::AngleAxisd(
                            sc_yaw_rad,
                            Eigen::Vector3d::UnitZ())
                            .toRotationMatrix();

                    sc_guesses.push_back(
                        positive_guess);

                    ScShadowInitialGuess
                        negative_guess;

                    negative_guess.name =
                        "SC_YAW_NEGATIVE";

                    negative_guess.transform =
                        Eigen::Isometry3d::Identity();

                    negative_guess.transform.linear() =
                        Eigen::AngleAxisd(
                            -sc_yaw_rad,
                            Eigen::Vector3d::UnitZ())
                            .toRotationMatrix();

                    sc_guesses.push_back(
                        negative_guess);
                }

                // ------------------------------------------------------------
                // Cheap prescore all hypotheses.
                // ------------------------------------------------------------
                for (ScShadowInitialGuess &guess :
                     sc_guesses)
                {
                    guess.score_ok =
                        sc_single_shadow_verifier
                            .ScoreInitialGuess(
                                current_keyframe.cloud,
                                sc_target_K,
                                guess.transform,
                                guess.prescore);

                    guess.prescore.valid =
                        guess.score_ok &&
                        guess.prescore.valid;
                }

                std::sort(
                    sc_guesses.begin(),
                    sc_guesses.end(),
                    [](const ScShadowInitialGuess &lhs,
                       const ScShadowInitialGuess &rhs)
                    {
                        if (lhs.prescore.valid !=
                            rhs.prescore.valid)
                        {
                            return lhs.prescore.valid;
                        }

                        if (lhs.prescore.overlap_ratio !=
                            rhs.prescore.overlap_ratio)
                        {
                            return lhs.prescore.overlap_ratio >
                                   rhs.prescore.overlap_ratio;
                        }

                        return lhs.prescore.rmse <
                               rhs.prescore.rmse;
                    });

                const ScShadowInitialGuess &
                    sc_best_prescore =
                        sc_guesses.front();

                bool sc_verify_ran = false;
                bool sc_best_verify_ok = false;
                bool sc_have_verification = false;

                const char *
                    sc_selected_guess =
                        "NONE";

                LoopVerificationResult
                    sc_best_verification;

                const double
                    sc_prescore_gate =
                        sc_single_shadow_verifier
                            .GetConfig()
                            .prescore_min_overlap_ratio;

                // ------------------------------------------------------------
                // Try hypotheses in descending prescore order.
                //
                // Stop once one passes the unchanged LoopVerifier geometry
                // gate. If none passes, retain the best failed trial for
                // diagnostics.
                // ------------------------------------------------------------
                for (const ScShadowInitialGuess &guess :
                     sc_guesses)
                {
                    if (!guess.prescore.valid ||
                        !std::isfinite(
                            guess.prescore.overlap_ratio) ||
                        guess.prescore.overlap_ratio <
                            sc_prescore_gate)
                    {
                        continue;
                    }

                    sc_verify_ran = true;

                    LoopVerificationResult
                        sc_trial;

                    const bool sc_trial_ok =
                        sc_single_shadow_verifier.Verify(
                            current_keyframe.cloud,
                            sc_target_K,
                            guess.transform,
                            sc_trial);

                    bool sc_trial_better =
                        !sc_have_verification;

                    if (!sc_trial_better &&
                        sc_trial.accepted !=
                            sc_best_verification.accepted)
                    {
                        sc_trial_better =
                            sc_trial.accepted;
                    }
                    else if (
                        !sc_trial_better &&
                        sc_trial.accepted ==
                            sc_best_verification.accepted &&
                        sc_trial.overlap_ratio >
                            sc_best_verification.overlap_ratio +
                                1.0e-9)
                    {
                        sc_trial_better = true;
                    }
                    else if (
                        !sc_trial_better &&
                        sc_trial.accepted ==
                            sc_best_verification.accepted &&
                        std::abs(
                            sc_trial.overlap_ratio -
                            sc_best_verification.overlap_ratio) <=
                            1.0e-9 &&
                        sc_trial.rmse <
                            sc_best_verification.rmse)
                    {
                        sc_trial_better = true;
                    }

                    if (sc_trial_better)
                    {
                        sc_have_verification = true;

                        sc_best_verify_ok =
                            sc_trial_ok;

                        sc_selected_guess =
                            guess.name;

                        sc_best_verification =
                            sc_trial;
                    }

                    if (sc_trial_ok &&
                        sc_trial.accepted)
                    {
                        break;
                    }
                }

                const char *sc_decision =
                    "VERIFY_REJECT";

                if (!sc_best_prescore.prescore.valid)
                {
                    sc_decision =
                        "NO_VALID_PRESCORE";
                }
                else if (
                    sc_best_prescore
                            .prescore.overlap_ratio <
                        sc_prescore_gate)
                {
                    sc_decision =
                        "PRESCORE_REJECT";
                }
                else if (!sc_verify_ran)
                {
                    sc_decision =
                        "VERIFY_NOT_RUN";
                }
                else if (
                    sc_have_verification &&
                    sc_best_verify_ok &&
                    sc_best_verification.accepted)
                {
                    sc_decision =
                        "GEOMETRY_ACCEPT";
                }

                sc_verify_csv
                    << current_loop_sample.sample_id << ','
                    << current_keyframe.id << ','
                    << (sc_rank + 1) << ','
                    << sc_candidate.candidate_id << ','
                    << sc_candidate.scan_context_distance << ','
                    << sc_candidate.scan_context_similarity << ','
                    << sc_candidate.yaw_shift_deg << ','
                    << 1 << ','
                    << sc_target_keyframes.size() << ','
                    << sc_target_points << ','
                    << sc_best_prescore.name << ','
                    << (sc_best_prescore.prescore.valid ? 1 : 0)
                    << ','
                    << sc_best_prescore
                           .prescore.overlap_ratio
                    << ','
                    << sc_best_prescore
                           .prescore.rmse
                    << ','
                    << sc_selected_guess << ','
                    << (sc_verify_ran ? 1 : 0) << ','
                    << (sc_best_verify_ok ? 1 : 0) << ',';

                if (sc_have_verification)
                {
                    sc_verify_csv
                        << (sc_best_verification.success ? 1 : 0)
                        << ','
                        << (sc_best_verification.converged ? 1 : 0)
                        << ','
                        << (sc_best_verification.accepted ? 1 : 0)
                        << ','
                        << sc_best_verification.source_points
                        << ','
                        << sc_best_verification.target_points
                        << ','
                        << sc_best_verification.inliers
                        << ','
                        << sc_best_verification.overlap_ratio
                        << ','
                        << sc_best_verification.rmse
                        << ','
                        << sc_best_verification
                               .correction_translation
                        << ','
                        << sc_best_verification
                               .correction_rotation_deg
                        << ',';
                }
                else
                {
                    sc_verify_csv
                        << 0 << ','
                        << 0 << ','
                        << 0 << ','
                        << current_keyframe.cloud->size()
                        << ','
                        << sc_target_points << ','
                        << 0 << ','
                        << sc_nan << ','
                        << sc_nan << ','
                        << sc_nan << ','
                        << sc_nan << ',';
                }

                sc_verify_csv
                    << sc_decision
                    << '\n';
            }
        }
    }
    catch (const std::exception &)
    {
        // Shadow diagnostics must never affect production loop closure.
    }

    }

    // BTC-SINGLE SHADOW V4
    //
    // One accepted Loop Retrieval Sample -> one Keyframe::btc_cloud.
    //
    // window_size   = 1
    // window_stride = 1
    //
    // DIAGNOSTIC ONLY.
    // Does NOT enter verifier / consistency / PoseGraph / PGO.
    // ====================================================================
    static fr_slam_btc::OfficialBtcAdapter
        btc_single_shadow_adapter(
            loop_runtime_config_.btc_config_profile,
            loop_runtime_config_.btc_skip_near_num,
            loop_runtime_config_.btc_proj_plane_num,
            1,
            1);

    bool btc_single_cloud_valid = false;

    std::size_t btc_single_source_points = 0;

    fr_slam_btc::OfficialBtcSubmapResult
        btc_single_result;

    if (kEnableExperimentalLoopShadows &&
        current_keyframe.btc_cloud &&
        !current_keyframe.btc_cloud->empty() &&
        current_keyframe.T_WL.matrix().allFinite())
    {
        btc_single_cloud_valid = true;

        btc_single_source_points =
            current_keyframe.btc_cloud->size();

        btc_single_result =
            btc_single_shadow_adapter.ProcessSubmap(
                current_keyframe.id,
                current_keyframe.btc_cloud);
    }


    // ====================================================================
    // BTC-SINGLE DIAGNOSTICS V4
    //
    // Important diagnostic chain:
    //
    // btc_points
    //      ->
    // btc_descriptor_count
    //      ->
    // queried
    //      ->
    // score / matched_triangle_pairs
    //      ->
    // has_candidate
    // ====================================================================
    try
    {
        const std::filesystem::path
            btc_single_directory =
                FrontendLoopDirectory();

        std::filesystem::create_directories(
            btc_single_directory);

        const std::filesystem::path
            btc_single_csv_path =
                btc_single_directory /
                "btc_single_shadow.csv";

        const bool btc_single_csv_exists =
            std::filesystem::exists(
                btc_single_csv_path);

        std::ofstream btc_single_csv(
            btc_single_csv_path,
            std::ios::app);

        if (btc_single_csv.is_open())
        {
            btc_single_csv
                << std::fixed
                << std::setprecision(9);

            if (!btc_single_csv_exists)
            {
                btc_single_csv
                    << "sample_id,"
                    << "mapping_keyframe_id,"
                    << "timestamp,"
                    << "btc_points,"
                    << "cloud_valid,"
                    << "newly_processed,"
                    << "queried,"
                    << "btc_descriptor_count,"
                    << "score,"
                    << "matched_triangle_pairs,"
                    << "has_candidate,"
                    << "historical_keyframe_id"
                    << '\n';
            }

            btc_single_csv
                << current_loop_sample.sample_id << ','
                << current_keyframe.id << ','
                << current_keyframe.timestamp << ','
                << btc_single_source_points << ','
                << (btc_single_cloud_valid ? 1 : 0) << ','
                << (btc_single_result.newly_processed ? 1 : 0) << ','
                << (btc_single_result.queried ? 1 : 0) << ','
                << btc_single_result.btc_descriptor_count << ','
                << btc_single_result.score << ','
                << btc_single_result.matched_triangle_pairs << ','
                << (btc_single_result.has_candidate ? 1 : 0) << ',';

            if (btc_single_result.has_candidate)
            {
                btc_single_csv
                    << btc_single_result.historical_submap_id;
            }
            else
            {
                btc_single_csv
                    << -1;
            }

            btc_single_csv
                << '\n';
        }
    }
    catch (const std::exception &)
    {
        // Diagnostics must never affect SLAM.
    }

    const std::chrono::steady_clock::time_point btc_start =
        std::chrono::steady_clock::now();

    // ====================================================================
    // Build the query from the trailing Loop Retrieval Samples.
    //
    // sample_id remains contiguous.
    // mapping_keyframe_id does NOT need to be contiguous.
    // ====================================================================
    const std::size_t query_last_sample_index =
        loop_retrieval_samples.size() - 1;

    const std::size_t query_requested_keyframes =
        std::min(
            btc_window_size,
            loop_retrieval_samples.size());

    const std::size_t query_first_sample_index =
        loop_retrieval_samples.size() -
        query_requested_keyframes;

    const std::size_t query_anchor_sample_index =
        query_first_sample_index +
        query_requested_keyframes / 2;

    const std::size_t query_first_kf =
        loop_retrieval_samples[
            query_first_sample_index]
            .mapping_keyframe_id;

    const std::size_t query_last_kf =
        loop_retrieval_samples[
            query_last_sample_index]
            .mapping_keyframe_id;

    const std::size_t query_anchor_kf =
        loop_retrieval_samples[
            query_anchor_sample_index]
            .mapping_keyframe_id;

    const bool query_is_full_window =
        query_requested_keyframes == btc_window_size;

    const bool query_will_enter_database =
        query_is_full_window &&
        query_last_sample_index >= btc_window_size - 1 &&
        ((query_last_sample_index -
              (btc_window_size - 1)) %
             btc_window_stride ==
         0);

    const bool window_query_needed =
        use_sc_window_retrieval ||
        (use_btc_retrieval &&
         !official_btc_adapter.IsProcessedSubmap(
             query_last_kf));

    if (query_requested_keyframes >=
            btc_min_valid_dense_keyframes &&
        window_query_needed)
    {
        const Keyframe *query_anchor =
            FindBackendKeyframeById(query_anchor_kf);

        std::size_t query_valid_dense_keyframes = 0;
        std::size_t query_missing_dense_keyframes = 0;
        std::size_t query_source_points = 0;

        if (query_anchor != nullptr &&
            query_anchor->T_WL.matrix().allFinite())
        {
            for (std::size_t query_sample_index =
                     query_first_sample_index;
                 query_sample_index <=
                     query_last_sample_index;
                 ++query_sample_index)
            {
                const std::size_t keyframe_id =
                    loop_retrieval_samples[
                        query_sample_index]
                        .mapping_keyframe_id;
                const Keyframe *keyframe =
                    FindBackendKeyframeById(keyframe_id);

                if (keyframe == nullptr ||
                    !keyframe->T_WL.matrix().allFinite() ||
                    !keyframe->btc_cloud ||
                    keyframe->btc_cloud->empty())
                {
                    ++query_missing_dense_keyframes;
                    continue;
                }

                ++query_valid_dense_keyframes;
                query_source_points +=
                    keyframe->btc_cloud->size();
            }
        }
        else
        {
            query_missing_dense_keyframes =
                query_requested_keyframes;
        }

        if (!use_window_cloud_retrieval ||
            query_anchor == nullptr ||
            !query_anchor->T_WL.matrix().allFinite() ||
            query_valid_dense_keyframes < btc_min_valid_dense_keyframes)
        {
        }
        else
        {
            pcl::PointCloud<LIDAR_POINT>::Ptr btc_cloud_C =
                pcl::make_shared<pcl::PointCloud<LIDAR_POINT>>();

            btc_cloud_C->reserve(query_source_points);

            const Eigen::Isometry3d T_CW =
                query_anchor->T_WL.inverse();

            std::size_t btc_accumulated_keyframes = 0;

            for (std::size_t query_sample_index =
                     query_first_sample_index;
                 query_sample_index <=
                     query_last_sample_index;
                 ++query_sample_index)
            {
                const std::size_t keyframe_id =
                    loop_retrieval_samples[
                        query_sample_index]
                        .mapping_keyframe_id;
                const Keyframe *keyframe =
                    FindBackendKeyframeById(keyframe_id);

                if (keyframe == nullptr ||
                    !keyframe->T_WL.matrix().allFinite() ||
                    !keyframe->btc_cloud ||
                    keyframe->btc_cloud->empty())
                {
                    continue;
                }

                const Eigen::Isometry3d T_C_L =
                    T_CW * keyframe->T_WL;

                if (!T_C_L.matrix().allFinite())
                {
                    continue;
                }

                ++btc_accumulated_keyframes;

                for (const LIDAR_POINT &point_L :
                     keyframe->btc_cloud->points)
                {
                    if (!std::isfinite(point_L.x) ||
                        !std::isfinite(point_L.y) ||
                        !std::isfinite(point_L.z))
                    {
                        continue;
                    }

                    const Eigen::Vector3d p_L(
                        static_cast<double>(point_L.x),
                        static_cast<double>(point_L.y),
                        static_cast<double>(point_L.z));

                    const Eigen::Vector3d p_C =
                        T_C_L * p_L;

                    if (!p_C.allFinite())
                    {
                        continue;
                    }

                    LIDAR_POINT point_C = point_L;
                    point_C.x = static_cast<float>(p_C.x());
                    point_C.y = static_cast<float>(p_C.y());
                    point_C.z = static_cast<float>(p_C.z());
                    btc_cloud_C->push_back(point_C);
                }
            }

            btc_cloud_C->width =
                static_cast<std::uint32_t>(btc_cloud_C->size());
            btc_cloud_C->height = 1;
            btc_cloud_C->is_dense = true;


            if (!btc_cloud_C->empty())
            {

                // ========================================================
                if (use_sc_window_retrieval)
                {
                // SC-WINDOW SHADOW RETRIEVAL
                //
                // btc_cloud_C is EXACTLY the same accumulated query cloud
                // that is passed into BTC below.
                //
                // IMPORTANT ORDER:
                //
                //     1. Query historical SC database.
                //     2. Record diagnostics.
                //     3. If this is a BTC database-stride window,
                //        insert it into the SC database.
                //
                // Query-before-insert prevents self matching.
                // ========================================================
                ScanContextWindowDiagnostics
                    sc_window_diagnostics;

                const std::vector<ScanContextWindowCandidate>
                    sc_window_candidates =
                        sc_window_shadow->QueryWindow(
                            query_first_kf,
                            query_last_kf,
                            query_anchor_kf,
                            query_anchor->timestamp,
                            query_anchor->T_WL,
                            btc_cloud_C,
                            &sc_window_diagnostics);


                // ========================================================
                // ========================================================
                // SC-WINDOW FORMAL V1 ADAPTER
                //
                // Window retrieval:
                //     C = current query-window anchor
                //     H = historical window anchor
                //
                // Formal backend:
                //     L = current trigger Mapping KF
                //     K = historical anchor Mapping KF (= H)
                //
                // T_C_L bridges current trigger KF into query-anchor frame.
                // SC yaw later supplies the coarse H <- C rotation.
                // ========================================================
                const Eigen::Isometry3d
                    sc_window_T_C_L =
                        query_anchor->T_WL.inverse() *
                        current_keyframe.T_WL;

                if (sc_window_T_C_L.matrix().allFinite())
                {
                    sc_window_formal_T_C_L =
                        sc_window_T_C_L;

                    sc_window_formal_bridge_valid =
                        true;

                    sc_window_formal_candidates.clear();

                    sc_window_formal_candidates.reserve(
                        sc_window_candidates.size());

                    for (const ScanContextWindowCandidate &
                             window_candidate :
                         sc_window_candidates)
                    {
                        const Keyframe *historical_anchor =
                            FindBackendKeyframeById(
                                window_candidate
                                    .historical_anchor_kf);

                        if (historical_anchor == nullptr ||
                            !historical_anchor->T_WL
                                 .matrix()
                                 .allFinite())
                        {
                            continue;
                        }

                        ScanContextShadowCandidate
                            formal_candidate;

                        formal_candidate.current_id =
                            current_keyframe.id;

                        formal_candidate.candidate_id =
                            window_candidate
                                .historical_anchor_kf;

                        // Diagnostic only; never used as an online gate.
                        formal_candidate.pose_distance =
                            (current_keyframe.T_WL.translation() -
                             historical_anchor->T_WL.translation())
                                .norm();

                        formal_candidate.time_separation_sec =
                            current_keyframe.timestamp -
                            historical_anchor->timestamp;

                        formal_candidate.scan_context_distance =
                            window_candidate
                                .scan_context_distance;

                        formal_candidate.scan_context_similarity =
                            window_candidate
                                .scan_context_similarity;

                        formal_candidate.raw_cosine_similarity =
                            window_candidate
                                .raw_cosine_similarity;

                        formal_candidate.sector_coverage_ratio =
                            window_candidate
                                .sector_coverage_ratio;

                        formal_candidate.cell_coverage_ratio =
                            window_candidate
                                .cell_coverage_ratio;

                        formal_candidate.compared_sectors =
                            window_candidate
                                .compared_sectors;

                        formal_candidate.sector_shift =
                            window_candidate
                                .sector_shift;

                        formal_candidate.yaw_shift_deg =
                            window_candidate
                                .yaw_shift_deg;

                        sc_window_formal_candidates.push_back(
                            formal_candidate);
                    }
                }

                if (sc_window_formal_bridge_valid)
                {
                    formal_loop_candidates.clear();
                    formal_loop_candidates.reserve(
                        sc_window_formal_candidates.size());

                    for (std::size_t rank_index = 0;
                         rank_index < sc_window_formal_candidates.size();
                         ++rank_index)
                    {
                        const ScanContextShadowCandidate &candidate =
                            sc_window_formal_candidates[rank_index];

                        FormalLoopCandidate formal_candidate;

                        formal_candidate.source =
                            FormalLoopRetrievalSource::ScanContextWindow;

                        formal_candidate.current_id =
                            candidate.current_id;

                        formal_candidate.candidate_id =
                            candidate.candidate_id;

                        formal_candidate.retrieval_rank =
                            rank_index;

                        formal_candidate.pose_distance =
                            candidate.pose_distance;

                        formal_candidate.time_separation_sec =
                            candidate.time_separation_sec;

                        formal_candidate.scan_context_distance =
                            candidate.scan_context_distance;

                        formal_candidate.scan_context_similarity =
                            candidate.scan_context_similarity;

                        formal_candidate.compared_sectors =
                            candidate.compared_sectors;

                        formal_candidate.has_yaw_hint =
                            std::isfinite(candidate.yaw_shift_deg);

                        formal_candidate.yaw_hint_deg =
                            candidate.yaw_shift_deg;

                        formal_candidate.has_base_initial_guess =
                            true;

                        formal_candidate.base_initial_guess =
                            sc_window_formal_T_C_L;

                        formal_loop_candidates.push_back(
                            formal_candidate);
                    }
                }

                // SC-WINDOW CSV
                // ========================================================
                try
                {
                    const std::filesystem::path
                        sc_output_directory =
                            FrontendLoopDirectory();

                    std::filesystem::create_directories(
                        sc_output_directory);

                    const std::filesystem::path
                        sc_csv_path =
                            sc_output_directory /
                            "scan_context_window_shadow.csv";

                    const bool sc_csv_exists =
                        std::filesystem::exists(
                            sc_csv_path);

                    std::ofstream sc_csv(
                        sc_csv_path,
                        std::ios::app);

                    if (sc_csv.is_open())
                    {
                        if (!sc_csv_exists)
                        {
                            sc_csv
                                << "query_first_kf,"
                                << "query_last_kf,"
                                << "query_anchor_kf,"
                                << "query_points,"
                                << "database_entries,"
                                << "eligible_entries,"
                                << "valid_matches,"
                                << "accepted_candidates,"
                                << "best_raw_hist_first_kf,"
                                << "best_raw_hist_last_kf,"
                                << "best_raw_hist_anchor_kf,"
                                << "best_raw_distance,"
                                << "best_raw_similarity,"
                                << "best_raw_yaw_deg,"
                                << "accepted_hist_first_kf,"
                                << "accepted_hist_last_kf,"
                                << "accepted_hist_anchor_kf,"
                                << "accepted_distance,"
                                << "accepted_similarity,"
                                << "accepted_yaw_deg"
                                << '\n';
                        }

                        sc_csv
                            << query_first_kf << ','
                            << query_last_kf << ','
                            << query_anchor_kf << ','
                            << btc_cloud_C->size() << ','
                            << sc_window_diagnostics.database_entries << ','
                            << sc_window_diagnostics.separation_eligible << ','
                            << sc_window_diagnostics.valid_matches << ','
                            << sc_window_diagnostics.accepted_candidates << ',';


                        // ----------------------------------------------
                        // Best RAW SC match.
                        //
                        // This is recorded even when SC distance > 0.40.
                        // ----------------------------------------------
                        if (sc_window_diagnostics.has_best_match)
                        {
                            const ScanContextWindowCandidate &
                                best_raw =
                                    sc_window_diagnostics.best_match;

                            sc_csv
                                << best_raw.historical_first_kf << ','
                                << best_raw.historical_last_kf << ','
                                << best_raw.historical_anchor_kf << ','
                                << best_raw.scan_context_distance << ','
                                << best_raw.scan_context_similarity << ','
                                << best_raw.yaw_shift_deg << ',';
                        }
                        else
                        {
                            sc_csv
                                << ",,,,,,";
                        }


                        // ----------------------------------------------
                        // Best ACCEPTED SC match.
                        //
                        // Already passed:
                        //
                        //     SC distance <= 0.40
                        // ----------------------------------------------
                        if (!sc_window_candidates.empty())
                        {
                            const ScanContextWindowCandidate &
                                best =
                                    sc_window_candidates.front();

                            sc_csv
                                << best.historical_first_kf << ','
                                << best.historical_last_kf << ','
                                << best.historical_anchor_kf << ','
                                << best.scan_context_distance << ','
                                << best.scan_context_similarity << ','
                                << best.yaw_shift_deg;
                        }
                        else
                        {
                            sc_csv
                                << ",,,,,";
                        }

                        sc_csv << '\n';
                    }
                }
                catch (const std::exception &)
                {
                    // Shadow diagnostics MUST NEVER affect SLAM.
                }


                // ========================================================

                // ========================================================
                // SC-WINDOW TOP-K LONG-FORM CSV
                //
                // One row = one accepted Scan Context candidate.
                //
                // This file is designed for:
                //
                //     Recall@1
                //     Recall@5
                //     Recall@10
                //
                // and retrieval-ranking diagnostics.
                //
                // IMPORTANT:
                //     This remains SHADOW ONLY.
                //     It never creates loop edges.
                // ========================================================
                try
                {
                    const std::filesystem::path
                        sc_topk_output_directory =
                            FrontendLoopDirectory();

                    std::filesystem::create_directories(
                        sc_topk_output_directory);

                    const std::filesystem::path
                        sc_topk_csv_path =
                            sc_topk_output_directory /
                            "scan_context_window_topk.csv";

                    const bool sc_topk_csv_exists =
                        std::filesystem::exists(
                            sc_topk_csv_path);

                    std::ofstream sc_topk_csv(
                        sc_topk_csv_path,
                        std::ios::app);

                    if (sc_topk_csv.is_open())
                    {
                        if (!sc_topk_csv_exists)
                        {
                            sc_topk_csv
                                << "query_first_kf,"
                                << "query_last_kf,"
                                << "query_anchor_kf,"
                                << "query_points,"
                                << "database_entries,"
                                << "rank,"
                                << "historical_first_kf,"
                                << "historical_last_kf,"
                                << "historical_anchor_kf,"
                                << "sc_distance,"
                                << "sc_similarity,"
                                << "raw_cosine_similarity,"
                                << "sector_coverage_ratio,"
                                << "cell_coverage_ratio,"
                                << "compared_sectors,"
                                << "yaw_shift_deg,"
                                << "time_separation_sec,"
                                << "anchor_pose_distance"
                                << '\n';
                        }

                        for (std::size_t rank_index = 0;
                             rank_index <
                                 sc_window_candidates.size();
                             ++rank_index)
                        {
                            const ScanContextWindowCandidate &
                                candidate =
                                    sc_window_candidates[
                                        rank_index];

                            sc_topk_csv
                                << query_first_kf << ','
                                << query_last_kf << ','
                                << query_anchor_kf << ','
                                << btc_cloud_C->size() << ','
                                << sc_window_diagnostics.database_entries << ','
                                << (rank_index + 1) << ','
                                << candidate.historical_first_kf << ','
                                << candidate.historical_last_kf << ','
                                << candidate.historical_anchor_kf << ','
                                << candidate.scan_context_distance << ','
                                << candidate.scan_context_similarity << ','
                                << candidate.raw_cosine_similarity << ','
                                << candidate.sector_coverage_ratio << ','
                                << candidate.cell_coverage_ratio << ','
                                << candidate.compared_sectors << ','
                                << candidate.yaw_shift_deg << ','
                                << candidate.time_separation_sec << ','
                                << candidate.anchor_pose_distance
                                << '\n';
                        }
                    }
                }
                catch (const std::exception &)
                {
                    // Shadow diagnostics MUST NEVER affect SLAM.
                }

                // BTC-STYLE SC DATABASE POLICY
                //
                // For window_size=7, stride=4:
                //
                //     [0..6]
                //     [4..10]
                //     [8..14]
                //     ...
                //
                // Query has already happened above.
                // We can now safely insert the current database window.
                // ========================================================
                if (query_will_enter_database &&
                    !sc_window_shadow->HasDatabaseWindow(
                        query_last_kf))
                {
                    sc_window_shadow->AddDatabaseWindow(
                        query_first_kf,
                        query_last_kf,
                        query_anchor_kf,
                        query_anchor->timestamp,
                        query_anchor->T_WL,
                        btc_cloud_C);
                }


                // ========================================================
                }

                // ORIGINAL BTC PIPELINE CONTINUES BELOW.
                //
                // Nothing in BTC / geometry / consistency / PGO is changed.
                // ========================================================

                if (use_btc_retrieval)
                {
                const fr_slam_btc::OfficialBtcSubmapResult btc_result =
                    official_btc_adapter.ProcessSubmap(
                        query_last_kf,
                        btc_cloud_C);

                if (btc_result.newly_processed)
                {
                    std::size_t historical_last_kf = 0;
                    std::size_t historical_first_kf = 0;
                    std::size_t historical_anchor_kf = 0;
                    std::size_t historical_keyframe_count = 0;

                    const Keyframe *historical_keyframe = nullptr;

                    if (btc_result.has_candidate)
                    {
                        historical_last_kf =
                            btc_result.historical_submap_id;

                        bool historical_sample_found =
                            false;

                        std::size_t
                            historical_last_sample_index = 0;

                        for (std::size_t sample_index = 0;
                             sample_index <
                                 loop_retrieval_samples.size();
                             ++sample_index)
                        {
                            if (loop_retrieval_samples[
                                    sample_index]
                                    .mapping_keyframe_id ==
                                historical_last_kf)
                            {
                                historical_last_sample_index =
                                    sample_index;

                                historical_sample_found =
                                    true;

                                break;
                            }
                        }

                        if (historical_sample_found &&
                            historical_last_sample_index + 1 >=
                                btc_window_size)
                        {
                            const std::size_t
                                historical_first_sample_index =
                                    historical_last_sample_index +
                                    1 -
                                    btc_window_size;

                            const std::size_t
                                historical_anchor_sample_index =
                                    historical_first_sample_index +
                                    btc_window_size / 2;

                            historical_first_kf =
                                loop_retrieval_samples[
                                    historical_first_sample_index]
                                    .mapping_keyframe_id;

                            historical_anchor_kf =
                                loop_retrieval_samples[
                                    historical_anchor_sample_index]
                                    .mapping_keyframe_id;

                            historical_keyframe_count =
                                btc_window_size;

                            historical_keyframe =
                                FindBackendKeyframeById(
                                    historical_anchor_kf);
                        }
                    }

                    bool forwarded = false;

                    if (btc_result.has_candidate &&
                        historical_keyframe != nullptr &&
                        historical_keyframe->T_WL.matrix().allFinite())
                    {
                        Eigen::Isometry3d T_H_C =
                            Eigen::Isometry3d::Identity();
                        T_H_C.linear() = btc_result.R_H_C;
                        T_H_C.translation() = btc_result.t_H_C;

                        // H is the historical BTC-window anchor Keyframe frame.
                        // C is the current BTC-window anchor Keyframe frame.
                        // L is the current trigger Keyframe frame.
                        const Eigen::Isometry3d T_C_L =
                            query_anchor->T_WL.inverse() *
                            current_keyframe.T_WL;

                        const Eigen::Isometry3d T_K_L =
                            T_H_C * T_C_L;

                        if (T_K_L.matrix().allFinite())
                        {
                            LoopCandidate candidate;
                            candidate.current_id = current_keyframe.id;
                            candidate.candidate_id = historical_anchor_kf;
                            candidate.distance =
                                (current_keyframe.T_WL.translation() -
                                 historical_keyframe->T_WL.translation())
                                    .norm();
                            candidate.time_separation_sec =
                                current_keyframe.timestamp -
                                historical_keyframe->timestamp;
                            candidate.retrieval_score = btc_result.score;
                            candidate.matched_descriptor_pairs =
                                btc_result.matched_triangle_pairs;

                            const auto duplicate =
                                std::find_if(
                                    btc_candidates.begin(),
                                    btc_candidates.end(),
                                    [historical_anchor_kf](
                                        const LoopCandidate &existing)
                                    {
                                        return existing.candidate_id ==
                                               historical_anchor_kf;
                                    });

                            if (duplicate == btc_candidates.end())
                            {
                                btc_candidates.push_back(candidate);
                            }

                            btc_initial_guess_by_kf[historical_anchor_kf] =
                                T_K_L;
                            forwarded = true;


                        }
                    }

                    try
                    {
                        const std::filesystem::path directory =
                            FrontendLoopDirectory();
                        std::filesystem::create_directories(directory);

                        const std::filesystem::path csv_path =
                            directory / "btc_submap_only_candidates.csv";

                        std::ios_base::openmode mode = std::ios::out;
                        mode |= btc_csv_initialized
                                    ? std::ios::app
                                    : std::ios::trunc;

                        std::ofstream file(csv_path, mode);

                        if (file.is_open())
                        {
                            file << std::fixed << std::setprecision(9);

                            if (!btc_csv_initialized)
                            {
                                file
                                    << "trigger_current_kf,query_endpoint_kf,"
                                    << "query_first_kf,query_last_kf,query_anchor_kf,"
                                    << "query_requested_kfs,query_valid_dense_kfs,"
                                    << "query_missing_dense_kfs,database_insert,"
                                    << "internal_btc_frame,btc_accumulated_keyframes,"
                                    << "btc_source_points,input_points,btc_descriptor_count,"
                                    << "queried,has_candidate,historical_endpoint_kf,"
                                    << "historical_first_kf,historical_last_kf,"
                                    << "historical_anchor_kf,historical_keyframes,"
                                    << "btc_score,matched_triangle_pairs,"
                                    << "t_H_C_x,t_H_C_y,t_H_C_z,"
                                    << "roll_H_C_deg,pitch_H_C_deg,yaw_H_C_deg\n";
                            }

                            file
                                << current_keyframe.id << ","
                                << query_last_kf << ","
                                << query_first_kf << ","
                                << query_last_kf << ","
                                << query_anchor_kf << ","
                                << query_requested_keyframes << ","
                                << query_valid_dense_keyframes << ","
                                << query_missing_dense_keyframes << ","
                                << (query_will_enter_database ? 1 : 0) << ","
                                << btc_result.internal_frame_id << ","
                                << btc_accumulated_keyframes << ","
                                << query_source_points << ","
                                << btc_result.input_points << ","
                                << btc_result.btc_descriptor_count << ","
                                << (btc_result.queried ? 1 : 0) << ","
                                << (btc_result.has_candidate ? 1 : 0) << ",";

                            if (btc_result.has_candidate &&
                                historical_keyframe != nullptr)
                            {
                                file
                                    << historical_last_kf << ","
                                    << historical_first_kf << ","
                                    << historical_last_kf << ","
                                    << historical_anchor_kf << ","
                                    << historical_keyframe_count << ","
                                    << btc_result.score << ","
                                    << btc_result.matched_triangle_pairs << ","
                                    << btc_result.t_H_C.x() << ","
                                    << btc_result.t_H_C.y() << ","
                                    << btc_result.t_H_C.z() << ","
                                    << btc_result.roll_deg << ","
                                    << btc_result.pitch_deg << ","
                                    << btc_result.yaw_deg << "\n";
                            }
                            else
                            {
                                const double nan =
                                    std::numeric_limits<double>::quiet_NaN();
                                file
                                    << "-1,-1,-1,-1,0,"
                                    << btc_result.score << ","
                                    << btc_result.matched_triangle_pairs << ","
                                    << nan << "," << nan << "," << nan << ","
                                    << nan << "," << nan << "," << nan << "\n";
                            }

                            file.flush();
                            btc_csv_initialized = true;
                        }
                    }
                    catch (const std::exception &exception)
                    {
                        RCLCPP_WARN(
                            rclcpp::get_logger("scan2local_map.btc"),
                            "BTC CSV write failed: %s",
                            exception.what());
                    }

                }
            }
        }
    }

                }

    if (use_btc_retrieval)
    {
        formal_loop_candidates.clear();
        formal_loop_candidates.reserve(
            btc_candidates.size());

        for (std::size_t rank_index = 0;
             rank_index < btc_candidates.size();
             ++rank_index)
        {
            const LoopCandidate &candidate =
                btc_candidates[rank_index];

            const auto guess_it =
                btc_initial_guess_by_kf.find(
                    candidate.candidate_id);

            if (guess_it ==
                    btc_initial_guess_by_kf.end() ||
                !guess_it->second.matrix().allFinite())
            {
                continue;
            }

            FormalLoopCandidate formal_candidate;

            formal_candidate.source =
                FormalLoopRetrievalSource::Btc;

            formal_candidate.current_id =
                candidate.current_id;

            formal_candidate.candidate_id =
                candidate.candidate_id;

            formal_candidate.retrieval_rank =
                rank_index;

            formal_candidate.pose_distance =
                candidate.distance;

            formal_candidate.time_separation_sec =
                candidate.time_separation_sec;

            formal_candidate.retrieval_score =
                candidate.retrieval_score;

            formal_candidate.matched_descriptor_pairs =
                candidate.matched_descriptor_pairs;

            formal_candidate.has_yaw_hint = false;

            formal_candidate.has_base_initial_guess = true;

            formal_candidate.base_initial_guess =
                guess_it->second;

            formal_loop_candidates.push_back(
                formal_candidate);
        }
    }

    // BTC remains diagnostic/shadow only from this point onward.
    loop_timing.btc_retrieval_ms +=
        ElapsedMilliseconds(
            btc_start,
            std::chrono::steady_clock::now());

    // ====================================================================
    // ====================================================================
    // ================================================================
    // SC-WINDOW FORMAL V1
    //
    // Experimental A/B switch:
    //
    //     true  -> SC-WINDOW is the FORMAL retrieval source
    //     false -> existing SC-SINGLE formal path
    //
    // All downstream geometry / consistency / graph / PGO logic
    // remains unchanged.
    // ================================================================

    // SC-SINGLE FORMAL V2
    //
    // Top-K Explore + Single Stateful Commit
    //
    // Formal retrieval:
    //
    //     accepted SC-SINGLE Top-K
    //              ->
    //     each candidate:
    //         candidate-centered historical target
    //              ->
    //         IDENTITY / +SC_YAW / -SC_YAW
    //              ->
    //         prescore
    //              ->
    //         LoopVerifier
    //              ->
    //     choose ONE best geometry-valid candidate
    //              ->
    //     Temporal + Cycle consistency ONCE
    //              ->
    //     spacing / graph-relative safety
    //              ->
    //     Dynamic FULL 6x6
    //              ->
    //     PoseGraph + G2O
    //
    // IMPORTANT:
    //   1. No candidate calls LoopConsistencyChecker::Check() here.
    //   2. pose_distance remains diagnostic only.
    //   3. Candidate promotion uses geometry:
    //          overlap higher is better
    //          RMSE lower is better
    //          SC rank breaks exact ties naturally.
    //   4. At most ONE candidate reaches stateful consistency.
    // ====================================================================
    if (formal_loop_candidates.empty())
    {
        return;
    }

    loop_timing.candidates =
        formal_loop_candidates.size();

    // ================================================================
    // SC-SINGLE FORMAL V2 performance diagnostics.
    //
    // Diagnostic only. These counters/timers MUST NOT change any
    // retrieval, geometry, consistency or PoseGraph decision.
    // ================================================================
    const std::chrono::steady_clock::time_point
        sc_v2_total_start =
            std::chrono::steady_clock::now();

    double sc_v2_target_ms =
        0.0;

    double sc_v2_prescore_ms =
        0.0;

    double sc_v2_verify_ms =
        0.0;

    std::size_t sc_v2_target_attempts =
        0;

    std::size_t sc_v2_targets_built =
        0;

    std::size_t sc_v2_prescore_calls =
        0;

    std::size_t sc_v2_verify_calls =
        0;

    bool sc_have_selected_candidate =
        false;

    std::size_t sc_selected_rank =
        0;

    FormalLoopCandidate
        sc_selected_candidate;

    LoopCandidate
        sc_formal_candidate;

    pcl::PointCloud<LIDAR_POINT>::Ptr
        best_target_K;

    std::vector<std::size_t>
        best_target_keyframes;

    LoopVerificationResult
        best_verification;

    LoopVerifierInitialGuessScore
        sc_best_prescore;

    bool sc_best_verify_ok =
        false;

    const char *
        sc_selected_guess =
            "NONE";

    // ================================================================
    // SC PRESCORE AUDIT V2.5
    //
    // Diagnostic only.
    //
    // For every SC candidate, store its BEST cheap prescore BEFORE
    // looking at the Full LoopVerifier result.
    //
    // IMPORTANT:
    //   * Does NOT prune candidates.
    //   * Does NOT alter Full Verify order.
    //   * Does NOT alter geometry winner selection.
    //   * Does NOT alter consistency / graph / PGO.
    //
    // Purpose:
    //   Measure whether the final Full-Verify winner would have been
    //   retained by future Prescore Top-M pruning.
    // ================================================================
    struct ScPrescoreAuditEntry
    {
        std::size_t sc_rank =
            0;

        std::size_t candidate_id =
            0;

        bool valid =
            false;

        double overlap =
            0.0;

        double rmse =
            std::numeric_limits<double>::infinity();

        const char *guess_name =
            "NONE";
    };

    std::vector<ScPrescoreAuditEntry>
        sc_prescore_audit;

    sc_prescore_audit.reserve(
        formal_loop_candidates.size());

    // ================================================================
    // SC-SINGLE PARALLEL VERIFICATION
    //
    // Correctness-first CPU parallelization.
    //
    // IMPORTANT:
    //   1. ALL SC Top-K candidates are still preserved.
    //   2. Prescore still ranks ONLY the three initial guesses of the
    //      SAME historical candidate.
    //   3. Full Verify still runs for every candidate that passes the
    //      existing prescore gate.
    //   4. Each worker owns an independent LoopVerifier + cache.
    //   5. Candidate promotion happens AFTER all workers finish.
    //   6. Promotion iterates in original SC rank order, preserving the
    //      existing exact-tie behaviour.
    //   7. Stateful consistency remains downstream and is still called
    //      exactly ONCE.
    // ================================================================

    constexpr std::size_t kScParallelWorkerCount =
        2;

    // ------------------------------------------------------------
    // Per-current-KF workers.
    //
    // We intentionally do NOT make these static yet:
    //     - Reset safety
    //     - no stale pointer-identity cache across SLAM runs
    //
    // Within this current KF the worker cache is still reused across
    // its assigned candidates and between prescore / Verify.
    // ------------------------------------------------------------
    std::array<
        std::unique_ptr<LoopVerifier>,
        kScParallelWorkerCount>
        sc_parallel_workers;

    for (std::size_t worker_id = 0;
         worker_id < kScParallelWorkerCount;
         ++worker_id)
    {
        sc_parallel_workers[worker_id] =
            std::make_unique<LoopVerifier>(
                loop_verifier_.GetConfig());
    }

    struct ScFormalInitialGuess
    {
        const char *name =
            "NONE";

        Eigen::Isometry3d transform =
            Eigen::Isometry3d::Identity();

        LoopVerifierInitialGuessScore
            prescore;

        bool score_ok =
            false;

        // ============================================================

            // ========================================================
// SC LAZY EXACT VERIFY REUSE SHADOW V1
        //
        // Keep the actual Full-Verify result for this exact initial
        // transform.  Diagnostic bookkeeping only.
        // ============================================================
        bool verify_ran =
            false;

        bool verify_ok =
            false;

        LoopVerificationResult verification;

        double verify_ms =
            0.0;
    };

    struct ScParallelCandidateJob
    {
        std::size_t sc_rank =
            0;

        std::size_t worker_id =
            0;

        FormalLoopCandidate
            sc_candidate;

        LoopCandidate
            formal_candidate;

        pcl::PointCloud<LIDAR_POINT>::Ptr
            target_K;

        std::vector<std::size_t>
            target_keyframes;

        std::vector<ScFormalInitialGuess>
            guesses;

        bool have_verification =
            false;

        bool best_verify_ok =
            false;

        const char *
            selected_guess =
                "NONE";

        LoopVerificationResult
            best_verification;

        LoopVerifierInitialGuessScore
            best_prescore;

        std::size_t verify_calls =
            0;

        double verify_sum_ms =
            0.0;
    };

    std::vector<ScParallelCandidateJob>
        sc_parallel_jobs;

    sc_parallel_jobs.reserve(
        formal_loop_candidates.size());

    const double sc_prescore_gate =
        loop_verifier_
            .GetConfig()
            .prescore_min_overlap_ratio;

    // ================================================================
    // STAGE A
    //
    // Serial preparation:
    //   target build
    //   three yaw hypotheses
    //   cheap prescore
    //
    // Each candidate is assigned round-robin to one verifier worker.
    // Its prescore also uses that same worker so the prepared target
    // cache is reusable by the later Full Verify.
    // ================================================================
    for (std::size_t sc_rank = 0;
         sc_rank < formal_loop_candidates.size();
         ++sc_rank)
    {
        const FormalLoopCandidate &
            sc_candidate =
                formal_loop_candidates[sc_rank];

        // ------------------------------------------------------------
        // Preserve existing defensive separation policy exactly.
        // ------------------------------------------------------------
        if (sc_candidate.candidate_id >=
            current_keyframe.id)
        {
            continue;
        }

        const std::size_t sc_keyframe_gap =
            current_keyframe.id -
            sc_candidate.candidate_id;

        if (sc_keyframe_gap <
                loop_config.min_keyframe_id_separation ||
            !std::isfinite(
                sc_candidate.time_separation_sec) ||
            sc_candidate.time_separation_sec <
                loop_config.min_time_separation_sec)
        {
            continue;
        }

        LoopCandidate
            formal_candidate_trial;

        formal_candidate_trial.current_id =
            current_keyframe.id;

        formal_candidate_trial.candidate_id =
            sc_candidate.candidate_id;

        // Pose distance remains diagnostic only.
        formal_candidate_trial.distance =
            sc_candidate.pose_distance;

        formal_candidate_trial.time_separation_sec =
            sc_candidate.time_separation_sec;

        if (sc_candidate.source ==
            FormalLoopRetrievalSource::Btc)
        {
            formal_candidate_trial.retrieval_score =
                sc_candidate.retrieval_score;

            formal_candidate_trial.matched_descriptor_pairs =
                sc_candidate.matched_descriptor_pairs;
        }
        else
        {
            formal_candidate_trial.retrieval_score =
                sc_candidate.scan_context_similarity;

            formal_candidate_trial.matched_descriptor_pairs =
                sc_candidate.compared_sectors;
        }

        pcl::PointCloud<LIDAR_POINT>::Ptr
            target_K_trial;

        std::vector<std::size_t>
            target_keyframes_trial;

        ++sc_v2_target_attempts;

        const std::chrono::steady_clock::time_point
            sc_v2_target_start =
                std::chrono::steady_clock::now();

        const bool sc_v2_target_built =
            BuildCandidateCenteredHistoricalTarget(
                formal_candidate_trial.candidate_id,
                target_K_trial,
                &target_keyframes_trial);

        sc_v2_target_ms +=
            ElapsedMilliseconds(
                sc_v2_target_start,
                std::chrono::steady_clock::now());

        if (!sc_v2_target_built ||
            !target_K_trial ||
            target_K_trial->empty())
        {
            continue;
        }

        ++sc_v2_targets_built;

        std::vector<ScFormalInitialGuess>
            sc_formal_guesses;

        const Eigen::Isometry3d
            sc_formal_base_guess =
                sc_candidate.has_base_initial_guess
                    ? sc_candidate.base_initial_guess
                    : Eigen::Isometry3d::Identity();

        sc_formal_guesses.reserve(5);

        // ------------------------------------------------------------
        // SC Formal V15-compatible yaw ambiguity handling.
        //
        // Candidate-centered Window semantics:
        //
        //   T_H_L_guess = T_H_C_yaw * T_C_L
        //
        // Besides +/- Scan Context yaw, explicitly test the 180-degree
        // complementary basin. This is required in repetitive rows where
        // Scan Context may report yaw ~= 0 even for reverse traversal.
        //
        // Equivalent yaw modes are de-duplicated.
        // ------------------------------------------------------------
        std::vector<double>
            sc_formal_yaw_modes_rad;

        sc_formal_yaw_modes_rad.reserve(5);
        sc_formal_yaw_modes_rad.push_back(0.0);

        {
            ScFormalInitialGuess guess;

            guess.name =
                (sc_candidate.source ==
                 FormalLoopRetrievalSource::Btc)
                    ? "BTC_COARSE"
                    : "IDENTITY";

            guess.transform =
                sc_formal_base_guess;

            sc_formal_guesses.push_back(
                guess);
        }

        const auto append_sc_formal_yaw_guess =
            [&sc_formal_guesses,
             &sc_formal_yaw_modes_rad,
             &sc_formal_base_guess](
                const char *name,
                double yaw_rad)
        {
            const double normalized_yaw =
                std::atan2(
                    std::sin(yaw_rad),
                    std::cos(yaw_rad));

            constexpr double
                kYawDuplicateToleranceRad =
                    1.0e-3;

            for (const double existing_yaw :
                 sc_formal_yaw_modes_rad)
            {
                const double angular_difference =
                    std::atan2(
                        std::sin(
                            normalized_yaw -
                            existing_yaw),
                        std::cos(
                            normalized_yaw -
                            existing_yaw));

                if (std::abs(
                        angular_difference) <=
                    kYawDuplicateToleranceRad)
                {
                    return;
                }
            }

            Eigen::Isometry3d yaw_guess =
                Eigen::Isometry3d::Identity();

            yaw_guess.linear() =
                Eigen::AngleAxisd(
                    normalized_yaw,
                    Eigen::Vector3d::UnitZ())
                    .toRotationMatrix();

            ScFormalInitialGuess guess;

            guess.name =
                name;

            guess.transform =
                yaw_guess *
                sc_formal_base_guess;

            sc_formal_guesses.push_back(
                guess);

            sc_formal_yaw_modes_rad.push_back(
                normalized_yaw);
        };

        if (sc_candidate.has_yaw_hint &&
            std::isfinite(
                sc_candidate.yaw_hint_deg))
        {
            constexpr double kPi =
                3.14159265358979323846;

            constexpr double kDegToRad =
                kPi / 180.0;

            const double sc_yaw_rad =
                sc_candidate.yaw_hint_deg *
                kDegToRad;

            append_sc_formal_yaw_guess(
                "SC_YAW_POSITIVE",
                sc_yaw_rad);

            append_sc_formal_yaw_guess(
                "SC_YAW_NEGATIVE",
                -sc_yaw_rad);

            append_sc_formal_yaw_guess(
                "SC_YAW_POSITIVE_FLIP180",
                sc_yaw_rad + kPi);

            append_sc_formal_yaw_guess(
                "SC_YAW_NEGATIVE_FLIP180",
                -sc_yaw_rad + kPi);
        }

        // ------------------------------------------------------------
        // Round-robin load distribution.
        //
        // Use number of VALID prepared jobs rather than SC rank, so
        // rejected/missing targets do not create an uneven assignment.
        // ------------------------------------------------------------
        const std::size_t worker_id =
            sc_parallel_jobs.size() %
            kScParallelWorkerCount;

        LoopVerifier &candidate_verifier =
            *sc_parallel_workers[worker_id];

        // ------------------------------------------------------------
        // Existing hypothesis prescore policy, unchanged.
        // ------------------------------------------------------------
        for (ScFormalInitialGuess &guess :
             sc_formal_guesses)
        {
            const std::chrono::steady_clock::time_point
                prescore_start =
                    std::chrono::steady_clock::now();

            guess.score_ok =
                candidate_verifier.ScoreInitialGuess(
                    current_keyframe.cloud,
                    target_K_trial,
                    guess.transform,
                    guess.prescore);

            const double prescore_elapsed_ms =
                ElapsedMilliseconds(
                    prescore_start,
                    std::chrono::steady_clock::now());

            loop_timing.verifier_prescore_ms +=
                prescore_elapsed_ms;

            ++loop_timing.verifier_prescore_calls;

            sc_v2_prescore_ms +=
                prescore_elapsed_ms;

            ++sc_v2_prescore_calls;

            guess.prescore.valid =
                guess.score_ok &&
                guess.prescore.valid;
        }

        std::sort(
            sc_formal_guesses.begin(),
            sc_formal_guesses.end(),
            [](const ScFormalInitialGuess &lhs,
               const ScFormalInitialGuess &rhs)
            {
                if (lhs.prescore.valid !=
                    rhs.prescore.valid)
                {
                    return lhs.prescore.valid;
                }

                if (lhs.prescore.overlap_ratio !=
                    rhs.prescore.overlap_ratio)
                {
                    return lhs.prescore.overlap_ratio >
                           rhs.prescore.overlap_ratio;
                }

                return lhs.prescore.rmse <
                       rhs.prescore.rmse;
            });

        // ------------------------------------------------------------
        // Preserve V2.5 Prescore Audit.
        // ------------------------------------------------------------
        ScPrescoreAuditEntry
            sc_prescore_entry;

        sc_prescore_entry.sc_rank =
            sc_rank;

        sc_prescore_entry.candidate_id =
            sc_candidate.candidate_id;

        for (const ScFormalInitialGuess &guess :
             sc_formal_guesses)
        {
            if (!guess.prescore.valid ||
                !std::isfinite(
                    guess.prescore.overlap_ratio) ||
                !std::isfinite(
                    guess.prescore.rmse))
            {
                continue;
            }

            sc_prescore_entry.valid =
                true;

            sc_prescore_entry.overlap =
                guess.prescore.overlap_ratio;

            sc_prescore_entry.rmse =
                guess.prescore.rmse;

            sc_prescore_entry.guess_name =
                guess.name;

            break;
        }

        sc_prescore_audit.push_back(
            sc_prescore_entry);

        ScParallelCandidateJob
            job;

        job.sc_rank =
            sc_rank;

        job.worker_id =
            worker_id;

        job.sc_candidate =
            sc_candidate;

        job.formal_candidate =
            formal_candidate_trial;

        job.target_K =
            target_K_trial;

        job.target_keyframes =
            std::move(
                target_keyframes_trial);

        job.guesses =
            std::move(
                sc_formal_guesses);

        sc_parallel_jobs.push_back(
            std::move(job));
    }

    // ================================================================
    // STAGE B
    //
    // Parallel FULL geometry verification.
    //
    // Exactly one thread owns each LoopVerifier worker.
    // One worker processes its own candidates sequentially.
    // Different workers run concurrently.
    // ================================================================
    const std::chrono::steady_clock::time_point
        sc_parallel_verify_wall_start =
            std::chrono::steady_clock::now();

    std::vector<std::future<void>>
        sc_parallel_futures;

    sc_parallel_futures.reserve(
        kScParallelWorkerCount);

    for (std::size_t worker_id = 0;
         worker_id < kScParallelWorkerCount;
         ++worker_id)
    {
        bool worker_has_job =
            false;

        for (const ScParallelCandidateJob &job :
             sc_parallel_jobs)
        {
            if (job.worker_id == worker_id)
            {
                worker_has_job =
                    true;
                break;
            }
        }

        if (!worker_has_job)
        {
            continue;
        }

        sc_parallel_futures.emplace_back(
            std::async(
                std::launch::async,
                [&, worker_id]()
                {
                    LoopVerifier &worker_verifier =
                        *sc_parallel_workers[worker_id];

                    for (ScParallelCandidateJob &job :
                         sc_parallel_jobs)
                    {
                        if (job.worker_id !=
                            worker_id)
                        {
                            continue;
                        }

                        for (ScFormalInitialGuess &guess :
                             job.guesses)
                        {
                            if (!guess.prescore.valid ||
                                !std::isfinite(
                                    guess.prescore.overlap_ratio) ||
                                guess.prescore.overlap_ratio <
                                    sc_prescore_gate)
                            {
                                continue;
                            }

                            LoopVerificationResult
                                trial;

                            const std::chrono::steady_clock::time_point
                                verify_start =
                                    std::chrono::steady_clock::now();

                            const bool trial_ok =
                                worker_verifier.Verify(
                                    current_keyframe.cloud,
                                    job.target_K,
                                    guess.transform,
                                    trial);

                            const double verify_elapsed_ms =
                                ElapsedMilliseconds(
                                    verify_start,
                                    std::chrono::steady_clock::now());

                            job.verify_sum_ms +=
                                verify_elapsed_ms;

                            ++job.verify_calls;

                            // =========================================
                            // SC LAZY EXACT VERIFY REUSE SHADOW V1
                            //
                            // Store the result of THIS exact transform.
                            // No selection logic is changed.
                            // =========================================
                            guess.verify_ran =
                                true;

                            guess.verify_ok =
                                trial_ok;

                            guess.verification =
                                trial;

                            guess.verify_ms =
                                verify_elapsed_ms;

                            bool better_trial =
                                !job.have_verification;

                            if (!better_trial &&
                                trial.accepted !=
                                    job.best_verification.accepted)
                            {
                                better_trial =
                                    trial.accepted;
                            }
                            else if (
                                !better_trial &&
                                trial.accepted ==
                                    job.best_verification.accepted &&
                                trial.overlap_ratio >
                                    job.best_verification
                                            .overlap_ratio +
                                        1.0e-9)
                            {
                                better_trial =
                                    true;
                            }
                            else if (
                                !better_trial &&
                                trial.accepted ==
                                    job.best_verification.accepted &&
                                std::abs(
                                    trial.overlap_ratio -
                                    job.best_verification
                                        .overlap_ratio) <=
                                    1.0e-9 &&
                                trial.rmse <
                                    job.best_verification.rmse)
                            {
                                better_trial =
                                    true;
                            }

                            if (better_trial)
                            {
                                job.have_verification =
                                    true;

                                job.best_verify_ok =
                                    trial_ok;

                                job.selected_guess =
                                    guess.name;

                                job.best_verification =
                                    trial;

                                job.best_prescore =
                                    guess.prescore;
                            }

                            // Restore original production policy:
                            // first geometry-accepted hypothesis wins.
                            if (trial_ok &&
                                trial.accepted)
                            {
                                break;
                            }
                        }
                    }
                }));
    }

    for (std::future<void> &future :
         sc_parallel_futures)
    {
        future.get();
    }

    const double sc_parallel_verify_wall_ms =
        ElapsedMilliseconds(
            sc_parallel_verify_wall_start,
            std::chrono::steady_clock::now());

    std::size_t sc_parallel_verify_calls =
        0;

    double sc_parallel_verify_sum_ms =
        0.0;

    for (const ScParallelCandidateJob &job :
         sc_parallel_jobs)
    {
        sc_parallel_verify_calls +=
            job.verify_calls;

        sc_parallel_verify_sum_ms +=
            job.verify_sum_ms;
    }

    // Existing verifier timing now represents REAL backend wall time.
    sc_v2_verify_ms +=
        sc_parallel_verify_wall_ms;

    sc_v2_verify_calls +=
        sc_parallel_verify_calls;

    loop_timing.verifier_ms +=
        sc_parallel_verify_wall_ms;

    loop_timing.verifier_calls +=
        sc_parallel_verify_calls;

    const std::size_t sc_parallel_workers_used =
        sc_parallel_futures.size();

    const double sc_parallel_sum_over_wall =
        (sc_parallel_verify_wall_ms > 1.0e-9)
            ? sc_parallel_verify_sum_ms /
                  sc_parallel_verify_wall_ms
            : 0.0;

    std::cout
        << "SC_PARALLEL_TIMING"
        << " | current_kf="
        << current_keyframe.id
        << " | topk="
        << formal_loop_candidates.size()
        << " | prepared_jobs="
        << sc_parallel_jobs.size()
        << " | workers="
        << sc_parallel_workers_used
        << " | verify_calls="
        << sc_parallel_verify_calls
        << " | verify_sum_ms="
        << sc_parallel_verify_sum_ms
        << " | verify_wall_ms="
        << sc_parallel_verify_wall_ms
        << " | sum_over_wall="
        << sc_parallel_sum_over_wall
        << std::endl;

    // ================================================================
    // POSE SUPPLEMENT SOURCE LOOKUP FIX V1
    //
    // ScParallelCandidateJob::sc_candidate is currently a
    // FormalLoopCandidate, so it does not carry the retrieval-source flag.
    //
    // Recover that flag from the original SC-SINGLE retrieval vector.
    // Candidate IDs are unique within one current query.
    // ================================================================
    const auto is_pose_supplement_candidate =
        [&sc_single_candidates](
            const std::size_t candidate_id)
            -> bool
        {
            for (const ScanContextShadowCandidate &candidate :
                 sc_single_candidates)
            {
                if (candidate.candidate_id ==
                        candidate_id &&
                    candidate.pose_supplement)
                {
                    return true;
                }
            }

            return false;
        };

    // ================================================================
    // STAGE C
    //
    // Single-thread deterministic promotion.
    //
    // Jobs are stored in original SC rank order. Therefore exact ties
    // still keep the earlier SC rank exactly as Formal V2 did.
    // ================================================================
    for (const ScParallelCandidateJob &job :
         sc_parallel_jobs)
    {
        // ============================================================
        // POSE SUPPLEMENT PRECISION GUARD V1
        //
        // Pose-guided retrieval is allowed to feed TrackRecovery below,
        // but it must NEVER replace the ordinary SC geometry winner by
        // itself.
        // ============================================================
        if (is_pose_supplement_candidate(
                job.sc_candidate.candidate_id))
        {
            continue;
        }

        const bool candidate_geometry_ok =
            job.have_verification &&
            job.best_verify_ok &&
            job.best_verification.accepted;

        if (!candidate_geometry_ok)
        {
            continue;
        }

        bool better_candidate =
            !sc_have_selected_candidate;

        if (!better_candidate &&
            job.best_verification.overlap_ratio >
                best_verification.overlap_ratio +
                    1.0e-9)
        {
            better_candidate =
                true;
        }
        else if (
            !better_candidate &&
            std::abs(
                job.best_verification.overlap_ratio -
                best_verification.overlap_ratio) <=
                1.0e-9 &&
            job.best_verification.rmse <
                best_verification.rmse)
        {
            better_candidate =
                true;
        }

        if (!better_candidate)
        {
            continue;
        }

        sc_have_selected_candidate =
            true;

        sc_selected_rank =
            job.sc_rank;

        sc_selected_candidate =
            job.sc_candidate;

        sc_formal_candidate =
            job.formal_candidate;

        best_target_K =
            job.target_K;

        best_target_keyframes =
            job.target_keyframes;

        best_verification =
            job.best_verification;

        sc_best_prescore =
            job.best_prescore;

        sc_best_verify_ok =
            job.best_verify_ok;

        sc_selected_guess =
            job.selected_guess;
    }

    // ================================================================
    // SC V2 CANDIDATE GEOMETRY DIAGNOSTIC V1
    //
    // Diagnostic only.
    //
    // IMPORTANT:
    //   - Uses Full Verify results already computed by worker threads.
    //   - Adds ZERO Verify / ICP calls.
    //   - Runs only after all workers have joined.
    //   - Does NOT change winner selection.
    //   - Does NOT change consistency / graph / PGO.
    // ================================================================
    try
    {
        const std::filesystem::path
            sc_candidate_geometry_directory =
                FrontendLoopDirectory();

        std::filesystem::create_directories(
            sc_candidate_geometry_directory);

        const std::filesystem::path
            sc_candidate_geometry_path =
                sc_candidate_geometry_directory /
                "sc_v2_candidate_geometry.csv";

        const bool
            sc_candidate_geometry_exists =
                std::filesystem::exists(
                    sc_candidate_geometry_path);

        std::ofstream
            sc_candidate_geometry_csv(
                sc_candidate_geometry_path,
                std::ios::app);

        if (sc_candidate_geometry_csv.is_open())
        {
            sc_candidate_geometry_csv
                << std::fixed
                << std::setprecision(9);

            if (!sc_candidate_geometry_exists)
            {
                sc_candidate_geometry_csv
                    << "current_kf,"
                    << "sc_rank,"
                    << "historical_kf,"
                    << "sc_distance,"
                    << "sc_similarity,"
                    << "sc_yaw_deg,"
                    << "prescore_valid,"
                    << "prescore_overlap,"
                    << "prescore_rmse,"
                    << "verify_ok,"
                    << "geometry_accepted,"
                    << "verify_overlap,"
                    << "verify_rmse,"
                    << "correction_translation_m,"
                    << "correction_rotation_deg,"
                    << "selected_guess,"
                    << "is_global_winner"
                    << '\n';
            }

            for (const ScParallelCandidateJob &job :
                 sc_parallel_jobs)
            {
                const bool geometry_accepted =
                    job.have_verification &&
                    job.best_verify_ok &&
                    job.best_verification.accepted;

                const bool is_global_winner =
                    sc_have_selected_candidate &&
                    job.sc_rank == sc_selected_rank;

                sc_candidate_geometry_csv
                    << current_keyframe.id << ','
                    << (job.sc_rank + 1) << ','
                    << job.sc_candidate.candidate_id << ','
                    << job.sc_candidate.scan_context_distance << ','
                    << job.sc_candidate.scan_context_similarity << ','
                    << job.sc_candidate.yaw_hint_deg << ','
                    << (job.best_prescore.valid ? 1 : 0) << ','
                    << job.best_prescore.overlap_ratio << ','
                    << job.best_prescore.rmse << ','
                    << (job.best_verify_ok ? 1 : 0) << ','
                    << (geometry_accepted ? 1 : 0) << ','
                    << job.best_verification.overlap_ratio << ','
                    << job.best_verification.rmse << ','
                    << job.best_verification
                           .correction_translation << ','
                    << job.best_verification
                           .correction_rotation_deg << ','
                    << job.selected_guess << ','
                    << (is_global_winner ? 1 : 0)
                    << '\n';
            }
        }
    }
    catch (const std::exception &)
    {
        // Diagnostic output must never affect SLAM.
    }

    // ================================================================
    // SC TRACK RESCUE CAUSAL SHADOW V1 DIAG PACK
    //
    // DIAGNOSTIC ONLY.
    //
    // Online causal reproduction of offline V6/V8/V9.
    //
    // Uses existing Full Verify results.  Adds ZERO ICP / Verify calls.
    //
    // Outputs:
    //
    //   sc_track_rescue_endpoints_shadow_v1.csv
    //       Every causal confirmed (q,h) endpoint.
    //
    //   sc_track_rescue_query_shadow_v1.csv
    //       One summary row per query:
    //       original winner, track counts, rescue decision,
    //       winner/rescue graph consistency diagnostics.
    //
    // IMPORTANT:
    //   * Does NOT change sc_selected_candidate.
    //   * Does NOT change best_verification.
    //   * Does NOT call LoopConsistencyChecker.
    //   * Does NOT add PoseGraph edges.
    //   * Does NOT affect PGO.
    //
    // Track models:
    //
    //   SAME    s=+1 : q-h ~= C
    //   REVERSE s=-1 : q+h ~= C
    //
    // Offline Strawberry_4 rescue policy:
    //
    //   winner pose distance > 2.0 m
    //   alternate pose distance <= 1.0 m
    //   alternate causal track nodes >= 8
    //
    // Track confirmation itself begins at 6 nodes so the CSV keeps
    // enough information for later threshold sweeps without rerunning.
    // ================================================================
    {
        struct ScTrackShadowState
        {
            int direction = 0;

            std::size_t first_q = 0;
            std::size_t first_h = 0;

            std::size_t last_q = 0;
            std::size_t last_h = 0;

            std::size_t nodes = 0;

            double score = 0.0;

            std::vector<double> corridor_values;
        };

        struct ScTrackShadowEndpoint
        {
            std::size_t historical_id = 0;

            int direction = 0;

            std::size_t nodes = 0;
            std::size_t span = 0;

            double corridor_median =
                std::numeric_limits<double>::quiet_NaN();

            double progress =
                std::numeric_limits<double>::quiet_NaN();

            double pose_distance =
                std::numeric_limits<double>::infinity();

            double overlap =
                std::numeric_limits<double>::quiet_NaN();

            double rmse =
                std::numeric_limits<double>::infinity();

            std::size_t sc_rank = 0;

            double score = 0.0;

            Eigen::Isometry3d measurement =
                Eigen::Isometry3d::Identity();
        };

        static std::array<
            std::vector<ScTrackShadowState>,
            2>
            sc_track_shadow_active;

        static bool
            sc_track_shadow_have_last_query =
                false;

        static std::size_t
            sc_track_shadow_last_query =
                0U;

        const std::size_t current_q =
            current_keyframe.id;

        // Reset safety for replay / node reset.
        if (sc_track_shadow_have_last_query &&
            current_q <=
                sc_track_shadow_last_query)
        {
            for (auto &states :
                 sc_track_shadow_active)
            {
                states.clear();
            }
        }

        sc_track_shadow_have_last_query =
            true;

        sc_track_shadow_last_query =
            current_q;

        constexpr std::size_t
            kTrackMinHistoryGap = 100U;

        constexpr std::size_t
            kTrackMaxQueryStep = 8U;

        constexpr double
            kTrackMaxCorridorStep = 3.0;

        constexpr double
            kTrackGlobalCorridorTolerance = 6.0;

        constexpr double
            kTrackMaxMotionError = 4.0;

        // Preserve V6 diagnostic coverage.
        constexpr std::size_t
            kTrackConfirmMinNodes = 6U;

        constexpr std::size_t
            kTrackMinSpan = 8U;

        constexpr double
            kTrackMinProgress = 0.55;

        constexpr double
            kTrackMaxProgress = 1.45;

        // V8/V9 rescue policy.
        constexpr std::size_t
            kRescueMinNodes = 8U;

        constexpr double
            kWinnerSuspiciousDistanceM = 2.0;

        constexpr double
            kAlternateMaxDistanceM = 1.0;

        const auto direction_index =
            [](const int direction)
                -> std::size_t
            {
                return
                    direction > 0
                        ? 0U
                        : 1U;
            };

        const auto median_value =
            [](const std::vector<double> &values)
                -> double
            {
                if (values.empty())
                {
                    return
                        std::numeric_limits<double>::
                            quiet_NaN();
                }

                std::vector<double> temp =
                    values;

                std::sort(
                    temp.begin(),
                    temp.end());

                const std::size_t n =
                    temp.size();

                if ((n % 2U) != 0U)
                {
                    return temp[n / 2U];
                }

                return
                    0.5 *
                    (temp[n / 2U - 1U] +
                     temp[n / 2U]);
            };

        // ------------------------------------------------------------
        // Remove states too old to connect causally to current query.
        // ------------------------------------------------------------
        for (auto &states :
             sc_track_shadow_active)
        {
            states.erase(
                std::remove_if(
                    states.begin(),
                    states.end(),
                    [current_q](
                        const ScTrackShadowState &state)
                    {
                        if (current_q <= state.last_q)
                        {
                            return true;
                        }

                        return
                            current_q -
                                state.last_q >
                            8U;
                    }),
                states.end());
        }

        const std::size_t active_same_before =
            sc_track_shadow_active[0].size();

        const std::size_t active_reverse_before =
            sc_track_shadow_active[1].size();

        std::array<
            std::vector<ScTrackShadowState>,
            2>
            new_states;

        std::vector<ScTrackShadowEndpoint>
            confirmed_raw;

        confirmed_raw.reserve(
            sc_parallel_jobs.size() * 2U);

        std::size_t
            long_geometry_candidates =
                0U;

        // ------------------------------------------------------------
        // Causal DP:
        // each current geometry-valid candidate is evaluated under
        // SAME and REVERSE fixed-direction hypotheses.
        // ------------------------------------------------------------
        for (const ScParallelCandidateJob &job :
             sc_parallel_jobs)
        {
            const bool geometry_ok =
                job.have_verification &&
                job.best_verify_ok &&
                job.best_verification.accepted;

            if (!geometry_ok)
            {
                continue;
            }

            const std::size_t historical_h =
                job.sc_candidate.candidate_id;

            if (historical_h >= current_q)
            {
                continue;
            }

            if (current_q - historical_h <
                kTrackMinHistoryGap)
            {
                continue;
            }

            ++long_geometry_candidates;

            const Keyframe *historical_keyframe =
                FindBackendKeyframeById(
                    historical_h);

            if (historical_keyframe == nullptr ||
                !historical_keyframe
                     ->T_WL
                     .matrix()
                     .allFinite() ||
                !current_keyframe
                     .T_WL
                     .matrix()
                     .allFinite())
            {
                continue;
            }

            const double pose_distance =
                (current_keyframe
                     .T_WL
                     .translation() -
                 historical_keyframe
                     ->T_WL
                     .translation())
                    .norm();

            const double overlap =
                job.best_verification
                    .overlap_ratio;

            const double rmse =
                job.best_verification
                    .rmse;

            if (!std::isfinite(pose_distance) ||
                !std::isfinite(overlap) ||
                !std::isfinite(rmse))
            {
                continue;
            }

            // sc_rank is zero based internally.
            // This matches the offline V6 quality term.
            const double candidate_quality =
                2.0 * overlap -
                0.8 * rmse -
                0.01 *
                    static_cast<double>(
                        job.sc_rank);

            for (const int direction :
                 {+1, -1})
            {
                const std::size_t index =
                    direction_index(
                        direction);

                const long long q_ll =
                    static_cast<long long>(
                        current_q);

                const long long h_ll =
                    static_cast<long long>(
                        historical_h);

                // Unified invariant:
                //
                // SAME:    q-h
                // REVERSE: q+h
                const double corridor =
                    static_cast<double>(
                        q_ll -
                        static_cast<long long>(
                            direction) *
                            h_ll);

                ScTrackShadowState best;

                best.direction =
                    direction;

                best.first_q =
                    current_q;

                best.first_h =
                    historical_h;

                best.last_q =
                    current_q;

                best.last_h =
                    historical_h;

                best.nodes =
                    1U;

                best.score =
                    candidate_quality;

                best.corridor_values =
                    {corridor};

                for (const ScTrackShadowState &prev :
                     sc_track_shadow_active[index])
                {
                    if (current_q <=
                        prev.last_q)
                    {
                        continue;
                    }

                    const std::size_t dq_u =
                        current_q -
                        prev.last_q;

                    if (dq_u == 0U ||
                        dq_u >
                            kTrackMaxQueryStep)
                    {
                        continue;
                    }

                    const long long dh =
                        static_cast<long long>(
                            historical_h) -
                        static_cast<long long>(
                            prev.last_h);

                    if (direction > 0 &&
                        dh < 0)
                    {
                        continue;
                    }

                    if (direction < 0 &&
                        dh > 0)
                    {
                        continue;
                    }

                    const double dq =
                        static_cast<double>(
                            dq_u);

                    const double motion_error =
                        std::abs(
                            dq -
                            static_cast<double>(
                                direction) *
                                static_cast<double>(
                                    dh));

                    if (motion_error >
                        kTrackMaxMotionError)
                    {
                        continue;
                    }

                    const double prev_corridor =
                        static_cast<double>(
                            static_cast<long long>(
                                prev.last_q) -
                            static_cast<long long>(
                                direction) *
                                static_cast<long long>(
                                    prev.last_h));

                    if (std::abs(
                            corridor -
                            prev_corridor) >
                        kTrackMaxCorridorStep)
                    {
                        continue;
                    }

                    const double prev_median =
                        median_value(
                            prev.corridor_values);

                    if (!std::isfinite(
                            prev_median) ||
                        std::abs(
                            corridor -
                            prev_median) >
                            kTrackGlobalCorridorTolerance)
                    {
                        continue;
                    }

                    const double transition_score =
                        1.0 -
                        0.15 * motion_error -
                        0.10 *
                            std::abs(
                                corridor -
                                prev_corridor) -
                        0.02 *
                            static_cast<double>(
                                dq_u > 1U
                                    ? dq_u - 1U
                                    : 0U);

                    ScTrackShadowState extended =
                        prev;

                    extended.last_q =
                        current_q;

                    extended.last_h =
                        historical_h;

                    extended.nodes =
                        prev.nodes + 1U;

                    extended.score =
                        prev.score +
                        candidate_quality +
                        transition_score;

                    extended
                        .corridor_values
                        .push_back(
                            corridor);

                    if (extended.nodes >
                            best.nodes ||
                        (extended.nodes ==
                             best.nodes &&
                         extended.score >
                             best.score))
                    {
                        best =
                            std::move(
                                extended);
                    }
                }

                new_states[index]
                    .push_back(
                        best);

                if (best.nodes <
                    kTrackConfirmMinNodes)
                {
                    continue;
                }

                const std::size_t span =
                    current_q -
                    best.first_q;

                if (span <
                    kTrackMinSpan)
                {
                    continue;
                }

                const double historical_motion =
                    std::abs(
                        static_cast<double>(
                            static_cast<long long>(
                                historical_h) -
                            static_cast<long long>(
                                best.first_h)));

                const double progress =
                    historical_motion /
                    std::max(
                        static_cast<double>(
                            span),
                        1.0);

                if (!std::isfinite(progress) ||
                    progress <
                        kTrackMinProgress ||
                    progress >
                        kTrackMaxProgress)
                {
                    continue;
                }

                ScTrackShadowEndpoint endpoint;

                endpoint.historical_id =
                    historical_h;

                endpoint.direction =
                    direction;

                endpoint.nodes =
                    best.nodes;

                endpoint.span =
                    span;

                endpoint.corridor_median =
                    median_value(
                        best.corridor_values);

                endpoint.progress =
                    progress;

                endpoint.pose_distance =
                    pose_distance;

                endpoint.overlap =
                    overlap;

                endpoint.rmse =
                    rmse;

                endpoint.sc_rank =
                    job.sc_rank;

                endpoint.score =
                    best.score;

                endpoint.measurement =
                    job.best_verification
                        .T_target_source;

                confirmed_raw.push_back(
                    endpoint);
            }
        }

        // Current endpoints become causal history only AFTER every
        // candidate from this current query has been evaluated.
        for (std::size_t i = 0U;
             i < 2U;
             ++i)
        {
            sc_track_shadow_active[i]
                .insert(
                    sc_track_shadow_active[i]
                        .end(),
                    new_states[i]
                        .begin(),
                    new_states[i]
                        .end());
        }

        // ------------------------------------------------------------
        // Deduplicate SAME / REVERSE realizations of same (q,h).
        //
        // Match V8/V9:
        //     larger nodes first,
        //     then larger track score.
        // ------------------------------------------------------------
        std::vector<ScTrackShadowEndpoint>
            confirmed_unique;

        for (const auto &endpoint :
             confirmed_raw)
        {
            auto it =
                std::find_if(
                    confirmed_unique.begin(),
                    confirmed_unique.end(),
                    [&endpoint](
                        const ScTrackShadowEndpoint &other)
                    {
                        return
                            other.historical_id ==
                            endpoint.historical_id;
                    });

            if (it ==
                confirmed_unique.end())
            {
                confirmed_unique
                    .push_back(
                        endpoint);
            }
            else
            {
                const bool stronger =
                    endpoint.nodes >
                        it->nodes ||
                    (endpoint.nodes ==
                         it->nodes &&
                     endpoint.score >
                         it->score);

                if (stronger)
                {
                    *it =
                        endpoint;
                }
            }
        }

        // ------------------------------------------------------------
        // Original formal winner pair distance.
        // ------------------------------------------------------------
        bool have_winner =
            sc_have_selected_candidate;

        std::size_t winner_h =
            0U;

        double winner_pose_d =
            std::numeric_limits<double>::
                quiet_NaN();

        if (have_winner)
        {
            winner_h =
                sc_selected_candidate
                    .candidate_id;

            const Keyframe *winner_keyframe =
                FindBackendKeyframeById(
                    winner_h);

            if (winner_keyframe != nullptr &&
                winner_keyframe
                    ->T_WL
                    .matrix()
                    .allFinite() &&
                current_keyframe
                    .T_WL
                    .matrix()
                    .allFinite())
            {
                winner_pose_d =
                    (current_keyframe
                         .T_WL
                         .translation() -
                     winner_keyframe
                         ->T_WL
                         .translation())
                        .norm();
            }
        }

        const bool rescue_triggered =
            have_winner &&
            std::isfinite(
                winner_pose_d) &&
            winner_pose_d >
                kWinnerSuspiciousDistanceM;

        // ------------------------------------------------------------
        // V8/V9 deployable alternate ordering.
        // ------------------------------------------------------------
        const auto rescue_order =
            [](const ScTrackShadowEndpoint &a,
               const ScTrackShadowEndpoint &b)
            {
                constexpr double eps =
                    1.0e-9;

                if (a.pose_distance <
                    b.pose_distance - eps)
                {
                    return true;
                }

                if (b.pose_distance <
                    a.pose_distance - eps)
                {
                    return false;
                }

                if (a.rmse <
                    b.rmse - eps)
                {
                    return true;
                }

                if (b.rmse <
                    a.rmse - eps)
                {
                    return false;
                }

                if (a.overlap >
                    b.overlap + eps)
                {
                    return true;
                }

                if (b.overlap >
                    a.overlap + eps)
                {
                    return false;
                }

                if (a.nodes !=
                    b.nodes)
                {
                    return
                        a.nodes >
                        b.nodes;
                }

                return
                    a.sc_rank <
                    b.sc_rank;
            };

        std::vector<ScTrackShadowEndpoint>
            eligible;

        if (rescue_triggered)
        {
            for (const auto &endpoint :
                 confirmed_unique)
            {
                if (endpoint.historical_id ==
                    winner_h)
                {
                    continue;
                }

                if (endpoint.nodes <
                    kRescueMinNodes)
                {
                    continue;
                }

                if (!std::isfinite(
                        endpoint.pose_distance) ||
                    endpoint.pose_distance >
                        kAlternateMaxDistanceM)
                {
                    continue;
                }

                eligible.push_back(
                    endpoint);
            }

            std::sort(
                eligible.begin(),
                eligible.end(),
                rescue_order);
        }

        const bool rescue_found =
            rescue_triggered &&
            !eligible.empty();

        // ------------------------------------------------------------
        // Graph-consistency shadow diagnostic.
        //
        // Same definition as downstream graph gate:
        //
        //   predicted =
        //       T_WK(h)^-1 * T_WK(q)
        //
        //   error =
        //       predicted^-1 * verified_measurement
        //
        // This DOES NOT mutate PoseGraph.
        // ------------------------------------------------------------
        const std::vector<PoseGraphNode>
            shadow_pose_snapshot =
                pose_graph_.GetNodes();

        const auto compute_graph_error =
            [&shadow_pose_snapshot,
             current_q](
                const std::size_t historical_id,
                const Eigen::Isometry3d &measurement,
                double &translation_error,
                double &rotation_error_deg)
                -> bool
            {
                const PoseGraphNode *h_node =
                    nullptr;

                const PoseGraphNode *q_node =
                    nullptr;

                for (const PoseGraphNode &node :
                     shadow_pose_snapshot)
                {
                    if (node.id ==
                        historical_id)
                    {
                        h_node =
                            &node;
                    }

                    if (node.id ==
                        current_q)
                    {
                        q_node =
                            &node;
                    }
                }

                if (h_node == nullptr ||
                    q_node == nullptr ||
                    !h_node->T_WK
                         .matrix()
                         .allFinite() ||
                    !q_node->T_WK
                         .matrix()
                         .allFinite() ||
                    !measurement
                         .matrix()
                         .allFinite())
                {
                    return false;
                }

                const Eigen::Isometry3d predicted =
                    h_node->T_WK.inverse() *
                    q_node->T_WK;

                const Eigen::Isometry3d error =
                    predicted.inverse() *
                    measurement;

                if (!error.matrix().allFinite())
                {
                    return false;
                }

                translation_error =
                    error.translation().norm();

                Eigen::Quaterniond q_err(
                    error.rotation());

                if (!q_err
                         .coeffs()
                         .allFinite() ||
                    q_err.norm() <
                        1.0e-12)
                {
                    return false;
                }

                q_err.normalize();

                const double w =
                    std::clamp(
                        std::abs(q_err.w()),
                        0.0,
                        1.0);

                rotation_error_deg =
                    2.0 *
                    std::acos(w) *
                    180.0 /
                    3.14159265358979323846;

                return
                    std::isfinite(
                        translation_error) &&
                    std::isfinite(
                        rotation_error_deg);
            };

        double winner_graph_dt =
            std::numeric_limits<double>::
                quiet_NaN();

        double winner_graph_dr =
            std::numeric_limits<double>::
                quiet_NaN();

        bool winner_graph_valid =
            false;

        if (have_winner)
        {
            winner_graph_valid =
                compute_graph_error(
                    winner_h,
                    best_verification
                        .T_target_source,
                    winner_graph_dt,
                    winner_graph_dr);
        }

        double rescue_graph_dt =
            std::numeric_limits<double>::
                quiet_NaN();

        double rescue_graph_dr =
            std::numeric_limits<double>::
                quiet_NaN();

        bool rescue_graph_valid =
            false;

        bool rescue_graph_like_pass =
            false;

        // ============================================================
        // SC TRACK RESCUE LAZY MULTI-YAW FORMAL V1
        //
        // Normal SC candidates:
        //
        //     original first-accepted yaw -> BREAK
        //
        // Only AFTER TrackRecovery has selected ONE rescue historical
        // candidate do we spend extra computation on multiple yaw basins.
        //
        // No GT is used.
        // ============================================================

        struct ScLazyYawAudit
        {
            std::string name;

            double yaw_deg =
                std::numeric_limits<double>::
                    quiet_NaN();

            bool prescore_valid =
                false;

            double prescore_overlap =
                std::numeric_limits<double>::
                    quiet_NaN();

            double prescore_rmse =
                std::numeric_limits<double>::
                    quiet_NaN();

            bool verify_ran =
                false;

            bool verify_ok =
                false;

            bool accepted =
                false;

            double overlap =
                std::numeric_limits<double>::
                    quiet_NaN();

            double rmse =
                std::numeric_limits<double>::
                    quiet_NaN();

            double graph_dt =
                std::numeric_limits<double>::
                    quiet_NaN();

            double graph_dr =
                std::numeric_limits<double>::
                    quiet_NaN();

            bool graph_valid =
                false;

            bool graph_pass =
                false;

            double graph_score =
                std::numeric_limits<double>::
                    infinity();

            bool selected =
                false;
        };


        const ScParallelCandidateJob *
            rescue_selected_job =
                nullptr;


        LoopVerificationResult
            rescue_selected_verification;

        LoopVerifierInitialGuessScore
            rescue_selected_prescore;


        std::string
            rescue_selected_yaw_name =
                "NONE";


        bool rescue_have_selected_yaw =
            false;


        bool rescue_yaw_gate_pass =
            false;


        double rescue_yaw_score =
            std::numeric_limits<double>::
                infinity();


        std::size_t lazy_prescore_calls =
            0U;

        std::size_t lazy_verify_calls =
            0U;

        double lazy_verify_sum_ms =
            0.0;


        std::vector<ScLazyYawAudit>
            lazy_yaw_audit;


        if (rescue_found)
        {
            // --------------------------------------------------------
            // Find the already prepared SC job corresponding to the
            // ONE TrackRecovery-selected historical KF.
            // --------------------------------------------------------
            for (const ScParallelCandidateJob &job :
                 sc_parallel_jobs)
            {
                if (job.sc_candidate.candidate_id ==
                    eligible.front().historical_id)
                {
                    rescue_selected_job =
                        &job;

                    break;
                }
            }
        }


        if (rescue_found &&
            rescue_selected_job != nullptr &&
            rescue_selected_job->target_K &&
            !rescue_selected_job
                 ->target_K
                 ->empty())
        {
            // --------------------------------------------------------
            // Fresh verifier:
            //
            // Lazy rescue diagnostics/verification must not disturb
            // the normal worker verifier state/cache.
            // --------------------------------------------------------
            LoopVerifier lazy_verifier(
                loop_verifier_.GetConfig());


            constexpr double kPi =
                3.14159265358979323846;


            const auto normalize_yaw =
                [](double yaw_deg)
                {
                    while (yaw_deg >= 180.0)
                    {
                        yaw_deg -= 360.0;
                    }

                    while (yaw_deg < -180.0)
                    {
                        yaw_deg += 360.0;
                    }

                    return yaw_deg;
                };


            struct LazyInitialGuess
            {
                std::string name;

                double yaw_deg = 0.0;

                Eigen::Isometry3d
                    transform =
                        Eigen::Isometry3d::
                            Identity();

                LoopVerifierInitialGuessScore
                    prescore;
            };


            std::vector<LazyInitialGuess>
                lazy_guesses;


            const auto add_yaw_guess =
                [&lazy_guesses,
                 &normalize_yaw](
                    const std::string &name,
                    const double raw_yaw_deg)
                {
                    const double yaw_deg =
                        normalize_yaw(
                            raw_yaw_deg);

                    for (const LazyInitialGuess &old :
                         lazy_guesses)
                    {
                        if (std::abs(
                                normalize_yaw(
                                    old.yaw_deg -
                                    yaw_deg)) <
                            1.0e-6)
                        {
                            return;
                        }
                    }

                    LazyInitialGuess guess;

                    guess.name =
                        name;

                    guess.yaw_deg =
                        yaw_deg;

                    lazy_guesses.push_back(
                        guess);
                };


            const double sc_yaw_deg =
                rescue_selected_job
                    ->sc_candidate
                    .yaw_hint_deg;


            // --------------------------------------------------------
            // Rescue-only yaw set.
            //
            // Includes the original SC hypotheses PLUS the missing
            // complementary 180-degree basin.
            //
            // Early reverse revisits:
            //     SC yaw ~ 0 deg
            //     need ~180 deg
            //
            // Later reverse revisits:
            //     SC yaw ~180 deg
            //     existing SC guess remains available.
            // --------------------------------------------------------
            add_yaw_guess(
                "LAZY_IDENTITY",
                0.0);

            if (std::isfinite(
                    sc_yaw_deg))
            {
                add_yaw_guess(
                    "LAZY_SC_POS",
                    sc_yaw_deg);

                add_yaw_guess(
                    "LAZY_SC_NEG",
                    -sc_yaw_deg);

                add_yaw_guess(
                    "LAZY_SC_POS_180",
                    sc_yaw_deg + 180.0);

                add_yaw_guess(
                    "LAZY_SC_NEG_180",
                    -sc_yaw_deg + 180.0);
            }

            add_yaw_guess(
                "LAZY_180",
                180.0);


            // --------------------------------------------------------
            // Prescore all rescue-only guesses.
            // --------------------------------------------------------
            for (LazyInitialGuess &guess :
                 lazy_guesses)
            {
                guess.transform =
                    Eigen::Isometry3d::
                        Identity();

                const double yaw_rad =
                    guess.yaw_deg *
                    kPi /
                    180.0;

                guess.transform.linear() =
                    Eigen::AngleAxisd(
                        yaw_rad,
                        Eigen::Vector3d::UnitZ())
                        .toRotationMatrix();


                ++lazy_prescore_calls;


                const bool score_ok =
                    lazy_verifier
                        .ScoreInitialGuess(
                            current_keyframe.cloud,
                            rescue_selected_job
                                ->target_K,
                            guess.transform,
                            guess.prescore);


                guess.prescore.valid =
                    score_ok &&
                    guess.prescore.valid;
            }


            // ========================================================
            // SC LAZY EXACT VERIFY REUSE SHADOW V1
            //
            // SHADOW ONLY:
            //   * no Verify call is skipped
            //   * no candidate result is replaced
            //   * no graph decision is changed
            //
            // We only measure whether the exact same initial transform
            // was already Full-Verified by Normal SC-SINGLE.
            // ========================================================
            const bool reuse_shadow_sc_single_v1 =
                rescue_selected_job->sc_candidate.source ==
                    FormalLoopRetrievalSource::ScanContextSingle;

            std::size_t reuse_shadow_exact_match_v1 =
                0U;

            std::size_t reuse_shadow_available_v1 =
                0U;

            std::size_t reuse_active_calls_v2 =
                0U;

            std::size_t reuse_shadow_status_mismatch_v1 =
                0U;

            double reuse_shadow_would_save_ms_v1 =
                0.0;

            double reuse_shadow_max_overlap_diff_v1 =
                0.0;

            double reuse_shadow_max_rmse_diff_v1 =
                0.0;

            double reuse_shadow_max_pose_dt_v1 =
                0.0;

            double reuse_shadow_max_pose_dr_deg_v1 =
                0.0;

            // ========================================================
            // SC PARALLEL LOOP VERIFICATION
            //
            // Compute only the NON-REUSED Full-Verify requests here.
            //
            // IMPORTANT:
            //   * prescore is unchanged
            //   * V2 exact reuse is unchanged
            //   * graph scoring is NOT done in worker threads
            //   * winner selection is NOT done in worker threads
            //   * the existing serial yaw-order decision loop below
            //     remains authoritative
            // ========================================================
            struct LazyParallelVerifyTaskV3
            {
                std::size_t guess_index =
                    0U;

                std::size_t worker_id =
                    0U;

                bool verify_ran =
                    false;

                bool verify_ok =
                    false;

                LoopVerificationResult
                    verification;

                double verify_ms =
                    0.0;
            };


            constexpr std::size_t
                kLazyParallelWorkerCountV3 =
                    2U;


            std::vector<LazyParallelVerifyTaskV3>
                lazy_parallel_tasks_v3;

            lazy_parallel_tasks_v3.reserve(
                lazy_guesses.size());


            std::size_t
                lazy_parallel_next_worker_v3 =
                    0U;


            // --------------------------------------------------------
            // Build task list.
            //
            // If Normal SC-SINGLE already Full-Verified the exact same
            // transform, V2 will reuse it later and NO task is created.
            // --------------------------------------------------------
            for (std::size_t lazy_index_v3 = 0U;
                 lazy_index_v3 < lazy_guesses.size();
                 ++lazy_index_v3)
            {
                const LazyInitialGuess &guess =
                    lazy_guesses[lazy_index_v3];

                if (!guess.prescore.valid ||
                    !std::isfinite(
                        guess.prescore.overlap_ratio) ||
                    guess.prescore.overlap_ratio <
                        sc_prescore_gate)
                {
                    continue;
                }


                bool reusable_normal_verify_v3 =
                    false;


                if (reuse_shadow_sc_single_v1)
                {
                    constexpr double
                        kReuseTransformToleranceV3 =
                            1.0e-12;

                    for (const ScFormalInitialGuess &normal_guess :
                         rescue_selected_job->guesses)
                    {
                        const double transform_max_abs =
                            (
                                normal_guess.transform.matrix() -
                                guess.transform.matrix()
                            )
                                .cwiseAbs()
                                .maxCoeff();

                        if (std::isfinite(transform_max_abs) &&
                            transform_max_abs <=
                                kReuseTransformToleranceV3 &&
                            normal_guess.verify_ran)
                        {
                            reusable_normal_verify_v3 =
                                true;

                            break;
                        }
                    }
                }


                if (reusable_normal_verify_v3)
                {
                    continue;
                }


                LazyParallelVerifyTaskV3 task;

                task.guess_index =
                    lazy_index_v3;

                task.worker_id =
                    lazy_parallel_next_worker_v3 %
                    kLazyParallelWorkerCountV3;

                ++lazy_parallel_next_worker_v3;

                lazy_parallel_tasks_v3.push_back(
                    std::move(task));
            }


            std::array<
                std::unique_ptr<LoopVerifier>,
                kLazyParallelWorkerCountV3>
                lazy_parallel_workers_v3;


            std::array<
                bool,
                kLazyParallelWorkerCountV3>
                lazy_parallel_worker_used_v3 =
                    {false, false};


            for (const LazyParallelVerifyTaskV3 &task :
                 lazy_parallel_tasks_v3)
            {
                lazy_parallel_worker_used_v3[
                    task.worker_id] =
                        true;
            }


            for (std::size_t worker_id = 0U;
                 worker_id <
                     kLazyParallelWorkerCountV3;
                 ++worker_id)
            {
                if (!lazy_parallel_worker_used_v3[
                        worker_id])
                {
                    continue;
                }

                lazy_parallel_workers_v3[worker_id] =
                    std::make_unique<LoopVerifier>(
                        loop_verifier_.GetConfig());
            }


            const auto
                lazy_parallel_wall_begin_v3 =
                    std::chrono::
                        steady_clock::
                        now();


            std::vector<std::future<void>>
                lazy_parallel_futures_v3;

            lazy_parallel_futures_v3.reserve(
                kLazyParallelWorkerCountV3);


            for (std::size_t worker_id = 0U;
                 worker_id <
                     kLazyParallelWorkerCountV3;
                 ++worker_id)
            {
                if (!lazy_parallel_worker_used_v3[
                        worker_id])
                {
                    continue;
                }


                lazy_parallel_futures_v3.emplace_back(
                    std::async(
                        std::launch::async,
                        [&, worker_id]()
                        {
                            LoopVerifier &worker_verifier =
                                *lazy_parallel_workers_v3[
                                    worker_id];


                            for (LazyParallelVerifyTaskV3 &task :
                                 lazy_parallel_tasks_v3)
                            {
                                if (task.worker_id !=
                                    worker_id)
                                {
                                    continue;
                                }


                                const LazyInitialGuess &guess =
                                    lazy_guesses[
                                        task.guess_index];


                                const auto verify_begin =
                                    std::chrono::
                                        steady_clock::
                                        now();


                                task.verify_ok =
                                    worker_verifier.Verify(
                                        current_keyframe.cloud,
                                        rescue_selected_job
                                            ->target_K,
                                        guess.transform,
                                        task.verification);


                                task.verify_ms =
                                    ElapsedMilliseconds(
                                        verify_begin,
                                        std::chrono::
                                            steady_clock::
                                            now());


                                task.verify_ran =
                                    true;
                            }
                        }));
            }


            for (std::future<void> &future :
                 lazy_parallel_futures_v3)
            {
                future.get();
            }


            const double
                lazy_parallel_wall_ms_v3 =
                    ElapsedMilliseconds(
                        lazy_parallel_wall_begin_v3,
                        std::chrono::
                            steady_clock::
                            now());


            const std::size_t
                lazy_parallel_workers_used_v3 =
                    lazy_parallel_futures_v3.size();


            std::size_t
                lazy_parallel_fallback_calls_v3 =
                    0U;


            // --------------------------------------------------------
            // Full Verify ONLY for this one rescue candidate and ONLY
            // for rescue yaw guesses that pass prescore.
            // --------------------------------------------------------
            for (std::size_t lazy_index_v3 = 0U;
                 lazy_index_v3 < lazy_guesses.size();
                 ++lazy_index_v3)
            {
                const LazyInitialGuess &guess =
                    lazy_guesses[lazy_index_v3];
                ScLazyYawAudit audit;

                audit.name =
                    guess.name;

                audit.yaw_deg =
                    guess.yaw_deg;

                audit.prescore_valid =
                    guess.prescore.valid;

                audit.prescore_overlap =
                    guess.prescore
                        .overlap_ratio;

                audit.prescore_rmse =
                    guess.prescore
                        .rmse;


                if (!guess.prescore.valid ||
                    !std::isfinite(
                        guess.prescore
                            .overlap_ratio) ||
                    guess.prescore
                            .overlap_ratio <
                        sc_prescore_gate)
                {
                    lazy_yaw_audit
                        .push_back(
                            audit);

                    continue;
                }


                // ================================================
                // SC LAZY EXACT VERIFY REUSE SHADOW V1
                //
                // Match by the complete initial transform rather than
                // by hypothesis name.
                //
                // For formal SC-SINGLE the base transform is Identity,
                // therefore matching transforms are semantically the
                // exact same Full-Verify request.
                // ================================================
                const ScFormalInitialGuess *
                    reuse_shadow_normal_guess_v1 =
                        nullptr;

                if (reuse_shadow_sc_single_v1)
                {
                    constexpr double
                        kReuseTransformToleranceV1 =
                            1.0e-12;

                    for (const ScFormalInitialGuess &normal_guess :
                         rescue_selected_job->guesses)
                    {
                        const double transform_max_abs =
                            (
                                normal_guess.transform.matrix() -
                                guess.transform.matrix()
                            )
                                .cwiseAbs()
                                .maxCoeff();

                        if (std::isfinite(transform_max_abs) &&
                            transform_max_abs <=
                                kReuseTransformToleranceV1)
                        {
                            ++reuse_shadow_exact_match_v1;

                            reuse_shadow_normal_guess_v1 =
                                &normal_guess;

                            break;
                        }
                    }
                }

                audit.verify_ran =
                    true;


                LoopVerificationResult trial;

                bool trial_ok =
                    false;

                double verify_ms =
                    0.0;

                bool reused_normal_verify_v2 =
                    false;


                // ================================================
                // SC EXACT VERIFY REUSE ACTIVE V2
                //
                // Reuse ONLY when:
                //   1. formal source is SC-SINGLE
                //   2. exact initial transform matched
                //   3. Normal stage really ran Full Verify
                //
                // Otherwise preserve the original Lazy Verify path.
                // ================================================
                if (reuse_shadow_normal_guess_v1 !=
                        nullptr &&
                    reuse_shadow_normal_guess_v1
                        ->verify_ran)
                {
                    trial =
                        reuse_shadow_normal_guess_v1
                            ->verification;

                    trial_ok =
                        reuse_shadow_normal_guess_v1
                            ->verify_ok;

                    reused_normal_verify_v2 =
                        true;

                    ++reuse_active_calls_v2;
                }
                else
                {
                    const LazyParallelVerifyTaskV3 *
                        parallel_task_v3 =
                            nullptr;


                    for (const LazyParallelVerifyTaskV3 &task :
                         lazy_parallel_tasks_v3)
                    {
                        if (task.guess_index ==
                            lazy_index_v3)
                        {
                            parallel_task_v3 =
                                &task;

                            break;
                        }
                    }


                    if (parallel_task_v3 != nullptr &&
                        parallel_task_v3->verify_ran)
                    {
                        trial =
                            parallel_task_v3
                                ->verification;

                        trial_ok =
                            parallel_task_v3
                                ->verify_ok;

                        verify_ms =
                            parallel_task_v3
                                ->verify_ms;
                    }
                    else
                    {
                        // --------------------------------------------
                        // Defensive fallback.
                        //
                        // Expected count in a correct V3 run: ZERO.
                        // Preserve correctness if task construction
                        // ever misses a request.
                        // --------------------------------------------
                        const auto verify_begin =
                            std::chrono::
                                steady_clock::
                                now();


                        trial_ok =
                            lazy_verifier.Verify(
                                current_keyframe.cloud,
                                rescue_selected_job
                                    ->target_K,
                                guess.transform,
                                trial);


                        verify_ms =
                            ElapsedMilliseconds(
                                verify_begin,
                                std::chrono::
                                    steady_clock::
                                    now());


                        ++lazy_parallel_fallback_calls_v3;
                    }


                    ++lazy_verify_calls;

                    lazy_verify_sum_ms +=
                        verify_ms;
                }


                // ================================================
                // SC LAZY EXACT VERIFY REUSE SHADOW V1
                //
                // Estimate the Lazy Verify wall time that could have
                // been removed if V2 reused the existing Normal result.
                //
                // We still keep the actual Lazy result below.
                // ================================================
                if (reuse_shadow_normal_guess_v1 != nullptr &&
                    reuse_shadow_normal_guess_v1->verify_ran)
                {
                    ++reuse_shadow_available_v1;

                    if (!reused_normal_verify_v2)
                    {
                        reuse_shadow_would_save_ms_v1 +=
                            verify_ms;
                    }

                    const LoopVerificationResult &
                        normal_trial =
                            reuse_shadow_normal_guess_v1
                                ->verification;

                    if (reuse_shadow_normal_guess_v1->verify_ok !=
                            trial_ok ||
                        normal_trial.accepted !=
                            trial.accepted)
                    {
                        ++reuse_shadow_status_mismatch_v1;
                    }

                    if (std::isfinite(
                            normal_trial.overlap_ratio) &&
                        std::isfinite(
                            trial.overlap_ratio))
                    {
                        reuse_shadow_max_overlap_diff_v1 =
                            std::max(
                                reuse_shadow_max_overlap_diff_v1,
                                std::abs(
                                    normal_trial.overlap_ratio -
                                    trial.overlap_ratio));
                    }

                    if (std::isfinite(
                            normal_trial.rmse) &&
                        std::isfinite(
                            trial.rmse))
                    {
                        reuse_shadow_max_rmse_diff_v1 =
                            std::max(
                                reuse_shadow_max_rmse_diff_v1,
                                std::abs(
                                    normal_trial.rmse -
                                    trial.rmse));
                    }

                    if (normal_trial
                            .T_target_source
                            .matrix()
                            .allFinite() &&
                        trial
                            .T_target_source
                            .matrix()
                            .allFinite())
                    {
                        const Eigen::Isometry3d
                            reuse_pose_error_v1 =
                                normal_trial
                                    .T_target_source
                                    .inverse() *
                                trial
                                    .T_target_source;

                        reuse_shadow_max_pose_dt_v1 =
                            std::max(
                                reuse_shadow_max_pose_dt_v1,
                                reuse_pose_error_v1
                                    .translation()
                                    .norm());

                        Eigen::Quaterniond reuse_q_v1(
                            reuse_pose_error_v1.rotation());

                        if (reuse_q_v1
                                .coeffs()
                                .allFinite() &&
                            reuse_q_v1.norm() >
                                1.0e-12)
                        {
                            reuse_q_v1.normalize();

                            const double reuse_w_v1 =
                                std::clamp(
                                    std::abs(
                                        reuse_q_v1.w()),
                                    0.0,
                                    1.0);

                            const double reuse_dr_deg_v1 =
                                2.0 *
                                std::acos(
                                    reuse_w_v1) *
                                180.0 /
                                3.14159265358979323846;

                            reuse_shadow_max_pose_dr_deg_v1 =
                                std::max(
                                    reuse_shadow_max_pose_dr_deg_v1,
                                    reuse_dr_deg_v1);
                        }
                    }
                }


                audit.verify_ok =
                    trial_ok;

                audit.accepted =
                    trial_ok &&
                    trial.accepted;

                audit.overlap =
                    trial.overlap_ratio;

                audit.rmse =
                    trial.rmse;


                if (!audit.accepted)
                {
                    lazy_yaw_audit
                        .push_back(
                            audit);

                    continue;
                }


                audit.graph_valid =
                    compute_graph_error(
                        eligible.front()
                            .historical_id,
                        trial.T_target_source,
                        audit.graph_dt,
                        audit.graph_dr);


                audit.graph_pass =
                    audit.graph_valid &&
                    audit.graph_dt <=
                        0.25 &&
                    audit.graph_dr <=
                        60.0;


                if (audit.graph_valid)
                {
                    audit.graph_score =
                        std::pow(
                            audit.graph_dt /
                                0.25,
                            2.0) +
                        std::pow(
                            audit.graph_dr /
                                60.0,
                            2.0);
                }


                const bool better =
                    !rescue_have_selected_yaw ||

                    (audit.graph_pass &&
                     !rescue_yaw_gate_pass) ||

                    (audit.graph_pass ==
                         rescue_yaw_gate_pass &&
                     audit.graph_score <
                         rescue_yaw_score);


                if (better)
                {
                    rescue_have_selected_yaw =
                        true;

                    rescue_yaw_gate_pass =
                        audit.graph_pass;

                    rescue_yaw_score =
                        audit.graph_score;

                    rescue_selected_yaw_name =
                        guess.name;

                    rescue_selected_verification =
                        trial;

                    rescue_selected_prescore =
                        guess.prescore;

                    rescue_graph_valid =
                        audit.graph_valid;

                    rescue_graph_dt =
                        audit.graph_dt;

                    rescue_graph_dr =
                        audit.graph_dr;
                }


                lazy_yaw_audit
                    .push_back(
                        audit);
            }


            // ========================================================
            // SC LAZY EXACT VERIFY REUSE SHADOW V1
            // ========================================================
            const double
                lazy_parallel_sum_over_wall_v3 =
                    (lazy_parallel_wall_ms_v3 >
                         1.0e-9)
                        ? lazy_verify_sum_ms /
                              lazy_parallel_wall_ms_v3
                        : 0.0;


            std::cout
                << "SC_PARALLEL_VERIFY"
                << " | current_kf="
                << current_q
                << " | historical_kf="
                << eligible.front().historical_id
                << " | workers="
                << lazy_parallel_workers_used_v3
                << " | actual_verify_calls="
                << lazy_verify_calls
                << " | verify_sum_ms="
                << lazy_verify_sum_ms
                << " | verify_wall_ms="
                << lazy_parallel_wall_ms_v3
                << " | sum_over_wall="
                << lazy_parallel_sum_over_wall_v3
                << " | fallback_calls="
                << lazy_parallel_fallback_calls_v3
                << std::endl;


            std::cout
                << "SC_LAZY_REUSE_ACTIVE_V2"
                << " | current_kf="
                << current_q
                << " | historical_kf="
                << eligible.front().historical_id
                << " | reused_verify_calls="
                << reuse_active_calls_v2
                << " | actual_lazy_verify_calls="
                << lazy_verify_calls
                << " | actual_lazy_verify_ms="
                << lazy_verify_sum_ms
                << std::endl;


            std::cout
                << "SC_LAZY_REUSE_SHADOW_V1"
                << " | current_kf="
                << current_q
                << " | historical_kf="
                << eligible.front().historical_id
                << " | sc_single="
                << (reuse_shadow_sc_single_v1 ? 1 : 0)
                << " | lazy_verify_calls="
                << lazy_verify_calls
                << " | exact_transform_matches="
                << reuse_shadow_exact_match_v1
                << " | reusable_verify_calls="
                << reuse_shadow_available_v1
                << " | would_save_ms="
                << reuse_shadow_would_save_ms_v1
                << " | status_mismatch="
                << reuse_shadow_status_mismatch_v1
                << " | max_overlap_diff="
                << reuse_shadow_max_overlap_diff_v1
                << " | max_rmse_diff="
                << reuse_shadow_max_rmse_diff_v1
                << " | max_pose_dt="
                << reuse_shadow_max_pose_dt_v1
                << " | max_pose_dr_deg="
                << reuse_shadow_max_pose_dr_deg_v1
                << std::endl;


            // Mark selected row after final comparison.
            for (ScLazyYawAudit &audit :
                 lazy_yaw_audit)
            {
                audit.selected =
                    rescue_have_selected_yaw &&
                    audit.name ==
                        rescue_selected_yaw_name;
            }
        }


        rescue_graph_like_pass =
            rescue_have_selected_yaw &&
            rescue_yaw_gate_pass;


        // ============================================================
        // FORMAL PROMOTION
        //
        // TrackRecovery itself NEVER adds an edge.
        //
        // Promotion requires:
        //
        //   winner d > 2 m
        //   rescue d <= 1 m
        //   track nodes >= 8
        //   geometry accepted
        //   lazy yaw basin accepted
        //   graph dt <= 0.25 m
        //   graph dR <= 60 deg
        //
        // After promotion the candidate STILL goes through the normal
        // Temporal/Cycle checker and normal graph gate.
        // ============================================================
        const bool
            track_rescue_formal_promoted =
                rescue_found &&
                rescue_selected_job !=
                    nullptr &&
                rescue_have_selected_yaw &&
                rescue_graph_like_pass;


        if (track_rescue_formal_promoted)
        {
            sc_have_selected_candidate =
                true;


            sc_selected_rank =
                rescue_selected_job
                    ->sc_rank;


            sc_selected_candidate =
                rescue_selected_job
                    ->sc_candidate;


            sc_formal_candidate =
                rescue_selected_job
                    ->formal_candidate;


            best_target_K =
                rescue_selected_job
                    ->target_K;


            best_target_keyframes =
                rescue_selected_job
                    ->target_keyframes;


            best_verification =
                rescue_selected_verification;


            sc_best_prescore =
                rescue_selected_prescore;


            sc_best_verify_ok =
                true;


            sc_selected_guess =
                rescue_selected_yaw_name
                    .c_str();


            std::cout
                << "SC_TRACK_RESCUE_LAZY_FORMAL_V1"
                << " | current_kf="
                << current_q
                << " | old_winner_h="
                << winner_h
                << " | old_winner_pose_d="
                << winner_pose_d
                << " | rescue_h="
                << sc_selected_candidate
                       .candidate_id
                << " | rescue_pose_d="
                << eligible.front()
                       .pose_distance
                << " | direction="
                << (eligible.front()
                            .direction > 0
                        ? "SAME"
                        : "REVERSE")
                << " | track_nodes="
                << eligible.front()
                       .nodes
                << " | yaw_guess="
                << rescue_selected_yaw_name
                << " | graph_dt="
                << rescue_graph_dt
                << " | graph_dR="
                << rescue_graph_dr
                << " | overlap="
                << best_verification
                       .overlap_ratio
                << " | rmse="
                << best_verification
                       .rmse
                << " | lazy_prescore_calls="
                << lazy_prescore_calls
                << " | lazy_verify_calls="
                << lazy_verify_calls
                << " | lazy_verify_sum_ms="
                << lazy_verify_sum_ms
                << std::endl;
        }


        // ============================================================
        // FULL LAZY YAW DIAGNOSTIC
        //
        // Keeps enough information for later offline tuning without
        // rerunning the bag.
        // ============================================================
        if (rescue_found &&
            rescue_selected_job != nullptr)
        {
            try
            {
                const std::filesystem::path directory =
                    FrontendLoopDirectory();


                std::filesystem::create_directories(
                    directory);


                const std::filesystem::path csv_path =
                    directory /
                    "sc_track_rescue_lazy_multiyaw_v1.csv";


                const bool exists =
                    std::filesystem::exists(
                        csv_path);


                std::ofstream csv(
                    csv_path,
                    std::ios::app);


                if (csv.is_open())
                {
                    csv
                        << std::fixed
                        << std::setprecision(9);


                    if (!exists)
                    {
                        csv
                            << "current_kf,"
                            << "old_winner_h,"
                            << "old_winner_pose_distance_m,"
                            << "rescue_h,"
                            << "rescue_pose_distance_m,"
                            << "track_direction,"
                            << "track_nodes,"
                            << "yaw_guess,"
                            << "yaw_deg,"
                            << "prescore_valid,"
                            << "prescore_overlap,"
                            << "prescore_rmse,"
                            << "verify_ran,"
                            << "verify_ok,"
                            << "geometry_accepted,"
                            << "overlap,"
                            << "rmse,"
                            << "graph_valid,"
                            << "graph_error_dt_m,"
                            << "graph_error_dR_deg,"
                            << "graph_gate_pass,"
                            << "graph_score,"
                            << "selected_yaw,"
                            << "formal_promoted,"
                            << "lazy_prescore_calls,"
                            << "lazy_verify_calls,"
                            << "lazy_verify_sum_ms"
                            << '\n';
                    }


                    for (const ScLazyYawAudit &audit :
                         lazy_yaw_audit)
                    {
                        csv
                            << current_q << ','
                            << winner_h << ','
                            << winner_pose_d << ','
                            << eligible.front()
                                   .historical_id
                            << ','
                            << eligible.front()
                                   .pose_distance
                            << ','
                            << eligible.front()
                                   .direction
                            << ','
                            << eligible.front()
                                   .nodes
                            << ','
                            << audit.name << ','
                            << audit.yaw_deg << ','
                            << (audit.prescore_valid
                                    ? 1
                                    : 0)
                            << ','
                            << audit.prescore_overlap
                            << ','
                            << audit.prescore_rmse
                            << ','
                            << (audit.verify_ran
                                    ? 1
                                    : 0)
                            << ','
                            << (audit.verify_ok
                                    ? 1
                                    : 0)
                            << ','
                            << (audit.accepted
                                    ? 1
                                    : 0)
                            << ','
                            << audit.overlap
                            << ','
                            << audit.rmse
                            << ','
                            << (audit.graph_valid
                                    ? 1
                                    : 0)
                            << ','
                            << audit.graph_dt
                            << ','
                            << audit.graph_dr
                            << ','
                            << (audit.graph_pass
                                    ? 1
                                    : 0)
                            << ','
                            << audit.graph_score
                            << ','
                            << (audit.selected
                                    ? 1
                                    : 0)
                            << ','
                            << (audit.selected &&
                                track_rescue_formal_promoted
                                    ? 1
                                    : 0)
                            << ','
                            << lazy_prescore_calls
                            << ','
                            << lazy_verify_calls
                            << ','
                            << lazy_verify_sum_ms
                            << '\n';
                    }
                }
            }
            catch (const std::exception &)
            {
                // Diagnostics must never affect SLAM.
            }
        }

        // ------------------------------------------------------------
        // DIAGNOSTIC CSV #1:
        // every confirmed unique causal endpoint.
        // ------------------------------------------------------------
        try
        {
            const std::filesystem::path directory =
                FrontendLoopDirectory();

            std::filesystem::create_directories(
                directory);

            const std::filesystem::path endpoint_path =
                directory /
                "sc_track_rescue_endpoints_shadow_v1.csv";

            const bool exists =
                std::filesystem::exists(
                    endpoint_path);

            std::ofstream csv(
                endpoint_path,
                std::ios::app);

            if (csv.is_open())
            {
                csv
                    << std::fixed
                    << std::setprecision(9);

                if (!exists)
                {
                    csv
                        << "current_kf,"
                        << "historical_kf,"
                        << "direction,"
                        << "track_nodes,"
                        << "track_span,"
                        << "corridor_median,"
                        << "progress,"
                        << "pose_distance_m,"
                        << "overlap,"
                        << "rmse,"
                        << "sc_rank,"
                        << "track_score,"
                        << "alternate_gate_ok,"
                        << "rescue_eligible,"
                        << "selected_rescue,"
                        << "graph_valid,"
                        << "graph_error_dt_m,"
                        << "graph_error_dR_deg,"
                        << "graph_like_pass"
                        << '\n';
                }

                for (const auto &endpoint :
                     confirmed_unique)
                {
                    const bool alt_gate_ok =
                        endpoint.nodes >=
                            kRescueMinNodes &&
                        std::isfinite(
                            endpoint.pose_distance) &&
                        endpoint.pose_distance <=
                            kAlternateMaxDistanceM &&
                        (!have_winner ||
                         endpoint.historical_id !=
                             winner_h);

                    const bool rescue_eligible =
                        rescue_triggered &&
                        alt_gate_ok;

                    const bool selected =
                        rescue_found &&
                        endpoint.historical_id ==
                            eligible.front()
                                .historical_id;

                    double graph_dt =
                        std::numeric_limits<double>::
                            quiet_NaN();

                    double graph_dr =
                        std::numeric_limits<double>::
                            quiet_NaN();

                    const bool graph_valid =
                        compute_graph_error(
                            endpoint.historical_id,
                            endpoint.measurement,
                            graph_dt,
                            graph_dr);

                    const bool graph_pass =
                        graph_valid &&
                        graph_dt <= 0.25 &&
                        graph_dr <= 60.0;

                    csv
                        << current_q << ','
                        << endpoint.historical_id << ','
                        << endpoint.direction << ','
                        << endpoint.nodes << ','
                        << endpoint.span << ','
                        << endpoint.corridor_median << ','
                        << endpoint.progress << ','
                        << endpoint.pose_distance << ','
                        << endpoint.overlap << ','
                        << endpoint.rmse << ','
                        << (endpoint.sc_rank + 1U) << ','
                        << endpoint.score << ','
                        << (alt_gate_ok ? 1 : 0) << ','
                        << (rescue_eligible ? 1 : 0) << ','
                        << (selected ? 1 : 0) << ','
                        << (graph_valid ? 1 : 0) << ','
                        << graph_dt << ','
                        << graph_dr << ','
                        << (graph_pass ? 1 : 0)
                        << '\n';
                }
            }
        }
        catch (const std::exception &)
        {
            // Diagnostics must never affect SLAM.
        }

        // ------------------------------------------------------------
        // DIAGNOSTIC CSV #2:
        // one row per query.
        // ------------------------------------------------------------
        try
        {
            const std::filesystem::path directory =
                FrontendLoopDirectory();

            std::filesystem::create_directories(
                directory);

            const std::filesystem::path query_path =
                directory /
                "sc_track_rescue_query_shadow_v1.csv";

            const bool exists =
                std::filesystem::exists(
                    query_path);

            std::ofstream csv(
                query_path,
                std::ios::app);

            if (csv.is_open())
            {
                csv
                    << std::fixed
                    << std::setprecision(9);

                if (!exists)
                {
                    csv
                        << "current_kf,"
                        << "formal_winner_valid,"
                        << "winner_historical_kf,"
                        << "winner_pose_distance_m,"
                        << "long_geometry_candidates,"
                        << "confirmed_raw_count,"
                        << "confirmed_unique_count,"
                        << "active_same_before,"
                        << "active_reverse_before,"
                        << "active_same_after,"
                        << "active_reverse_after,"
                        << "rescue_triggered,"
                        << "eligible_count,"
                        << "rescue_found,"
                        << "rescue_historical_kf,"
                        << "rescue_direction,"
                        << "rescue_pose_distance_m,"
                        << "rescue_track_nodes,"
                        << "rescue_track_span,"
                        << "rescue_corridor_median,"
                        << "rescue_progress,"
                        << "rescue_overlap,"
                        << "rescue_rmse,"
                        << "rescue_sc_rank,"
                        << "winner_graph_valid,"
                        << "winner_graph_error_dt_m,"
                        << "winner_graph_error_dR_deg,"
                        << "rescue_graph_valid,"
                        << "rescue_graph_error_dt_m,"
                        << "rescue_graph_error_dR_deg,"
                        << "rescue_graph_like_pass"
                        << '\n';
                }

                csv
                    << current_q << ','
                    << (have_winner ? 1 : 0) << ',';

                if (have_winner)
                {
                    csv
                        << winner_h << ','
                        << winner_pose_d << ',';
                }
                else
                {
                    csv
                        << "-1,nan,";
                }

                csv
                    << long_geometry_candidates << ','
                    << confirmed_raw.size() << ','
                    << confirmed_unique.size() << ','
                    << active_same_before << ','
                    << active_reverse_before << ','
                    << sc_track_shadow_active[0].size() << ','
                    << sc_track_shadow_active[1].size() << ','
                    << (rescue_triggered ? 1 : 0) << ','
                    << eligible.size() << ','
                    << (rescue_found ? 1 : 0) << ',';

                if (rescue_found)
                {
                    const auto &best =
                        eligible.front();

                    csv
                        << best.historical_id << ','
                        << best.direction << ','
                        << best.pose_distance << ','
                        << best.nodes << ','
                        << best.span << ','
                        << best.corridor_median << ','
                        << best.progress << ','
                        << best.overlap << ','
                        << best.rmse << ','
                        << (best.sc_rank + 1U) << ',';
                }
                else
                {
                    csv
                        << "-1,0,nan,0,0,nan,nan,nan,nan,0,";
                }

                csv
                    << (winner_graph_valid ? 1 : 0) << ','
                    << winner_graph_dt << ','
                    << winner_graph_dr << ','
                    << (rescue_graph_valid ? 1 : 0) << ','
                    << rescue_graph_dt << ','
                    << rescue_graph_dr << ','
                    << (rescue_graph_like_pass ? 1 : 0)
                    << '\n';
            }
        }
        catch (const std::exception &)
        {
            // Diagnostics must never affect SLAM.
        }

        // Console only for suspicious formal winners.
        if (rescue_triggered)
        {
            std::cout
                << "SC_TRACK_RESCUE_SHADOW_V1"
                << " | current_kf="
                << current_q
                << " | winner_h="
                << winner_h
                << " | winner_pose_d="
                << winner_pose_d
                << " | confirmed_unique="
                << confirmed_unique.size()
                << " | eligible="
                << eligible.size()
                << " | rescue_found="
                << (rescue_found ? 1 : 0);

            if (rescue_found)
            {
                const auto &best =
                    eligible.front();

                std::cout
                    << " | rescue_h="
                    << best.historical_id
                    << " | direction="
                    << (best.direction > 0
                            ? "SAME"
                            : "REVERSE")
                    << " | rescue_pose_d="
                    << best.pose_distance
                    << " | track_nodes="
                    << best.nodes
                    << " | track_span="
                    << best.span
                    << " | corridor="
                    << best.corridor_median
                    << " | progress="
                    << best.progress
                    << " | overlap="
                    << best.overlap
                    << " | rmse="
                    << best.rmse
                    << " | sc_rank="
                    << (best.sc_rank + 1U)
                    << " | shadow_graph_valid="
                    << (rescue_graph_valid ? 1 : 0)
                    << " | shadow_graph_dt="
                    << rescue_graph_dt
                    << " | shadow_graph_dR="
                    << rescue_graph_dr
                    << " | shadow_graph_pass="
                    << (rescue_graph_like_pass ? 1 : 0);
            }

            std::cout
                << std::endl;
        }
    }

    // ================================================================
    // SC V2 TRACK RECOVERY SHADOW V1
    //
    // DIAGNOSTIC ONLY.
    //
    // Offline-established reverse-track anchor for this experiment:
    //
    //     current KF 442 -> historical KF 31
    //
    // with approximate signed slope:
    //
    //     dh / dc = -1
    //
    // Therefore:
    //
    //     predicted historical KF = 31 - (current_kf - 442)
    //                             = 473 - current_kf
    //
    // IMPORTANT:
    //   * No result from this block enters formal candidate selection.
    //   * Does NOT modify sc_parallel_jobs.
    //   * Does NOT modify best_verification.
    //   * Does NOT call LoopConsistencyChecker.
    //   * Does NOT add PoseGraph edges.
    //   * Does NOT affect PGO.
    //
    // This is ONLY a controlled shadow experiment for:
    //     KF448
    //     KF452
    //     KF458 ... KF462
    //
    // Recovery corridor:
    //     predicted + {-4,-2,0,+2,+4}
    //
    // Since recovery candidates have no Scan Context yaw estimate,
    // diagnostic coarse-yaw hypotheses are:
    //     0, +90, 180, -90 deg
    // ================================================================
    const bool sc_track_recovery_shadow_query =
        kEnableExperimentalLoopShadows &&
        (current_keyframe.id == 448 ||
        current_keyframe.id == 452 ||
        (current_keyframe.id >= 458 &&
          current_keyframe.id <= 462));

    if (sc_track_recovery_shadow_query)
    {
        try
        {
            const long long predicted_historical_kf =
                473LL -
                static_cast<long long>(
                    current_keyframe.id);

            const std::filesystem::path
                recovery_directory =
                    FrontendLoopDirectory();

            std::filesystem::create_directories(
                recovery_directory);

            const std::filesystem::path
                recovery_csv_path =
                    recovery_directory /
                    "sc_v2_track_recovery_shadow.csv";

            const bool recovery_csv_exists =
                std::filesystem::exists(
                    recovery_csv_path);

            std::ofstream recovery_csv(
                recovery_csv_path,
                std::ios::app);

            if (recovery_csv.is_open())
            {
                recovery_csv
                    << std::fixed
                    << std::setprecision(9);

                if (!recovery_csv_exists)
                {
                    recovery_csv
                        << "current_kf,"
                        << "predicted_historical_kf,"
                        << "test_historical_kf,"
                        << "offset,"
                        << "target_built,"
                        << "target_keyframes,"
                        << "target_points,"
                        << "best_prescore_guess,"
                        << "best_prescore_valid,"
                        << "best_prescore_overlap,"
                        << "best_prescore_rmse,"
                        << "verify_ran,"
                        << "verify_ok,"
                        << "geometry_accepted,"
                        << "best_verify_guess,"
                        << "overlap,"
                        << "rmse,"
                        << "correction_translation_m,"
                        << "correction_rotation_deg"
                        << '\n';
                }

                const int recovery_offsets[] =
                {
                    -4,
                    -2,
                    0,
                    2,
                    4
                };

                constexpr double kRecoveryPi =
                    3.14159265358979323846;

                const double recovery_yaw_deg[] =
                {
                    0.0,
                    90.0,
                    180.0,
                    -90.0
                };

                const char *recovery_yaw_name[] =
                {
                    "RECOVERY_YAW_0",
                    "RECOVERY_YAW_POS_90",
                    "RECOVERY_YAW_180",
                    "RECOVERY_YAW_NEG_90"
                };

                // Independent verifier:
                // recovery diagnostics must not share production caches.
                LoopVerifier recovery_verifier(
                    loop_verifier_.GetConfig());

                const double recovery_prescore_gate =
                    recovery_verifier
                        .GetConfig()
                        .prescore_min_overlap_ratio;

                const double recovery_nan =
                    std::numeric_limits<double>::
                        quiet_NaN();

                for (const int recovery_offset :
                     recovery_offsets)
                {
                    const long long recovery_hist_signed =
                        predicted_historical_kf +
                        static_cast<long long>(
                            recovery_offset);

                    if (recovery_hist_signed < 0)
                    {
                        continue;
                    }

                    const std::size_t recovery_hist =
                        static_cast<std::size_t>(
                            recovery_hist_signed);

                    if (recovery_hist >=
                        current_keyframe.id)
                    {
                        continue;
                    }

                    pcl::PointCloud<LIDAR_POINT>::Ptr
                        recovery_target_K;

                    std::vector<std::size_t>
                        recovery_target_keyframes;

                    const bool recovery_target_built =
                        BuildCandidateCenteredHistoricalTarget(
                            recovery_hist,
                            recovery_target_K,
                            &recovery_target_keyframes);

                    const std::size_t recovery_target_points =
                        recovery_target_K
                            ? recovery_target_K->size()
                            : 0;

                    if (!recovery_target_built ||
                        !recovery_target_K ||
                        recovery_target_K->empty())
                    {
                        recovery_csv
                            << current_keyframe.id << ','
                            << predicted_historical_kf << ','
                            << recovery_hist << ','
                            << recovery_offset << ','
                            << 0 << ','
                            << recovery_target_keyframes.size() << ','
                            << recovery_target_points << ','
                            << "NONE,"
                            << 0 << ','
                            << recovery_nan << ','
                            << recovery_nan << ','
                            << 0 << ','
                            << 0 << ','
                            << 0 << ','
                            << "NONE,"
                            << recovery_nan << ','
                            << recovery_nan << ','
                            << recovery_nan << ','
                            << recovery_nan
                            << '\n';

                        continue;
                    }

                    std::vector<ScFormalInitialGuess>
                        recovery_guesses;

                    recovery_guesses.reserve(4);

                    for (std::size_t yaw_index = 0;
                         yaw_index < 4;
                         ++yaw_index)
                    {
                        ScFormalInitialGuess guess;

                        guess.name =
                            recovery_yaw_name[yaw_index];

                        guess.transform =
                            Eigen::Isometry3d::Identity();

                        const double yaw_rad =
                            recovery_yaw_deg[yaw_index] *
                            kRecoveryPi /
                            180.0;

                        guess.transform.linear() =
                            Eigen::AngleAxisd(
                                yaw_rad,
                                Eigen::Vector3d::UnitZ())
                                .toRotationMatrix();

                        guess.score_ok =
                            recovery_verifier
                                .ScoreInitialGuess(
                                    current_keyframe.cloud,
                                    recovery_target_K,
                                    guess.transform,
                                    guess.prescore);

                        guess.prescore.valid =
                            guess.score_ok &&
                            guess.prescore.valid;

                        recovery_guesses.push_back(
                            guess);
                    }

                    std::sort(
                        recovery_guesses.begin(),
                        recovery_guesses.end(),
                        [](const ScFormalInitialGuess &lhs,
                           const ScFormalInitialGuess &rhs)
                        {
                            if (lhs.prescore.valid !=
                                rhs.prescore.valid)
                            {
                                return lhs.prescore.valid;
                            }

                            if (lhs.prescore.overlap_ratio !=
                                rhs.prescore.overlap_ratio)
                            {
                                return
                                    lhs.prescore.overlap_ratio >
                                    rhs.prescore.overlap_ratio;
                            }

                            return
                                lhs.prescore.rmse <
                                rhs.prescore.rmse;
                        });

                    bool recovery_prescore_valid =
                        false;

                    const char *
                        recovery_best_prescore_guess =
                            "NONE";

                    double recovery_best_prescore_overlap =
                        recovery_nan;

                    double recovery_best_prescore_rmse =
                        recovery_nan;

                    for (const ScFormalInitialGuess &guess :
                         recovery_guesses)
                    {
                        if (!guess.prescore.valid ||
                            !std::isfinite(
                                guess.prescore.overlap_ratio) ||
                            !std::isfinite(
                                guess.prescore.rmse))
                        {
                            continue;
                        }

                        recovery_prescore_valid =
                            true;

                        recovery_best_prescore_guess =
                            guess.name;

                        recovery_best_prescore_overlap =
                            guess.prescore.overlap_ratio;

                        recovery_best_prescore_rmse =
                            guess.prescore.rmse;

                        break;
                    }

                    bool recovery_verify_ran =
                        false;

                    bool recovery_have_verification =
                        false;

                    bool recovery_best_verify_ok =
                        false;

                    const char *
                        recovery_best_verify_guess =
                            "NONE";

                    LoopVerificationResult
                        recovery_best_verification;

                    // ------------------------------------------------
                    // Diagnostic mode intentionally evaluates ALL
                    // coarse-yaw hypotheses that pass prescore.
                    //
                    // We want to answer:
                    // "Does usable geometry exist in this predicted
                    // historical corridor at all?"
                    //
                    // This does NOT mimic production promotion and
                    // cannot affect production state.
                    // ------------------------------------------------
                    for (const ScFormalInitialGuess &guess :
                         recovery_guesses)
                    {
                        if (!guess.prescore.valid ||
                            !std::isfinite(
                                guess.prescore.overlap_ratio) ||
                            guess.prescore.overlap_ratio <
                                recovery_prescore_gate)
                        {
                            continue;
                        }

                        recovery_verify_ran =
                            true;

                        LoopVerificationResult
                            recovery_trial;

                        const bool recovery_trial_ok =
                            recovery_verifier.Verify(
                                current_keyframe.cloud,
                                recovery_target_K,
                                guess.transform,
                                recovery_trial);

                        bool recovery_trial_better =
                            !recovery_have_verification;

                        if (!recovery_trial_better &&
                            recovery_trial.accepted !=
                                recovery_best_verification.accepted)
                        {
                            recovery_trial_better =
                                recovery_trial.accepted;
                        }
                        else if (
                            !recovery_trial_better &&
                            recovery_trial.accepted ==
                                recovery_best_verification.accepted &&
                            recovery_trial.overlap_ratio >
                                recovery_best_verification
                                        .overlap_ratio +
                                    1.0e-9)
                        {
                            recovery_trial_better =
                                true;
                        }
                        else if (
                            !recovery_trial_better &&
                            recovery_trial.accepted ==
                                recovery_best_verification.accepted &&
                            std::abs(
                                recovery_trial.overlap_ratio -
                                recovery_best_verification
                                    .overlap_ratio) <=
                                1.0e-9 &&
                            recovery_trial.rmse <
                                recovery_best_verification.rmse)
                        {
                            recovery_trial_better =
                                true;
                        }

                        if (recovery_trial_better)
                        {
                            recovery_have_verification =
                                true;

                            recovery_best_verify_ok =
                                recovery_trial_ok;

                            recovery_best_verify_guess =
                                guess.name;

                            recovery_best_verification =
                                recovery_trial;
                        }
                    }

                    const bool recovery_geometry_accepted =
                        recovery_have_verification &&
                        recovery_best_verify_ok &&
                        recovery_best_verification.accepted;

                    recovery_csv
                        << current_keyframe.id << ','
                        << predicted_historical_kf << ','
                        << recovery_hist << ','
                        << recovery_offset << ','
                        << 1 << ','
                        << recovery_target_keyframes.size() << ','
                        << recovery_target_points << ','
                        << recovery_best_prescore_guess << ','
                        << (recovery_prescore_valid ? 1 : 0) << ','
                        << recovery_best_prescore_overlap << ','
                        << recovery_best_prescore_rmse << ','
                        << (recovery_verify_ran ? 1 : 0) << ','
                        << (recovery_best_verify_ok ? 1 : 0) << ','
                        << (recovery_geometry_accepted ? 1 : 0) << ','
                        << recovery_best_verify_guess << ',';

                    if (recovery_have_verification)
                    {
                        recovery_csv
                            << recovery_best_verification
                                   .overlap_ratio
                            << ','
                            << recovery_best_verification.rmse
                            << ','
                            << recovery_best_verification
                                   .correction_translation
                            << ','
                            << recovery_best_verification
                                   .correction_rotation_deg;
                    }
                    else
                    {
                        recovery_csv
                            << recovery_nan << ','
                            << recovery_nan << ','
                            << recovery_nan << ','
                            << recovery_nan;
                    }

                    recovery_csv
                        << '\n';
                }
            }
        }
        catch (const std::exception &)
        {
            // Shadow diagnostics must never affect production SLAM.
        }
    }

    if (!sc_have_selected_candidate ||
        !sc_best_verify_ok ||
        !best_verification.accepted)
    {
        const double sc_v2_total_ms =
            ElapsedMilliseconds(
                sc_v2_total_start,
                std::chrono::steady_clock::now());

        std::cout
            << "LOOP_FORMAL_TIMING"
            << " | current_kf="
            << current_keyframe.id
            << " | topk="
            << formal_loop_candidates.size()
            << " | target_attempts="
            << sc_v2_target_attempts
            << " | targets_built="
            << sc_v2_targets_built
            << " | target_ms="
            << sc_v2_target_ms
            << " | prescore_calls="
            << sc_v2_prescore_calls
            << " | prescore_ms="
            << sc_v2_prescore_ms
            << " | verify_calls="
            << sc_v2_verify_calls
            << " | verify_ms="
            << sc_v2_verify_ms
            << " | total_ms="
            << sc_v2_total_ms
            << " | winner=0"
            << std::endl;

        return;
    }

    const double sc_v2_total_ms =
        ElapsedMilliseconds(
            sc_v2_total_start,
            std::chrono::steady_clock::now());

    const double sc_v2_other_ms =
        sc_v2_total_ms -
        sc_v2_target_ms -
        sc_v2_prescore_ms -
        sc_v2_verify_ms;

    std::cout
        << "LOOP_FORMAL_TIMING"
        << " | current_kf="
        << current_keyframe.id
        << " | topk="
        << formal_loop_candidates.size()
        << " | target_attempts="
        << sc_v2_target_attempts
        << " | targets_built="
        << sc_v2_targets_built
        << " | target_ms="
        << sc_v2_target_ms
        << " | prescore_calls="
        << sc_v2_prescore_calls
        << " | prescore_ms="
        << sc_v2_prescore_ms
        << " | verify_calls="
        << sc_v2_verify_calls
        << " | verify_ms="
        << sc_v2_verify_ms
        << " | other_ms="
        << sc_v2_other_ms
        << " | total_ms="
        << sc_v2_total_ms
        << " | selected_rank="
        << (sc_selected_rank + 1)
        << " | winner=1"
        << std::endl;

    const LoopCandidate *
        best_candidate =
            &sc_formal_candidate;

    loop_timing.geometry_accepted =
        true;

    // ----------------------------------------------------------------
    // Keep one winner-level geometry trace for compatibility with the
    // existing formal diagnostic stream.
    // ----------------------------------------------------------------
    {
        const auto &verifier_config =
            loop_verifier_.GetConfig();

        LoopVerificationTraceRow trace;

        trace.current_kf =
            current_keyframe.id;

        trace.historical_kf =
            best_candidate->candidate_id;

        trace.retrieval_score =
            best_candidate->retrieval_score;

        trace.stage =
            "LOOP_GEOMETRY";

        trace.prescore_valid =
            sc_best_prescore.valid;

        trace.prescore_overlap =
            sc_best_prescore.overlap_ratio;

        trace.prescore_rmse =
            sc_best_prescore.rmse;

        trace.verify_ok =
            sc_best_verify_ok;

        trace.geometry_accepted =
            true;

        trace.overlap =
            best_verification.overlap_ratio;

        trace.rmse =
            best_verification.rmse;

        trace.correction_translation =
            best_verification
                .correction_translation;

        trace.correction_rotation_deg =
            best_verification
                .correction_rotation_deg;

        trace.gate_overlap_pass =
            std::isfinite(
                best_verification.overlap_ratio) &&
            best_verification.overlap_ratio >=
                verifier_config.min_overlap_ratio;

        trace.gate_rmse_pass =
            std::isfinite(
                best_verification.rmse) &&
            best_verification.rmse <=
                verifier_config.max_rmse;

        trace.gate_translation_pass =
            std::isfinite(
                best_verification
                    .correction_translation) &&
            best_verification
                    .correction_translation <=
                verifier_config
                    .max_correction_translation;

        trace.gate_rotation_pass =
            std::isfinite(
                best_verification
                    .correction_rotation_deg) &&
            best_verification
                    .correction_rotation_deg <=
                verifier_config
                    .max_correction_rotation_deg;

        trace.decision =
            "LOOP_GEOMETRY_PASS";

        append_loop_verification_trace(
            trace);
    }

    // ================================================================
    // SC PRESCORE AUDIT V2.5
    //
    // Rank candidate-level prescores AFTER Full Verify has already
    // selected the production geometry winner.
    //
    // This is counterfactual diagnostics only:
    //
    //   "If we had retained Prescore Top-M only, would the actual
    //    Full-Verify winner have survived?"
    // ================================================================
    std::vector<ScPrescoreAuditEntry>
        sc_prescore_sorted =
            sc_prescore_audit;

    std::stable_sort(
        sc_prescore_sorted.begin(),
        sc_prescore_sorted.end(),
        [](const ScPrescoreAuditEntry &lhs,
           const ScPrescoreAuditEntry &rhs)
        {
            if (lhs.valid != rhs.valid)
            {
                return lhs.valid;
            }

            if (!lhs.valid)
            {
                return lhs.sc_rank <
                       rhs.sc_rank;
            }

            if (lhs.overlap != rhs.overlap)
            {
                return lhs.overlap >
                       rhs.overlap;
            }

            if (lhs.rmse != rhs.rmse)
            {
                return lhs.rmse <
                       rhs.rmse;
            }

            return lhs.sc_rank <
                   rhs.sc_rank;
        });

    std::size_t sc_winner_prescore_rank =
        0;

    double sc_winner_prescore_overlap =
        0.0;

    double sc_winner_prescore_rmse =
        std::numeric_limits<double>::infinity();

    const char *
        sc_winner_prescore_guess =
            "NONE";

    for (std::size_t i = 0;
         i < sc_prescore_sorted.size();
         ++i)
    {
        const ScPrescoreAuditEntry &entry =
            sc_prescore_sorted[i];

        if (!entry.valid)
        {
            continue;
        }

        if (entry.sc_rank ==
            sc_selected_rank)
        {
            sc_winner_prescore_rank =
                i + 1;

            sc_winner_prescore_overlap =
                entry.overlap;

            sc_winner_prescore_rmse =
                entry.rmse;

            sc_winner_prescore_guess =
                entry.guess_name;

            break;
        }
    }

    std::size_t sc_prescore_top1_kf =
        0;

    std::size_t sc_prescore_top1_sc_rank =
        0;

    double sc_prescore_top1_overlap =
        0.0;

    double sc_prescore_top1_rmse =
        std::numeric_limits<double>::infinity();

    const char *
        sc_prescore_top1_guess =
            "NONE";

    bool sc_prescore_have_top1 =
        false;

    for (const ScPrescoreAuditEntry &entry :
         sc_prescore_sorted)
    {
        if (!entry.valid)
        {
            continue;
        }

        sc_prescore_have_top1 =
            true;

        sc_prescore_top1_kf =
            entry.candidate_id;

        sc_prescore_top1_sc_rank =
            entry.sc_rank + 1;

        sc_prescore_top1_overlap =
            entry.overlap;

        sc_prescore_top1_rmse =
            entry.rmse;

        sc_prescore_top1_guess =
            entry.guess_name;

        break;
    }

    std::cout
        << "LOOP_PRESCORE_AUDIT"
        << " | current_kf="
        << current_keyframe.id
        << " | topk="
        << formal_loop_candidates.size()
        << " | prescore_candidates="
        << sc_prescore_audit.size()
        << " | winner_kf="
        << sc_selected_candidate.candidate_id
        << " | winner_sc_rank="
        << (sc_selected_rank + 1)
        << " | winner_prescore_rank="
        << sc_winner_prescore_rank
        << " | top1_hit="
        << ((sc_winner_prescore_rank == 1) ? 1 : 0)
        << " | top2_hit="
        << ((sc_winner_prescore_rank > 0 &&
             sc_winner_prescore_rank <= 2)
                ? 1
                : 0)
        << " | top3_hit="
        << ((sc_winner_prescore_rank > 0 &&
             sc_winner_prescore_rank <= 3)
                ? 1
                : 0)
        << " | top5_hit="
        << ((sc_winner_prescore_rank > 0 &&
             sc_winner_prescore_rank <= 5)
                ? 1
                : 0)
        << " | winner_prescore_overlap="
        << sc_winner_prescore_overlap
        << " | winner_prescore_rmse="
        << sc_winner_prescore_rmse
        << " | winner_prescore_guess="
        << sc_winner_prescore_guess
        << " | prescore_top1_valid="
        << (sc_prescore_have_top1 ? 1 : 0)
        << " | prescore_top1_kf="
        << sc_prescore_top1_kf
        << " | prescore_top1_sc_rank="
        << sc_prescore_top1_sc_rank
        << " | prescore_top1_overlap="
        << sc_prescore_top1_overlap
        << " | prescore_top1_rmse="
        << sc_prescore_top1_rmse
        << " | prescore_top1_guess="
        << sc_prescore_top1_guess
        << " | verify_winner_overlap="
        << best_verification.overlap_ratio
        << " | verify_winner_rmse="
        << best_verification.rmse
        << std::endl;

    std::cout
        << "LOOP_FORMAL"
        << " | current_kf="
        << current_keyframe.id
        << " | historical_kf="
        << best_candidate->candidate_id
        << " | selected_rank="
        << (sc_selected_rank + 1)
        << " | topk="
        << formal_loop_candidates.size()
        << " | sc_distance="
        << sc_selected_candidate.scan_context_distance
        << " | sc_similarity="
        << sc_selected_candidate.scan_context_similarity
        << " | sc_yaw_deg="
        << sc_selected_candidate.yaw_hint_deg
        << " | initial_guess="
        << sc_selected_guess
        << " | overlap="
        << best_verification.overlap_ratio
        << " | rmse="
        << best_verification.rmse
        << std::endl;

    // ====================================================================
    // Generic temporal/cycle consistency after geometry.
    // BTC retrieval itself never bypasses this stage.
    // ====================================================================
    LoopConsistencyProposal proposal;
    proposal.current_id = current_keyframe.id;
    proposal.historical_id = best_candidate->candidate_id;
    proposal.T_historical_current =
        best_verification.T_target_source;
    proposal.rmse = best_verification.rmse;
    proposal.overlap_ratio = best_verification.overlap_ratio;

    const LoopConsistencyResult consistency =
        loop_consistency_checker_.Check(
            pose_graph_,
            proposal);


    if (!consistency.accepted)
    {
        // ------------------------------------------------------------
        // SC-SINGLE temporal/cycle confirmation.
        //
        // A geometry-valid proposal with no temporal neighbour is a
        // PENDING SEED only.  LoopConsistencyChecker stores the seed
        // internally, but it MUST NOT enter the PoseGraph yet.
        // ------------------------------------------------------------
        const bool pending_seed =
            consistency.valid &&
            !consistency.temporal_available;

        {
            LoopVerificationTraceRow trace;

            trace.current_kf =
                proposal.current_id;

            trace.historical_kf =
                proposal.historical_id;

            trace.retrieval_score =
                best_candidate->retrieval_score;

            trace.stage =
                "CONSISTENCY";

            trace.verify_ok = true;
            trace.geometry_accepted = true;

            trace.overlap =
                best_verification.overlap_ratio;

            trace.rmse =
                best_verification.rmse;

            trace.correction_translation =
                best_verification
                    .correction_translation;

            trace.correction_rotation_deg =
                best_verification
                    .correction_rotation_deg;

            trace.consistency_valid =
                consistency.valid;

            trace.temporal_available =
                consistency.temporal_available;

            trace.temporal_support =
                consistency.temporal_support;

            trace.temporal_consistent =
                consistency.temporal_consistent;

            trace.cycle_available =
                consistency.cycle_available;

            trace.cycle_translation_error =
                consistency
                    .cycle_translation_error;

            trace.cycle_rotation_error_deg =
                consistency
                    .cycle_rotation_error_deg;

            trace.cycle_consistent =
                consistency.cycle_consistent;

            trace.consistency_accepted =
                false;

            trace.pending_seed =
                pending_seed;

            trace.decision =
                pending_seed
                    ? "PENDING_SEED"
                    : "CONSISTENCY_REJECT";

            append_loop_verification_trace(
                trace);
        }

        std::cout
            << "LOOP_CONSISTENCY_WAIT"
            << " | current_kf="
            << proposal.current_id
            << " | historical_kf="
            << proposal.historical_id
            << " | temporal_available="
            << (consistency.temporal_available ? 1 : 0)
            << " | temporal_support="
            << consistency.temporal_support
            << " | temporal_consistent="
            << (consistency.temporal_consistent ? 1 : 0)
            << " | cycle_available="
            << (consistency.cycle_available ? 1 : 0)
            << " | cycle_consistent="
            << (consistency.cycle_consistent ? 1 : 0)
            << std::endl;

        return;
    }

    {
        LoopVerificationTraceRow trace;

        trace.current_kf =
            proposal.current_id;
        trace.historical_kf =
            proposal.historical_id;
        trace.retrieval_score =
            best_candidate->retrieval_score;

        trace.stage =
            "CONSISTENCY";

        trace.verify_ok = true;
        trace.geometry_accepted = true;

        trace.overlap =
            best_verification.overlap_ratio;
        trace.rmse =
            best_verification.rmse;
        trace.correction_translation =
            best_verification.correction_translation;
        trace.correction_rotation_deg =
            best_verification.correction_rotation_deg;

        trace.consistency_valid =
            consistency.valid;

        trace.temporal_available =
            consistency.temporal_available;
        trace.temporal_support =
            consistency.temporal_support;
        trace.temporal_consistent =
            consistency.temporal_consistent;

        trace.cycle_available =
            consistency.cycle_available;
        trace.cycle_translation_error =
            consistency.cycle_translation_error;
        trace.cycle_rotation_error_deg =
            consistency.cycle_rotation_error_deg;
        trace.cycle_consistent =
            consistency.cycle_consistent;

        trace.consistency_accepted =
            consistency.accepted;

        trace.decision =
            "CONSISTENCY_PASS";

        append_loop_verification_trace(
            trace);
    }

    // SC-SINGLE temporal/cycle consistency has passed.
    // Continue to spacing, graph-relative gate, and PGO.

    // Avoid adding several highly correlated loop factors from nearly the
    // same local revisit segment.
    if (has_last_online_loop_edge_)
    {
        const std::size_t current_gap =
            current_keyframe.id >= last_online_loop_current_keyframe_id_
                ? current_keyframe.id - last_online_loop_current_keyframe_id_
                : last_online_loop_current_keyframe_id_ - current_keyframe.id;

        const std::size_t historical_gap =
            best_candidate->candidate_id >= last_online_loop_historical_keyframe_id_
                ? best_candidate->candidate_id - last_online_loop_historical_keyframe_id_
                : last_online_loop_historical_keyframe_id_ - best_candidate->candidate_id;

        if (current_gap < min_online_loop_edge_current_keyframe_spacing_ ||
            historical_gap < min_online_loop_edge_historical_keyframe_spacing_)
        {
            return;
        }
    }

    if (!pose_graph_.HasNode(best_candidate->candidate_id) ||
        !pose_graph_.HasNode(current_keyframe.id))
    {
        return;
    }

    Eigen::Matrix<double, 6, 6> loop_information =
        Eigen::Matrix<double, 6, 6>::Identity();

    std::size_t shadow_correspondences = 0;
    double median_range =
        std::numeric_limits<double>::quiet_NaN();
    double min_relative =
        std::numeric_limits<double>::quiet_NaN();

    const bool dynamic_information =
        best_target_K &&
        BuildLoopShadowInformationFull6x6(
            current_keyframe.cloud,
            best_target_K,
            best_verification.T_target_source,
            loop_verifier_.GetConfig(),
            loop_information,
            shadow_correspondences,
            median_range,
            min_relative);

    const std::size_t historical_id =
        best_candidate->candidate_id;
    const std::size_t current_id =
        current_keyframe.id;
    const Eigen::Isometry3d measurement =
        best_verification.T_target_source;

    const std::vector<PoseGraphNode> pose_snapshot =
        pose_graph_.GetNodes();

    // ================================================================
    // SC-SINGLE GRAPH-RELATIVE GROSS-ALIAS GATE V1
    //
    // Compare the verified loop measurement against the relative pose
    // currently implied by the PoseGraph BEFORE inserting the loop.
    //
    // This gate is intentionally loose.  It only rejects catastrophic
    // contradictions such as the ~180 deg agricultural-row aliases seen
    // in the BTC D-run.
    // ================================================================
    const PoseGraphNode *
        historical_graph_node =
            nullptr;

    const PoseGraphNode *
        current_graph_node =
            nullptr;

    for (const PoseGraphNode &node :
         pose_snapshot)
    {
        if (node.id == historical_id)
        {
            historical_graph_node =
                &node;
        }

        if (node.id == current_id)
        {
            current_graph_node =
                &node;
        }
    }

    if (historical_graph_node == nullptr ||
        current_graph_node == nullptr ||
        !historical_graph_node->T_WK
             .matrix()
             .allFinite() ||
        !current_graph_node->T_WK
             .matrix()
             .allFinite())
    {
        return;
    }

    const Eigen::Isometry3d
        graph_predicted_measurement =
            historical_graph_node
                ->T_WK.inverse() *
            current_graph_node
                ->T_WK;

    const Eigen::Isometry3d
        graph_measurement_error =
            graph_predicted_measurement
                .inverse() *
            measurement;

    if (!graph_measurement_error
             .matrix()
             .allFinite())
    {
        return;
    }

    const double
        graph_error_translation_m =
            graph_measurement_error
                .translation()
                .norm();

    Eigen::Quaterniond
        graph_error_quaternion(
            graph_measurement_error
                .rotation());

    if (!graph_error_quaternion
             .coeffs()
             .allFinite() ||
        graph_error_quaternion.norm() <
            1.0e-12)
    {
        return;
    }

    graph_error_quaternion.normalize();

    const double
        graph_error_w =
            std::clamp(
                std::abs(
                    graph_error_quaternion.w()),
                0.0,
                1.0);

    const double
        graph_error_rotation_deg =
            2.0 *
            std::acos(
                graph_error_w) *
            180.0 /
            3.14159265358979323846;

    constexpr double
        kScSingleMaxGraphErrorTranslationM =
            0.25;

    constexpr double
        kScSingleMaxGraphErrorRotationDeg =
            60.0;

    if (!std::isfinite(
            graph_error_translation_m) ||
        !std::isfinite(
            graph_error_rotation_deg) ||
        graph_error_translation_m >
            kScSingleMaxGraphErrorTranslationM ||
        graph_error_rotation_deg >
            kScSingleMaxGraphErrorRotationDeg)
    {
        std::cerr
            << "LOOP_GRAPH_REJECT"
            << " | from_kf="
            << historical_id
            << " | to_kf="
            << current_id
            << " | graph_error_dt="
            << graph_error_translation_m
            << " m"
            << " | graph_error_dR="
            << graph_error_rotation_deg
            << " deg"
            << std::endl;

        return;
    }

    if (!pose_graph_.AddLoopEdge(
            historical_id,
            current_id,
            measurement,
            loop_information))
    {
        return;
    }

    double loop_max_offdiag = 0.0;
    double loop_max_tr_coupling = 0.0;
    ComputeLoopInformationStats(
        loop_information,
        loop_max_offdiag,
        loop_max_tr_coupling);


    // ================================================================
    // BACKEND PGO TIMING V1
    //
    // Diagnostic only.
    //
    // Keep the existing per-edge G2O + rollback semantics completely
    // unchanged while measuring where backend time is actually spent.
    // ================================================================
    const std::chrono::steady_clock::time_point
        backend_accept_start_v1 =
            std::chrono::steady_clock::now();

    PoseGraphOptimizationResult optimization_result;

    const std::chrono::steady_clock::time_point pgo_start =
        std::chrono::steady_clock::now();

    const bool pgo_ok =
        pose_graph_optimizer_.Optimize(
            pose_graph_,
            optimization_result);

    const double backend_pgo_ms_v1 =
        ElapsedMilliseconds(
            pgo_start,
            std::chrono::steady_clock::now());

    loop_timing.pose_graph_optimize_ms +=
        backend_pgo_ms_v1;

    ++loop_timing.pose_graph_optimize_calls;

    const bool update_guard_passed =
        pgo_ok &&
        std::isfinite(optimization_result.max_translation_update) &&
        std::isfinite(optimization_result.max_rotation_update_deg) &&
        optimization_result.max_translation_update <=
            online_loop_pgo_max_translation_update_ &&
        optimization_result.max_rotation_update_deg <=
            online_loop_pgo_max_rotation_update_deg_;

    if (!update_guard_passed)
    {
        for (const PoseGraphNode &node : pose_snapshot)
        {
            pose_graph_.SetNodePose(node.id, node.T_WK);
        }

        pose_graph_.RemoveLoopEdge(historical_id, current_id);

        std::cerr
            << "SC_WINDOW PoseGraph optimization rejected"
            << " | from_kf=" << historical_id
            << " | to_kf=" << current_id
            << " | optimizer_ok=" << (pgo_ok ? "true" : "false")
            << " | max_update_dt="
            << optimization_result.max_translation_update << " m"
            << " | max_update_dR="
            << optimization_result.max_rotation_update_deg << " deg"
            << " | action=ROLLBACK"
            << std::endl;
        return;
    }

    loop_timing.optimization_accepted = true;
    loop_timing.loop_edge_accepted = true;

    has_last_online_loop_edge_ = true;
    last_online_loop_current_keyframe_id_ = current_id;
    last_online_loop_historical_keyframe_id_ = historical_id;
    last_online_loop_measurement_ = measurement;

    fr_slam_debug::RemoveLoopDecisionDebugEdge(
        historical_id,
        current_id);

    std::cout
        << "ONLINE Keyframe PoseGraph loop edge accepted"
        << " | from_kf=" << historical_id
        << " | to_kf=" << current_id
        << " | source="
        << formal_retrieval_source_name
        << " | loop_edges=" << pose_graph_.LoopEdgeCount()
        << std::endl;

    std::cout
        << "G2O Keyframe PoseGraph optimized"
        << " | chi2_before=" << optimization_result.chi2_before
        << " | chi2_after=" << optimization_result.chi2_after
        << " | max_translation_update="
        << optimization_result.max_translation_update << " m"
        << " | max_rotation_update="
        << optimization_result.max_rotation_update_deg << " deg"
        << " | loop_edges=" << optimization_result.loop_edges
        << std::endl;

    const std::chrono::steady_clock::time_point map_odom_start =
        std::chrono::steady_clock::now();

    const bool map_odom_ok =
        UpdateMapOdomCorrection(current_id);

    const double backend_map_odom_ms_v1 =
        ElapsedMilliseconds(
            map_odom_start,
            std::chrono::steady_clock::now());

    loop_timing.map_odom_ms +=
        backend_map_odom_ms_v1;

    if (!map_odom_ok)
    {
        std::cerr
            << "Map->odom correction update failed"
            << " | anchor_kf=" << current_id
            << std::endl;
    }

    const std::chrono::steady_clock::time_point map_start =
        std::chrono::steady_clock::now();

    const bool global_map_rebuilt =
        RebuildGlobalMapSnapshots();

    const double backend_map_rebuild_ms_v1 =
        ElapsedMilliseconds(
            map_start,
            std::chrono::steady_clock::now());

    loop_timing.global_map_rebuild_ms +=
        backend_map_rebuild_ms_v1;

    // ================================================================
    // LAZY POST-PGO REFINEMENT V2
    //
    // Main backend state stays eager:
    //   - accepted LoopEdge
    //   - G2O
    //   - rollback guard
    //   - map->odom
    //   - optimized global map
    //
    // Post-PGO local geometric refinement is DERIVED MAP ONLY.
    // It never overwrites the main PoseGraph or frontend state.
    //
    // It dominated backend runtime in V1, therefore run it sparsely.
    // ================================================================
    double backend_refine_ms_v1 =
        0.0;

    constexpr std::size_t
        kPostPgoRefineLoopStrideV2 =
            8U;

    const std::size_t
        loop_edge_count_v2 =
            pose_graph_.LoopEdgeCount();

    const bool
        run_post_pgo_refine_v2 =
            global_map_rebuilt &&
            (
                loop_edge_count_v2 == 1U ||
                (
                    loop_edge_count_v2 %
                    kPostPgoRefineLoopStrideV2
                ) == 0U
            );

    if (run_post_pgo_refine_v2)
    {
        const std::chrono::steady_clock::time_point
            refine_start =
                std::chrono::steady_clock::now();

        RebuildPostPgoRefinedMap();

        backend_refine_ms_v1 =
            ElapsedMilliseconds(
                refine_start,
                std::chrono::steady_clock::now());

        loop_timing.refinement_ms +=
            backend_refine_ms_v1;

        std::cout
            << "BACKEND_REFINE_RUN_V2"
            << " | current_kf="
            << current_id
            << " | loop_edges="
            << loop_edge_count_v2
            << " | stride="
            << kPostPgoRefineLoopStrideV2
            << " | refine_ms="
            << backend_refine_ms_v1
            << std::endl;
    }
    else if (global_map_rebuilt)
    {
        std::cout
            << "BACKEND_REFINE_DEFER_V2"
            << " | current_kf="
            << current_id
            << " | loop_edges="
            << loop_edge_count_v2
            << " | next_multiple="
            << (
                (
                    loop_edge_count_v2 /
                    kPostPgoRefineLoopStrideV2
                ) + 1U
            ) *
                kPostPgoRefineLoopStrideV2
            << std::endl;
    }

    const double backend_total_ms_v1 =
        ElapsedMilliseconds(
            backend_accept_start_v1,
            std::chrono::steady_clock::now());

    std::cout
        << "BACKEND_PGO_TIMING_V1"
        << " | current_kf="
        << current_id
        << " | historical_kf="
        << historical_id
        << " | nodes="
        << pose_graph_.GetNodes().size()
        << " | loop_edges="
        << pose_graph_.LoopEdgeCount()
        << " | pgo_ms="
        << backend_pgo_ms_v1
        << " | map_odom_ms="
        << backend_map_odom_ms_v1
        << " | map_rebuild_ms="
        << backend_map_rebuild_ms_v1
        << " | refine_ms="
        << backend_refine_ms_v1
        << " | refine_ran="
        << (run_post_pgo_refine_v2 ? 1 : 0)
        << " | post_accept_total_ms="
        << backend_total_ms_v1
        << " | map_rebuilt="
        << (global_map_rebuilt ? 1 : 0)
        << std::endl;
}

// ============================================================================
// UpdateIncrementalGlobalMaps()
//
// Keyframe clouds + poses remain the authoritative backend data.  The global
// point clouds are derived caches split into small BACKEND-ONLY Keyframe blocks.
//
// This is intentionally independent from frontend SubmapManager lifecycle:
//
//     frontend Submap -> Scan-to-LocalMap / loop geometry
//     backend block   -> visualization/export cache only
//
// New Keyframe:
//     normally dirties one raw block + one optimized block.
//
// PoseGraph optimization:
//     compares cached T_WK against the new graph solution and rebuilds only
//     blocks containing Keyframes whose pose changed beyond the dirty gate.
//
// Per-block VoxelGrid is applied during block rebuild.  The published global
// cloud is assembled from already-filtered blocks, so there is no million-point
// global VoxelGrid pass on every update.
// ============================================================================
