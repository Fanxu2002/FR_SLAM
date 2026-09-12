#include "fr_slam/loop/btc_adapter.hpp"
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

void RegistrationScan2LocalMap::DetectAndVerifyLoopFromKeyframe(
    const Keyframe &current_keyframe,
    std::size_t current_submap_id)
{
    LoopTimingDiagnostics loop_timing;
    loop_timing.current_keyframe_id = current_keyframe.id;
    loop_timing.current_submap_id = current_submap_id;

    LoopTimingReporter loop_timing_reporter(loop_timing);

    const LoopDetectorConfig &loop_config =
        loop_detector_.GetConfig();

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

    // ================================================================
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

    const std::chrono::steady_clock::time_point btc_start =
        std::chrono::steady_clock::now();

    const std::size_t query_last_kf = current_keyframe.id;
    const std::size_t query_first_kf =
        (query_last_kf + 1 > btc_window_size)
            ? query_last_kf + 1 - btc_window_size
            : 0;
    const std::size_t query_requested_keyframes =
        query_last_kf - query_first_kf + 1;
    const std::size_t query_anchor_kf =
        query_first_kf + query_requested_keyframes / 2;

    const bool query_is_full_window =
        query_requested_keyframes == btc_window_size;

    const bool query_will_enter_database =
        query_is_full_window &&
        query_last_kf >= btc_window_size - 1 &&
        ((query_last_kf - (btc_window_size - 1)) %
             btc_window_stride ==
         0);

    if (query_requested_keyframes >= btc_min_valid_dense_keyframes &&
        !official_btc_adapter.IsProcessedSubmap(query_last_kf))
    {
        const Keyframe *query_anchor =
            FindBackendKeyframeById(query_anchor_kf);

        std::size_t query_valid_dense_keyframes = 0;
        std::size_t query_missing_dense_keyframes = 0;
        std::size_t query_source_points = 0;

        if (query_anchor != nullptr &&
            query_anchor->T_WL.matrix().allFinite())
        {
            for (std::size_t keyframe_id = query_first_kf;
                 keyframe_id <= query_last_kf;
                 ++keyframe_id)
            {
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

        if (query_anchor == nullptr ||
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

            for (std::size_t keyframe_id = query_first_kf;
                 keyframe_id <= query_last_kf;
                 ++keyframe_id)
            {
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

                        if (historical_last_kf + 1 >= btc_window_size)
                        {
                            historical_first_kf =
                                historical_last_kf + 1 - btc_window_size;
                            historical_keyframe_count =
                                btc_window_size;
                            historical_anchor_kf =
                                historical_first_kf +
                                btc_window_size / 2;

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

    loop_timing.btc_retrieval_ms +=
        ElapsedMilliseconds(
            btc_start,
            std::chrono::steady_clock::now());

    if (btc_candidates.empty())
    {
        return;
    }

    std::sort(
        btc_candidates.begin(),
        btc_candidates.end(),
        [](const LoopCandidate &lhs,
           const LoopCandidate &rhs)
        {
            if (lhs.retrieval_score != rhs.retrieval_score)
            {
                return lhs.retrieval_score > rhs.retrieval_score;
            }
            return lhs.candidate_id < rhs.candidate_id;
        });

    if (btc_candidates.size() > loop_config.max_candidates)
    {
        btc_candidates.resize(loop_config.max_candidates);
    }

    loop_timing.candidates = btc_candidates.size();

    const LoopCandidate *best_candidate = nullptr;
    LoopVerificationResult best_verification;
    pcl::PointCloud<LIDAR_POINT>::Ptr best_target_K;
    std::vector<std::size_t> best_target_keyframes;

    for (const LoopCandidate &candidate : btc_candidates)
    {
        if (candidate.candidate_id >= current_keyframe.id)
        {
            continue;
        }

        const std::size_t keyframe_gap =
            current_keyframe.id - candidate.candidate_id;

        if (keyframe_gap < loop_config.min_keyframe_id_separation ||
            !std::isfinite(candidate.time_separation_sec) ||
            candidate.time_separation_sec <
                loop_config.min_time_separation_sec)
        {
            continue;
        }

        if (loop_config.use_pose_distance_gate &&
            (!std::isfinite(candidate.distance) ||
             candidate.distance > loop_config.max_candidate_distance))
        {
            continue;
        }

        pcl::PointCloud<LIDAR_POINT>::Ptr target_K;
        std::vector<std::size_t> target_keyframes;

        if (!BuildCandidateCenteredHistoricalTarget(
                candidate.candidate_id,
                target_K,
                &target_keyframes))
        {
            continue;
        }

        const auto guess_iterator =
            btc_initial_guess_by_kf.find(candidate.candidate_id);

        if (guess_iterator == btc_initial_guess_by_kf.end() ||
            !guess_iterator->second.matrix().allFinite())
        {
            continue;
        }

        const Eigen::Isometry3d &initial_guess =
            guess_iterator->second;


        LoopVerifierInitialGuessScore prescore;
        const std::chrono::steady_clock::time_point prescore_start =
            std::chrono::steady_clock::now();

        const bool prescore_ok =
            loop_verifier_.ScoreInitialGuess(
                current_keyframe.cloud,
                target_K,
                initial_guess,
                prescore);

        loop_timing.verifier_prescore_ms +=
            ElapsedMilliseconds(
                prescore_start,
                std::chrono::steady_clock::now());
        ++loop_timing.verifier_prescore_calls;

        if (!prescore_ok ||
            !prescore.valid ||
            prescore.overlap_ratio <
                loop_verifier_.GetConfig().prescore_min_overlap_ratio)
        {
            continue;
        }

        LoopVerificationResult verification;
        const std::chrono::steady_clock::time_point verify_start =
            std::chrono::steady_clock::now();

        const bool verify_ok =
            loop_verifier_.Verify(
                current_keyframe.cloud,
                target_K,
                initial_guess,
                verification);

        loop_timing.verifier_ms +=
            ElapsedMilliseconds(
                verify_start,
                std::chrono::steady_clock::now());
        ++loop_timing.verifier_calls;

        // ================================================================
        // PURE_BTC_V30.1 geometry gate diagnostic
        // Diagnostic only. Does NOT change any acceptance logic.
        // ================================================================
        const auto &verifier_config =
            loop_verifier_.GetConfig();

        const bool gate_overlap_pass =
            std::isfinite(verification.overlap_ratio) &&
            verification.overlap_ratio >=
                verifier_config.min_overlap_ratio;

        const bool gate_rmse_pass =
            std::isfinite(verification.rmse) &&
            verification.rmse <=
                verifier_config.max_rmse;

        const bool gate_translation_pass =
            std::isfinite(
                verification.correction_translation) &&
            verification.correction_translation <=
                verifier_config.max_correction_translation;

        const bool gate_rotation_pass =
            std::isfinite(
                verification.correction_rotation_deg) &&
            verification.correction_rotation_deg <=
                verifier_config.max_correction_rotation_deg;



        if (!verify_ok || !verification.accepted)
        {
            continue;
        }

        const bool better =
            best_candidate == nullptr ||
            verification.overlap_ratio >
                best_verification.overlap_ratio + 1.0e-9 ||
            (std::abs(
                 verification.overlap_ratio -
                 best_verification.overlap_ratio) <= 1.0e-9 &&
             verification.rmse < best_verification.rmse);

        if (better)
        {
            best_candidate = &candidate;
            best_verification = verification;
            best_target_K = target_K;
            best_target_keyframes = target_keyframes;
        }
    }

    if (best_candidate == nullptr ||
        !best_verification.accepted)
    {
        return;
    }

    loop_timing.geometry_accepted = true;


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
        // FR-SLAM V31.11 PENDING rescue.
        //
        // ARM only from a geometry-valid proposal that currently has
        // no temporal neighbour.
        //
        // Once armed, later BTC retrieval is relaxed ONLY around this
        // historical area.
        // ------------------------------------------------------------
        const bool pending_seed =
            consistency.valid &&
            !consistency.temporal_available;

        if (!btc_pending_rescue_active_v31_11 &&
            pending_seed)
        {
            const bool configured =
                official_btc_adapter
                    .ConfigurePendingRescue(
                        proposal.historical_id,
                        btc_consistency_config_v31_11
                            .temporal_max_historical_gap);

            if (configured)
            {
                btc_pending_rescue_active_v31_11 =
                    true;

                btc_pending_current_kf_v31_11 =
                    proposal.current_id;

                btc_pending_historical_kf_v31_11 =
                    proposal.historical_id;
            }

        }

        return;
    }

    // Consistency has passed.
    // Return BTC retrieval to strict global rough >=5.
    if (btc_pending_rescue_active_v31_11)
    {

        official_btc_adapter
            .ClearPendingRescue();

        btc_pending_rescue_active_v31_11 =
            false;
    }

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

        if (current_gap < min_online_loop_edge_current_keyframe_spacing_ &&
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


    PoseGraphOptimizationResult optimization_result;
    const std::chrono::steady_clock::time_point pgo_start =
        std::chrono::steady_clock::now();

    const bool pgo_ok =
        pose_graph_optimizer_.Optimize(
            pose_graph_,
            optimization_result);

    loop_timing.pose_graph_optimize_ms +=
        ElapsedMilliseconds(
            pgo_start,
            std::chrono::steady_clock::now());
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
            << "BTC PoseGraph optimization rejected"
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
        << " | source=BTC"
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

    loop_timing.map_odom_ms +=
        ElapsedMilliseconds(
            map_odom_start,
            std::chrono::steady_clock::now());

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

    loop_timing.global_map_rebuild_ms +=
        ElapsedMilliseconds(
            map_start,
            std::chrono::steady_clock::now());

    if (global_map_rebuilt)
    {
        const std::chrono::steady_clock::time_point refine_start =
            std::chrono::steady_clock::now();

        RebuildPostPgoRefinedMap();

        loop_timing.refinement_ms +=
            ElapsedMilliseconds(
                refine_start,
                std::chrono::steady_clock::now());
    }
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
