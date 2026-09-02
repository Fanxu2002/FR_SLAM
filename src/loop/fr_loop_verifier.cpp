#include "fr_slam/loop/fr_loop_verifier.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iostream>
#include <limits>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/search/kdtree.h>

namespace
{

    constexpr double kPi =
        3.14159265358979323846;

    // ============================================================================
    // Loop Shadow Hessian Diagnostic V1
    //
    // IMPORTANT:
    //   * PCL point-to-point ICP still computes the loop pose.
    //   * The diagnostic below does NOT change ICP, loop acceptance, or PoseGraph.
    //   * After ICP converges, we rebuild local point-to-plane geometry only to
    //     estimate directional observability / coupling.
    // ============================================================================
    constexpr int kShadowPlaneKnn = 5;
    constexpr double kShadowMaxPlaneFitError = 0.15;
    constexpr double kShadowMinimumScaleRange = 1.0;
    constexpr double kShadowMaximumScaleRange = 50.0;
    constexpr double kShadowRelativeEigenvalueFloor = 0.01;
    constexpr std::size_t kShadowMinimumCorrespondences = 50;

    pcl::PointCloud<pcl::PointXYZ>::Ptr
    ConvertToXYZ(
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &cloud)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr result(
            new pcl::PointCloud<pcl::PointXYZ>);

        if (!cloud)
        {
            return result;
        }

        result->reserve(cloud->size());

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

            result->push_back(xyz);
        }

        result->width =
            static_cast<std::uint32_t>(result->size());

        result->height = 1;
        result->is_dense = true;

        return result;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr
    VoxelFilter(
        const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud,
        double leaf_size)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(
            new pcl::PointCloud<pcl::PointXYZ>);

        if (!cloud || cloud->empty())
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

    double NormalizeAngleRad(
        double angle)
    {
        while (angle > kPi)
        {
            angle -= 2.0 * kPi;
        }

        while (angle < -kPi)
        {
            angle += 2.0 * kPi;
        }

        return angle;
    }

    double YawFromRotation(
        const Eigen::Matrix3d &rotation)
    {
        return std::atan2(
            rotation(1, 0),
            rotation(0, 0));
    }

    double RelativeRotationDeg(
        const Eigen::Isometry3d &T_A,
        const Eigen::Isometry3d &T_B)
    {
        if (!T_A.matrix().allFinite() ||
            !T_B.matrix().allFinite())
        {
            return std::numeric_limits<double>::infinity();
        }

        Eigen::Quaterniond q(
            T_A.rotation().transpose() *
            T_B.rotation());

        if (!q.coeffs().allFinite() ||
            q.norm() < 1.0e-12)
        {
            return std::numeric_limits<double>::infinity();
        }

        q.normalize();

        const double w =
            std::clamp(
                std::abs(q.w()),
                0.0,
                1.0);

        return 2.0 *
               std::acos(w) *
               180.0 /
               kPi;
    }

    Eigen::Isometry3d ReplaceYaw(
        const Eigen::Isometry3d &graph_guess,
        double yaw_new)
    {
        Eigen::Isometry3d result =
            graph_guess;

        const double yaw_graph =
            YawFromRotation(
                graph_guess.rotation());

        const Eigen::Matrix3d Rz_graph =
            Eigen::AngleAxisd(
                yaw_graph,
                Eigen::Vector3d::UnitZ())
                .toRotationMatrix();

        const Eigen::Matrix3d R_tilt =
            Rz_graph.transpose() *
            graph_guess.rotation();

        const Eigen::Matrix3d Rz_new =
            Eigen::AngleAxisd(
                yaw_new,
                Eigen::Vector3d::UnitZ())
                .toRotationMatrix();

        result.linear() =
            Rz_new * R_tilt;

        return result;
    }

    Eigen::Isometry3d MakeScanContextInitialGuess(
        const Eigen::Isometry3d &graph_guess,
        double scan_context_yaw)
    {
        Eigen::Isometry3d result =
            ReplaceYaw(
                graph_guess,
                scan_context_yaw);

        result.translation().setZero();

        return result;
    }

    // ----------------------------------------------------------------------------
    // Explicit final geometry quality measurement using an already-built target
    // KD-tree.
    // ----------------------------------------------------------------------------
    bool EvaluateAlignment(
        const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &source,
        const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &target,
        const pcl::search::KdTree<pcl::PointXYZ>::Ptr &target_kdtree,
        const Eigen::Isometry3d &T_target_source,
        double inlier_distance,
        std::size_t &inliers,
        double &overlap_ratio,
        double &rmse)
    {
        inliers = 0;
        overlap_ratio = 0.0;
        rmse =
            std::numeric_limits<double>::infinity();

        if (!source ||
            !target ||
            !target_kdtree ||
            source->empty() ||
            target->empty() ||
            !T_target_source.matrix().allFinite() ||
            !std::isfinite(inlier_distance) ||
            inlier_distance <= 0.0)
        {
            return false;
        }

        const double max_squared_distance =
            inlier_distance *
            inlier_distance;

        double squared_error_sum = 0.0;

        std::vector<int> nearest_index(1);
        std::vector<float> nearest_squared_distance(1);

        for (const pcl::PointXYZ &point_source : source->points)
        {
            const Eigen::Vector3d p_source(
                static_cast<double>(point_source.x),
                static_cast<double>(point_source.y),
                static_cast<double>(point_source.z));

            const Eigen::Vector3d p_target =
                T_target_source *
                p_source;

            if (!p_target.allFinite())
            {
                continue;
            }

            pcl::PointXYZ query;
            query.x = static_cast<float>(p_target.x());
            query.y = static_cast<float>(p_target.y());
            query.z = static_cast<float>(p_target.z());

            if (target_kdtree->nearestKSearch(
                    query,
                    1,
                    nearest_index,
                    nearest_squared_distance) <= 0)
            {
                continue;
            }

            const double squared_distance =
                static_cast<double>(
                    nearest_squared_distance[0]);

            if (!std::isfinite(squared_distance) ||
                squared_distance > max_squared_distance)
            {
                continue;
            }

            squared_error_sum +=
                squared_distance;

            ++inliers;
        }

        overlap_ratio =
            static_cast<double>(inliers) /
            static_cast<double>(source->size());

        if (inliers == 0)
        {
            return true;
        }

        rmse =
            std::sqrt(
                squared_error_sum /
                static_cast<double>(inliers));

        return std::isfinite(rmse);
    }

    struct PreparedSource
    {
        pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud;
    };

    struct PreparedTarget
    {
        pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud;
        pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree;
    };

    struct LoopShadowHessianDiagnostic
    {
        bool valid = false;

        std::size_t correspondences = 0;
        std::size_t plane_fit_failures = 0;

        double median_range =
            std::numeric_limits<double>::quiet_NaN();

        double scale_L =
            std::numeric_limits<double>::quiet_NaN();

        double condition_number =
            std::numeric_limits<double>::infinity();

        Eigen::Matrix<double, 6, 1> eigenvalues =
            Eigen::Matrix<double, 6, 1>::Zero();

        Eigen::Matrix<double, 6, 1> relative_eigenvalues =
            Eigen::Matrix<double, 6, 1>::Zero();

        // Diagnostic-only candidate Full 6x6 loop information.
        // Order = [tx ty tz rx ry rz], frame = current/source LiDAR.
        Eigen::Matrix<double, 6, 6> base_information =
            Eigen::Matrix<double, 6, 6>::Identity();

        double maximum_absolute_off_diagonal = 0.0;
        double maximum_translation_rotation_coupling = 0.0;
    };

    bool FitShadowPlane(
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

            const Eigen::Vector3d p(
                static_cast<double>(point.x),
                static_cast<double>(point.y),
                static_cast<double>(point.z));

            const Eigen::Vector3d delta =
                p - centroid;

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

        if (eigen_solver.info() !=
            Eigen::Success)
        {
            return false;
        }

        Eigen::Vector3d normal =
            eigen_solver
                .eigenvectors()
                .col(0);

        const double normal_norm =
            normal.norm();

        if (!std::isfinite(normal_norm) ||
            normal_norm < 1.0e-12)
        {
            return false;
        }

        normal /= normal_norm;

        // Match the frontend plane-validity idea: all K neighbours must lie close
        // to the fitted plane.  This is diagnostic only and does not reject ICP.
        for (const int index : neighbor_indices)
        {
            const pcl::PointXYZ &point =
                target->points[static_cast<std::size_t>(index)];

            const Eigen::Vector3d p(
                static_cast<double>(point.x),
                static_cast<double>(point.y),
                static_cast<double>(point.z));

            const double distance =
                std::abs(
                    normal.dot(
                        p - centroid));

            if (!std::isfinite(distance) ||
                distance > kShadowMaxPlaneFitError)
            {
                return false;
            }
        }

        plane_point = centroid;
        plane_normal = normal;

        return true;
    }

    double MedianValue(
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

    bool BuildLoopShadowHessianDiagnostic(
        const PreparedSource &source,
        const PreparedTarget &target,
        const Eigen::Isometry3d &T_target_source,
        double inlier_distance,
        LoopShadowHessianDiagnostic &diagnostic)
    {
        diagnostic =
            LoopShadowHessianDiagnostic();

        if (!source.cloud ||
            !target.cloud ||
            !target.kdtree ||
            source.cloud->empty() ||
            target.cloud->empty() ||
            !T_target_source.matrix().allFinite() ||
            !std::isfinite(inlier_distance) ||
            inlier_distance <= 0.0)
        {
            return false;
        }

        Eigen::Matrix<double, 6, 6> H_raw =
            Eigen::Matrix<double, 6, 6>::Zero();

        std::vector<double> correspondence_ranges;
        correspondence_ranges.reserve(
            source.cloud->size());

        std::vector<int> neighbor_indices(
            static_cast<std::size_t>(
                kShadowPlaneKnn));

        std::vector<float> neighbor_squared_distances(
            static_cast<std::size_t>(
                kShadowPlaneKnn));

        const double max_squared_distance =
            inlier_distance *
            inlier_distance;

        const Eigen::Vector3d sensor_origin_target =
            T_target_source.translation();

        for (const pcl::PointXYZ &source_point :
             source.cloud->points)
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
            query.x = static_cast<float>(p_target.x());
            query.y = static_cast<float>(p_target.y());
            query.z = static_cast<float>(p_target.z());

            const int found =
                target.kdtree->nearestKSearch(
                    query,
                    kShadowPlaneKnn,
                    neighbor_indices,
                    neighbor_squared_distances);

            if (found < kShadowPlaneKnn)
            {
                continue;
            }

            const double nearest_squared_distance =
                static_cast<double>(
                    neighbor_squared_distances[0]);

            if (!std::isfinite(nearest_squared_distance) ||
                nearest_squared_distance > max_squared_distance)
            {
                continue;
            }

            Eigen::Vector3d plane_point;
            Eigen::Vector3d plane_normal;

            const bool plane_ok =
                FitShadowPlane(
                    target.cloud,
                    neighbor_indices,
                    plane_point,
                    plane_normal);

            if (!plane_ok)
            {
                ++diagnostic.plane_fit_failures;
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

            // Same sensor-centered point-to-plane convention as realtime odometry:
            //     J = [(lever_arm x n)^T, n^T]
            // order = [rx ry rz tx ty tz].
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

            ++diagnostic.correspondences;
        }

        if (diagnostic.correspondences <
                kShadowMinimumCorrespondences ||
            correspondence_ranges.empty() ||
            !H_raw.allFinite())
        {
            return false;
        }

        diagnostic.median_range =
            MedianValue(
                correspondence_ranges);

        if (!std::isfinite(
                diagnostic.median_range) ||
            diagnostic.median_range <= 0.0)
        {
            return false;
        }

        diagnostic.scale_L =
            std::clamp(
                diagnostic.median_range,
                kShadowMinimumScaleRange,
                kShadowMaximumScaleRange);

        Eigen::Matrix<double, 6, 6> parameter_unscale =
            Eigen::Matrix<double, 6, 6>::Identity();

        const double inverse_scale =
            1.0 /
            diagnostic.scale_L;

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

        if (hessian_solver.info() !=
            Eigen::Success)
        {
            return false;
        }

        diagnostic.eigenvalues =
            hessian_solver.eigenvalues();

        if (!diagnostic.eigenvalues.allFinite())
        {
            return false;
        }

        const double lambda_min =
            diagnostic.eigenvalues(0);

        const double lambda_max =
            diagnostic.eigenvalues(5);

        if (!std::isfinite(lambda_max) ||
            lambda_max <= 1.0e-12)
        {
            return false;
        }

        diagnostic.relative_eigenvalues =
            diagnostic.eigenvalues /
            lambda_max;

        if (!diagnostic.relative_eigenvalues.allFinite())
        {
            return false;
        }

        if (std::isfinite(lambda_min) &&
            lambda_min > 1.0e-12)
        {
            diagnostic.condition_number =
                lambda_max /
                lambda_min;
        }

        // ------------------------------------------------------------------------
        // Relative covariance shape in historical-target frame [r,t].
        // Absolute Hessian scale is deliberately removed.
        // ------------------------------------------------------------------------
        Eigen::Matrix<double, 6, 6> inverse_relative_eigenvalues =
            Eigen::Matrix<double, 6, 6>::Zero();

        for (int i = 0;
             i < 6;
             ++i)
        {
            const double safe_relative =
                std::clamp(
                    diagnostic.relative_eigenvalues(i),
                    kShadowRelativeEigenvalueFloor,
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

        // Target/historical H -> current/source LiDAR L.
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

        // Reorder [r,t] -> g2o [t,r].
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

        // ------------------------------------------------------------------------
        // Same V1.1 directional-confidence rule used by odometry.
        // ------------------------------------------------------------------------
        constexpr double minimum_directional_confidence =
            0.01;

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
                    minimum_directional_confidence,
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
                    minimum_directional_confidence,
                    1.0);
        }

        // ------------------------------------------------------------------------
        // Full precision SHAPE and standardized coupling.
        // ------------------------------------------------------------------------
        Eigen::SelfAdjointEigenSolver<
            Eigen::Matrix<double, 6, 6>>
            covariance_solver(
                covariance_tr);

        if (covariance_solver.info() !=
            Eigen::Success)
        {
            return false;
        }

        const Eigen::Matrix<double, 6, 1>
            covariance_eigenvalues =
                covariance_solver.eigenvalues();

        if (!covariance_eigenvalues.allFinite() ||
            covariance_eigenvalues.minCoeff() <= 1.0e-12)
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
                covariance_eigenvalues(i);
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

        diagnostic.base_information =
            confidence_scale *
            standardized_precision *
            confidence_scale;

        diagnostic.base_information =
            0.5 *
            (diagnostic.base_information +
             diagnostic.base_information.transpose());

        if (!diagnostic.base_information.allFinite())
        {
            return false;
        }

        Eigen::SelfAdjointEigenSolver<
            Eigen::Matrix<double, 6, 6>>
            information_solver(
                diagnostic.base_information,
                Eigen::EigenvaluesOnly);

        if (information_solver.info() !=
                Eigen::Success ||
            information_solver.eigenvalues().minCoeff() <=
                1.0e-9)
        {
            return false;
        }

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

                diagnostic.maximum_absolute_off_diagonal =
                    std::max(
                        diagnostic.maximum_absolute_off_diagonal,
                        std::abs(
                            diagnostic.base_information(i, j)));

                const bool translation_rotation_pair =
                    (i < 3 && j >= 3) ||
                    (i >= 3 && j < 3);

                if (translation_rotation_pair)
                {
                    diagnostic.maximum_translation_rotation_coupling =
                        std::max(
                            diagnostic.maximum_translation_rotation_coupling,
                            std::abs(
                                diagnostic.base_information(i, j)));
                }
            }
        }

        diagnostic.valid = true;
        return true;
    }

    void PrintLoopShadowHessianDiagnostic(
        const LoopShadowHessianDiagnostic &diagnostic,
        bool p2p_accepted)
    {
        std::cout
            << "LOOP_SHADOW_HESSIAN_V1"
            << " | p2p_accepted="
            << (p2p_accepted ? "true" : "false")
            << " | valid="
            << (diagnostic.valid ? "true" : "false")
            << " | shadow_corr="
            << diagnostic.correspondences
            << " | plane_fit_failures="
            << diagnostic.plane_fit_failures
            << " | knn="
            << kShadowPlaneKnn
            << " | plane_fit_error="
            << kShadowMaxPlaneFitError
            << " | median_range="
            << diagnostic.median_range
            << " | scale_L="
            << diagnostic.scale_L
            << " | eigen=["
            << diagnostic.eigenvalues.transpose()
            << "]"
            << " | relative=["
            << diagnostic.relative_eigenvalues.transpose()
            << "]"
            << " | min_relative="
            << diagnostic.relative_eigenvalues.minCoeff()
            << " | condition="
            << diagnostic.condition_number
            << " | base_diag=["
            << diagnostic.base_information(0, 0) << " "
            << diagnostic.base_information(1, 1) << " "
            << diagnostic.base_information(2, 2) << " "
            << diagnostic.base_information(3, 3) << " "
            << diagnostic.base_information(4, 4) << " "
            << diagnostic.base_information(5, 5)
            << "]"
            << " | max_offdiag="
            << diagnostic.maximum_absolute_off_diagonal
            << " | max_tr_coupling="
            << diagnostic.maximum_translation_rotation_coupling
            << " | action=DIAGNOSTIC_ONLY_EDGE_INFORMATION_BUILT_AT_COMMIT"
            << std::endl;
    }

    LoopVerificationResult RunHypothesis(
        const PreparedSource &source,
        const PreparedTarget &target,
        const Eigen::Isometry3d &initial_guess,
        LoopVerifierHypothesis hypothesis,
        const LoopVerifierConfig &config)
    {
        LoopVerificationResult result;
        result.initial_guess = initial_guess;
        result.T_target_source = initial_guess;
        result.hypothesis = hypothesis;
        result.source_points =
            source.cloud ? source.cloud->size() : 0;
        result.target_points =
            target.cloud ? target.cloud->size() : 0;

        if (!source.cloud ||
            !target.cloud ||
            !target.kdtree ||
            source.cloud->size() < config.min_cloud_points ||
            target.cloud->size() < config.min_cloud_points ||
            !initial_guess.matrix().allFinite())
        {
            return result;
        }

        pcl::IterativeClosestPoint<
            pcl::PointXYZ,
            pcl::PointXYZ>
            icp;

        icp.setInputSource(source.cloud);
        icp.setInputTarget(target.cloud);

        icp.setSearchMethodTarget(
            target.kdtree,
            true);

        icp.setMaximumIterations(
            static_cast<int>(
                config.max_iterations));

        icp.setMaxCorrespondenceDistance(
            config.max_correspondence_distance);

        icp.setTransformationEpsilon(
            config.transformation_epsilon);

        icp.setEuclideanFitnessEpsilon(
            config.euclidean_fitness_epsilon);

        pcl::PointCloud<pcl::PointXYZ> aligned;

        icp.align(
            aligned,
            initial_guess
                .matrix()
                .cast<float>());

        result.converged =
            icp.hasConverged();

        if (!result.converged)
        {
            return result;
        }

        const Eigen::Matrix4d final_matrix =
            icp.getFinalTransformation()
                .cast<double>();

        if (!final_matrix.allFinite())
        {
            return result;
        }

        result.T_target_source =
            Eigen::Isometry3d::Identity();

        result.T_target_source.matrix() =
            final_matrix;

        if (!result.T_target_source
                 .matrix()
                 .allFinite())
        {
            return result;
        }

        result.fitness_score =
            icp.getFitnessScore();

        const bool metrics_ok =
            EvaluateAlignment(
                source.cloud,
                target.cloud,
                target.kdtree,
                result.T_target_source,
                config.verification_inlier_distance,
                result.inliers,
                result.overlap_ratio,
                result.rmse);

        if (!metrics_ok)
        {
            return result;
        }

        result.correction_translation =
            (result.T_target_source.translation() -
             initial_guess.translation())
                .norm();

        result.correction_rotation_deg =
            RelativeRotationDeg(
                initial_guess,
                result.T_target_source);

        result.success = true;

        // Acceptance logic intentionally unchanged.
        result.accepted =
            result.inliers >= config.min_inliers &&
            result.overlap_ratio >= config.min_overlap_ratio &&
            std::isfinite(result.rmse) &&
            result.rmse <= config.max_rmse &&
            std::isfinite(result.correction_translation) &&
            result.correction_translation <=
                config.max_correction_translation &&
            std::isfinite(result.correction_rotation_deg) &&
            result.correction_rotation_deg <=
                config.max_correction_rotation_deg;

        // ------------------------------------------------------------------------
        // Diagnostic-only shadow point-to-plane Hessian.
        // This does NOT modify result.accepted or the loop measurement.
        // ------------------------------------------------------------------------
        LoopShadowHessianDiagnostic shadow_diagnostic;

        BuildLoopShadowHessianDiagnostic(
            source,
            target,
            result.T_target_source,
            config.verification_inlier_distance,
            shadow_diagnostic);

        PrintLoopShadowHessianDiagnostic(
            shadow_diagnostic,
            result.accepted);

        return result;
    }

    bool IsBetterResult(
        const LoopVerificationResult &candidate,
        const LoopVerificationResult &current_best)
    {
        if (!candidate.success)
        {
            return false;
        }

        if (!current_best.success)
        {
            return true;
        }

        if (candidate.accepted != current_best.accepted)
        {
            return candidate.accepted;
        }

        if (candidate.overlap_ratio != current_best.overlap_ratio)
        {
            return candidate.overlap_ratio >
                   current_best.overlap_ratio;
        }

        return candidate.rmse <
               current_best.rmse;
    }

} // namespace

// ============================================================================
// Cache implementation hidden from the public header.
// ============================================================================
struct LoopVerifier::Cache
{
    struct SourceEntry
    {
        const void *identity = nullptr;
        std::size_t original_size = 0;
        pcl::PointCloud<pcl::PointXYZ>::ConstPtr filtered;
    };

    struct TargetEntry
    {
        std::size_t original_size = 0;
        PreparedTarget prepared;
    };

    SourceEntry source;

    std::unordered_map<const void *, TargetEntry>
        targets;

    std::deque<const void *>
        target_order;
};

LoopVerifier::LoopVerifier(
    const LoopVerifierConfig &config)
    : config_(config),
      cache_(std::make_unique<Cache>())
{
    if (!std::isfinite(config_.voxel_leaf_size) ||
        config_.voxel_leaf_size <= 0.0)
    {
        config_.voxel_leaf_size = 0.50;
    }

    if (config_.max_iterations == 0)
    {
        config_.max_iterations = 50;
    }

    if (!std::isfinite(config_.max_correspondence_distance) ||
        config_.max_correspondence_distance <= 0.0)
    {
        config_.max_correspondence_distance = 5.0;
    }

    if (!std::isfinite(config_.transformation_epsilon) ||
        config_.transformation_epsilon <= 0.0)
    {
        config_.transformation_epsilon = 1.0e-6;
    }

    if (!std::isfinite(config_.euclidean_fitness_epsilon) ||
        config_.euclidean_fitness_epsilon <= 0.0)
    {
        config_.euclidean_fitness_epsilon = 1.0e-5;
    }

    if (!std::isfinite(config_.verification_inlier_distance) ||
        config_.verification_inlier_distance <= 0.0)
    {
        config_.verification_inlier_distance = 1.0;
    }

    if (!std::isfinite(config_.max_rmse) ||
        config_.max_rmse <= 0.0)
    {
        config_.max_rmse = 0.65;
    }

    if (config_.min_inliers == 0)
    {
        config_.min_inliers = 300;
    }

    if (!std::isfinite(config_.min_overlap_ratio) ||
        config_.min_overlap_ratio <= 0.0 ||
        config_.min_overlap_ratio > 1.0)
    {
        config_.min_overlap_ratio = 0.15;
    }

    if (config_.min_cloud_points == 0)
    {
        config_.min_cloud_points = 300;
    }

    if (!std::isfinite(config_.prescore_inlier_distance) ||
        config_.prescore_inlier_distance <= 0.0)
    {
        config_.prescore_inlier_distance = 2.0;
    }

    if (!std::isfinite(config_.prescore_min_overlap_ratio) ||
        config_.prescore_min_overlap_ratio < 0.0 ||
        config_.prescore_min_overlap_ratio > 1.0)
    {
        config_.prescore_min_overlap_ratio = 0.03;
    }

    if (!std::isfinite(config_.max_correction_translation) ||
        config_.max_correction_translation <= 0.0)
    {
        config_.max_correction_translation = 15.0;
    }

    if (!std::isfinite(config_.max_correction_rotation_deg) ||
        config_.max_correction_rotation_deg <= 0.0)
    {
        config_.max_correction_rotation_deg = 45.0;
    }

    if (config_.max_cached_targets == 0)
    {
        config_.max_cached_targets = 1;
    }
}

LoopVerifier::~LoopVerifier() = default;

bool LoopVerifier::ScoreInitialGuess(
    const pcl::PointCloud<LIDAR_POINT>::ConstPtr &source_current,
    const pcl::PointCloud<LIDAR_POINT>::ConstPtr &target_historical,
    const Eigen::Isometry3d &initial_guess,
    LoopVerifierInitialGuessScore &score) const
{
    score = LoopVerifierInitialGuessScore();

    if (!source_current ||
        !target_historical ||
        source_current->empty() ||
        target_historical->empty() ||
        !initial_guess.matrix().allFinite())
    {
        return false;
    }

    if (!cache_)
    {
        cache_ = std::make_unique<Cache>();
    }

    const void *source_identity =
        static_cast<const void *>(
            source_current.get());

    if (cache_->source.identity != source_identity ||
        cache_->source.original_size != source_current->size() ||
        !cache_->source.filtered)
    {
        const pcl::PointCloud<pcl::PointXYZ>::Ptr source_xyz =
            ConvertToXYZ(source_current);

        const pcl::PointCloud<pcl::PointXYZ>::Ptr source_filtered =
            VoxelFilter(
                source_xyz,
                config_.voxel_leaf_size);

        if (!source_filtered ||
            source_filtered->size() < config_.min_cloud_points)
        {
            return false;
        }

        cache_->source.identity = source_identity;
        cache_->source.original_size = source_current->size();
        cache_->source.filtered = source_filtered;
    }

    const void *target_identity =
        static_cast<const void *>(
            target_historical.get());

    auto target_iterator =
        cache_->targets.find(target_identity);

    const bool target_cache_entry_valid =
        target_iterator != cache_->targets.end() &&
        target_iterator->second.original_size == target_historical->size() &&
        target_iterator->second.prepared.cloud &&
        target_iterator->second.prepared.kdtree;

    if (!target_cache_entry_valid)
    {
        if (target_iterator != cache_->targets.end())
        {
            cache_->targets.erase(target_iterator);

            cache_->target_order.erase(
                std::remove(
                    cache_->target_order.begin(),
                    cache_->target_order.end(),
                    target_identity),
                cache_->target_order.end());
        }

        const pcl::PointCloud<pcl::PointXYZ>::Ptr target_xyz =
            ConvertToXYZ(target_historical);

        const pcl::PointCloud<pcl::PointXYZ>::Ptr target_filtered =
            VoxelFilter(
                target_xyz,
                config_.voxel_leaf_size);

        if (!target_filtered ||
            target_filtered->size() < config_.min_cloud_points)
        {
            return false;
        }

        pcl::search::KdTree<pcl::PointXYZ>::Ptr target_kdtree(
            new pcl::search::KdTree<pcl::PointXYZ>());

        target_kdtree->setInputCloud(
            target_filtered);

        while (cache_->targets.size() >=
                   config_.max_cached_targets &&
               !cache_->target_order.empty())
        {
            const void *oldest_identity =
                cache_->target_order.front();

            cache_->target_order.pop_front();
            cache_->targets.erase(oldest_identity);
        }

        Cache::TargetEntry entry;
        entry.original_size =
            target_historical->size();
        entry.prepared.cloud =
            target_filtered;
        entry.prepared.kdtree =
            target_kdtree;

        cache_->targets.emplace(
            target_identity,
            std::move(entry));

        cache_->target_order.push_back(
            target_identity);

        target_iterator =
            cache_->targets.find(target_identity);
    }

    if (target_iterator == cache_->targets.end())
    {
        return false;
    }

    const PreparedTarget &prepared_target =
        target_iterator->second.prepared;

    if (!prepared_target.cloud ||
        !prepared_target.kdtree ||
        !cache_->source.filtered)
    {
        return false;
    }

    const bool evaluation_ok =
        EvaluateAlignment(
            cache_->source.filtered,
            prepared_target.cloud,
            prepared_target.kdtree,
            initial_guess,
            config_.prescore_inlier_distance,
            score.inliers,
            score.overlap_ratio,
            score.rmse);

    score.valid =
        evaluation_ok &&
        std::isfinite(score.rmse);

    if (evaluation_ok &&
        score.inliers == 0)
    {
        score.valid = true;
    }

    return evaluation_ok;
}

bool LoopVerifier::Verify(
    const pcl::PointCloud<LIDAR_POINT>::ConstPtr &source_current,
    const pcl::PointCloud<LIDAR_POINT>::ConstPtr &target_historical,
    const Eigen::Isometry3d &graph_initial_guess,
    double scan_context_yaw_shift_deg,
    LoopVerificationResult &result) const
{
    result = LoopVerificationResult();

    if (!source_current ||
        !target_historical ||
        source_current->empty() ||
        target_historical->empty() ||
        !graph_initial_guess.matrix().allFinite())
    {
        return false;
    }

    if (!cache_)
    {
        cache_ = std::make_unique<Cache>();
    }

    const void *source_identity =
        static_cast<const void *>(
            source_current.get());

    if (cache_->source.identity != source_identity ||
        cache_->source.original_size != source_current->size() ||
        !cache_->source.filtered)
    {
        const pcl::PointCloud<pcl::PointXYZ>::Ptr source_xyz =
            ConvertToXYZ(source_current);

        const pcl::PointCloud<pcl::PointXYZ>::Ptr source_filtered =
            VoxelFilter(
                source_xyz,
                config_.voxel_leaf_size);

        if (!source_filtered ||
            source_filtered->size() < config_.min_cloud_points)
        {
            return false;
        }

        cache_->source.identity = source_identity;
        cache_->source.original_size = source_current->size();
        cache_->source.filtered = source_filtered;
    }

    PreparedSource prepared_source;
    prepared_source.cloud =
        cache_->source.filtered;

    const void *target_identity =
        static_cast<const void *>(
            target_historical.get());

    auto target_iterator =
        cache_->targets.find(target_identity);

    const bool target_cache_entry_valid =
        target_iterator != cache_->targets.end() &&
        target_iterator->second.original_size == target_historical->size() &&
        target_iterator->second.prepared.cloud &&
        target_iterator->second.prepared.kdtree;

    if (!target_cache_entry_valid)
    {
        if (target_iterator != cache_->targets.end())
        {
            cache_->targets.erase(target_iterator);

            cache_->target_order.erase(
                std::remove(
                    cache_->target_order.begin(),
                    cache_->target_order.end(),
                    target_identity),
                cache_->target_order.end());
        }

        const pcl::PointCloud<pcl::PointXYZ>::Ptr target_xyz =
            ConvertToXYZ(target_historical);

        const pcl::PointCloud<pcl::PointXYZ>::Ptr target_filtered =
            VoxelFilter(
                target_xyz,
                config_.voxel_leaf_size);

        if (!target_filtered ||
            target_filtered->size() < config_.min_cloud_points)
        {
            return false;
        }

        pcl::search::KdTree<pcl::PointXYZ>::Ptr target_kdtree(
            new pcl::search::KdTree<pcl::PointXYZ>());

        target_kdtree->setInputCloud(
            target_filtered);

        while (cache_->targets.size() >=
                   config_.max_cached_targets &&
               !cache_->target_order.empty())
        {
            const void *oldest_identity =
                cache_->target_order.front();

            cache_->target_order.pop_front();
            cache_->targets.erase(oldest_identity);
        }

        Cache::TargetEntry entry;
        entry.original_size =
            target_historical->size();
        entry.prepared.cloud =
            target_filtered;
        entry.prepared.kdtree =
            target_kdtree;

        cache_->targets.emplace(
            target_identity,
            std::move(entry));

        cache_->target_order.push_back(
            target_identity);

        target_iterator =
            cache_->targets.find(target_identity);
    }

    if (target_iterator == cache_->targets.end())
    {
        return false;
    }

    const PreparedTarget &prepared_target =
        target_iterator->second.prepared;

    std::vector<
        std::pair<
            LoopVerifierHypothesis,
            Eigen::Isometry3d>>
        hypotheses;

    hypotheses.emplace_back(
        LoopVerifierHypothesis::GraphPose,
        graph_initial_guess);

    if (std::isfinite(scan_context_yaw_shift_deg))
    {
        const double yaw_shift_rad =
            NormalizeAngleRad(
                scan_context_yaw_shift_deg *
                kPi /
                180.0);

        const Eigen::Isometry3d positive_yaw_guess =
            MakeScanContextInitialGuess(
                graph_initial_guess,
                yaw_shift_rad);

        const Eigen::Isometry3d negative_yaw_guess =
            MakeScanContextInitialGuess(
                graph_initial_guess,
                -yaw_shift_rad);

        hypotheses.emplace_back(
            LoopVerifierHypothesis::ScanContextPositiveYaw,
            positive_yaw_guess);

        hypotheses.emplace_back(
            LoopVerifierHypothesis::ScanContextNegativeYaw,
            negative_yaw_guess);
    }

    LoopVerificationResult best;

    for (const auto &entry : hypotheses)
    {
        const LoopVerificationResult candidate =
            RunHypothesis(
                prepared_source,
                prepared_target,
                entry.second,
                entry.first,
                config_);

        if (IsBetterResult(candidate, best))
        {
            best = candidate;
        }
    }

    result = best;

    return result.success;
}

void LoopVerifier::ClearCache()
{
    if (!cache_)
    {
        cache_ = std::make_unique<Cache>();
        return;
    }

    cache_->source = Cache::SourceEntry();
    cache_->targets.clear();
    cache_->target_order.clear();
}

std::size_t LoopVerifier::CachedTargetCount() const
{
    if (!cache_)
    {
        return 0;
    }

    return cache_->targets.size();
}

const LoopVerifierConfig &
LoopVerifier::GetConfig() const
{
    return config_;
}

const char *LoopVerifier::HypothesisName(
    LoopVerifierHypothesis hypothesis)
{
    switch (hypothesis)
    {
    case LoopVerifierHypothesis::GraphPose:
        return "GRAPH";

    case LoopVerifierHypothesis::ScanContextPositiveYaw:
        return "SCAN_CONTEXT_POSITIVE_YAW";

    case LoopVerifierHypothesis::ScanContextNegativeYaw:
        return "SCAN_CONTEXT_NEGATIVE_YAW";

    default:
        return "UNKNOWN";
    }
}
