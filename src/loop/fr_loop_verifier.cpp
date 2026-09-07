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
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>

namespace
{

    constexpr double kPi =
        3.14159265358979323846;

    // ============================================================================
    // Loop Point-to-Plane V2
    //
    // V2 keeps V1's candidate generation, prescore and final gates unchanged,
    // but makes the pose solver observability-aware:
    //
    //   E = E_point_to_plane
    //     + E_ground(roll,pitch,z)
    //     + E_seed_prior(geometry-weak directions only, trusted seeds only)
    //
    // Degeneracy analysis / suppression is performed in normalized coordinates
    //
    //   [L*rx, L*ry, L*rz, tx, ty, tz]
    //
    // with L = median correspondence range.  This prevents rad/m unit scaling
    // from contaminating eigenvalue comparisons.
    // ============================================================================
    constexpr int kShadowPlaneKnn = 5;
    constexpr double kShadowMaxPlaneFitError = 0.15;
    constexpr double kShadowMinimumScaleRange = 1.0;
    constexpr double kShadowMaximumScaleRange = 50.0;
    constexpr double kShadowRelativeEigenvalueFloor = 0.01;
    constexpr std::size_t kShadowMinimumCorrespondences = 50;

    constexpr double kLoopP2PlaneNumericalDampingRatio = 1.0e-6;
    constexpr std::size_t kLoopP2PlaneMinimumCorrespondences = 50;

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
        bool solver_accepted)
    {
        std::cout
            << "LOOP_SHADOW_HESSIAN_V1"
            << " | solver=POINT_TO_PLANE_V2_DEGENERACY_GROUND"
            << " | solver_accepted="
            << (solver_accepted ? "true" : "false")
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
            << " | action=FINAL_POSE_OBSERVABILITY_DIAGNOSTIC_ONLY"
            << std::endl;
    }

    struct LoopP2PlaneIterationDiagnostic
    {
        bool valid = false;
        bool ground_active = false;
        bool trusted_seed_prior = false;

        std::size_t correspondences = 0;
        std::size_t plane_fit_failures = 0;
        std::size_t downweighted = 0;

        std::size_t ground_correspondences = 0;
        std::size_t ground_downweighted = 0;

        std::size_t geometry_weak_directions = 0;
        std::size_t prior_directions = 0;
        std::size_t suppressed_directions = 0;

        double raw_rmse =
            std::numeric_limits<double>::infinity();

        double robust_rmse =
            std::numeric_limits<double>::infinity();

        double median_range =
            std::numeric_limits<double>::quiet_NaN();

        double scale_L =
            std::numeric_limits<double>::quiet_NaN();

        double condition_number =
            std::numeric_limits<double>::infinity();

        double total_condition_number =
            std::numeric_limits<double>::infinity();

        Eigen::Matrix<double, 6, 1> eigenvalues =
            Eigen::Matrix<double, 6, 1>::Zero();

        Eigen::Matrix<double, 6, 1> relative_eigenvalues =
            Eigen::Matrix<double, 6, 1>::Zero();

        Eigen::Matrix<double, 6, 6> eigenvectors =
            Eigen::Matrix<double, 6, 6>::Identity();

        Eigen::Matrix<double, 6, 6> parameter_unscale =
            Eigen::Matrix<double, 6, 6>::Identity();

        // IMPORTANT: this vector lives in normalized coordinates:
        // [L*rx, L*ry, L*rz, tx, ty, tz].
        Eigen::Matrix<double, 6, 1> weakest_direction_normalized =
            Eigen::Matrix<double, 6, 1>::Zero();

        Eigen::Matrix<double, 6, 1> total_relative_eigenvalues =
            Eigen::Matrix<double, 6, 1>::Zero();
    };

    Eigen::Vector3d RotationLogVector(
        const Eigen::Matrix3d &rotation)
    {
        if (!rotation.allFinite())
        {
            return Eigen::Vector3d::Zero();
        }

        Eigen::AngleAxisd angle_axis(rotation);

        const double angle =
            angle_axis.angle();

        const Eigen::Vector3d axis =
            angle_axis.axis();

        if (!std::isfinite(angle) ||
            !axis.allFinite() ||
            std::abs(angle) < 1.0e-12)
        {
            return Eigen::Vector3d::Zero();
        }

        return angle * axis;
    }

    bool AnalyzeLoopP2PlaneHessianV2(
        const Eigen::Matrix<double, 6, 6> &H_geometry,
        const std::vector<double> &correspondence_ranges,
        const LoopVerifierConfig &config,
        LoopP2PlaneIterationDiagnostic &diagnostic)
    {
        if (!H_geometry.allFinite() ||
            correspondence_ranges.empty())
        {
            return false;
        }

        diagnostic.median_range =
            MedianValue(
                correspondence_ranges);

        if (!std::isfinite(diagnostic.median_range) ||
            diagnostic.median_range <= 0.0)
        {
            return false;
        }

        diagnostic.scale_L =
            std::clamp(
                diagnostic.median_range,
                kShadowMinimumScaleRange,
                kShadowMaximumScaleRange);

        diagnostic.parameter_unscale.setIdentity();

        const double inverse_scale =
            1.0 /
            diagnostic.scale_L;

        // delta_raw = parameter_unscale * delta_normalized.
        diagnostic.parameter_unscale(0, 0) = inverse_scale;
        diagnostic.parameter_unscale(1, 1) = inverse_scale;
        diagnostic.parameter_unscale(2, 2) = inverse_scale;

        Eigen::Matrix<double, 6, 6> H_analysis =
            diagnostic.parameter_unscale.transpose() *
            H_geometry *
            diagnostic.parameter_unscale;

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
            eigen_solver(
                H_analysis,
                Eigen::ComputeEigenvectors);

        if (eigen_solver.info() !=
            Eigen::Success)
        {
            return false;
        }

        diagnostic.eigenvalues =
            eigen_solver.eigenvalues();

        diagnostic.eigenvectors =
            eigen_solver.eigenvectors();

        if (!diagnostic.eigenvalues.allFinite() ||
            !diagnostic.eigenvectors.allFinite())
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

        diagnostic.weakest_direction_normalized =
            diagnostic.eigenvectors.col(0);

        diagnostic.geometry_weak_directions = 0;

        for (int i = 0;
             i < 6;
             ++i)
        {
            if (diagnostic.relative_eigenvalues(i) <
                config.degeneracy_weak_relative_threshold)
            {
                ++diagnostic.geometry_weak_directions;
            }
        }

        diagnostic.valid = true;
        return true;
    }

    bool RunPointToPlaneGaussNewtonV2(
        const PreparedSource &source,
        const PreparedTarget &target,
        const Eigen::Isometry3d &initial_guess,
        bool enable_trusted_seed_prior,
        const LoopVerifierConfig &config,
        Eigen::Isometry3d &T_target_source,
        std::size_t &iterations_completed,
        bool &stopped_by_increment,
        bool &stopped_by_fitness)
    {
        T_target_source =
            initial_guess;

        iterations_completed = 0;
        stopped_by_increment = false;
        stopped_by_fitness = false;

        if (!source.cloud ||
            !target.cloud ||
            !target.kdtree ||
            source.cloud->empty() ||
            target.cloud->empty() ||
            !initial_guess.matrix().allFinite())
        {
            return false;
        }

        const double maximum_squared_distance =
            config.max_correspondence_distance *
            config.max_correspondence_distance;

        const double increment_epsilon =
            std::sqrt(
                std::max(
                    config.transformation_epsilon,
                    1.0e-16));

        const double ground_normal_cosine_threshold =
            std::cos(
                config.ground_normal_max_tilt_deg *
                kPi /
                180.0);

        double previous_raw_rmse =
            std::numeric_limits<double>::infinity();

        bool solved_at_least_once = false;

        std::vector<int> neighbor_indices(
            static_cast<std::size_t>(
                kShadowPlaneKnn));

        std::vector<float> neighbor_squared_distances(
            static_cast<std::size_t>(
                kShadowPlaneKnn));

        for (std::size_t iteration = 0;
             iteration < config.max_iterations;
             ++iteration)
        {
            Eigen::Matrix<double, 6, 6> H_geometry =
                Eigen::Matrix<double, 6, 6>::Zero();

            Eigen::Matrix<double, 6, 1> b_geometry =
                Eigen::Matrix<double, 6, 1>::Zero();

            Eigen::Matrix<double, 6, 6> H_ground =
                Eigen::Matrix<double, 6, 6>::Zero();

            Eigen::Matrix<double, 6, 1> b_ground =
                Eigen::Matrix<double, 6, 1>::Zero();

            double raw_squared_error_sum = 0.0;
            double robust_squared_error_sum = 0.0;
            double robust_weight_sum = 0.0;

            LoopP2PlaneIterationDiagnostic diagnostic;
            diagnostic.trusted_seed_prior =
                enable_trusted_seed_prior &&
                config.enable_weak_direction_seed_prior;

            std::vector<double> correspondence_ranges;
            correspondence_ranges.reserve(
                source.cloud->size());

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
                query.x =
                    static_cast<float>(p_target.x());
                query.y =
                    static_cast<float>(p_target.y());
                query.z =
                    static_cast<float>(p_target.z());

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
                    nearest_squared_distance >
                        maximum_squared_distance)
                {
                    continue;
                }

                Eigen::Vector3d plane_point;
                Eigen::Vector3d plane_normal;

                if (!FitShadowPlane(
                        target.cloud,
                        neighbor_indices,
                        plane_point,
                        plane_normal))
                {
                    ++diagnostic.plane_fit_failures;
                    continue;
                }

                const double residual =
                    plane_normal.dot(
                        p_target -
                        plane_point);

                if (!std::isfinite(residual) ||
                    std::abs(residual) >
                        config.point_to_plane_max_residual)
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

                // Sensor-centered perturbation, order = [rx ry rz tx ty tz].
                Eigen::Matrix<double, 1, 6> J =
                    Eigen::Matrix<double, 1, 6>::Zero();

                J.block<1, 3>(0, 0) =
                    lever_arm_target.cross(
                                        plane_normal)
                        .transpose();

                J.block<1, 3>(0, 3) =
                    plane_normal.transpose();

                const double absolute_residual =
                    std::abs(residual);

                double huber_weight = 1.0;

                if (absolute_residual >
                        config.point_to_plane_huber_delta &&
                    absolute_residual > 1.0e-12)
                {
                    huber_weight =
                        config.point_to_plane_huber_delta /
                        absolute_residual;

                    ++diagnostic.downweighted;
                }

                H_geometry.noalias() +=
                    huber_weight *
                    J.transpose() *
                    J;

                b_geometry.noalias() +=
                    huber_weight *
                    J.transpose() *
                    residual;

                raw_squared_error_sum +=
                    residual *
                    residual;

                robust_squared_error_sum +=
                    huber_weight *
                    residual *
                    residual;

                robust_weight_sum +=
                    huber_weight;

                const double range =
                    lever_arm_target.norm();

                if (std::isfinite(range) &&
                    range > 1.0e-9)
                {
                    correspondence_ranges.push_back(
                        range);
                }

                ++diagnostic.correspondences;

                // ------------------------------------------------------------
                // Loop-local ground-like extra information.
                //
                // Unlike realtime frontend Ground V1.3, LoopVerifier currently
                // does not receive a persisted GroundSegmentationResult for a
                // Keyframe.  Therefore V2 infers ground-like correspondences
                // conservatively from the already-fitted target plane:
                //   * normal close to target-frame +/-Z,
                //   * source point is below the current LiDAR origin,
                //   * small point-to-plane residual.
                //
                // The Jacobian is then projected to [roll,pitch,z] exactly as
                // in frontend Ground V1.3: rz / tx / ty are zeroed.
                // ------------------------------------------------------------
                if (config.enable_ground_constraint &&
                    std::abs(plane_normal.z()) >=
                        ground_normal_cosine_threshold &&
                    lever_arm_target.z() <=
                        -config.ground_min_below_sensor_m &&
                    absolute_residual <=
                        config.ground_max_residual)
                {
                    Eigen::Matrix<double, 1, 6> J_ground =
                        J;

                    J_ground(0, 2) = 0.0; // yaw
                    J_ground(0, 3) = 0.0; // x
                    J_ground(0, 4) = 0.0; // y

                    double ground_robust_weight =
                        1.0;

                    if (absolute_residual >
                            config.ground_huber_delta &&
                        absolute_residual > 1.0e-12)
                    {
                        ground_robust_weight =
                            config.ground_huber_delta /
                            absolute_residual;

                        ++diagnostic.ground_downweighted;
                    }

                    const double final_ground_weight =
                        config.ground_weight *
                        ground_robust_weight;

                    H_ground.noalias() +=
                        final_ground_weight *
                        J_ground.transpose() *
                        J_ground;

                    b_ground.noalias() +=
                        final_ground_weight *
                        J_ground.transpose() *
                        residual;

                    ++diagnostic.ground_correspondences;
                }
            }

            if (diagnostic.correspondences <
                    kLoopP2PlaneMinimumCorrespondences ||
                !H_geometry.allFinite() ||
                !b_geometry.allFinite())
            {
                std::cout
                    << "LOOP_P2PLANE_V2"
                    << " | iteration=" << iteration
                    << " | action=STOP_INSUFFICIENT_GEOMETRY"
                    << " | corr="
                    << diagnostic.correspondences
                    << " | min_corr="
                    << kLoopP2PlaneMinimumCorrespondences
                    << " | plane_fit_failures="
                    << diagnostic.plane_fit_failures
                    << std::endl;
                break;
            }

            diagnostic.ground_active =
                config.enable_ground_constraint &&
                diagnostic.ground_correspondences >=
                    config.min_ground_correspondences;

            if (!diagnostic.ground_active)
            {
                H_ground.setZero();
                b_ground.setZero();
            }

            diagnostic.raw_rmse =
                std::sqrt(
                    raw_squared_error_sum /
                    static_cast<double>(
                        diagnostic.correspondences));

            if (robust_weight_sum > 0.0)
            {
                diagnostic.robust_rmse =
                    std::sqrt(
                        robust_squared_error_sum /
                        robust_weight_sum);
            }

            if (!AnalyzeLoopP2PlaneHessianV2(
                    H_geometry,
                    correspondence_ranges,
                    config,
                    diagnostic))
            {
                std::cout
                    << "LOOP_P2PLANE_V2"
                    << " | iteration=" << iteration
                    << " | action=STOP_HESSIAN_ANALYSIS_FAILED"
                    << std::endl;
                break;
            }

            // Geometry and Ground are converted to the same normalized space.
            Eigen::Matrix<double, 6, 6> H_total_analysis =
                diagnostic.parameter_unscale.transpose() *
                (H_geometry + H_ground) *
                diagnostic.parameter_unscale;

            Eigen::Matrix<double, 6, 1> b_total_analysis =
                diagnostic.parameter_unscale.transpose() *
                (b_geometry + b_ground);

            H_total_analysis =
                0.5 *
                (H_total_analysis +
                 H_total_analysis.transpose());

            // ------------------------------------------------------------
            // Weak-direction seed prior.
            //
            // The prior is added ONLY when the caller explicitly marks this
            // initial guess trusted, and ONLY along eigenvectors that geometry
            // itself says are weak.  It never constrains a strong geometry
            // direction.
            // ------------------------------------------------------------
            if (diagnostic.trusted_seed_prior)
            {
                Eigen::Matrix<double, 6, 1> seed_error_raw =
                    Eigen::Matrix<double, 6, 1>::Zero();

                seed_error_raw.head<3>() =
                    RotationLogVector(
                        T_target_source.rotation() *
                        initial_guess.rotation().transpose());

                seed_error_raw.tail<3>() =
                    T_target_source.translation() -
                    initial_guess.translation();

                Eigen::Matrix<double, 6, 1> seed_error_analysis =
                    seed_error_raw;

                seed_error_analysis.head<3>() *=
                    diagnostic.scale_L;

                const double geometry_lambda_max =
                    diagnostic.eigenvalues(5);

                const double target_relative =
                    std::max(
                        config.degeneracy_weak_relative_threshold,
                        config.weak_direction_prior_target_relative);

                const double target_information =
                    target_relative *
                    geometry_lambda_max;

                for (int i = 0;
                     i < 6;
                     ++i)
                {
                    const double relative_lambda =
                        diagnostic.relative_eigenvalues(i);

                    if (!std::isfinite(relative_lambda) ||
                        relative_lambda >=
                            config.degeneracy_weak_relative_threshold)
                    {
                        continue;
                    }

                    const double missing_information =
                        std::max(
                            0.0,
                            target_information -
                                diagnostic.eigenvalues(i));

                    const double prior_information =
                        config.weak_direction_prior_gain *
                        missing_information;

                    if (!std::isfinite(prior_information) ||
                        prior_information <= 0.0)
                    {
                        continue;
                    }

                    const Eigen::Matrix<double, 6, 1> direction =
                        diagnostic.eigenvectors.col(i);

                    const double projected_seed_error =
                        direction.dot(
                            seed_error_analysis);

                    H_total_analysis.noalias() +=
                        prior_information *
                        direction *
                        direction.transpose();

                    b_total_analysis.noalias() +=
                        prior_information *
                        direction *
                        projected_seed_error;

                    ++diagnostic.prior_directions;
                }
            }

            H_total_analysis =
                0.5 *
                (H_total_analysis +
                 H_total_analysis.transpose());

            if (!H_total_analysis.allFinite() ||
                !b_total_analysis.allFinite())
            {
                break;
            }

            Eigen::SelfAdjointEigenSolver<
                Eigen::Matrix<double, 6, 6>>
                total_solver(
                    H_total_analysis,
                    Eigen::ComputeEigenvectors);

            if (total_solver.info() !=
                Eigen::Success)
            {
                std::cout
                    << "LOOP_P2PLANE_V2"
                    << " | iteration=" << iteration
                    << " | action=STOP_SOLVE_FAILURE"
                    << std::endl;
                break;
            }

            const Eigen::Matrix<double, 6, 1>
                total_eigenvalues =
                    total_solver.eigenvalues();

            const Eigen::Matrix<double, 6, 6>
                total_eigenvectors =
                    total_solver.eigenvectors();

            if (!total_eigenvalues.allFinite() ||
                !total_eigenvectors.allFinite())
            {
                break;
            }

            const double total_lambda_max =
                total_eigenvalues(5);

            if (!std::isfinite(total_lambda_max) ||
                total_lambda_max <= 1.0e-12)
            {
                break;
            }

            diagnostic.total_relative_eigenvalues =
                total_eigenvalues /
                total_lambda_max;

            if (total_eigenvalues(0) > 1.0e-12)
            {
                diagnostic.total_condition_number =
                    total_lambda_max /
                    total_eigenvalues(0);
            }

            const Eigen::Matrix<double, 6, 1>
                gradient_eigen =
                    total_eigenvectors.transpose() *
                    b_total_analysis;

            Eigen::Matrix<double, 6, 1>
                delta_eigen =
                    Eigen::Matrix<double, 6, 1>::Zero();

            const double damping =
                kLoopP2PlaneNumericalDampingRatio *
                std::max(
                    1.0,
                    total_lambda_max);

            diagnostic.suppressed_directions = 0;

            int usable_directions = 0;

            for (int i = 0;
                 i < 6;
                 ++i)
            {
                const double lambda =
                    total_eigenvalues(i);

                const double relative_lambda =
                    diagnostic.total_relative_eigenvalues(i);

                if (!std::isfinite(lambda) ||
                    lambda <= 1.0e-12 ||
                    !std::isfinite(relative_lambda))
                {
                    ++diagnostic.suppressed_directions;
                    continue;
                }

                double direction_scale = 1.0;

                if (config.enable_degeneracy_suppression)
                {
                    if (relative_lambda <
                        config.degeneracy_hard_relative_threshold)
                    {
                        ++diagnostic.suppressed_directions;
                        continue;
                    }

                    if (relative_lambda <
                        config.degeneracy_weak_relative_threshold)
                    {
                        direction_scale =
                            std::clamp(
                                relative_lambda /
                                    config.degeneracy_weak_relative_threshold,
                                0.0,
                                1.0);

                        ++diagnostic.suppressed_directions;
                    }
                }

                delta_eigen(i) =
                    direction_scale *
                    (-gradient_eigen(i) /
                     (lambda + damping));

                ++usable_directions;
            }

            if (usable_directions <= 0)
            {
                std::cout
                    << "LOOP_P2PLANE_V2"
                    << " | iteration=" << iteration
                    << " | action=STOP_ALL_DIRECTIONS_SUPPRESSED"
                    << std::endl;
                break;
            }

            const Eigen::Matrix<double, 6, 1>
                delta_analysis =
                    total_eigenvectors *
                    delta_eigen;

            const Eigen::Matrix<double, 6, 1>
                delta =
                    diagnostic.parameter_unscale *
                    delta_analysis;

            if (!delta.allFinite())
            {
                break;
            }

            const Eigen::Vector3d delta_rotation =
                delta.head<3>();

            const Eigen::Vector3d delta_translation =
                delta.tail<3>();

            const double delta_rotation_rad =
                delta_rotation.norm();

            const double delta_translation_norm =
                delta_translation.norm();

            std::cout
                << "LOOP_P2PLANE_V2"
                << " | iteration=" << iteration
                << " | corr="
                << diagnostic.correspondences
                << " | plane_fit_failures="
                << diagnostic.plane_fit_failures
                << " | general_huber_delta="
                << config.point_to_plane_huber_delta
                << " m"
                << " | downweighted="
                << diagnostic.downweighted
                << " | ground_active="
                << (diagnostic.ground_active ? "true" : "false")
                << " | ground_corr="
                << diagnostic.ground_correspondences
                << " | ground_weight="
                << config.ground_weight
                << " | trusted_seed_prior="
                << (diagnostic.trusted_seed_prior ? "true" : "false")
                << " | geometry_weak_dirs="
                << diagnostic.geometry_weak_directions
                << " | prior_dirs="
                << diagnostic.prior_directions
                << " | suppressed_dirs="
                << diagnostic.suppressed_directions
                << " | raw_rmse="
                << diagnostic.raw_rmse
                << " m"
                << " | robust_rmse="
                << diagnostic.robust_rmse
                << " m"
                << " | dT="
                << delta_translation_norm
                << " m"
                << " | dR="
                << delta_rotation_rad *
                       180.0 / kPi
                << " deg"
                << std::endl;

            std::cout
                << "LOOP_P2PLANE_HESSIAN_V2"
                << " | iteration=" << iteration
                << " | valid="
                << (diagnostic.valid ? "true" : "false")
                << " | coordinates=[Lrx Lry Lrz tx ty tz]"
                << " | median_range="
                << diagnostic.median_range
                << " m"
                << " | scale_L="
                << diagnostic.scale_L
                << " m"
                << " | geometry_eigen=["
                << diagnostic.eigenvalues.transpose()
                << "]"
                << " | geometry_relative=["
                << diagnostic.relative_eigenvalues.transpose()
                << "]"
                << " | geometry_min_relative="
                << diagnostic.relative_eigenvalues.minCoeff()
                << " | geometry_condition="
                << diagnostic.condition_number
                << " | weakest_normalized=["
                << diagnostic.weakest_direction_normalized.transpose()
                << "]"
                << " | total_relative=["
                << diagnostic.total_relative_eigenvalues.transpose()
                << "]"
                << " | total_condition="
                << diagnostic.total_condition_number
                << std::endl;

            Eigen::Matrix3d delta_R =
                Eigen::Matrix3d::Identity();

            if (delta_rotation_rad >
                1.0e-12)
            {
                delta_R =
                    Eigen::AngleAxisd(
                        delta_rotation_rad,
                        delta_rotation /
                            delta_rotation_rad)
                        .toRotationMatrix();
            }

            T_target_source.linear() =
                delta_R *
                T_target_source.rotation();

            T_target_source.translation() +=
                delta_translation;

            if (!T_target_source.matrix().allFinite())
            {
                return false;
            }

            ++iterations_completed;
            solved_at_least_once = true;

            const bool increment_converged =
                delta_translation_norm <=
                    increment_epsilon &&
                delta_rotation_rad <=
                    increment_epsilon;

            const bool fitness_converged =
                std::isfinite(previous_raw_rmse) &&
                std::isfinite(diagnostic.raw_rmse) &&
                std::abs(
                    previous_raw_rmse -
                    diagnostic.raw_rmse) <=
                    config.euclidean_fitness_epsilon;

            previous_raw_rmse =
                diagnostic.raw_rmse;

            if (increment_converged)
            {
                stopped_by_increment = true;
                break;
            }

            if (fitness_converged)
            {
                stopped_by_fitness = true;
                break;
            }
        }

        return solved_at_least_once &&
               T_target_source.matrix().allFinite();
    }

    LoopVerificationResult RunHypothesis(
        const PreparedSource &source,
        const PreparedTarget &target,
        const Eigen::Isometry3d &initial_guess,
        LoopVerifierHypothesis hypothesis,
        bool enable_trusted_seed_prior,
        bool enable_trusted_reverse_sequence_gate,
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

        const std::size_t minimum_source_cloud_points =
            enable_trusted_reverse_sequence_gate
                ? config.trusted_reverse_min_cloud_points
                : config.min_cloud_points;

        if (!source.cloud ||
            !target.cloud ||
            !target.kdtree ||
            source.cloud->size() < minimum_source_cloud_points ||
            target.cloud->size() < config.min_cloud_points ||
            !initial_guess.matrix().allFinite())
        {
            return result;
        }

        std::size_t iterations_completed = 0;
        bool stopped_by_increment = false;
        bool stopped_by_fitness = false;

        const bool solver_ok =
            RunPointToPlaneGaussNewtonV2(
                source,
                target,
                initial_guess,
                enable_trusted_seed_prior,
                config,
                result.T_target_source,
                iterations_completed,
                stopped_by_increment,
                stopped_by_fitness);

        result.converged =
            solver_ok;

        if (!solver_ok ||
            !result.T_target_source
                 .matrix()
                 .allFinite())
        {
            return result;
        }

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

        // Keep the legacy fitness_score field meaningful for diagnostics.
        // Final acceptance still uses the unchanged explicit point-to-point
        // overlap/RMSE measurement below.
        result.fitness_score =
            std::isfinite(result.rmse)
                ? result.rmse * result.rmse
                : std::numeric_limits<double>::infinity();

        result.correction_translation =
            (result.T_target_source.translation() -
             initial_guess.translation())
                .norm();

        result.correction_rotation_deg =
            RelativeRotationDeg(
                initial_guess,
                result.T_target_source);

        const Eigen::Vector3d initial_t =
            initial_guess.translation();

        const Eigen::Vector3d final_t =
            result.T_target_source.translation();

        const Eigen::Vector3d delta_t =
            final_t - initial_t;

        // Keep the old grep token so existing log scripts still work, but
        // identify the new pose solver explicitly.
        std::cout
            << "LOOP_ICP_POSE_DEBUG"
            << " | solver=POINT_TO_PLANE_V2_DEGENERACY_GROUND"
            << " | hypothesis="
            << static_cast<int>(hypothesis)
            << " | initial_t=["
            << initial_t.transpose()
            << "]"
            << " | initial_norm="
            << initial_t.norm()
            << " m"
            << " | final_t=["
            << final_t.transpose()
            << "]"
            << " | final_norm="
            << final_t.norm()
            << " m"
            << " | delta_t=["
            << delta_t.transpose()
            << "]"
            << " | delta_norm="
            << delta_t.norm()
            << " m"
            << " | correction_rotation="
            << result.correction_rotation_deg
            << " deg"
            << " | iterations="
            << iterations_completed
            << " | stop="
            << (stopped_by_increment
                    ? "INCREMENT"
                    : (stopped_by_fitness
                           ? "FITNESS"
                           : "MAX_ITER_OR_OTHER"))
            << std::endl;

        result.success = true;

        const bool standard_geometry_accepted =
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

        // V2.1: confirmed reverse traversal has intrinsically asymmetric
        // single-frame visibility.  Relax ONLY point-count / overlap support,
        // while making RMSE and pose-correction limits much tighter.
        //
        // This flag is supplied only for CANDIDATE_REVERSE_SEQUENCE after the
        // RAW-SC temporal sequence has been confirmed in fr_loop_backend.cpp.
        const bool trusted_reverse_geometry_accepted =
            enable_trusted_reverse_sequence_gate &&
            result.inliers >= config.trusted_reverse_min_inliers &&
            result.overlap_ratio >=
                config.trusted_reverse_min_overlap_ratio &&
            std::isfinite(result.rmse) &&
            result.rmse <=
                config.trusted_reverse_max_rmse &&
            std::isfinite(result.correction_translation) &&
            result.correction_translation <=
                config.trusted_reverse_max_correction_translation &&
            std::isfinite(result.correction_rotation_deg) &&
            result.correction_rotation_deg <=
                config.trusted_reverse_max_correction_rotation_deg;

        result.accepted =
            standard_geometry_accepted ||
            trusted_reverse_geometry_accepted;

        if (enable_trusted_reverse_sequence_gate)
        {
            std::cout
                << "LOOP_REVERSE_SEQUENCE_GATE_V2_1"
                << " | source_points=" << result.source_points
                << " | target_points=" << result.target_points
                << " | inliers=" << result.inliers
                << " | overlap=" << result.overlap_ratio
                << " | rmse=" << result.rmse << " m"
                << " | correction_translation="
                << result.correction_translation << " m"
                << " | correction_rotation="
                << result.correction_rotation_deg << " deg"
                << " | standard="
                << (standard_geometry_accepted ? "PASS" : "FAIL")
                << " | trusted_reverse="
                << (trusted_reverse_geometry_accepted ? "PASS" : "FAIL")
                << std::endl;
        }

        // Final-pose observability diagnostic remains diagnostic-only.
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

    if (config_.trusted_reverse_min_cloud_points == 0)
    {
        config_.trusted_reverse_min_cloud_points = 150;
    }

    if (config_.trusted_reverse_min_inliers == 0)
    {
        config_.trusted_reverse_min_inliers = 100;
    }

    if (!std::isfinite(config_.trusted_reverse_min_overlap_ratio) ||
        config_.trusted_reverse_min_overlap_ratio <= 0.0 ||
        config_.trusted_reverse_min_overlap_ratio > 1.0)
    {
        config_.trusted_reverse_min_overlap_ratio = 0.35;
    }

    if (!std::isfinite(config_.trusted_reverse_max_rmse) ||
        config_.trusted_reverse_max_rmse <= 0.0)
    {
        config_.trusted_reverse_max_rmse = 0.50;
    }

    if (!std::isfinite(config_.trusted_reverse_max_correction_translation) ||
        config_.trusted_reverse_max_correction_translation <= 0.0)
    {
        config_.trusted_reverse_max_correction_translation = 1.50;
    }

    if (!std::isfinite(config_.trusted_reverse_max_correction_rotation_deg) ||
        config_.trusted_reverse_max_correction_rotation_deg <= 0.0)
    {
        config_.trusted_reverse_max_correction_rotation_deg = 15.0;
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

    if (!std::isfinite(config_.point_to_plane_max_residual) ||
        config_.point_to_plane_max_residual <= 0.0)
    {
        config_.point_to_plane_max_residual = 1.00;
    }

    if (!std::isfinite(config_.point_to_plane_huber_delta) ||
        config_.point_to_plane_huber_delta <= 0.0)
    {
        config_.point_to_plane_huber_delta = 0.10;
    }

    if (!std::isfinite(config_.degeneracy_hard_relative_threshold) ||
        config_.degeneracy_hard_relative_threshold <= 0.0 ||
        config_.degeneracy_hard_relative_threshold >= 1.0)
    {
        config_.degeneracy_hard_relative_threshold = 0.01;
    }

    if (!std::isfinite(config_.degeneracy_weak_relative_threshold) ||
        config_.degeneracy_weak_relative_threshold <=
            config_.degeneracy_hard_relative_threshold ||
        config_.degeneracy_weak_relative_threshold >= 1.0)
    {
        config_.degeneracy_weak_relative_threshold = 0.03;
    }

    if (!std::isfinite(config_.weak_direction_prior_target_relative) ||
        config_.weak_direction_prior_target_relative <= 0.0 ||
        config_.weak_direction_prior_target_relative >= 1.0)
    {
        config_.weak_direction_prior_target_relative = 0.05;
    }

    if (!std::isfinite(config_.weak_direction_prior_gain) ||
        config_.weak_direction_prior_gain < 0.0)
    {
        config_.weak_direction_prior_gain = 1.0;
    }

    if (!std::isfinite(config_.ground_weight) ||
        config_.ground_weight < 0.0)
    {
        config_.ground_weight = 4.0;
    }

    if (!std::isfinite(config_.ground_normal_max_tilt_deg) ||
        config_.ground_normal_max_tilt_deg <= 0.0 ||
        config_.ground_normal_max_tilt_deg >= 89.0)
    {
        config_.ground_normal_max_tilt_deg = 25.0;
    }

    if (!std::isfinite(config_.ground_min_below_sensor_m) ||
        config_.ground_min_below_sensor_m < 0.0)
    {
        config_.ground_min_below_sensor_m = 0.30;
    }

    if (!std::isfinite(config_.ground_max_residual) ||
        config_.ground_max_residual <= 0.0)
    {
        config_.ground_max_residual = 0.25;
    }

    if (!std::isfinite(config_.ground_huber_delta) ||
        config_.ground_huber_delta <= 0.0)
    {
        config_.ground_huber_delta = 0.10;
    }

    if (config_.min_ground_correspondences == 0)
    {
        config_.min_ground_correspondences = 30;
    }

    if (config_.max_cached_targets == 0)
    {
        config_.max_cached_targets = 1;
    }

    std::cout
        << "LoopVerifier Point-to-Plane V2"
        << " | normalized_coordinates=[Lrx Lry Lrz tx ty tz]"
        << " | hard_rel=" << config_.degeneracy_hard_relative_threshold
        << " | weak_rel=" << config_.degeneracy_weak_relative_threshold
        << " | weak_seed_prior="
        << (config_.enable_weak_direction_seed_prior ? "ON" : "OFF")
        << " | prior_target_rel="
        << config_.weak_direction_prior_target_relative
        << " | ground="
        << (config_.enable_ground_constraint ? "ON" : "OFF")
        << " | ground_weight=" << config_.ground_weight
        << " | ground_dofs=ROLL_PITCH_Z"
        << " | ground_source=LOOP_LOCAL_PLANE_INFERENCE"
        << " | trusted_reverse_sequence_gate=ON"
        << " | reverse_min_source_points="
        << config_.trusted_reverse_min_cloud_points
        << " | reverse_min_inliers="
        << config_.trusted_reverse_min_inliers
        << " | reverse_min_overlap="
        << config_.trusted_reverse_min_overlap_ratio
        << " | reverse_max_rmse="
        << config_.trusted_reverse_max_rmse
        << " | reverse_max_corr_t="
        << config_.trusted_reverse_max_correction_translation
        << " | reverse_max_corr_R="
        << config_.trusted_reverse_max_correction_rotation_deg
        << std::endl;
}

LoopVerifier::~LoopVerifier() = default;

bool LoopVerifier::ScoreInitialGuess(
    const pcl::PointCloud<LIDAR_POINT>::ConstPtr &source_current,
    const pcl::PointCloud<LIDAR_POINT>::ConstPtr &target_historical,
    const Eigen::Isometry3d &initial_guess,
    LoopVerifierInitialGuessScore &score) const
{
    return ScoreInitialGuess(
        source_current,
        target_historical,
        initial_guess,
        false,
        score);
}

bool LoopVerifier::ScoreInitialGuess(
    const pcl::PointCloud<LIDAR_POINT>::ConstPtr &source_current,
    const pcl::PointCloud<LIDAR_POINT>::ConstPtr &target_historical,
    const Eigen::Isometry3d &initial_guess,
    bool enable_trusted_reverse_sequence_gate,
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

        const std::size_t minimum_source_cloud_points =
            enable_trusted_reverse_sequence_gate
                ? config_.trusted_reverse_min_cloud_points
                : config_.min_cloud_points;

        if (!source_filtered ||
            source_filtered->size() < minimum_source_cloud_points)
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
    return Verify(
        source_current,
        target_historical,
        graph_initial_guess,
        scan_context_yaw_shift_deg,
        false,
        result);
}

bool LoopVerifier::Verify(
    const pcl::PointCloud<LIDAR_POINT>::ConstPtr &source_current,
    const pcl::PointCloud<LIDAR_POINT>::ConstPtr &target_historical,
    const Eigen::Isometry3d &graph_initial_guess,
    double scan_context_yaw_shift_deg,
    bool enable_trusted_seed_prior,
    LoopVerificationResult &result) const
{
    return Verify(
        source_current,
        target_historical,
        graph_initial_guess,
        scan_context_yaw_shift_deg,
        enable_trusted_seed_prior,
        false,
        result);
}

bool LoopVerifier::Verify(
    const pcl::PointCloud<LIDAR_POINT>::ConstPtr &source_current,
    const pcl::PointCloud<LIDAR_POINT>::ConstPtr &target_historical,
    const Eigen::Isometry3d &graph_initial_guess,
    double scan_context_yaw_shift_deg,
    bool enable_trusted_seed_prior,
    bool enable_trusted_reverse_sequence_gate,
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

        const std::size_t minimum_source_cloud_points =
            enable_trusted_reverse_sequence_gate
                ? config_.trusted_reverse_min_cloud_points
                : config_.min_cloud_points;

        if (!source_filtered ||
            source_filtered->size() < minimum_source_cloud_points)
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
                enable_trusted_seed_prior,
                enable_trusted_reverse_sequence_gate,
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
