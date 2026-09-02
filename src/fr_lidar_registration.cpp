#include "fr_slam/fr_lidar_registration.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#include <Eigen/Eigenvalues>

#include <sophus/so3.hpp>

LidarRegistration::LidarRegistration(
    const LidarRegistrationConfig &config)
    : config_(config)
{
}

bool LidarRegistration::FitLocalPlane(
    const pcl::PointCloud<LIDAR_POINT>::ConstPtr &target,
    const std::vector<int> &neighbor_indices,
    Eigen::Vector3d &plane_point,
    Eigen::Vector3d &plane_normal) const
{
    if (!target ||
        neighbor_indices.size() < 3)
    {
        return false;
    }

    Eigen::Vector3d centroid =
        Eigen::Vector3d::Zero();

    for (const int index :
         neighbor_indices)
    {
        if (index < 0 ||
            static_cast<std::size_t>(index) >=
                target->size())
        {
            return false;
        }

        const LIDAR_POINT &point =
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

    double c_xx = 0.0;
    double c_xy = 0.0;
    double c_xz = 0.0;
    double c_yy = 0.0;
    double c_yz = 0.0;
    double c_zz = 0.0;

    const double cx =
        centroid.x();

    const double cy =
        centroid.y();

    const double cz =
        centroid.z();

    for (const int index :
         neighbor_indices)
    {
        const LIDAR_POINT &point =
            target->points[static_cast<std::size_t>(index)];

        const double dx =
            static_cast<double>(point.x) - cx;

        const double dy =
            static_cast<double>(point.y) - cy;

        const double dz =
            static_cast<double>(point.z) - cz;

        c_xx += dx * dx;
        c_xy += dx * dy;
        c_xz += dx * dz;
        c_yy += dy * dy;
        c_yz += dy * dz;
        c_zz += dz * dz;
    }

    const double inv_n =
        1.0 /
        static_cast<double>(
            neighbor_indices.size());

    Eigen::Matrix3d covariance;

    covariance << c_xx * inv_n,
        c_xy * inv_n,
        c_xz * inv_n,
        c_xy * inv_n,
        c_yy * inv_n,
        c_yz * inv_n,
        c_xz * inv_n,
        c_yz * inv_n,
        c_zz * inv_n;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>
        eigen_solver;

    eigen_solver.computeDirect(
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

    normal /=
        normal_norm;

    for (const int index :
         neighbor_indices)
    {
        const LIDAR_POINT &point =
            target->points[static_cast<std::size_t>(index)];

        const Eigen::Vector3d p(
            static_cast<double>(point.x),
            static_cast<double>(point.y),
            static_cast<double>(point.z));

        const double point_to_plane_distance =
            std::abs(
                normal.dot(
                    p - centroid));

        if (point_to_plane_distance >
            config_.max_plane_fit_error)
        {
            return false;
        }
    }

    plane_point =
        centroid;

    plane_normal =
        normal;

    return true;
}

bool LidarRegistration::PrepareTarget(
    const pcl::PointCloud<LIDAR_POINT>::ConstPtr &target,
    PreparedLidarTarget &prepared_target) const
{
    prepared_target =
        PreparedLidarTarget();

    if (!target ||
        target->empty())
    {
        std::cerr
            << "LidarRegistration::PrepareTarget(): "
            << "target is empty."
            << std::endl;

        return false;
    }

    if (config_.knn < 3)
    {
        std::cerr
            << "LidarRegistration::PrepareTarget(): "
            << "knn must be >= 3 for plane fitting."
            << std::endl;

        return false;
    }

    prepared_target.cloud =
        target;

    prepared_target.kdtree =
        std::make_shared<
            pcl::KdTreeFLANN<LIDAR_POINT>>();

    prepared_target.kdtree->setInputCloud(
        target);

    prepared_target.planes.resize(
        target->size());

    std::vector<int>
        neighbor_indices(
            static_cast<std::size_t>(
                config_.knn));

    std::vector<float>
        neighbor_squared_distances(
            static_cast<std::size_t>(
                config_.knn));

    for (std::size_t i = 0;
         i < target->size();
         ++i)
    {
        const LIDAR_POINT &query_point =
            target->points[i];

        const int found =
            prepared_target.kdtree->nearestKSearch(
                query_point,
                config_.knn,
                neighbor_indices,
                neighbor_squared_distances);

        TargetPlane &plane =
            prepared_target.planes[i];

        if (found <
            config_.knn)
        {
            plane.state =
                TargetPlane::State::Invalid;

            ++prepared_target.invalid_planes;

            continue;
        }

        Eigen::Vector3d plane_point;
        Eigen::Vector3d plane_normal;

        const bool plane_ok =
            FitLocalPlane(
                target,
                neighbor_indices,
                plane_point,
                plane_normal);

        if (!plane_ok)
        {
            plane.state =
                TargetPlane::State::Invalid;

            ++prepared_target.invalid_planes;

            continue;
        }

        plane.point =
            plane_point;

        plane.normal =
            plane_normal;

        plane.state =
            TargetPlane::State::Valid;

        ++prepared_target.valid_planes;
    }

    prepared_target.ready =
        prepared_target.valid_planes >=
        config_.min_correspondences;

    if (!prepared_target.ready)
    {
        std::cerr
            << "LidarRegistration::PrepareTarget(): "
            << "not enough valid target planes. valid="
            << prepared_target.valid_planes
            << " required="
            << config_.min_correspondences
            << std::endl;
    }

    return prepared_target.ready;
}

bool LidarRegistration::Align(
    const pcl::PointCloud<LIDAR_POINT>::ConstPtr &source,
    const PreparedLidarTarget &target,
    const Eigen::Isometry3d &initial_guess,
    LidarRegistrationResult &result) const
{
    // ============================================================
    // 0. Initialize result
    // ============================================================

    result =
        LidarRegistrationResult();

    result.T_target_source =
        initial_guess;

    result.robust_kernel_enabled =
        config_.enable_huber_loss;

    result.robust_kernel_delta =
        config_.huber_delta;

    result.hessian_scale_normalization_enabled =
        config_.enable_sensor_centered_perturbation &&
        config_.enable_hessian_scale_normalization;

    if (result.hessian_scale_normalization_enabled &&
        (!std::isfinite(config_.hessian_scale_min_range) ||
         !std::isfinite(config_.hessian_scale_max_range) ||
         config_.hessian_scale_min_range <= 0.0 ||
         config_.hessian_scale_max_range <
             config_.hessian_scale_min_range))
    {
        std::cerr
            << "LidarRegistration::Align(): "
            << "invalid V2B Hessian scale range. min="
            << config_.hessian_scale_min_range
            << " max="
            << config_.hessian_scale_max_range
            << std::endl;

        return false;
    }

    if (config_.enable_huber_loss &&
        (!std::isfinite(config_.huber_delta) ||
         config_.huber_delta <= 0.0))
    {
        std::cerr
            << "LidarRegistration::Align(): "
            << "Huber delta must be finite and > 0. delta="
            << config_.huber_delta
            << std::endl;

        return false;
    }

    if (!source ||
        source->empty())
    {
        std::cerr
            << "LidarRegistration::Align(): "
            << "source is empty."
            << std::endl;

        return false;
    }

    if (!target.ready ||
        !target.cloud ||
        target.cloud->empty() ||
        !target.kdtree ||
        target.planes.size() !=
            target.cloud->size())
    {
        std::cerr
            << "LidarRegistration::Align(): "
            << "target has not been prepared correctly."
            << std::endl;

        return false;
    }

    if (config_.knn <= 0)
    {
        std::cerr
            << "LidarRegistration::Align(): "
            << "knn must be > 0."
            << std::endl;

        return false;
    }

    const std::vector<TargetPlane> &target_planes =
        target.planes;

    pcl::KdTreeFLANN<LIDAR_POINT> &kdtree =
        *target.kdtree;

    Eigen::Isometry3d T_target_source =
        initial_guess;

    std::vector<int>
        neighbor_indices(
            static_cast<std::size_t>(
                config_.knn));

    std::vector<float>
        neighbor_squared_distances(
            static_cast<std::size_t>(
                config_.knn));

    const double max_correspondence_distance_squared =
        config_.max_correspondence_distance *
        config_.max_correspondence_distance;

    // ============================================================
    // 2. Gauss-Newton
    // ============================================================

    for (int iteration = 0;
         iteration < config_.max_iterations;
         ++iteration)
    {
        Eigen::Matrix<double, 6, 6> H =
            Eigen::Matrix<double, 6, 6>::Zero();

        Eigen::Matrix<double, 6, 1> b =
            Eigen::Matrix<double, 6, 1>::Zero();

        double squared_error_sum =
            0.0;

        double robust_weighted_squared_error_sum =
            0.0;

        double robust_weight_sum =
            0.0;

        double minimum_robust_weight =
            1.0;

        std::size_t valid_correspondences =
            0;

        std::size_t downweighted_correspondences =
            0;

        const Eigen::Vector3d sensor_origin_target =
            T_target_source.translation();

        const bool use_hessian_scale_normalization =
            config_.enable_sensor_centered_perturbation &&
            config_.enable_hessian_scale_normalization;

        std::vector<double> correspondence_ranges;

        if (use_hessian_scale_normalization)
        {
            correspondence_ranges.reserve(
                source->size());
        }

        for (const LIDAR_POINT &source_point :
             source->points)
        {
            const Eigen::Vector3d p_source(
                static_cast<double>(source_point.x),
                static_cast<double>(source_point.y),
                static_cast<double>(source_point.z));

            const Eigen::Vector3d p_target =
                T_target_source *
                p_source;

            LIDAR_POINT query_point{};

            query_point.x =
                static_cast<float>(
                    p_target.x());

            query_point.y =
                static_cast<float>(
                    p_target.y());

            query_point.z =
                static_cast<float>(
                    p_target.z());

            const int found =
                kdtree.nearestKSearch(
                    query_point,
                    config_.knn,
                    neighbor_indices,
                    neighbor_squared_distances);

            if (found <= 0)
            {
                continue;
            }

            int plane_index =
                -1;

            for (int j = 0;
                 j < found;
                 ++j)
            {
                const double neighbor_distance_squared =
                    static_cast<double>(
                        neighbor_squared_distances[static_cast<std::size_t>(j)]);

                if (neighbor_distance_squared >
                    max_correspondence_distance_squared)
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

                if (candidate >=
                    target_planes.size())
                {
                    continue;
                }

                if (target_planes[candidate].state !=
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
                target_planes[static_cast<std::size_t>(
                    plane_index)];

            const double residual =
                plane.normal.dot(
                    p_target -
                    plane.point);

            if (std::abs(residual) >
                config_.max_point_to_plane_distance)
            {
                continue;
            }

            Eigen::Matrix<double, 1, 6> J;

            if (config_.enable_sensor_centered_perturbation)
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
                        lever_arm_range >
                            1.0e-9)
                    {
                        correspondence_ranges.push_back(
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
                std::abs(
                    residual);

            if (config_.enable_huber_loss &&
                absolute_residual >
                    config_.huber_delta)
            {
                robust_weight =
                    config_.huber_delta /
                    absolute_residual;

                ++downweighted_correspondences;
            }

            minimum_robust_weight =
                std::min(
                    minimum_robust_weight,
                    robust_weight);

            // ====================================================
            // REAL ICP accumulation
            // ====================================================

            H.noalias() +=
                robust_weight *
                J.transpose() *
                J;

            b.noalias() +=
                robust_weight *
                J.transpose() *
                residual;

            squared_error_sum +=
                residual *
                residual;

            robust_weighted_squared_error_sum +=
                robust_weight *
                residual *
                residual;

            robust_weight_sum +=
                robust_weight;

            ++valid_correspondences;
        }

        // ====================================================
        // 2.5 Check correspondence count
        // ====================================================

        if (valid_correspondences <
            config_.min_correspondences)
        {
            std::cerr
                << "LidarRegistration::Align(): "
                << "not enough valid correspondences. valid="
                << valid_correspondences
                << std::endl;

            result.success =
                false;

            result.converged =
                false;

            result.iterations =
                iteration + 1;

            result.correspondences =
                valid_correspondences;

            result.robust_downweighted_correspondences =
                downweighted_correspondences;

            result.robust_downweighted_ratio =
                valid_correspondences > 0
                    ? static_cast<double>(
                          downweighted_correspondences) /
                          static_cast<double>(
                              valid_correspondences)
                    : 0.0;

            result.robust_effective_weight_sum =
                robust_weight_sum;

            result.robust_min_weight =
                minimum_robust_weight;

            if (robust_weight_sum >
                1.0e-12)
            {
                result.robust_rmse =
                    std::sqrt(
                        robust_weighted_squared_error_sum /
                        robust_weight_sum);
            }

            result.T_target_source =
                T_target_source;

            return false;
        }

        const double rmse =
            std::sqrt(
                squared_error_sum /
                static_cast<double>(
                    valid_correspondences));

        const double robust_rmse =
            robust_weight_sum >
                    1.0e-12
                ? std::sqrt(
                      robust_weighted_squared_error_sum /
                      robust_weight_sum)
                : std::numeric_limits<double>::infinity();

        const double downweighted_ratio =
            static_cast<double>(
                downweighted_correspondences) /
            static_cast<double>(
                valid_correspondences);

        result.robust_downweighted_correspondences =
            downweighted_correspondences;

        result.robust_downweighted_ratio =
            downweighted_ratio;

        result.robust_effective_weight_sum =
            robust_weight_sum;

        result.robust_min_weight =
            minimum_robust_weight;

        result.robust_rmse =
            robust_rmse;

        // ====================================================
        // 2.6 Degeneracy V2B scale normalization
        // ====================================================

        double median_range =
            1.0;

        double characteristic_length =
            1.0;

        Eigen::Matrix<double, 6, 6> parameter_unscale =
            Eigen::Matrix<double, 6, 6>::Identity();

        if (use_hessian_scale_normalization)
        {
            if (correspondence_ranges.empty())
            {
                std::cerr
                    << "LidarRegistration::Align(): "
                    << "V2B has no valid sensor-centered ranges."
                    << std::endl;

                return false;
            }

            const std::size_t range_count =
                correspondence_ranges.size();

            std::vector<double>::iterator middle =
                correspondence_ranges.begin() +
                static_cast<std::ptrdiff_t>(
                    range_count / 2);

            std::nth_element(
                correspondence_ranges.begin(),
                middle,
                correspondence_ranges.end());

            median_range =
                *middle;

            if ((range_count % 2U) ==
                0U)
            {
                const std::vector<double>::iterator
                    lower_middle =
                        std::max_element(
                            correspondence_ranges.begin(),
                            middle);

                if (lower_middle !=
                    middle)
                {
                    median_range =
                        0.5 *
                        (median_range +
                         *lower_middle);
                }
            }

            if (!std::isfinite(median_range) ||
                median_range <=
                    0.0)
            {
                std::cerr
                    << "LidarRegistration::Align(): "
                    << "invalid V2B median range="
                    << median_range
                    << std::endl;

                return false;
            }

            characteristic_length =
                std::clamp(
                    median_range,
                    config_.hessian_scale_min_range,
                    config_.hessian_scale_max_range);

            const double inverse_length =
                1.0 /
                characteristic_length;

            parameter_unscale(0, 0) =
                inverse_length;

            parameter_unscale(1, 1) =
                inverse_length;

            parameter_unscale(2, 2) =
                inverse_length;
        }

        const Eigen::Matrix<double, 6, 6> H_analysis =
            parameter_unscale.transpose() *
            H *
            parameter_unscale;

        const Eigen::Matrix<double, 6, 1> b_analysis =
            parameter_unscale.transpose() *
            b;

        // ====================================================
        // Degeneracy V2D result diagnostics        // ====================================================
        // Existing V2B result diagnostics
        // ====================================================

        result.hessian_scale_normalization_enabled =
            use_hessian_scale_normalization;

        result.hessian_scale_range_count =
            correspondence_ranges.size();

        result.hessian_median_range =
            median_range;

        result.hessian_characteristic_length =
            characteristic_length;

        double raw_condition_number =
            std::numeric_limits<double>::infinity();

        if (use_hessian_scale_normalization)
        {
            Eigen::SelfAdjointEigenSolver<
                Eigen::Matrix<double, 6, 6>>
                raw_hessian_eigen_solver(
                    H);

            if (raw_hessian_eigen_solver.info() ==
                Eigen::Success)
            {
                const Eigen::Matrix<double, 6, 1>
                    raw_eigenvalues =
                        raw_hessian_eigen_solver.eigenvalues();

                const double raw_lambda_min =
                    raw_eigenvalues(0);

                const double raw_lambda_max =
                    raw_eigenvalues(5);

                if (std::isfinite(raw_lambda_min) &&
                    std::isfinite(raw_lambda_max) &&
                    raw_lambda_min >
                        config_
                            .degeneracy_absolute_eigenvalue_threshold)
                {
                    raw_condition_number =
                        raw_lambda_max /
                        raw_lambda_min;
                }
            }
        }

        Eigen::SelfAdjointEigenSolver<
            Eigen::Matrix<double, 6, 6>>
            hessian_eigen_solver(
                H_analysis);

        if (hessian_eigen_solver.info() !=
            Eigen::Success)
        {
            std::cerr
                << "LidarRegistration::Align(): "
                << "Hessian eigen decomposition failed."
                << std::endl;

            return false;
        }

        const Eigen::Matrix<double, 6, 1>
            hessian_eigenvalues =
                hessian_eigen_solver.eigenvalues();

        const Eigen::Matrix<double, 6, 6>
            hessian_eigenvectors =
                hessian_eigen_solver.eigenvectors();

        const double lambda_min =
            hessian_eigenvalues(0);

        const double lambda_max =
            hessian_eigenvalues(5);

        Eigen::Matrix<double, 6, 1>
            relative_eigenvalues =
                Eigen::Matrix<double, 6, 1>::Zero();

        // ====================================================
        // Degeneracy V2D thresholds
        //
        // The calibrated thresholds apply ONLY to the realtime
        // SENSOR_CENTERED_NORMALIZED frontend:
        //
        //     relative < 0.01
        //         -> DEGENERATE
        //         -> suppress this eigen-direction.
        //
        //     0.01 <= relative < 0.02
        //         -> WEAK
        //         -> diagnostic only; keep the LiDAR correction.
        //
        //     relative >= 0.02
        //         -> OBSERVABLE.
        //
        // Backend refinement intentionally remains in the legacy
        // experiment mode, so it keeps the old configured threshold.
        // ====================================================

        constexpr double v2d_hard_relative_threshold =
            0.01;

        constexpr double v2d_weak_relative_threshold =
            0.02;

        const double hard_relative_threshold =
            use_hessian_scale_normalization
                ? v2d_hard_relative_threshold
                : config_
                      .degeneracy_relative_eigenvalue_threshold;

        const double weak_relative_threshold =
            use_hessian_scale_normalization
                ? v2d_weak_relative_threshold
                : hard_relative_threshold;

        int degenerate_directions =
            0;

        int weak_directions =
            0;

        if (std::isfinite(lambda_max) &&
            lambda_max >
                config_
                    .degeneracy_absolute_eigenvalue_threshold)
        {
            relative_eigenvalues =
                hessian_eigenvalues /
                lambda_max;

            for (int i = 0;
                 i < 6;
                 ++i)
            {
                const double lambda =
                    hessian_eigenvalues(i);

                const double relative_lambda =
                    relative_eigenvalues(i);

                const bool strong_degenerate =
                    !std::isfinite(lambda) ||
                    lambda <=
                        config_
                            .degeneracy_absolute_eigenvalue_threshold ||
                    relative_lambda <
                        hard_relative_threshold;

                if (strong_degenerate)
                {
                    ++degenerate_directions;
                    continue;
                }

                const bool weak_observability =
                    use_hessian_scale_normalization &&
                    relative_lambda <
                        weak_relative_threshold;

                if (weak_observability)
                {
                    ++weak_directions;

                    std::cout
                        << "WEAK_DIRECTION_V2D"
                        << " | index="
                        << i
                        << " | eigenvalue="
                        << lambda
                        << " | relative="
                        << relative_lambda
                        << " | hard_threshold="
                        << hard_relative_threshold
                        << " | weak_threshold="
                        << weak_relative_threshold
                        << " | action=KEEP"
                        << " | eigenvector=["
                        << hessian_eigenvectors
                               .col(i)
                               .transpose()
                        << "]"
                        << std::endl;
                }
            }
        }
        else
        {
            degenerate_directions =
                6;

            weak_directions =
                0;
        }

        const bool degenerate =
            degenerate_directions >
            0;

        double condition_number =
            std::numeric_limits<double>::infinity();

        if (std::isfinite(lambda_min) &&
            std::isfinite(lambda_max) &&
            lambda_min >
                config_
                    .degeneracy_absolute_eigenvalue_threshold)
        {
            condition_number =
                lambda_max /
                lambda_min;
        }

        result.degenerate =
            degenerate;

        result.degenerate_directions =
            degenerate_directions;

        result.condition_number =
            condition_number;

        result.hessian_eigenvalues =
            hessian_eigenvalues;

        result.hessian_relative_eigenvalues =
            relative_eigenvalues;

        // ================================================================
        // Build relative covariance shape for G2O Information Matrix V1.
        //
        // H_analysis = V * Lambda * V^T
        //
        // relative_lambda_i = lambda_i / lambda_max
        //
        // C_relative =
        //     V * diag(1 / relative_lambda_i) * V^T
        //
        // Order:
        //     [rx ry rz tx ty tz]
        //
        // Frame:
        //     target / World frame
        // ================================================================
        result.hessian_relative_covariance_valid =
            false;

        result.hessian_relative_covariance =
            Eigen::Matrix<double, 6, 6>::Identity();

        if (use_hessian_scale_normalization &&
            std::isfinite(lambda_max) &&
            lambda_max >
                config_.degeneracy_absolute_eigenvalue_threshold &&
            hessian_eigenvectors.allFinite() &&
            relative_eigenvalues.allFinite())
        {
            constexpr double minimum_relative_information =
                0.01;

            Eigen::Matrix<double, 6, 6> relative_covariance =
                Eigen::Matrix<double, 6, 6>::Zero();

            bool covariance_valid = true;

            for (int i = 0;
                 i < 6;
                 ++i)
            {
                double relative_information =
                    relative_eigenvalues(i);

                if (!std::isfinite(relative_information))
                {
                    covariance_valid = false;
                    break;
                }

                relative_information =
                    std::clamp(
                        relative_information,
                        minimum_relative_information,
                        1.0);

                const Eigen::Matrix<double, 6, 1>
                    eigen_direction =
                        hessian_eigenvectors.col(i);

                relative_covariance.noalias() +=
                    (1.0 / relative_information) *
                    eigen_direction *
                    eigen_direction.transpose();
            }

            if (covariance_valid &&
                relative_covariance.allFinite())
            {
                bool diagonal_valid = true;

                for (int i = 0;
                     i < 6;
                     ++i)
                {
                    if (!std::isfinite(
                            relative_covariance(i, i)) ||
                        relative_covariance(i, i) <= 0.0)
                    {
                        diagonal_valid = false;
                        break;
                    }
                }

                if (diagonal_valid)
                {
                    result.hessian_relative_covariance =
                        relative_covariance;

                    result.hessian_relative_covariance_valid =
                        true;
                }
            }
        }

        Eigen::Matrix<double, 6, 1>
            hessian_eigenvalues_per_weight =
                Eigen::Matrix<double, 6, 1>::Zero();

        if (std::isfinite(robust_weight_sum) &&
            robust_weight_sum >
                1.0e-12)
        {
            hessian_eigenvalues_per_weight =
                hessian_eigenvalues /
                robust_weight_sum;
        }

        std::cout
            << "Hessian diagnostics"
            << " | mode="
            << (use_hessian_scale_normalization
                    ? "SENSOR_CENTERED_NORMALIZED_V2D"
                    : (config_.enable_sensor_centered_perturbation
                           ? "SENSOR_CENTERED_V2A"
                           : "LEGACY_WORLD_ORIGIN"))
            << " | sensor_origin=["
            << sensor_origin_target.transpose()
            << "]"
            << " | corr="
            << valid_correspondences
            << " | weight_sum="
            << robust_weight_sum
            << " | range_count="
            << correspondence_ranges.size()
            << " | median_range="
            << median_range
            << " | scale_L="
            << characteristic_length
            << " | raw_condition="
            << raw_condition_number
            << " | eigenvalues=["
            << hessian_eigenvalues.transpose()
            << "]"
            << " | eigen_per_weight=["
            << hessian_eigenvalues_per_weight.transpose()
            << "]"
            << " | relative=["
            << relative_eigenvalues.transpose()
            << "]"
            << " | condition="
            << condition_number
            << " | hard_threshold="
            << hard_relative_threshold
            << " | weak_threshold="
            << weak_relative_threshold
            << " | degenerate="
            << (degenerate
                    ? "true"
                    : "false")
            << " | degenerate_directions="
            << degenerate_directions
            << " | weak_directions="
            << weak_directions
            << std::endl;

        // ====================================================
        // 2.7 Solve in H_analysis coordinates
        // ====================================================

        Eigen::Matrix<double, 6, 1> dx =
            Eigen::Matrix<double, 6, 1>::Zero();

        Eigen::Matrix<double, 6, 1> delta_analysis =
            Eigen::Matrix<double, 6, 1>::Zero();

        if (!degenerate)
        {
            delta_analysis =
                H_analysis
                    .ldlt()
                    .solve(
                        -b_analysis);

            dx =
                parameter_unscale *
                delta_analysis;
        }
        else
        {
            if (degenerate_directions >=
                6)
            {
                std::cerr
                    << "LidarRegistration::Align(): "
                    << "all Hessian directions are degenerate."
                    << std::endl;

                result.success =
                    false;

                result.converged =
                    false;

                result.iterations =
                    iteration + 1;

                result.correspondences =
                    valid_correspondences;

                result.T_target_source =
                    T_target_source;

                return false;
            }

            const Eigen::Matrix<double, 6, 1>
                gradient_eigen =
                    hessian_eigenvectors.transpose() *
                    b_analysis;

            Eigen::Matrix<double, 6, 1>
                delta_eigen_before_suppression =
                    Eigen::Matrix<double, 6, 1>::Zero();

            Eigen::Matrix<double, 6, 1>
                delta_eigen =
                    Eigen::Matrix<double, 6, 1>::Zero();

            int usable_directions =
                0;

            for (int i = 0;
                 i < 6;
                 ++i)
            {
                const double lambda =
                    hessian_eigenvalues(i);

                const double relative_lambda =
                    relative_eigenvalues(i);

                const bool valid_lambda =
                    std::isfinite(lambda) &&
                    lambda >
                        config_
                            .degeneracy_absolute_eigenvalue_threshold;

                if (valid_lambda)
                {
                    delta_eigen_before_suppression(i) =
                        -gradient_eigen(i) /
                        lambda;
                }

                const bool strong_degenerate =
                    !valid_lambda ||
                    relative_lambda <
                        hard_relative_threshold;

                if (strong_degenerate)
                {
                    delta_eigen(i) =
                        0.0;

                    std::cout
                        << (use_hessian_scale_normalization
                                ? "DEGENERATE_DIRECTION_V2D"
                                : "Weak Hessian direction")
                        << " | index="
                        << i
                        << " | eigenvalue="
                        << lambda
                        << " | relative="
                        << relative_lambda
                        << " | threshold="
                        << hard_relative_threshold
                        << " | action=SUPPRESS"
                        << " | space="
                        << (use_hessian_scale_normalization
                                ? "NORMALIZED_EQUIVALENT_METERS"
                                : "PHYSICAL_RAD_M")
                        << " | eigenvector=["
                        << hessian_eigenvectors
                               .col(i)
                               .transpose()
                        << "]"
                        << std::endl;

                    continue;
                }

                delta_eigen(i) =
                    delta_eigen_before_suppression(i);

                ++usable_directions;
            }

            if (usable_directions <=
                0)
            {
                std::cerr
                    << "LidarRegistration::Align(): "
                    << "no observable Hessian direction remains."
                    << std::endl;

                return false;
            }

            const Eigen::Matrix<double, 6, 1>
                delta_analysis_before_suppression =
                    hessian_eigenvectors *
                    delta_eigen_before_suppression;

            const Eigen::Matrix<double, 6, 1>
                dx_before_suppression =
                    parameter_unscale *
                    delta_analysis_before_suppression;

            delta_analysis =
                hessian_eigenvectors *
                delta_eigen;

            dx =
                parameter_unscale *
                delta_analysis;

            if (use_hessian_scale_normalization)
            {
                std::cout
                    << "DEGENERACY_EVENT_V2D"
                    << " | hard_threshold="
                    << hard_relative_threshold
                    << " | weak_threshold="
                    << weak_relative_threshold
                    << " | degenerate_directions="
                    << degenerate_directions
                    << " | weak_directions="
                    << weak_directions
                    << " | usable="
                    << usable_directions
                    << " | corr="
                    << valid_correspondences
                    << " | weight_sum="
                    << robust_weight_sum
                    << " | condition="
                    << condition_number
                    << " | relative=["
                    << relative_eigenvalues.transpose()
                    << "]"
                    << " | eigen_step_before=["
                    << delta_eigen_before_suppression.transpose()
                    << "]"
                    << " | eigen_step_after=["
                    << delta_eigen.transpose()
                    << "]"
                    << " | dx_before=["
                    << dx_before_suppression.transpose()
                    << "]"
                    << " | dx_after=["
                    << dx.transpose()
                    << "]"
                    << std::endl;
            }
            else
            {
                std::cout
                    << "Degeneracy handling"
                    << " | suppressed="
                    << degenerate_directions
                    << " | usable="
                    << usable_directions
                    << " | eigen_step=["
                    << delta_eigen.transpose()
                    << "]"
                    << std::endl;
            }
        }

        if (!dx.allFinite())
        {
            std::cerr
                << "LidarRegistration::Align(): "
                << "dx contains NaN/Inf after degeneracy handling."
                << std::endl;

            return false;
        }

        const Eigen::Vector3d delta_rotation =
            dx.head<3>();

        const Eigen::Vector3d delta_translation =
            dx.tail<3>();

        const Eigen::Matrix3d delta_R =
            Sophus::SO3d::exp(
                delta_rotation)
                .matrix();

        if (config_.enable_sensor_centered_perturbation)
        {
            T_target_source.linear() =
                delta_R *
                T_target_source.rotation();

            T_target_source.translation() +=
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

            T_target_source =
                delta_T *
                T_target_source;
        }

        const double dR =
            delta_rotation.norm();

        const double dT =
            delta_translation.norm();

        std::cout
            << "ROBUST_ICP"
            << " | iteration="
            << iteration
            << " | enabled="
            << (config_.enable_huber_loss
                    ? "true"
                    : "false")
            << " | delta="
            << config_.huber_delta
            << " m"
            << " | corr="
            << valid_correspondences
            << " | downweighted="
            << downweighted_correspondences
            << " | downweighted_ratio="
            << downweighted_ratio
            << " | weight_sum="
            << robust_weight_sum
            << " | min_weight="
            << minimum_robust_weight
            << " | raw_rmse="
            << rmse
            << " | robust_rmse="
            << robust_rmse
            << " | hessian_scale="
            << (use_hessian_scale_normalization
                    ? "ON"
                    : "OFF")
            << " | median_range="
            << median_range
            << " | scale_L="
            << characteristic_length
            << std::endl;

        std::cout
            << "Registration iteration "
            << iteration
            << " | correspondence="
            << valid_correspondences
            << " | rmse="
            << rmse
            << " | dR="
            << dR
            << " | dT="
            << dT
            << std::endl;

        result.success =
            true;

        result.converged =
            false;

        result.iterations =
            iteration + 1;

        result.correspondences =
            valid_correspondences;

        result.rmse =
            rmse;

        result.T_target_source =
            T_target_source;

        if (dR <
                config_
                    .rotation_convergence_threshold &&
            dT <
                config_
                    .translation_convergence_threshold)
        {
            result.converged =
                true;

            break;
        }
    }

    return result.success;
}

bool LidarRegistration::Align(
    const pcl::PointCloud<LIDAR_POINT>::ConstPtr &source,
    const pcl::PointCloud<LIDAR_POINT>::ConstPtr &target,
    const Eigen::Isometry3d &initial_guess,
    LidarRegistrationResult &result) const
{
    PreparedLidarTarget prepared_target;

    if (!PrepareTarget(
            target,
            prepared_target))
    {
        result =
            LidarRegistrationResult();

        result.T_target_source =
            initial_guess;

        return false;
    }

    return Align(
        source,
        prepared_target,
        initial_guess,
        result);
}
