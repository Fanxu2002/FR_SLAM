#include "fr_slam/fr_lidar_registration.hpp"

#include <cmath>
#include <iostream>
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

    // ============================================================
    // 1. Centroid
    // ============================================================

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

    // ============================================================
    // 2. Covariance
    // ============================================================

    double c_xx = 0.0;
    double c_xy = 0.0;
    double c_xz = 0.0;
    double c_yy = 0.0;
    double c_yz = 0.0;
    double c_zz = 0.0;

    const double cx = centroid.x();
    const double cy = centroid.y();
    const double cz = centroid.z();

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

    // ============================================================
    // 3. Eigen solve
    //
    // The eigenvector corresponding to the smallest eigenvalue is
    // the local plane normal.
    // ============================================================

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
        normal_norm < 1e-12)
    {
        return false;
    }

    normal /=
        normal_norm;

    // ============================================================
    // 4. Plane-quality validation
    // ============================================================

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

    // ============================================================
    // 1. Store target cloud
    // ============================================================

    prepared_target.cloud =
        target;

    // ============================================================
    // 2. Build KDTree directly on the original LIDAR_POINT cloud.
    // ============================================================

    prepared_target.kdtree =
        std::make_shared<
            pcl::KdTreeFLANN<LIDAR_POINT>>();

    prepared_target.kdtree->setInputCloud(
        target);

    // ============================================================
    // 3. Prepare one plane-cache entry for every target point.
    // ============================================================

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
        target.planes.size() != target.cloud->size())
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

    // ============================================================
    // 1. Initial pose
    // ============================================================

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

        std::size_t valid_correspondences =
            0;

        for (const LIDAR_POINT &source_point :
             source->points)
        {
            // =================================================
            // 2.1 Transform source point
            // =================================================

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

            // =================================================
            // 2.2 Find nearby target points
            // =================================================

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

            // =================================================
            // 2.3 Select the nearest valid precomputed plane
            // =================================================

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

            // =================================================
            // 2.4 Point-to-plane residual + Jacobian
            //
            // r = n^T (p' - q)
            //
            // Left perturbation:
            // T <- Exp(dx) * T
            //
            // dx = [delta_rotation, delta_translation]
            //
            // J = [(p' x n)^T, n^T]
            // =================================================

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

            J.block<1, 3>(0, 0) =
                p_target.cross(
                            plane.normal)
                    .transpose();

            J.block<1, 3>(0, 3) =
                plane.normal.transpose();

            H.noalias() +=
                J.transpose() * J;

            b.noalias() +=
                J.transpose() * residual;

            squared_error_sum +=
                residual * residual;

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

            result.T_target_source =
                T_target_source;

            return false;
        }

        // ====================================================
        // 2.6 Degeneracy diagnostics from Hessian H
        //
        // H = sum(J^T J) is symmetric positive semi-definite in theory.
        // Small eigenvalues mean that some 6-DOF motion directions are
        // weakly constrained by the current point-to-plane geometry.
        //
        // IMPORTANT: for now this block only detects and reports
        // degeneracy. It does NOT modify dx or reject the registration.
        // ====================================================

        Eigen::SelfAdjointEigenSolver<
            Eigen::Matrix<double, 6, 6>>
            hessian_eigen_solver(H);

        if (hessian_eigen_solver.info() !=
            Eigen::Success)
        {
            std::cerr
                << "LidarRegistration::Align(): "
                << "Hessian eigen decomposition failed."
                << std::endl;

            return false;
        }

        const Eigen::Matrix<double, 6, 1> hessian_eigenvalues =
            hessian_eigen_solver.eigenvalues();

        const double lambda_min =
            hessian_eigenvalues(0);

        const double lambda_max =
            hessian_eigenvalues(5);

        Eigen::Matrix<double, 6, 1> relative_eigenvalues =
            Eigen::Matrix<double, 6, 1>::Zero();

        int degenerate_directions = 0;

        if (std::isfinite(lambda_max) &&
            lambda_max >
                config_.degeneracy_absolute_eigenvalue_threshold)
        {
            relative_eigenvalues =
                hessian_eigenvalues / lambda_max;

            for (int i = 0;
                 i < 6;
                 ++i)
            {
                const double lambda =
                    hessian_eigenvalues(i);

                const double relative_lambda =
                    relative_eigenvalues(i);

                if (!std::isfinite(lambda) ||
                    lambda <=
                        config_.degeneracy_absolute_eigenvalue_threshold ||
                    relative_lambda <
                        config_.degeneracy_relative_eigenvalue_threshold)
                {
                    ++degenerate_directions;
                }
            }
        }
        else
        {
            // If even the largest eigenvalue is almost zero, the entire
            // Hessian carries essentially no usable geometric information.
            degenerate_directions = 6;
        }

        const bool degenerate =
            degenerate_directions > 0;

        double condition_number =
            std::numeric_limits<double>::infinity();

        if (std::isfinite(lambda_min) &&
            std::isfinite(lambda_max) &&
            lambda_min >
                config_.degeneracy_absolute_eigenvalue_threshold)
        {
            condition_number =
                lambda_max / lambda_min;
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

        std::cout
            << "Hessian diagnostics"
            << " | eigenvalues=["
            << hessian_eigenvalues.transpose()
            << "]"
            << " | relative=["
            << relative_eigenvalues.transpose()
            << "]"
            << " | condition="
            << condition_number
            << " | degenerate="
            << (degenerate ? "true" : "false")
            << " | weak_directions="
            << degenerate_directions
            << std::endl;

        // ====================================================
        // 2.7 Solve H dx = -b
        // ====================================================

        const Eigen::Matrix<double, 6, 1> dx =
            H.ldlt().solve(
                -b);

        if (!dx.allFinite())
        {
            std::cerr
                << "LidarRegistration::Align(): "
                << "dx contains NaN/Inf."
                << std::endl;

            return false;
        }

        // ====================================================
        // 2.8 Left pose update
        // ====================================================

        const Eigen::Vector3d delta_rotation =
            dx.head<3>();

        const Eigen::Vector3d delta_translation =
            dx.tail<3>();

        Eigen::Isometry3d delta_T =
            Eigen::Isometry3d::Identity();

        delta_T.linear() =
            Sophus::SO3d::exp(
                delta_rotation)
                .matrix();

        delta_T.translation() =
            delta_translation;

        T_target_source =
            delta_T *
            T_target_source;

        // ====================================================
        // 2.9 Diagnostics
        // ====================================================

        const double rmse =
            std::sqrt(
                squared_error_sum /
                static_cast<double>(
                    valid_correspondences));

        const double dR =
            delta_rotation.norm();

        const double dT =
            delta_translation.norm();

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

        // ====================================================
        // 2.10 Convergence
        // ====================================================

        if (dR <
                config_.rotation_convergence_threshold &&
            dT <
                config_.translation_convergence_threshold)
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