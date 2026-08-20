#pragma once

#include <cstddef>
#include <limits>

#include <Eigen/Core>
#include <Eigen/Geometry>

struct LidarRegistrationConfig
{
    int max_iterations = 20;

    // Number of target neighbors queried during registration and
    // during local-plane preparation.
    int knn = 5;

    double max_correspondence_distance = 1.0;

    double max_plane_fit_error = 0.15;

    double max_point_to_plane_distance = 0.5;

    std::size_t min_correspondences = 50;

    double rotation_convergence_threshold = 1.5e-3;

    double translation_convergence_threshold = 3.0e-3;

    // Degeneracy diagnostics.
    // A Hessian direction is considered weak when its eigenvalue is
    // either absolutely tiny or very small relative to the largest one.
    // These thresholds are diagnostic defaults and should be tuned from logs.
    double degeneracy_relative_eigenvalue_threshold = 1.0e-4;

    double degeneracy_absolute_eigenvalue_threshold = 1.0e-9;
};

struct TargetPlane
{
    enum class State
    {
        Unknown,
        Valid,
        Invalid
    };

    Eigen::Vector3d point =
        Eigen::Vector3d::Zero();

    Eigen::Vector3d normal =
        Eigen::Vector3d::Zero();

    State state =
        State::Unknown;
};

struct LidarRegistrationResult
{
    bool success = false;

    bool converged = false;

    int iterations = 0;

    std::size_t correspondences = 0;

    double rmse =
        std::numeric_limits<double>::infinity();

    // Degeneracy diagnostics from the Hessian of the latest GN iteration.
    bool degenerate = false;

    int degenerate_directions = 0;

    double condition_number =
        std::numeric_limits<double>::infinity();

    Eigen::Matrix<double, 6, 1> hessian_eigenvalues =
        Eigen::Matrix<double, 6, 1>::Zero();

    Eigen::Matrix<double, 6, 1> hessian_relative_eigenvalues =
        Eigen::Matrix<double, 6, 1>::Zero();

    // source frame -> target frame
    Eigen::Isometry3d T_target_source =
        Eigen::Isometry3d::Identity();
};