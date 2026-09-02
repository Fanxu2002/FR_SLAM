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

    // Robust ICP V1.
    //
    // The hard point-to-plane gate above still rejects gross outliers.
    // Huber then softly down-weights the remaining medium residuals:
    //
    //     w = 1                    , |r| <= delta
    //     w = delta / |r|          , |r| >  delta
    //
    // The same weight is applied consistently to both Hessian and gradient:
    //
    //     H += w * J^T J
    //     b += w * J^T r
    //
    // Keep this enabled for the realtime frontend experiment.  The current
    // RegistrationScan2LocalMap constructor explicitly disables it for the
    // post-PGO backend refinement registration so that this V1 experiment
    // changes only frontend tracking.
    bool enable_huber_loss = true;

    double huber_delta = 0.10;

    // Degeneracy V2A: sensor-centered pose perturbation.
    //
    // When enabled, the point-to-plane rotational Jacobian uses the lever arm
    // from the current LiDAR origin instead of the target/world origin:
    //
    //     lever_arm = p_target - t_target_source
    //     J_rot     = lever_arm x normal
    //
    // and the matching pose retraction is:
    //
    //     R <- Exp(delta_rotation) * R
    //     t <- t + delta_translation
    //
    // This removes the Hessian degeneracy test's dependence on pure
    // translations of the arbitrary world-coordinate origin.
    //
    // RegistrationScan2LocalMap disables this for post-PGO backend refinement
    // during the V2A experiment so only realtime frontend tracking changes.
    bool enable_sensor_centered_perturbation = true;

    // Degeneracy V2B: rotation / translation parameter-scale normalization.
    //
    // V2A removed dependence on the arbitrary world origin by using the
    // sensor-centered lever arm.  V2B additionally removes the numerical
    // scale mismatch between rotation [rad] and translation [m] before
    // Hessian eigen-analysis.
    //
    // For the valid correspondences of the current GN iteration:
    //
    //     L = median( ||p_target - sensor_origin_target|| )
    //
    // clamped into [hessian_scale_min_range, hessian_scale_max_range].
    //
    // With y = [L*dtheta, dt], the normalized Jacobian is equivalent to:
    //
    //     J_norm = [ (lever_arm x normal) / L, normal ]
    //
    // The raw robust normal equations are transformed exactly as:
    //
    //     H_norm = D^-T * H * D^-1
    //     b_norm = D^-T * b
    //
    // where D = diag(L, L, L, 1, 1, 1).  Degeneracy detection and
    // weak-direction suppression are then performed in this normalized
    // coordinate space.  The final increment is mapped back to physical
    // [rad, m] before updating the pose.
    bool enable_hessian_scale_normalization = true;

    double hessian_scale_min_range = 1.0;

    double hessian_scale_max_range = 30.0;

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

    // Robust ICP diagnostics from the latest GN iteration.
    // `rmse` above deliberately remains the ordinary unweighted RMSE so the
    // existing Scan-to-LocalMap Quality Gate keeps exactly the same meaning.
    bool robust_kernel_enabled = false;

    double robust_kernel_delta = 0.0;

    std::size_t robust_downweighted_correspondences = 0;

    double robust_downweighted_ratio = 0.0;

    double robust_effective_weight_sum = 0.0;

    double robust_min_weight = 1.0;

    double robust_rmse =
        std::numeric_limits<double>::infinity();

    // Degeneracy V2B scale-normalization diagnostics from the latest GN
    // iteration.  The median is computed only from correspondences that pass
    // the normal correspondence/plane/point-to-plane gates.
    bool hessian_scale_normalization_enabled = false;

    std::size_t hessian_scale_range_count = 0;

    double hessian_median_range = 1.0;

    double hessian_characteristic_length = 1.0;

    // Degeneracy diagnostics from the Hessian of the latest GN iteration.
    // When V2B is enabled these eigenvalues belong to the normalized Hessian.
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

    // ========================================================================
    // V2B Hessian relative covariance shape.
    //
    // Order:
    //     [rx ry rz tx ty tz]
    //
    // Frame:
    //     Scan-to-LocalMap target / World frame.
    // ========================================================================
    bool hessian_relative_covariance_valid = false;

    Eigen::Matrix<double, 6, 6> hessian_relative_covariance =
        Eigen::Matrix<double, 6, 6>::Identity();
};