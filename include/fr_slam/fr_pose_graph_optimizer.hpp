#pragma once

#include <cstddef>

#include "fr_slam/fr_pose_graph.hpp"

// ============================================================================
// g2o Keyframe PoseGraph optimizer.
//
// Graph convention used by FR-SLAM:
//
//     X_i = T_WK_i
//
//     Z_ij = T_Ki_Kj
//          = T_WK_i^-1 * T_WK_j
//
// PoseGraph vertices are Keyframes. Submaps are not optimization variables.
// ============================================================================
struct PoseGraphOptimizerConfig
{
    int max_iterations = 30;

    bool verbose = false;

    // Only loop edges receive the robust kernel in V8.
    bool use_huber_for_loop_edges = true;
    double loop_huber_delta = 1.0;

    // Gravity Guard V1.  A gravity-direction unary factor is added for every
    // node that has an immutable frontend gravity reference.  The residual is
    // yaw invariant, so x/y/z/yaw can still move while artificial roll/pitch
    // tilt is discouraged.
    bool use_gravity_direction_prior = true;
    double gravity_information_scale = 50.0;

    // Transactional hard guard.  All optimized poses are validated BEFORE any
    // pose is committed to PoseGraph.  If any node exceeds this tilt error
    // relative to its frontend gravity reference, Optimize() returns false and
    // the caller can roll back the newly inserted loop edge.
    bool use_gravity_hard_guard = true;
    double max_gravity_tilt_error_deg = 3.0;

    // Trajectory Shape Guard V1.  This protects XY geometry from collapsing
    // when a small number of loop constraints tries to bend a long odometry
    // chain into an unrealistic shape.  The guard is evaluated transactionally
    // after g2o optimization and before ANY pose is committed.
    bool use_trajectory_shape_guard = true;

    // Let the optimized XY PCA minor/major ratio keep at least this fraction
    // of the pre-optimization ratio.  A 2D loop collapsing toward one line is
    // therefore rejected.
    double min_xy_pca_ratio_scale = 0.35;

    // Global XY path length should not shrink or grow implausibly.
    double min_path_length_ratio = 0.70;
    double max_path_length_ratio = 1.30;
};

struct PoseGraphOptimizationResult
{
    bool success = false;

    int iterations = 0;

    double chi2_before = 0.0;
    double chi2_after = 0.0;

    std::size_t optimized_nodes = 0;
    std::size_t odometry_edges = 0;
    std::size_t loop_edges = 0;

    double max_translation_update = 0.0;
    double max_rotation_update_deg = 0.0;

    std::size_t gravity_edges = 0;
    std::size_t gravity_reference_nodes = 0;

    double mean_gravity_tilt_error_deg = 0.0;
    double max_gravity_tilt_error_deg = 0.0;
    std::size_t worst_gravity_keyframe_id = 0;

    double max_roll_update_deg = 0.0;
    double max_pitch_update_deg = 0.0;
    double max_yaw_update_deg = 0.0;

    std::size_t max_roll_update_keyframe_id = 0;
    std::size_t max_pitch_update_keyframe_id = 0;
    std::size_t max_yaw_update_keyframe_id = 0;

    bool gravity_guard_passed = true;

    double xy_pca_ratio_before = 0.0;
    double xy_pca_ratio_after = 0.0;

    double path_length_before = 0.0;
    double path_length_after = 0.0;
    double path_length_ratio = 1.0;

    bool trajectory_shape_guard_passed = true;
};

class PoseGraphOptimizer
{
public:
    explicit PoseGraphOptimizer(
        const PoseGraphOptimizerConfig &config =
            PoseGraphOptimizerConfig());

    // Build a temporary g2o graph from PoseGraph, optimize it, then write the
    // optimized Keyframe poses back into PoseGraph::T_WK.
    //
    // IMPORTANT:
    //     This changes ONLY backend PoseGraph node estimates.
    //     It does NOT overwrite the live Scan-to-LocalMap frontend T_WL state.
    bool Optimize(
        PoseGraph &pose_graph,
        PoseGraphOptimizationResult &result) const;

private:
    PoseGraphOptimizerConfig config_;
};
