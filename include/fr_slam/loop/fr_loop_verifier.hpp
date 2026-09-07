#pragma once

#include "fr_slam/common/fr_point_types.hpp"

#include <cstddef>
#include <limits>
#include <memory>

#include <Eigen/Geometry>

#include <pcl/point_cloud.h>

// ============================================================================
// LoopVerifier V4 - Point-to-Plane V2 + degeneracy-aware solve + loop-ground
//
// Purpose:
//
//     Scan Context answers:
//
//         "Which old place looks similar?"
//
//     LoopVerifier answers:
//
//         "Can the two actual point clouds be geometrically aligned?"
//
// V4 SOLVER CHANGE:
//
//     The pose solver is Loop Point-to-Plane V2:
//       1) point-to-plane + Huber,
//       2) median-range rotation/translation scale normalization,
//       3) Hessian eigen-analysis and weak-direction suppression,
//       4) optional weak-direction seed prior for trusted reverse hypotheses,
//       5) an extra ground-like point-to-plane block projected to
//          [roll, pitch, z] only.
//
//     Prescore, final overlap/RMSE measurement, correction gates, graph gates
//     and PGO remain unchanged.
//
//     The expensive reusable work is cached:
//
//         current Keyframe cloud
//             -> PointXYZ conversion + VoxelGrid       (once per current KF)
//
//         historical frozen Submap cloud
//             -> PointXYZ conversion + VoxelGrid
//             -> target KD-tree                         (once per cached Submap)
//
//     The same prepared target KD-tree is reused by:
//
//         - Point-to-Plane nearest-neighbour / local-plane search
//         - explicit final overlap / RMSE evaluation
//         - repeated initial guesses for the same candidate
//
//     This is especially useful in FR_SLAM because the caller currently tests
//     CANDIDATE_POSE / +SC yaw / -SC yaw as separate Verify() calls.
//
// IMPORTANT:
//     This class only verifies geometry and returns a relative transform.
//     It does NOT add a PoseGraph Loop Edge and does NOT run g2o.
//
// Transform convention:
//
//     T_AB maps coordinates from B -> A:
//
//         p_A = T_AB * p_B
// ============================================================================

struct LoopVerifierConfig
{
    // Backend Submap clouds are already voxelized by LocalMap, but a slightly
    // coarser verifier cloud keeps candidate checking inexpensive.
    double voxel_leaf_size = 0.50;

    // Loop Point-to-Plane V2 solver.
    // max_correspondence_distance is the per-iteration neighbour-search radius;
    // it is NOT a global pose-displacement limit.
    std::size_t max_iterations = 50;
    double max_correspondence_distance = 5.0;
    double transformation_epsilon = 1.0e-6;
    double euclidean_fitness_epsilon = 1.0e-5;

    // General point-to-plane robust geometry.
    double point_to_plane_max_residual = 1.00;
    double point_to_plane_huber_delta = 0.10;

    // Degeneracy analysis is performed in the normalized coordinates
    //
    //   [L*rx, L*ry, L*rz, tx, ty, tz]
    //
    // where L is the median correspondence range.
    bool enable_degeneracy_suppression = true;
    double degeneracy_hard_relative_threshold = 0.01;
    double degeneracy_weak_relative_threshold = 0.03;

    // Trusted reverse hypotheses may use the supplied initial pose as a SOFT
    // prior, but ONLY inside geometry-weak eigen-directions.  The caller
    // explicitly opts in per Verify() call.
    bool enable_weak_direction_seed_prior = true;
    double weak_direction_prior_target_relative = 0.05;
    double weak_direction_prior_gain = 1.0;

    // Loop-local ground-like information.  The current Keyframe does not store
    // the frontend GroundSegmentationResult, so V2 infers ground-like
    // correspondences from locally fitted target planes.  Only planes whose
    // normals are close to target-frame +/-Z and whose source points lie below
    // the LiDAR origin are eligible.  Their Jacobian is projected to
    // [roll, pitch, z] exactly like frontend Ground V1.3.
    bool enable_ground_constraint = true;
    double ground_weight = 4.0;
    double ground_normal_max_tilt_deg = 25.0;
    double ground_min_below_sensor_m = 0.30;
    double ground_max_residual = 0.25;
    double ground_huber_delta = 0.10;
    std::size_t min_ground_correspondences = 30;

    // Final geometric quality measurement after Point-to-Plane solve.
    double verification_inlier_distance = 1.0;

    // Acceptance thresholds.
    double max_rmse = 0.65;
    std::size_t min_inliers = 300;
    double min_overlap_ratio = 0.15;

    // Final sanity gate on how far the solver may move from the selected hypothesis.
    double max_correction_translation = 15.0;
    double max_correction_rotation_deg = 45.0;

    // Reject extremely small clouds before doing normal geometry verification.
    std::size_t min_cloud_points = 300;

    // V2.1 trusted reverse-sequence gate.
    //
    // A confirmed reverse sequence is already protected by:
    //   - 3+ RAW-SC observations,
    //   - 2+ real reverse historical progress events,
    //   - opposite frontend heading,
    //   - compact K-1/K/K+1 target,
    //   - weak-direction seed prior,
    //   - graph consistency.
    //
    // In that narrow case, do not require the current 0.5 m voxelized source
    // to contain 300 points or 300 final inliers. Reverse traversal has much
    // lower single-frame field-of-view overlap by construction.
    std::size_t trusted_reverse_min_cloud_points = 150;
    std::size_t trusted_reverse_min_inliers = 100;
    double trusted_reverse_min_overlap_ratio = 0.35;
    double trusted_reverse_max_rmse = 0.50;
    double trusted_reverse_max_correction_translation = 1.50;
    double trusted_reverse_max_correction_rotation_deg = 15.0;

    // Cheap initial-guess pre-score.  Before running full ICP, the current
    // downsampled source is transformed by the hypothesis and evaluated against
    // the cached historical target KD-tree.  This performs nearest-neighbour
    // queries only; it does not iterate ICP.
    //
    // A generous distance keeps this stage conservative.  The overlap threshold
    // is intentionally much lower than the final 0.15 geometry gate so only
    // clearly implausible hypotheses are pruned.
    double prescore_inlier_distance = 2.0;
    double prescore_min_overlap_ratio = 0.03;

    // Maximum number of prepared historical Submap targets retained by the
    // verifier. Finished Submap clouds are immutable, so reusing them is safe.
    // The cache is FIFO-bounded to avoid unbounded memory growth.
    std::size_t max_cached_targets = 64;
};

enum class LoopVerifierHypothesis
{
    GraphPose = 0,
    ScanContextPositiveYaw = 1,
    ScanContextNegativeYaw = 2
};

struct LoopVerifierInitialGuessScore
{
    bool valid = false;
    std::size_t inliers = 0;
    double overlap_ratio = 0.0;
    double rmse =
        std::numeric_limits<double>::infinity();
};

struct LoopVerificationResult
{
    bool success = false;
    bool converged = false;
    bool accepted = false;

    Eigen::Isometry3d T_target_source =
        Eigen::Isometry3d::Identity();

    Eigen::Isometry3d initial_guess =
        Eigen::Isometry3d::Identity();

    LoopVerifierHypothesis hypothesis =
        LoopVerifierHypothesis::GraphPose;

    double fitness_score =
        std::numeric_limits<double>::infinity();

    std::size_t inliers = 0;
    double overlap_ratio = 0.0;
    double rmse =
        std::numeric_limits<double>::infinity();

    double correction_translation =
        std::numeric_limits<double>::infinity();

    double correction_rotation_deg =
        std::numeric_limits<double>::infinity();

    std::size_t source_points = 0;
    std::size_t target_points = 0;
};

class LoopVerifier
{
public:
    explicit LoopVerifier(
        const LoopVerifierConfig &config =
            LoopVerifierConfig());

    ~LoopVerifier();

    LoopVerifier(const LoopVerifier &) = delete;
    LoopVerifier &operator=(const LoopVerifier &) = delete;

    // ------------------------------------------------------------------------
    // Cheaply score ONE externally-generated initial guess.
    //
    // This reuses the same cached downsampled source / target KD-tree as Verify()
    // but does NOT run ICP.  It is intended to rank CANDIDATE_POSE / +SC / -SC
    // before spending time on the full Point-to-Plane solve.
    // ------------------------------------------------------------------------
    bool ScoreInitialGuess(
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &source_current,
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &target_historical,
        const Eigen::Isometry3d &initial_guess,
        LoopVerifierInitialGuessScore &score) const;

    // V2.1: relaxed source-size prescore is available ONLY to a confirmed
    // CANDIDATE_REVERSE_SEQUENCE. Target size and all other hypotheses keep
    // the normal thresholds.
    bool ScoreInitialGuess(
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &source_current,
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &target_historical,
        const Eigen::Isometry3d &initial_guess,
        bool enable_trusted_reverse_sequence_gate,
        LoopVerifierInitialGuessScore &score) const;

    // ------------------------------------------------------------------------
    // Verify one candidate using the geometry-only policy.
    // Existing callers remain source-compatible.
    // ------------------------------------------------------------------------
    bool Verify(
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &source_current,
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &target_historical,
        const Eigen::Isometry3d &graph_initial_guess,
        double scan_context_yaw_shift_deg,
        LoopVerificationResult &result) const;

    // ------------------------------------------------------------------------
    // Verify one candidate and optionally enable the V2 weak-direction
    // initial-pose prior.  Use true only for a trusted reverse-traversal seed
    // such as CANDIDATE_REVERSE_FRONTEND / CANDIDATE_REVERSE_SEQUENCE.
    // ------------------------------------------------------------------------
    bool Verify(
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &source_current,
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &target_historical,
        const Eigen::Isometry3d &graph_initial_guess,
        double scan_context_yaw_shift_deg,
        bool enable_trusted_seed_prior,
        LoopVerificationResult &result) const;

    // V2.1: the second flag enables the relaxed single-frame support gate ONLY
    // for a temporally confirmed CANDIDATE_REVERSE_SEQUENCE.
    bool Verify(
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &source_current,
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &target_historical,
        const Eigen::Isometry3d &graph_initial_guess,
        double scan_context_yaw_shift_deg,
        bool enable_trusted_seed_prior,
        bool enable_trusted_reverse_sequence_gate,
        LoopVerificationResult &result) const;

    // Clear prepared source/target data. Call this together with SLAM Reset().
    void ClearCache();

    std::size_t CachedTargetCount() const;

    const LoopVerifierConfig &GetConfig() const;

    static const char *HypothesisName(
        LoopVerifierHypothesis hypothesis);

private:
    struct Cache;

    LoopVerifierConfig config_;
    mutable std::unique_ptr<Cache> cache_;
};
