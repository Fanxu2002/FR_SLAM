#pragma once

#include "fr_slam/fr_point_types.hpp"

#include <cstddef>
#include <limits>
#include <memory>

#include <Eigen/Geometry>

#include <pcl/point_cloud.h>

// ============================================================================
// LoopVerifier V2 - cached preprocessing / cached historical target KD-tree
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
// V2 PERFORMANCE CHANGE:
//
//     The geometry / acceptance logic is intentionally unchanged.
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
//         - PCL ICP nearest-neighbour search
//         - explicit overlap / RMSE evaluation
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

    // Coarse point-to-point ICP.
    std::size_t max_iterations = 50;
    double max_correspondence_distance = 5.0;
    double transformation_epsilon = 1.0e-6;
    double euclidean_fitness_epsilon = 1.0e-5;

    // Final geometric quality measurement after ICP.
    double verification_inlier_distance = 1.0;

    // Acceptance thresholds.
    double max_rmse = 0.65;
    std::size_t min_inliers = 300;
    double min_overlap_ratio = 0.15;

    // Final sanity gate on how far ICP may move from the selected hypothesis.
    double max_correction_translation = 15.0;
    double max_correction_rotation_deg = 45.0;

    // Reject extremely small clouds before doing ICP.
    std::size_t min_cloud_points = 300;

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
    // before spending time on full 50-iteration ICP.
    // ------------------------------------------------------------------------
    bool ScoreInitialGuess(
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &source_current,
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &target_historical,
        const Eigen::Isometry3d &initial_guess,
        LoopVerifierInitialGuessScore &score) const;

    // ------------------------------------------------------------------------
    // Verify one candidate.
    //
    // The public API is intentionally unchanged from V1 so the existing loop
    // decision / temporal / PoseGraph code does not need to change.
    // ------------------------------------------------------------------------
    bool Verify(
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &source_current,
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &target_historical,
        const Eigen::Isometry3d &graph_initial_guess,
        double scan_context_yaw_shift_deg,
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
