#pragma once

#include "fr_slam/fr_point_types.hpp"

#include <cstddef>
#include <limits>

#include <Eigen/Geometry>

#include <pcl/point_cloud.h>

// ============================================================================
// LoopVerifier V1
//
// Purpose:
//
//     Scan Context answers:
//
//         "Which old Submap looks similar?"
//
//     LoopVerifier answers:
//
//         "Can the two actual point clouds be geometrically aligned?"
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
//
// For loop verification:
//
//     source = current Submap cloud_S
//     target = historical candidate Submap cloud_S
//
// therefore the returned transform is:
//
//     T_target_source
//       =
//     T_Shistorical_Scurrent
//
// which is exactly the measurement direction needed later by:
//
//     AddLoopEdge(
//         historical_id,
//         current_id,
//         T_Shistorical_Scurrent,
//         information);
// ============================================================================

struct LoopVerifierConfig
{
    // Backend Submap clouds are already voxelized by LocalMap, but a slightly
    // coarser verifier cloud keeps candidate checking inexpensive.
    double voxel_leaf_size = 0.50;

    // Coarse point-to-point ICP.
    //
    // This is intentionally wider than the final verification inlier gate.
    // Its job is to enter the correct basin of attraction.
    std::size_t max_iterations = 50;
    double max_correspondence_distance = 5.0;
    double transformation_epsilon = 1.0e-6;
    double euclidean_fitness_epsilon = 1.0e-5;

    // Final geometric quality measurement after ICP.
    //
    // A transformed source point is considered an inlier when its nearest
    // target point lies within this distance.
    double verification_inlier_distance = 1.0;

    // V1 acceptance thresholds.
    //
    // These are deliberately starting values, not final tuned parameters.
    // We will tune them from the real loop-verification log.
    double max_rmse = 0.65;
    std::size_t min_inliers = 300;
    double min_overlap_ratio = 0.15;

    // Final sanity gate on how far ICP is allowed to move away from the
    // selected initial hypothesis.
    //
    // This is NOT intended to force the loop measurement to stay close to the
    // raw odometry graph forever. The verifier already tests three different
    // hypotheses (graph / +SC yaw / -SC yaw). These limits only reject ICP
    // solutions that require an implausibly large correction from the winning
    // hypothesis, which is a common symptom of convergence to a wrong repeated
    // structure.
    //
    // V1 values are deliberately loose and should be tuned from real logs.
    double max_correction_translation = 15.0;
    double max_correction_rotation_deg = 45.0;

    // Reject extremely small clouds before doing ICP.
    std::size_t min_cloud_points = 300;
};

enum class LoopVerifierHypothesis
{
    GraphPose = 0,
    ScanContextPositiveYaw = 1,
    ScanContextNegativeYaw = 2
};

struct LoopVerificationResult
{
    bool success = false;
    bool converged = false;
    bool accepted = false;

    // Final current-Submap -> historical-Submap transform.
    Eigen::Isometry3d T_target_source =
        Eigen::Isometry3d::Identity();

    // Initial transform used by the winning hypothesis.
    Eigen::Isometry3d initial_guess =
        Eigen::Isometry3d::Identity();

    LoopVerifierHypothesis hypothesis =
        LoopVerifierHypothesis::GraphPose;

    // PCL ICP's own fitness diagnostic.
    double fitness_score =
        std::numeric_limits<double>::infinity();

    // Our explicit final nearest-neighbour verification metrics.
    std::size_t inliers = 0;
    double overlap_ratio = 0.0;
    double rmse =
        std::numeric_limits<double>::infinity();

    // How much ICP corrected the chosen initial guess.
    double correction_translation =
        std::numeric_limits<double>::infinity();

    double correction_rotation_deg =
        std::numeric_limits<double>::infinity();

    // Number of points after verifier voxel filtering.
    std::size_t source_points = 0;
    std::size_t target_points = 0;
};

class LoopVerifier
{
public:
    explicit LoopVerifier(
        const LoopVerifierConfig &config =
            LoopVerifierConfig());

    // ------------------------------------------------------------------------
    // Verify one Scan Context candidate.
    //
    // source_current:
    //     current finished Submap cloud, in S_current.
    //
    // target_historical:
    //     historical finished Submap cloud, in S_historical.
    //
    // graph_initial_guess:
    //
    //     T_Shistorical_Scurrent_guess
    //       =
    //     T_WS_historical^{-1} * T_WS_current
    //
    // scan_context_yaw_shift_deg:
    //     coarse yaw displacement reported by Scan Context.
    //
    // V1 tests THREE initial-rotation hypotheses:
    //
    //     1. graph relative pose
    //     2. +ScanContext yaw
    //     3. -ScanContext yaw
    //
    // Testing both signs is intentional because we have not yet calibrated the
    // Scan Context shift sign against the project's T_AB convention.
    // ------------------------------------------------------------------------
    bool Verify(
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &source_current,
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &target_historical,
        const Eigen::Isometry3d &graph_initial_guess,
        double scan_context_yaw_shift_deg,
        LoopVerificationResult &result) const;

    const LoopVerifierConfig &GetConfig() const;

    static const char *HypothesisName(
        LoopVerifierHypothesis hypothesis);

private:
    LoopVerifierConfig config_;
};
