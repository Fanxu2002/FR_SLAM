#pragma once

#include <cstddef>
#include <string>

#include "fr_slam/frontend/ground_segmenter.hpp"

// Configuration for the realtime Ground pose constraint.
//
// Final Ground V1 architecture:
//
//   ordinary ICP -> baseline pose
//   trusted current-scan Ground + piecewise-frozen world reference
//       -> Ground-only tilt/Z proposal
//       -> trust region
//       -> alpha line search
//       -> general-geometry safety evaluation
//       -> final frontend pose
//
// Ground never contributes Hessian/Jacobian terms to ordinary ICP.
// General ICP geometry never participates in the Ground proposal solve.
struct GroundConstraintConfig
{
    bool enabled = true;
    std::string mode = "piecewise_frozen";

    fr_slam::GroundSegmentationConfig segmentation;

    double analysis_voxel_leaf_m = 0.15;

    std::size_t anchor_bootstrap_frames = 8;
    double anchor_maximum_normal_spread_deg = 2.0;
    double anchor_maximum_plane_d_spread_m = 0.08;

    // Trusted-Ground quality scale.  Final Ground V1 uses this only as a
    // measurement-quality gate/diagnostic; it is NOT an information-matrix
    // multiplier because Ground no longer enters the ICP Hessian.
    double plane_information_scale = 1.0;

    // ------------------------------------------------------------------
    // Legacy compatibility fields.
    //
    // app/lio.cpp still declares the previous Ground V1.4 parameters.
    // Final Ground V1 does NOT use these fields in its Ground solver.
    // They are retained only so the shared GroundConstraintConfig remains
    // source-compatible with the current LIO development target.
    // ------------------------------------------------------------------
    double height_sigma_m = 0.005;
    double normal_sigma_deg = 2.0;
    double height_huber_delta_m = 0.05;
    double normal_huber_delta_deg = 2.0;

    int maximum_refinement_iterations = 2;
    double maximum_total_rotation_correction_deg = 0.50;
    double maximum_total_translation_correction_m = 0.050;

    // Gross residual sanity gates.  These do not drive reference switching.
    // A large residual primarily means pose/reference disagreement and must
    // never, by itself, move the piecewise-frozen reference.
    double maximum_height_residual_m = 5.0;
    double maximum_normal_residual_deg = 10.0;

    // Ground-only trust region.
    double maximum_tilt_correction_deg = 0.25;
    double maximum_z_correction_m = 0.030;

    // Candidate construction guards.
    double minimum_reference_normal_z = 0.50;
    double minimum_heading_projection_norm = 1.0e-4;
    double ground_sanity_height_tolerance_m = 0.005;
    double ground_sanity_normal_tolerance_deg = 1.0e-6;

    // Sensor/robot forward axis expressed in the current LiDAR frame.
    // It is projected into the measured Ground tangent plane before use.
    double forward_axis_x = 1.0;
    double forward_axis_y = 0.0;
    double forward_axis_z = 0.0;

    // Largest safe alpha is accepted first.
    double line_search_alpha_1 = 1.0;
    double line_search_alpha_2 = 0.5;
    double line_search_alpha_3 = 0.25;

    // Piecewise-frozen lifecycle.  Entering PENDING_SWITCH requires an
    // independent pose-invariant TerrainChangeDetector signal.  Final Ground
    // V1 intentionally does not infer terrain change from Ground residuals.
    std::size_t reference_switch_confirmation_frames = 8;

    // The Ground result is discarded if it damages the original all-scene
    // point-to-plane solution beyond any of these limits.
    double maximum_general_rmse_ratio = 1.03;
    double maximum_general_rmse_absolute_increase_m = 0.003;
    double minimum_general_correspondence_ratio = 0.85;

    bool diagnostics_enabled = true;
    std::size_t diagnostics_flush_interval = 20;
};
