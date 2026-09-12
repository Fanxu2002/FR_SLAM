#pragma once

#include <cstddef>
#include <string>

#include "fr_slam/frontend/ground_segmenter.hpp"

// Configuration for the realtime Ground pose constraint.
//
// "flat_anchor" learns one world-frame support plane from the first trusted
// Ground observations and then freezes it.  This is intentionally different
// from matching Ground points back to the drifting LocalMap: the frozen plane
// supplies an independent reference for roll, pitch and world Z.
struct GroundConstraintConfig
{
    bool enabled = true;
    std::string mode = "flat_anchor";

    fr_slam::GroundSegmentationConfig segmentation;

    double analysis_voxel_leaf_m = 0.15;

    std::size_t anchor_bootstrap_frames = 8;
    double anchor_maximum_normal_spread_deg = 2.0;
    double anchor_maximum_plane_d_spread_m = 0.08;

    // Quality multiplier applied before the statistically meaningful
    // 1/sigma^2 information terms below.
    double plane_information_scale = 1.0;

    double height_sigma_m = 0.05;
    double normal_sigma_deg = 2.0;
    double height_huber_delta_m = 0.05;
    double normal_huber_delta_deg = 2.0;

    double maximum_height_residual_m = 0.30;
    double maximum_normal_residual_deg = 10.0;

    int maximum_refinement_iterations = 2;
    double maximum_total_rotation_correction_deg = 0.50;
    double maximum_total_translation_correction_m = 0.050;

    // The Ground result is discarded if it damages the original all-scene
    // point-to-plane solution beyond any of these limits.
    double maximum_general_rmse_ratio = 1.03;
    double maximum_general_rmse_absolute_increase_m = 0.003;
    double minimum_general_correspondence_ratio = 0.85;

    bool diagnostics_enabled = true;
    std::size_t diagnostics_flush_interval = 20;
};
