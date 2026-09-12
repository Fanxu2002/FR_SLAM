#ifndef FR_SLAM_FR_GROUND_SEGMENTER_HPP_
#define FR_SLAM_FR_GROUND_SEGMENTER_HPP_

#include <cstddef>
#include <cstdint>
#include <deque>

#include <Eigen/Core>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace fr_slam
{

struct GroundSegmentationConfig
{
    // ------------------------------------------------------------
    // Input range used by the grid surface model.
    // ------------------------------------------------------------
    double minimum_range_m = 0.8;
    double maximum_range_m = 15.0;

    // ------------------------------------------------------------
    // XY grid.
    // ------------------------------------------------------------
    double grid_size_m = 0.40;
    std::size_t minimum_points_per_cell = 2;

    // Robust low-surface estimate inside one cell.
    // Example: 0.30 means use the median of the lowest 30% z samples.
    double low_surface_fraction = 0.30;

    // Reject cells whose low layer itself is too vertically scattered.
    // This suppresses wall / vegetation cells without assuming horizontal
    // ground. A genuine slope can still have low roughness locally.
    double maximum_cell_low_roughness_m = 0.10;

    // ------------------------------------------------------------
    // Near-field seed search for Ground-like terrain.
    // ------------------------------------------------------------
    double seed_minimum_range_m = 0.8;
    double seed_maximum_range_m = 4.0;

    std::size_t minimum_seed_component_cells = 6;
    std::size_t seed_angular_sector_count = 16;
    std::size_t minimum_seed_angular_sectors = 4;

    // ------------------------------------------------------------
    // Ground V3 propagation: local height-field prediction.
    //
    //     z = a x + b y + c
    //
    // The model is local and does NOT require z = constant.
    // ------------------------------------------------------------
    int prediction_neighbor_radius_cells = 2;
    std::size_t minimum_prediction_support_cells = 4;

    double maximum_predicted_surface_slope_deg = 40.0;

    double surface_prediction_base_tolerance_m = 0.07;
    double surface_prediction_roughness_scale = 2.5;
    double surface_prediction_rmse_scale = 2.0;
    double maximum_surface_prediction_tolerance_m = 0.20;

    // Fallback continuity used only when the local model is unavailable.
    double maximum_local_slope_deg = 30.0;
    double maximum_neighbor_height_jump_m = 0.40;

    // ------------------------------------------------------------
    // Point-level Ground-like classification.
    // ------------------------------------------------------------
    double point_below_surface_tolerance_m = 0.06;
    double point_height_base_threshold_m = 0.06;
    double point_height_roughness_scale = 3.0;
    double point_height_minimum_threshold_m = 0.05;
    double point_height_maximum_threshold_m = 0.16;

    // ------------------------------------------------------------
    // General local Ground plane (diagnostic only).
    // ------------------------------------------------------------
    double local_plane_minimum_range_m = 0.8;
    double local_plane_maximum_range_m = 4.0;
    std::size_t minimum_local_plane_cells = 12;

    double local_plane_base_residual_m = 0.05;
    double local_plane_mad_scale = 3.0;
    double local_plane_maximum_residual_m = 0.18;
    double minimum_local_plane_inlier_ratio = 0.55;

    // ============================================================
    // Ground V3.3: Gap-tolerant Multi-hypothesis Vehicle Support Surface selector.
    //
    // Ground-like terrain may contain road shoulder, curb ramps or another
    // low continuous surface. V3.3 keeps the V3.2 multi-hypothesis selector
    // but makes candidate connectivity tolerant to ONE missing grid cell.
    // Gap links are allowed only when the missing intermediate cell is truly
    // absent and the two endpoint surfaces agree under stricter geometry.
    //
    // LiDAR convention assumed by this selector:
    //     +X forward, +Y left/right, +Z up.
    // All values are configurable.
    // ============================================================

    // Rectangular corridor used to obtain the first support-plane hypothesis.
    double support_corridor_rear_m = 1.5;
    double support_corridor_forward_m = 4.0;
    double support_corridor_half_width_m = 1.10;

    // Support cells can be collected slightly farther than the fit corridor,
    // but must agree geometrically with the corridor plane.
    double support_maximum_range_m = 5.0;

    // ------------------------------------------------------------
    // V3.3 gap-tolerant multi-hypothesis component extraction.
    // A narrow primary strip gives strong evidence for the surface directly
    // under the vehicle, while the wider corridor is only a search region.
    // ------------------------------------------------------------
    double support_primary_half_width_m = 0.60;

    int support_component_local_model_radius_cells = 2;
    std::size_t minimum_support_component_local_cells = 4;

    // Adjacent cells remain in one candidate when their LOCAL surface normals
    // are similar. This preserves a genuine uniform slope, while splitting a
    // road from a shoulder when the local surface orientation changes.
    double support_component_maximum_normal_change_deg = 6.0;
    double support_component_maximum_model_residual_m = 0.08;

    // Fallback only when a local model cannot be estimated.
    double support_component_fallback_maximum_slope_deg = 18.0;
    double support_component_fallback_maximum_height_jump_m = 0.20;

    // V3.3 gap-tolerant connectivity. Radius=2 bridges at most one missing
    // cell. A gap link is never allowed to jump over an existing but
    // geometrically incompatible cell (important at a road shoulder).
    int support_component_gap_radius_cells = 2;
    double support_component_gap_normal_scale = 0.75;
    double support_component_gap_residual_scale = 0.75;
    double support_component_gap_fallback_slope_scale = 0.75;
    double support_component_gap_height_jump_scale = 0.75;

    // Candidate admission is deliberately a little more permissive than V3.2.
    // Final support-plane fitting and temporal gates remain strict.
    std::size_t minimum_support_candidate_cells = 4;
    std::size_t minimum_support_center_strip_cells = 2;

    // Candidate scoring. Higher score is better.
    double support_center_gaussian_sigma_m = 0.45;
    double support_score_center_strip_weight = 4.0;
    double support_score_center_proximity_weight = 2.0;
    double support_score_area_weight = 1.0;
    double support_score_fit_weight = 1.5;
    double support_score_rmse_weight = 1.0;
    double support_score_temporal_normal_weight = 3.0;
    double support_score_temporal_distance_weight = 3.0;
    std::size_t support_score_area_saturation_cells = 25;
    double support_score_temporal_normal_sigma_deg = 4.0;
    double support_score_temporal_distance_sigma_m = 0.06;

    // ------------------------------------------------------------
    // V3.3 persistent recovery (logic inherited from V3.2).
    // If a genuinely new support surface (e.g. vehicle enters a slope) is
    // repeatedly observed with strong center-strip support, allow controlled
    // re-initialization instead of remaining INVALID forever.
    // ------------------------------------------------------------
    std::size_t support_recovery_required_consecutive_frames = 6;
    double support_recovery_maximum_candidate_normal_change_deg = 2.5;
    double support_recovery_maximum_candidate_distance_change_m = 0.05;
    std::size_t support_recovery_minimum_center_strip_cells = 4;
    double support_recovery_minimum_inlier_ratio = 0.85;
    double support_recovery_maximum_rmse_m = 0.05;

    std::size_t minimum_support_corridor_cells = 8;
    std::size_t minimum_support_plane_cells = 10;

    // Robust corridor-plane fit.
    double support_plane_base_residual_m = 0.04;
    double support_plane_mad_scale = 3.0;
    double support_plane_maximum_residual_m = 0.12;
    double minimum_support_plane_inlier_ratio = 0.60;

    // A Ground cell is promoted to Support Ground only when its low surface
    // agrees with the selected support plane.
    double support_cell_base_residual_m = 0.06;
    double support_cell_roughness_scale = 2.0;
    double support_cell_maximum_residual_m = 0.14;

    // Temporal consistency is intentionally a QUALITY GATE, not a flat-ground
    // assumption. A real slope can be followed as long as the support normal
    // and sensor-to-surface distance evolve continuously.
    double maximum_support_normal_change_deg = 8.0;
    double maximum_support_distance_change_m = 0.12;

    // ============================================================
    // Ground V4.0: TRUSTED support-constraint gate.
    //
    // support_plane_valid means: "a geometrically plausible vehicle support
    // surface was found".  support_constraint_valid is intentionally stricter
    // and means: "this frame is trusted enough to influence Scan-to-LocalMap".
    //
    // The gate is pure LiDAR.  It never forces n=[0,0,1] and never forces a
    // fixed world Z.  Instead it learns the rigid sensor-to-support clearance
    // from high-quality observations and protects that clearance with robust
    // median/MAD statistics.
    // ============================================================

    // Minimum geometry needed before a support plane may become an ICP
    // constraint.  Candidate generation stays permissive; the final gate is
    // deliberately conservative.
    std::size_t support_constraint_minimum_points = 120;
    std::size_t support_constraint_minimum_cells = 35;
    std::size_t support_constraint_minimum_center_strip_cells = 3;
    double support_constraint_minimum_inlier_ratio = 0.90;
    double support_constraint_maximum_rmse_m = 0.030;
    double support_constraint_minimum_selected_score = 8.5;

    // Confidence hysteresis prevents one noisy frame from rapidly toggling a
    // future ICP residual on/off.  Hard safety gates always dominate.
    double support_constraint_enter_confidence = 0.72;
    double support_constraint_keep_confidence = 0.62;

    // Compare a candidate to the last TRUSTED constraint as an additional
    // short-term protection.  This is not a long-term normal prior: a genuine
    // slope is still followed frame-by-frame.
    double support_constraint_maximum_trusted_normal_change_deg = 8.0;
    double support_constraint_maximum_trusted_distance_change_m = 0.10;

    // Robust LiDAR-to-support clearance anchor.  The anchor is learned rather
    // than hard-coded.  During bootstrap only strong support frames are added.
    std::size_t support_clearance_bootstrap_samples = 8;
    std::size_t support_clearance_history_size = 31;
    std::size_t support_clearance_bootstrap_minimum_points = 150;
    std::size_t support_clearance_bootstrap_minimum_cells = 40;
    std::size_t support_clearance_bootstrap_minimum_center_strip_cells = 4;
    double support_clearance_bootstrap_minimum_inlier_ratio = 0.92;
    double support_clearance_bootstrap_maximum_rmse_m = 0.030;
    double support_clearance_bootstrap_minimum_selected_score = 7.5;

    // Once at least this many bootstrap samples exist, a provisional median
    // gate stops a stable wrong shoulder from silently building the anchor.
    std::size_t support_clearance_bootstrap_provisional_gate_samples = 3;
    double support_clearance_bootstrap_maximum_deviation_m = 0.07;

    // Final robust anchor tolerance:
    //   tol = clamp(max(base, MAD_scale * robust_sigma), base, maximum)
    double support_clearance_anchor_base_tolerance_m = 0.060;
    double support_clearance_anchor_mad_scale = 3.5;
    double support_clearance_anchor_maximum_tolerance_m = 0.090;

    // Only very consistent trusted frames are allowed to move the anchor.
    double support_clearance_history_update_maximum_error_m = 0.050;
};


enum SupportConstraintRejectMask : std::uint32_t
{
    SUPPORT_CONSTRAINT_REJECT_NONE = 0U,
    SUPPORT_CONSTRAINT_REJECT_NO_SUPPORT = 1U << 0,
    SUPPORT_CONSTRAINT_REJECT_BOOTSTRAP = 1U << 1,
    SUPPORT_CONSTRAINT_REJECT_LOW_SCORE = 1U << 2,
    SUPPORT_CONSTRAINT_REJECT_LOW_POINTS = 1U << 3,
    SUPPORT_CONSTRAINT_REJECT_LOW_CELLS = 1U << 4,
    SUPPORT_CONSTRAINT_REJECT_LOW_CENTER = 1U << 5,
    SUPPORT_CONSTRAINT_REJECT_LOW_INLIER = 1U << 6,
    SUPPORT_CONSTRAINT_REJECT_HIGH_RMSE = 1U << 7,
    SUPPORT_CONSTRAINT_REJECT_TEMPORAL = 1U << 8,
    SUPPORT_CONSTRAINT_REJECT_ANCHOR = 1U << 9,
    SUPPORT_CONSTRAINT_REJECT_TRUSTED_JUMP = 1U << 10,
    SUPPORT_CONSTRAINT_REJECT_LOW_CONFIDENCE = 1U << 11
};


struct GroundSegmentationResult
{
    bool success = false;

    // All Ground-like terrain found by V3 propagation.
    pcl::PointCloud<pcl::PointXYZ>::Ptr ground_cloud;

    // Everything not classified as Ground-like terrain.
    pcl::PointCloud<pcl::PointXYZ>::Ptr nonground_cloud;

    // V3.3 subset: the surface judged to be the vehicle support surface.
    pcl::PointCloud<pcl::PointXYZ>::Ptr support_ground_cloud;

    // ------------------------------------------------------------
    // General local Ground-like plane (diagnostic only).
    // ------------------------------------------------------------
    bool local_plane_valid = false;

    Eigen::Vector3d local_ground_normal_L =
        Eigen::Vector3d::UnitZ();

    double local_ground_plane_d = 0.0;
    double local_ground_distance_m = 0.0;
    double local_ground_tilt_deg = 0.0;
    double local_plane_inlier_ratio = 0.0;
    double local_plane_rmse_m = 0.0;

    // ------------------------------------------------------------
    // V3.3 support-surface geometry in current LiDAR coordinates:
    //
    //     n_L^T p + d = 0
    //
    // This is the geometric support estimate. It is NOT forced to be
    // horizontal. V4.0 requires support_constraint_valid before future ICP use.
    // ------------------------------------------------------------
    bool support_plane_valid = false;

    Eigen::Vector3d support_ground_normal_L =
        Eigen::Vector3d::UnitZ();

    double support_ground_plane_d = 0.0;
    double support_ground_distance_m = 0.0;
    double support_ground_tilt_deg = 0.0;
    double support_plane_inlier_ratio = 0.0;
    double support_plane_rmse_m = 0.0;

    // Temporal diagnostics. NaN means there was no previous accepted support
    // plane yet.
    bool support_temporal_gate_passed = false;
    double support_normal_change_deg = 0.0;
    double support_distance_change_m = 0.0;

    // ------------------------------------------------------------
    // Diagnostics.
    // ------------------------------------------------------------
    std::size_t input_points = 0;
    std::size_t valid_points = 0;

    std::size_t grid_cells = 0;
    std::size_t valid_surface_cells = 0;
    std::size_t rejected_rough_surface_cells = 0;

    std::size_t near_seed_candidate_cells = 0;
    std::size_t seed_component_cells = 0;
    std::size_t seed_component_angular_sectors = 0;

    std::size_t ground_cells = 0;
    std::size_t predicted_ground_cells = 0;
    std::size_t fallback_ground_cells = 0;

    std::size_t ground_points = 0;
    std::size_t nonground_points = 0;

    double mean_ground_cell_roughness_m = 0.0;
    double maximum_ground_cell_roughness_m = 0.0;

    std::size_t local_plane_cells = 0;
    std::size_t local_plane_inliers = 0;

    // V3.3 support diagnostics.
    std::size_t support_corridor_cells = 0;

    // Candidate-generation diagnostics. These make candidates=0 explainable.
    std::size_t support_component_count_total = 0;
    std::size_t support_component_gap_links = 0;
    std::size_t support_component_rejected_small = 0;
    std::size_t support_component_rejected_no_center = 0;
    std::size_t support_component_rejected_fit = 0;

    std::size_t support_candidate_count = 0;
    std::size_t support_selected_candidate_index = 0;
    std::size_t support_selected_component_cells = 0;
    std::size_t support_selected_center_strip_cells = 0;
    double support_selected_score = 0.0;

    std::size_t support_plane_cells = 0;
    std::size_t support_plane_inliers = 0;
    std::size_t support_ground_cells = 0;
    std::size_t support_ground_points = 0;

    bool support_recovery_pending = false;
    std::size_t support_recovery_pending_count = 0;
    bool support_reinitialized = false;

    // ------------------------------------------------------------
    // V4.0 trusted constraint output.
    // support_plane_valid may be true while support_constraint_valid is false.
    // Only support_constraint_valid is intended for future ICP residuals.
    // ------------------------------------------------------------
    bool support_constraint_valid = false;
    double support_constraint_confidence = 0.0;
    std::uint32_t support_constraint_rejection_mask =
        SUPPORT_CONSTRAINT_REJECT_NO_SUPPORT;

    bool support_clearance_anchor_valid = false;
    double support_clearance_anchor_m = 0.0;
    double support_clearance_anchor_sigma_m = 0.0;
    double support_clearance_anchor_tolerance_m = 0.0;
    double support_clearance_error_m = 0.0;
    std::size_t support_clearance_history_samples = 0;
    bool support_clearance_sample_accepted = false;

    // Difference from the last trusted constraint.  NaN means no trusted
    // constraint has been established yet.
    double support_trusted_normal_change_deg = 0.0;
    double support_trusted_distance_change_m = 0.0;
};


class GroundSegmenter
{
public:
    GroundSegmenter();

    explicit GroundSegmenter(
        const GroundSegmentationConfig &config);

    void SetConfig(
        const GroundSegmentationConfig &config);

    const GroundSegmentationConfig &GetConfig() const;

    void Reset();

    GroundSegmentationResult Segment(
        const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud);

private:
    GroundSegmentationConfig config_;

    bool has_previous_support_plane_ = false;

    Eigen::Vector3d previous_support_normal_L_ =
        Eigen::Vector3d::UnitZ();

    double previous_support_distance_m_ = 0.0;

    bool has_pending_support_plane_ = false;

    Eigen::Vector3d pending_support_normal_L_ =
        Eigen::Vector3d::UnitZ();

    double pending_support_distance_m_ = 0.0;

    std::size_t pending_support_consecutive_frames_ = 0;

    // V4.0 trusted-constraint state.
    std::deque<double> support_clearance_history_m_;

    bool support_constraint_was_valid_ = false;

    bool has_trusted_support_plane_ = false;

    Eigen::Vector3d trusted_support_normal_L_ =
        Eigen::Vector3d::UnitZ();

    double trusted_support_distance_m_ = 0.0;

    void EvaluateSupportConstraint(
        GroundSegmentationResult &result);
};

}  // namespace fr_slam

#endif  // FR_SLAM_FR_GROUND_SEGMENTER_HPP_
