#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <Eigen/Core>
#include <pcl/point_cloud.h>

#include "fr_slam/common/point_types.hpp"

namespace fr_slam
{

struct GroundSegmentationResult;

enum class PlaneSemanticType
{
    GroundCandidate,
    Horizontal,
    Vertical,
    Oblique
};

const char *PlaneSemanticTypeName(PlaneSemanticType type);

enum class SupportSurfaceMode
{
    None,
    Flat,
    Slope
};


const char *SupportSurfaceModeName(
    SupportSurfaceMode mode);


enum PlaneConstraintRejectMask : std::uint32_t
{
    PLANE_CONSTRAINT_REJECT_NONE = 0U,
    PLANE_CONSTRAINT_REJECT_LOW_POINTS = 1U << 0,
    PLANE_CONSTRAINT_REJECT_LOW_PATCHES = 1U << 1,
    PLANE_CONSTRAINT_REJECT_SMALL_RADIUS = 1U << 2,
    PLANE_CONSTRAINT_REJECT_WEAK_PLANARITY = 1U << 3,
    PLANE_CONSTRAINT_REJECT_GROUND_ANGLE = 1U << 4,
    PLANE_CONSTRAINT_REJECT_GROUND_V4_INVALID = 1U << 5,
    PLANE_CONSTRAINT_REJECT_GROUND_NORMAL_MISMATCH = 1U << 6,
    PLANE_CONSTRAINT_REJECT_GROUND_DISTANCE_MISMATCH = 1U << 7,
    PLANE_CONSTRAINT_REJECT_WALL_ANGLE = 1U << 8,
    PLANE_CONSTRAINT_REJECT_UNSUPPORTED_SEMANTIC = 1U << 9,

    // V1.4 Active Support Surface diagnostics.
    PLANE_CONSTRAINT_REJECT_SUPPORT_TOO_STEEP = 1U << 10,
    PLANE_CONSTRAINT_REJECT_SUPPORT_CENTER_TOO_HIGH = 1U << 11,
    PLANE_CONSTRAINT_REJECT_SUPPORT_DISTANCE_TOO_LARGE = 1U << 12,
    PLANE_CONSTRAINT_REJECT_NOT_ACTIVE_SUPPORT = 1U << 13,

    // V1.4.1 curb / road-edge rejection for slope support.
    PLANE_CONSTRAINT_REJECT_SUPPORT_SLOPE_TOO_NARROW = 1U << 14,
    PLANE_CONSTRAINT_REJECT_SUPPORT_SLOPE_OUTSIDE_CORRIDOR = 1U << 15,

    // V1.5 local-ground selection diagnostics.
    PLANE_CONSTRAINT_REJECT_LOCAL_GROUND_OUTSIDE_ROI = 1U << 16,
    PLANE_CONSTRAINT_REJECT_LOCAL_GROUND_TOO_STEEP = 1U << 17,
    PLANE_CONSTRAINT_REJECT_LOCAL_GROUND_SLOPE_DISTANCE_JUMP = 1U << 18
};

struct MultiPlaneConfig
{
    double voxel_size_m = 1.0;
    std::size_t minimum_points_per_voxel = 10;

    double plane_detection_eigenvalue_threshold = 0.01;
    double plane_merge_normal_threshold = 0.10;
    double plane_merge_distance_threshold_m = 0.30;

    std::size_t minimum_points_per_output_plane = 20;
    std::size_t maximum_output_planes = 12;

    // V1.2 classification relative to supplied up direction.
    double horizontal_max_angle_to_up_deg = 12.0;
    double vertical_min_angle_to_up_deg = 78.0;

    // Only one horizontal plane can become GroundCandidate.
    double ground_candidate_max_center_up_m = 0.25;
    double ground_candidate_max_distance_m = 3.0;
    std::size_t ground_candidate_minimum_points = 100;
    double ground_candidate_minimum_radius_m = 0.60;


    // V1.3 constraint-eligibility refinement.
    // Detection stays permissive. Only strong planes are allowed to become
    // future measurement candidates.
    std::size_t constraint_minimum_points = 150;
    std::size_t constraint_minimum_merged_patches = 3;
    double constraint_minimum_radius_m = 0.80;

    // Wall-only radius gate.
    // Keep 0.80 m for existing outdoor behavior.
    // Indoor corridor experiments may override this to 0.50 m.
    double wall_constraint_minimum_radius_m = 0.80;

    double constraint_maximum_min_eigenvalue = 0.008;

    double refined_ground_maximum_angle_to_up_deg = 8.0;
    double refined_ground_maximum_v4_normal_difference_deg = 5.0;
    double refined_ground_maximum_v4_plane_distance_difference_m = 0.15;

    double refined_wall_minimum_angle_to_up_deg = 82.0;


    // ------------------------------------------------------------------
    // Multi-Plane V1.4 Active Support Surface.
    //
    // Flat and slope surfaces may both become support candidates.
    // Multiple candidates may coexist, but exactly zero or one current
    // plane is marked as the active support surface.
    // ------------------------------------------------------------------
    double support_flat_maximum_angle_to_up_deg = 8.0;
    double support_maximum_slope_angle_to_up_deg = 30.0;

    double support_candidate_maximum_center_up_m = 0.35;
    double support_candidate_maximum_plane_distance_m = 3.0;

    double support_temporal_maximum_normal_difference_deg = 6.0;
    double support_temporal_maximum_plane_distance_difference_m = 0.20;

    std::size_t support_switch_confirmation_frames = 3;
    std::size_t support_maximum_missed_frames = 2;


    // MultiPlane V1.5: one local Ground, multiple Wall candidates.
    double local_ground_maximum_angle_to_up_deg = 30.0;
    double local_ground_roi_longitudinal_half_length_m = 4.0;
    double local_ground_roi_lateral_half_width_m = 1.0;
    double local_ground_footprint_sigma_multiplier = 2.50;
    double local_ground_maximum_origin_footprint_gap_m = 1.00;

    // V1.5.1: dedicated GroundCandidate may bridge a slightly larger
    // LiDAR blind-zone than generic Horizontal/Oblique planes.
    double local_ground_ground_candidate_maximum_origin_gap_m = 1.50;

    std::size_t local_ground_slope_confirmation_frames = 3;
    double local_ground_slope_temporal_maximum_normal_difference_deg = 6.0;
    double local_ground_slope_temporal_maximum_plane_distance_difference_m = 0.20;
    double local_ground_slope_temporal_maximum_center_distance_m = 1.50;

    // |d| is perpendicular distance from LiDAR origin to the support plane.
    double local_ground_slope_maximum_recent_support_distance_difference_m = 0.35;

    double local_ground_maximum_center_up_m = 0.35;
    double local_ground_minimum_center_up_m = -1.00;

    // V1.4.1 curb / road-edge rejection.
    double support_slope_minimum_lateral_sigma_m = 0.30;
    double support_slope_corridor_half_width_m = 0.75;
    double support_slope_lateral_sigma_multiplier = 2.50;
};

struct PlaneObservation
{
    Eigen::Vector3d center_L = Eigen::Vector3d::Zero();
    Eigen::Vector3d normal_L = Eigen::Vector3d::UnitZ();
    Eigen::Matrix3d covariance_L = Eigen::Matrix3d::Zero();

    double d = 0.0;
    double radius_m = 0.0;
    double minimum_eigenvalue = 0.0;

    std::size_t point_count = 0;
    std::size_t merged_patch_count = 0;

    PlaneSemanticType semantic_type = PlaneSemanticType::Oblique;
    double angle_to_up_deg = 90.0;
    double ground_candidate_score = 0.0;


    // V1.3 measurement-candidate refinement diagnostics.
    bool constraint_eligible = false;
    double constraint_quality_score = 0.0;
    std::uint32_t constraint_rejection_mask =
        PLANE_CONSTRAINT_REJECT_NONE;

    // NaN when this plane is not the GroundCandidate or V4 cross-check
    // cannot be evaluated.
    double ground_v4_normal_difference_deg = 0.0;
    double ground_v4_plane_distance_difference_m = 0.0;


    // V1.4 support-surface diagnostics.
    bool support_candidate = false;
    bool active_support = false;
    bool support_v4_confirmed = false;

    SupportSurfaceMode support_mode =
        SupportSurfaceMode::None;

    double support_selection_score = 0.0;

    double support_temporal_normal_difference_deg = 0.0;
    double support_temporal_plane_distance_difference_m = 0.0;


    // V1.5 local-ground / wall-candidate diagnostics.
    bool local_ground_candidate = false;
    bool active_ground = false;
    bool wall_constraint_candidate = false;

    double local_ground_longitudinal_center_m = 0.0;
    double local_ground_lateral_center_m = 0.0;
    double local_ground_longitudinal_sigma_m = 0.0;
    double local_ground_lateral_sigma_m = 0.0;
    double local_ground_longitudinal_roi_gap_m = 0.0;
    double local_ground_lateral_roi_gap_m = 0.0;
    double local_ground_origin_footprint_gap_m = 0.0;
    double local_ground_selection_score = 0.0;

    // V1.4.1 slope-corridor diagnostics.
    double support_lateral_center_m = 0.0;
    double support_lateral_sigma_m = 0.0;
    double support_corridor_gap_m = 0.0;
};

struct MultiPlaneExtractionResult
{
    std::size_t input_points = 0;
    std::size_t occupied_voxels = 0;
    std::size_t raw_plane_patches = 0;
    std::size_t merged_plane_count_before_filter = 0;

    std::size_t ground_candidate_count = 0;
    std::size_t horizontal_plane_count = 0;
    std::size_t vertical_plane_count = 0;
    std::size_t oblique_plane_count = 0;


    // Eligible = good enough for a future measurement.
    // V1.3 still does not apply these planes to pose.
    std::size_t eligible_ground_plane_count = 0;
    std::size_t eligible_wall_plane_count = 0;


    // V1.4 support-surface status.
    std::size_t support_candidate_count = 0;
    std::size_t active_support_plane_count = 0;

    int active_support_index = -1;

    SupportSurfaceMode active_support_mode =
        SupportSurfaceMode::None;

    std::size_t support_pending_confirmation_count = 0;


    // V1.5 result summary.
    std::size_t local_ground_candidate_count = 0;
    std::size_t active_ground_plane_count = 0;
    int active_ground_index = -1;
    std::size_t wall_constraint_candidate_count = 0;

    std::vector<PlaneObservation> planes;
};

struct MultiPlaneSnapshot
{
    MultiPlaneExtractionResult result;
    std::size_t revision = 0;
};

// Set Wall-only radius used by subsequently default-constructed
// MultiPlaneExtractor instances.
void SetDefaultMultiPlaneWallConstraintMinimumRadius(
    double radius_m);

class MultiPlaneExtractor
{
public:
    MultiPlaneExtractor();

    explicit MultiPlaneExtractor(const MultiPlaneConfig &config);

    void SetConfig(const MultiPlaneConfig &config);

    const MultiPlaneConfig &GetConfig() const;

    MultiPlaneExtractionResult Extract(
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &cloud,
        const Eigen::Vector3d &up_direction_L =
            Eigen::Vector3d::UnitZ()) const;

private:
    MultiPlaneConfig config_;
};


// V1.3 quality refinement + independent Ground V4 cross-validation.
// This marks future measurement candidates only; it does not change pose.
void RefinePlaneConstraintEligibility(
    MultiPlaneExtractionResult &result,
    const GroundSegmentationResult &ground_result,
    const MultiPlaneConfig &config,
    const Eigen::Matrix3d &R_WL,
    const Eigen::Vector3d &t_WL);

// Frontend -> ROS-node visualization bridge. Diagnostics only.
void StoreLatestMultiPlaneResult(
    const MultiPlaneExtractionResult &result);

bool GetLatestMultiPlaneSnapshot(
    MultiPlaneSnapshot &snapshot);

void ResetMultiPlaneSnapshot();

}  // namespace fr_slam
