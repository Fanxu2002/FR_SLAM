#include "fr_slam/frontend/multi_plane_extractor.hpp"
#include "fr_slam/frontend/ground_segmenter.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <mutex>
#include <utility>
#include <vector>

#include <Eigen/Eigenvalues>

namespace
{

constexpr double kPi = 3.14159265358979323846;

struct VoxelKey
{
    int x = 0;
    int y = 0;
    int z = 0;

    bool operator<(const VoxelKey &other) const
    {
        if (x != other.x)
        {
            return x < other.x;
        }
        if (y != other.y)
        {
            return y < other.y;
        }
        return z < other.z;
    }
};

struct VoxelAccumulator
{
    std::size_t point_count = 0;
    Eigen::Vector3d sum = Eigen::Vector3d::Zero();
    Eigen::Matrix3d sum_outer = Eigen::Matrix3d::Zero();
};

std::mutex g_multi_plane_snapshot_mutex;
fr_slam::MultiPlaneSnapshot g_multi_plane_snapshot;

// Configured once by the ROS node before frontend processing starts.
// Default 0.80 m exactly preserves previous behavior.
double g_default_wall_constraint_minimum_radius_m = 0.80;

bool IsFinitePoint(const LIDAR_POINT &point)
{
    return
        std::isfinite(point.x) &&
        std::isfinite(point.y) &&
        std::isfinite(point.z);
}

VoxelKey ComputeVoxelKey(
    const Eigen::Vector3d &point,
    double voxel_size_m)
{
    return VoxelKey{
        static_cast<int>(std::floor(point.x() / voxel_size_m)),
        static_cast<int>(std::floor(point.y() / voxel_size_m)),
        static_cast<int>(std::floor(point.z() / voxel_size_m))};
}

bool RecomputePlaneGeometry(fr_slam::PlaneObservation &plane)
{
    if (plane.point_count < 3 ||
        !plane.center_L.allFinite() ||
        !plane.covariance_L.allFinite())
    {
        return false;
    }

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>
        eigen_solver(plane.covariance_L);

    if (eigen_solver.info() != Eigen::Success)
    {
        return false;
    }

    const Eigen::Vector3d eigenvalues =
        eigen_solver.eigenvalues();
    const Eigen::Matrix3d eigenvectors =
        eigen_solver.eigenvectors();

    if (!eigenvalues.allFinite() ||
        !eigenvectors.allFinite())
    {
        return false;
    }

    Eigen::Vector3d normal =
        eigenvectors.col(0);

    if (!normal.allFinite() ||
        normal.norm() < 1.0e-9)
    {
        return false;
    }

    normal.normalize();

    // Stable sign for diagnostics only. Merge remains sign-invariant.
    if (normal.dot(plane.center_L) > 0.0)
    {
        normal = -normal;
    }

    plane.normal_L = normal;
    plane.minimum_eigenvalue = eigenvalues.x();
    plane.radius_m =
        std::sqrt(std::max(0.0, eigenvalues.z()));
    plane.d =
        -plane.normal_L.dot(plane.center_L);

    return
        std::isfinite(plane.minimum_eigenvalue) &&
        std::isfinite(plane.radius_m) &&
        std::isfinite(plane.d);
}

bool ArePlanesCompatible(
    const fr_slam::PlaneObservation &first,
    const fr_slam::PlaneObservation &second,
    const fr_slam::MultiPlaneConfig &config)
{
    const double normal_difference =
        std::min(
            (first.normal_L - second.normal_L).norm(),
            (first.normal_L + second.normal_L).norm());

    const double first_to_second_distance =
        std::abs(
            first.normal_L.dot(second.center_L) +
            first.d);

    const double second_to_first_distance =
        std::abs(
            second.normal_L.dot(first.center_L) +
            second.d);

    return
        std::isfinite(normal_difference) &&
        std::isfinite(first_to_second_distance) &&
        std::isfinite(second_to_first_distance) &&
        normal_difference <=
            config.plane_merge_normal_threshold &&
        first_to_second_distance <=
            config.plane_merge_distance_threshold_m &&
        second_to_first_distance <=
            config.plane_merge_distance_threshold_m;
}

bool MergePlaneObservation(
    fr_slam::PlaneObservation &destination,
    const fr_slam::PlaneObservation &source)
{
    if (destination.point_count == 0 ||
        source.point_count == 0)
    {
        return false;
    }

    const double n1 =
        static_cast<double>(destination.point_count);
    const double n2 =
        static_cast<double>(source.point_count);
    const double total = n1 + n2;

    const Eigen::Matrix3d second1 =
        (destination.covariance_L +
         destination.center_L *
             destination.center_L.transpose()) *
        n1;

    const Eigen::Matrix3d second2 =
        (source.covariance_L +
         source.center_L *
             source.center_L.transpose()) *
        n2;

    const Eigen::Vector3d merged_center =
        (destination.center_L * n1 +
         source.center_L * n2) /
        total;

    const Eigen::Matrix3d merged_covariance =
        (second1 + second2) /
            total -
        merged_center *
            merged_center.transpose();

    fr_slam::PlaneObservation merged =
        destination;

    merged.center_L =
        merged_center;
    merged.covariance_L =
        merged_covariance;
    merged.point_count +=
        source.point_count;
    merged.merged_patch_count +=
        source.merged_patch_count;

    if (!RecomputePlaneGeometry(merged))
    {
        return false;
    }

    destination = merged;
    return true;
}

double AngleToUpDeg(
    const Eigen::Vector3d &normal_L,
    const Eigen::Vector3d &up_L)
{
    const double c =
        std::clamp(
            std::abs(normal_L.dot(up_L)),
            0.0,
            1.0);

    return std::acos(c) * 180.0 / kPi;
}

void ClassifyPlanes(
    std::vector<fr_slam::PlaneObservation> &planes,
    const Eigen::Vector3d &up_direction_L,
    const fr_slam::MultiPlaneConfig &config)
{
    Eigen::Vector3d up_L =
        up_direction_L;

    if (!up_L.allFinite() ||
        up_L.norm() < 1.0e-9)
    {
        up_L =
            Eigen::Vector3d::UnitZ();
    }

    up_L.normalize();

    for (fr_slam::PlaneObservation &plane : planes)
    {
        plane.angle_to_up_deg =
            AngleToUpDeg(
                plane.normal_L,
                up_L);

        plane.ground_candidate_score = 0.0;

        if (plane.angle_to_up_deg <=
            config.horizontal_max_angle_to_up_deg)
        {
            plane.semantic_type =
                fr_slam::PlaneSemanticType::Horizontal;
        }
        else if (plane.angle_to_up_deg >=
                 config.vertical_min_angle_to_up_deg)
        {
            plane.semantic_type =
                fr_slam::PlaneSemanticType::Vertical;
        }
        else
        {
            plane.semantic_type =
                fr_slam::PlaneSemanticType::Oblique;
        }
    }

    // At most one ground candidate per frame.
    std::size_t best_ground_index = planes.size();
    double best_ground_score =
        -std::numeric_limits<double>::infinity();

    for (std::size_t i = 0;
         i < planes.size();
         ++i)
    {
        fr_slam::PlaneObservation &plane =
            planes[i];

        if (plane.semantic_type !=
            fr_slam::PlaneSemanticType::Horizontal)
        {
            continue;
        }

        const double center_up_m =
            plane.center_L.dot(up_L);

        const double distance_m =
            std::abs(plane.d);

        if (!std::isfinite(center_up_m) ||
            !std::isfinite(distance_m) ||
            center_up_m >
                config.ground_candidate_max_center_up_m ||
            distance_m >
                config.ground_candidate_max_distance_m ||
            plane.point_count <
                config.ground_candidate_minimum_points ||
            plane.radius_m <
                config.ground_candidate_minimum_radius_m)
        {
            continue;
        }

        const double support =
            static_cast<double>(plane.point_count);

        const double patch_bonus =
            std::sqrt(
                static_cast<double>(
                    std::max<std::size_t>(
                        1,
                        plane.merged_patch_count)));

        const double radius_bonus =
            std::max(0.10, plane.radius_m);

        const double planarity_penalty =
            1.0 +
            1000.0 *
                std::max(
                    0.0,
                    plane.minimum_eigenvalue);

        const double score =
            support *
            patch_bonus *
            radius_bonus /
            planarity_penalty;

        plane.ground_candidate_score =
            score;

        if (score > best_ground_score)
        {
            best_ground_score = score;
            best_ground_index = i;
        }
    }

    if (best_ground_index < planes.size())
    {
        planes[best_ground_index].semantic_type =
            fr_slam::PlaneSemanticType::GroundCandidate;
    }
}

void UpdateSemanticCounts(
    fr_slam::MultiPlaneExtractionResult &result)
{
    result.ground_candidate_count = 0;
    result.horizontal_plane_count = 0;
    result.vertical_plane_count = 0;
    result.oblique_plane_count = 0;

    for (const fr_slam::PlaneObservation &plane :
         result.planes)
    {
        switch (plane.semantic_type)
        {
        case fr_slam::PlaneSemanticType::GroundCandidate:
            ++result.ground_candidate_count;
            break;

        case fr_slam::PlaneSemanticType::Horizontal:
            ++result.horizontal_plane_count;
            break;

        case fr_slam::PlaneSemanticType::Vertical:
            ++result.vertical_plane_count;
            break;

        case fr_slam::PlaneSemanticType::Oblique:
        default:
            ++result.oblique_plane_count;
            break;
        }
    }
}


double SignInvariantNormalAngleDeg(
    const Eigen::Vector3d &first,
    const Eigen::Vector3d &second)
{
    if (!first.allFinite() ||
        !second.allFinite() ||
        first.norm() < 1.0e-9 ||
        second.norm() < 1.0e-9)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const Eigen::Vector3d first_unit =
        first.normalized();

    const Eigen::Vector3d second_unit =
        second.normalized();

    const double cosine =
        std::clamp(
            std::abs(
                first_unit.dot(second_unit)),
            0.0,
            1.0);

    return std::acos(cosine) * 180.0 / kPi;
}


double ComputeConstraintQualityScore(
    const fr_slam::PlaneObservation &plane,
    const fr_slam::MultiPlaneConfig &config)
{
    const double points_score =
        std::clamp(
            static_cast<double>(plane.point_count) /
                static_cast<double>(
                    std::max<std::size_t>(
                        1,
                        2 * config.constraint_minimum_points)),
            0.0,
            1.0);

    const double patch_score =
        std::clamp(
            static_cast<double>(plane.merged_patch_count) /
                static_cast<double>(
                    std::max<std::size_t>(
                        1,
                        2 * config.constraint_minimum_merged_patches)),
            0.0,
            1.0);

    const double radius_score =
        std::clamp(
            plane.radius_m /
                std::max(
                    1.0e-6,
                    2.0 * config.constraint_minimum_radius_m),
            0.0,
            1.0);

    const double planarity_score =
        std::clamp(
            1.0 -
                plane.minimum_eigenvalue /
                    std::max(
                        1.0e-9,
                        config.constraint_maximum_min_eigenvalue),
            0.0,
            1.0);

    return
        0.30 * points_score +
        0.25 * patch_score +
        0.20 * radius_score +
        0.25 * planarity_score;
}


struct SupportPlaneWorldGeometry
{
    bool valid = false;

    Eigen::Vector3d normal_W =
        Eigen::Vector3d::UnitZ();

    double d_W = 0.0;
};


struct ActiveSupportTrackerState
{
    bool active_valid = false;

    Eigen::Vector3d active_normal_W =
        Eigen::Vector3d::UnitZ();

    double active_d_W = 0.0;

    fr_slam::SupportSurfaceMode active_mode =
        fr_slam::SupportSurfaceMode::None;

    std::size_t missed_frames = 0;

    bool pending_valid = false;

    Eigen::Vector3d pending_normal_W =
        Eigen::Vector3d::UnitZ();

    double pending_d_W = 0.0;

    fr_slam::SupportSurfaceMode pending_mode =
        fr_slam::SupportSurfaceMode::None;

    std::size_t pending_confirmation_count = 0;
};


ActiveSupportTrackerState g_active_support_tracker;


SupportPlaneWorldGeometry
ComputeSupportPlaneWorldGeometry(
    const fr_slam::PlaneObservation &plane,
    const Eigen::Matrix3d &R_WL,
    const Eigen::Vector3d &t_WL)
{
    SupportPlaneWorldGeometry geometry;

    if (!plane.normal_L.allFinite() ||
        !R_WL.allFinite() ||
        !t_WL.allFinite() ||
        !std::isfinite(plane.d))
    {
        return geometry;
    }

    Eigen::Vector3d normal_W =
        R_WL *
        plane.normal_L;

    const double normal_norm =
        normal_W.norm();

    if (!(normal_norm > 1.0e-9) ||
        !std::isfinite(normal_norm))
    {
        return geometry;
    }

    normal_W /=
        normal_norm;

    double d_W =
        plane.d /
            normal_norm -
        normal_W.dot(
            t_WL);

    if (normal_W.dot(
            Eigen::Vector3d::UnitZ()) <
        0.0)
    {
        normal_W =
            -normal_W;

        d_W =
            -d_W;
    }

    geometry.valid = true;
    geometry.normal_W = normal_W;
    geometry.d_W = d_W;

    return geometry;
}


bool SupportWorldGeometryCompatible(
    const SupportPlaneWorldGeometry &first,
    const SupportPlaneWorldGeometry &second,
    double maximum_normal_difference_deg,
    double maximum_plane_distance_difference_m,
    double &normal_difference_deg,
    double &plane_distance_difference_m)
{
    normal_difference_deg =
        std::numeric_limits<double>::quiet_NaN();

    plane_distance_difference_m =
        std::numeric_limits<double>::quiet_NaN();

    if (!first.valid ||
        !second.valid)
    {
        return false;
    }

    normal_difference_deg =
        SignInvariantNormalAngleDeg(
            first.normal_W,
            second.normal_W);

    plane_distance_difference_m =
        std::abs(
            first.d_W -
            second.d_W);

    return
        std::isfinite(
            normal_difference_deg) &&
        std::isfinite(
            plane_distance_difference_m) &&
        normal_difference_deg <=
            maximum_normal_difference_deg &&
        plane_distance_difference_m <=
            maximum_plane_distance_difference_m;
}


bool ComputeSupportLateralCorridorMetrics(
    const fr_slam::PlaneObservation &plane,
    const Eigen::Vector3d &up_direction_L,
    const fr_slam::MultiPlaneConfig &config,
    double &lateral_center_m,
    double &lateral_sigma_m,
    double &corridor_gap_m)
{
    lateral_center_m =
        std::numeric_limits<double>::quiet_NaN();

    lateral_sigma_m =
        std::numeric_limits<double>::quiet_NaN();

    corridor_gap_m =
        std::numeric_limits<double>::quiet_NaN();

    if (!plane.center_L.allFinite() ||
        !plane.covariance_L.allFinite() ||
        !up_direction_L.allFinite() ||
        up_direction_L.norm() < 1.0e-9)
    {
        return false;
    }

    const Eigen::Vector3d up_L =
        up_direction_L.normalized();

    Eigen::Vector3d forward_L =
        Eigen::Vector3d::UnitX() -
        up_L *
            up_L.dot(
                Eigen::Vector3d::UnitX());

    if (forward_L.norm() < 1.0e-6)
    {
        forward_L =
            Eigen::Vector3d::UnitY() -
            up_L *
                up_L.dot(
                    Eigen::Vector3d::UnitY());
    }

    if (forward_L.norm() < 1.0e-6)
    {
        return false;
    }

    forward_L.normalize();

    Eigen::Vector3d lateral_L =
        up_L.cross(
            forward_L);

    if (lateral_L.norm() < 1.0e-6)
    {
        return false;
    }

    lateral_L.normalize();

    lateral_center_m =
        plane.center_L.dot(
            lateral_L);

    const double lateral_variance =
        lateral_L.dot(
            plane.covariance_L *
            lateral_L);

    if (!std::isfinite(lateral_variance))
    {
        return false;
    }

    lateral_sigma_m =
        std::sqrt(
            std::max(
                0.0,
                lateral_variance));

    const double observed_half_span_m =
        config.support_slope_lateral_sigma_multiplier *
        lateral_sigma_m;

    corridor_gap_m =
        std::max(
            0.0,
            std::abs(
                lateral_center_m) -
                observed_half_span_m -
                config.support_slope_corridor_half_width_m);

    return
        std::isfinite(lateral_center_m) &&
        std::isfinite(lateral_sigma_m) &&
        std::isfinite(corridor_gap_m);
}


bool EvaluateGroundV4Confirmation(
    fr_slam::PlaneObservation &plane,
    const fr_slam::GroundSegmentationResult &ground_result,
    const fr_slam::MultiPlaneConfig &config)
{
    plane.ground_v4_normal_difference_deg =
        std::numeric_limits<double>::quiet_NaN();

    plane.ground_v4_plane_distance_difference_m =
        std::numeric_limits<double>::quiet_NaN();

    if (!ground_result.support_plane_valid ||
        !ground_result.support_constraint_valid ||
        !ground_result.support_ground_normal_L.allFinite() ||
        !std::isfinite(
            ground_result.support_ground_plane_d))
    {
        return false;
    }

    Eigen::Vector3d v4_normal =
        ground_result.support_ground_normal_L;

    const double v4_normal_norm =
        v4_normal.norm();

    if (!(v4_normal_norm > 1.0e-9) ||
        !std::isfinite(v4_normal_norm))
    {
        return false;
    }

    v4_normal /=
        v4_normal_norm;

    const double v4_d =
        ground_result.support_ground_plane_d /
        v4_normal_norm;

    plane.ground_v4_normal_difference_deg =
        SignInvariantNormalAngleDeg(
            plane.normal_L,
            v4_normal);

    const double sign =
        plane.normal_L.dot(v4_normal) >= 0.0 ?
            1.0 :
            -1.0;

    const double aligned_multi_plane_d =
        sign *
        plane.d;

    plane.ground_v4_plane_distance_difference_m =
        std::abs(
            aligned_multi_plane_d -
            v4_d);

    return
        std::isfinite(
            plane.ground_v4_normal_difference_deg) &&
        std::isfinite(
            plane.ground_v4_plane_distance_difference_m) &&
        plane.ground_v4_normal_difference_deg <=
            config
                .refined_ground_maximum_v4_normal_difference_deg &&
        plane.ground_v4_plane_distance_difference_m <=
            config
                .refined_ground_maximum_v4_plane_distance_difference_m;
}


struct LocalGroundFootprintMetrics
{
    bool valid = false;
    double longitudinal_center_m = 0.0;
    double lateral_center_m = 0.0;
    double longitudinal_sigma_m = 0.0;
    double lateral_sigma_m = 0.0;
    double longitudinal_roi_gap_m = 0.0;
    double lateral_roi_gap_m = 0.0;
    double origin_footprint_gap_m = 0.0;
};


LocalGroundFootprintMetrics
ComputeLocalGroundFootprintMetrics(
    const fr_slam::PlaneObservation &plane,
    const Eigen::Vector3d &up_direction_L,
    const fr_slam::MultiPlaneConfig &config)
{
    LocalGroundFootprintMetrics metrics;

    if (!plane.center_L.allFinite() ||
        !plane.covariance_L.allFinite() ||
        !up_direction_L.allFinite() ||
        up_direction_L.norm() < 1.0e-9)
    {
        return metrics;
    }

    const Eigen::Vector3d up_L =
        up_direction_L.normalized();

    Eigen::Vector3d longitudinal_L =
        Eigen::Vector3d::UnitX() -
        up_L * up_L.dot(Eigen::Vector3d::UnitX());

    if (longitudinal_L.norm() < 1.0e-6)
    {
        longitudinal_L =
            Eigen::Vector3d::UnitY() -
            up_L * up_L.dot(Eigen::Vector3d::UnitY());
    }

    if (longitudinal_L.norm() < 1.0e-6)
    {
        return metrics;
    }

    longitudinal_L.normalize();

    Eigen::Vector3d lateral_L =
        up_L.cross(longitudinal_L);

    if (lateral_L.norm() < 1.0e-6)
    {
        return metrics;
    }

    lateral_L.normalize();

    metrics.longitudinal_center_m =
        plane.center_L.dot(longitudinal_L);
    metrics.lateral_center_m =
        plane.center_L.dot(lateral_L);

    const double longitudinal_variance =
        longitudinal_L.dot(plane.covariance_L * longitudinal_L);
    const double lateral_variance =
        lateral_L.dot(plane.covariance_L * lateral_L);

    if (!std::isfinite(longitudinal_variance) ||
        !std::isfinite(lateral_variance))
    {
        return metrics;
    }

    metrics.longitudinal_sigma_m =
        std::sqrt(std::max(0.0, longitudinal_variance));
    metrics.lateral_sigma_m =
        std::sqrt(std::max(0.0, lateral_variance));

    const double longitudinal_half_span_m =
        config.local_ground_footprint_sigma_multiplier *
        metrics.longitudinal_sigma_m;
    const double lateral_half_span_m =
        config.local_ground_footprint_sigma_multiplier *
        metrics.lateral_sigma_m;

    metrics.longitudinal_roi_gap_m =
        std::max(
            0.0,
            std::abs(metrics.longitudinal_center_m) -
                longitudinal_half_span_m -
                config.local_ground_roi_longitudinal_half_length_m);

    metrics.lateral_roi_gap_m =
        std::max(
            0.0,
            std::abs(metrics.lateral_center_m) -
                lateral_half_span_m -
                config.local_ground_roi_lateral_half_width_m);

    const double longitudinal_origin_gap_m =
        std::max(
            0.0,
            std::abs(metrics.longitudinal_center_m) -
                longitudinal_half_span_m);

    const double lateral_origin_gap_m =
        std::max(
            0.0,
            std::abs(metrics.lateral_center_m) -
                lateral_half_span_m);

    metrics.origin_footprint_gap_m =
        std::hypot(
            longitudinal_origin_gap_m,
            lateral_origin_gap_m);

    metrics.valid =
        std::isfinite(metrics.longitudinal_center_m) &&
        std::isfinite(metrics.lateral_center_m) &&
        std::isfinite(metrics.longitudinal_sigma_m) &&
        std::isfinite(metrics.lateral_sigma_m) &&
        std::isfinite(metrics.longitudinal_roi_gap_m) &&
        std::isfinite(metrics.lateral_roi_gap_m) &&
        std::isfinite(metrics.origin_footprint_gap_m);

    return metrics;
}

}  // namespace

namespace fr_slam
{

const char *PlaneSemanticTypeName(
    PlaneSemanticType type)
{
    switch (type)
    {
    case PlaneSemanticType::GroundCandidate:
        return "GROUND_CANDIDATE";
    case PlaneSemanticType::Horizontal:
        return "HORIZONTAL";
    case PlaneSemanticType::Vertical:
        return "VERTICAL";
    case PlaneSemanticType::Oblique:
    default:
        return "OBLIQUE";
    }
}

const char *SupportSurfaceModeName(
    SupportSurfaceMode mode)
{
    switch (mode)
    {
    case SupportSurfaceMode::Flat:
        return "FLAT_SUPPORT";

    case SupportSurfaceMode::Slope:
        return "SLOPE_SUPPORT";

    case SupportSurfaceMode::None:
    default:
        return "NONE";
    }
}


void SetDefaultMultiPlaneWallConstraintMinimumRadius(
    double radius_m)
{
    if (!std::isfinite(radius_m) ||
        radius_m <= 0.0)
    {
        return;
    }

    g_default_wall_constraint_minimum_radius_m =
        radius_m;
}

MultiPlaneExtractor::MultiPlaneExtractor()
    : config_()
{
    config_.wall_constraint_minimum_radius_m =
        g_default_wall_constraint_minimum_radius_m;
}

MultiPlaneExtractor::MultiPlaneExtractor(
    const MultiPlaneConfig &config)
    : config_(config)
{
}

void MultiPlaneExtractor::SetConfig(
    const MultiPlaneConfig &config)
{
    config_ = config;
}

const MultiPlaneConfig &
MultiPlaneExtractor::GetConfig() const
{
    return config_;
}

MultiPlaneExtractionResult
MultiPlaneExtractor::Extract(
    const pcl::PointCloud<LIDAR_POINT>::ConstPtr &cloud,
    const Eigen::Vector3d &up_direction_L) const
{
    MultiPlaneExtractionResult result;

    if (!cloud ||
        cloud->empty() ||
        !(config_.voxel_size_m > 0.0))
    {
        return result;
    }

    result.input_points =
        cloud->size();

    std::map<VoxelKey, VoxelAccumulator>
        voxel_map;

    for (const LIDAR_POINT &point : cloud->points)
    {
        if (!IsFinitePoint(point))
        {
            continue;
        }

        const Eigen::Vector3d point_L(
            static_cast<double>(point.x),
            static_cast<double>(point.y),
            static_cast<double>(point.z));

        const VoxelKey key =
            ComputeVoxelKey(
                point_L,
                config_.voxel_size_m);

        VoxelAccumulator &voxel =
            voxel_map[key];

        ++voxel.point_count;
        voxel.sum += point_L;
        voxel.sum_outer +=
            point_L *
            point_L.transpose();
    }

    result.occupied_voxels =
        voxel_map.size();

    std::vector<PlaneObservation>
        raw_planes;

    raw_planes.reserve(voxel_map.size());

    for (const auto &entry : voxel_map)
    {
        const VoxelAccumulator &voxel =
            entry.second;

        if (voxel.point_count <
            std::max<std::size_t>(
                3,
                config_.minimum_points_per_voxel))
        {
            continue;
        }

        const double count =
            static_cast<double>(voxel.point_count);

        PlaneObservation plane;
        plane.point_count = voxel.point_count;
        plane.merged_patch_count = 1;
        plane.center_L =
            voxel.sum / count;
        plane.covariance_L =
            voxel.sum_outer / count -
            plane.center_L *
                plane.center_L.transpose();

        if (!RecomputePlaneGeometry(plane))
        {
            continue;
        }

        if (plane.minimum_eigenvalue >
            config_.plane_detection_eigenvalue_threshold)
        {
            continue;
        }

        raw_planes.push_back(plane);
    }

    result.raw_plane_patches =
        raw_planes.size();

    std::sort(
        raw_planes.begin(),
        raw_planes.end(),
        [](
            const PlaneObservation &lhs,
            const PlaneObservation &rhs)
        {
            return lhs.point_count >
                   rhs.point_count;
        });

    std::vector<PlaneObservation>
        merged_planes;

    for (const PlaneObservation &raw_plane :
         raw_planes)
    {
        std::size_t best_index =
            merged_planes.size();

        double best_normal_difference =
            std::numeric_limits<double>::infinity();

        for (std::size_t i = 0;
             i < merged_planes.size();
             ++i)
        {
            const PlaneObservation &candidate =
                merged_planes[i];

            if (!ArePlanesCompatible(
                    candidate,
                    raw_plane,
                    config_))
            {
                continue;
            }

            const double normal_difference =
                std::min(
                    (candidate.normal_L -
                     raw_plane.normal_L).norm(),
                    (candidate.normal_L +
                     raw_plane.normal_L).norm());

            if (normal_difference <
                best_normal_difference)
            {
                best_normal_difference =
                    normal_difference;
                best_index = i;
            }
        }

        if (best_index ==
            merged_planes.size())
        {
            merged_planes.push_back(raw_plane);
            continue;
        }

        MergePlaneObservation(
            merged_planes[best_index],
            raw_plane);
    }

    result.merged_plane_count_before_filter =
        merged_planes.size();

    merged_planes.erase(
        std::remove_if(
            merged_planes.begin(),
            merged_planes.end(),
            [&](const PlaneObservation &plane)
            {
                return
                    plane.point_count <
                    config_.minimum_points_per_output_plane;
            }),
        merged_planes.end());

    ClassifyPlanes(
        merged_planes,
        up_direction_L,
        config_);

    // Keep the ground candidate even if the top-K support cap would otherwise
    // remove it.
    std::sort(
        merged_planes.begin(),
        merged_planes.end(),
        [](
            const PlaneObservation &lhs,
            const PlaneObservation &rhs)
        {
            const bool lhs_ground =
                lhs.semantic_type ==
                PlaneSemanticType::GroundCandidate;

            const bool rhs_ground =
                rhs.semantic_type ==
                PlaneSemanticType::GroundCandidate;

            if (lhs_ground != rhs_ground)
            {
                return lhs_ground;
            }

            return lhs.point_count >
                   rhs.point_count;
        });

    if (merged_planes.size() >
        config_.maximum_output_planes)
    {
        merged_planes.resize(
            config_.maximum_output_planes);
    }

    result.planes =
        std::move(merged_planes);

    UpdateSemanticCounts(result);

    return result;
}

void RefinePlaneConstraintEligibility(
    MultiPlaneExtractionResult &result,
    const GroundSegmentationResult &ground_result,
    const MultiPlaneConfig &config,
    const Eigen::Matrix3d &R_WL,
    const Eigen::Vector3d &t_WL)
{
    (void)t_WL;

    struct SlopeTransitionTracker
    {
        bool pending_valid = false;
        Eigen::Vector3d pending_normal_L =
            Eigen::Vector3d::UnitZ();
        Eigen::Vector3d pending_center_L =
            Eigen::Vector3d::Zero();
        double pending_plane_distance_m =
            std::numeric_limits<double>::quiet_NaN();
        std::size_t pending_count = 0;

        double recent_active_ground_plane_distance_m =
            std::numeric_limits<double>::quiet_NaN();
    };

    static SlopeTransitionTracker slope_tracker;

    result.eligible_ground_plane_count = 0;
    result.eligible_wall_plane_count = 0;

    result.support_candidate_count = 0;
    result.active_support_plane_count = 0;
    result.active_support_index = -1;
    result.active_support_mode = SupportSurfaceMode::None;
    result.support_pending_confirmation_count = 0;

    result.local_ground_candidate_count = 0;
    result.active_ground_plane_count = 0;
    result.active_ground_index = -1;
    result.wall_constraint_candidate_count = 0;

    const Eigen::Vector3d up_direction_L =
        R_WL.transpose() * Eigen::Vector3d::UnitZ();

    const std::uint32_t base_quality_mask =
        PLANE_CONSTRAINT_REJECT_LOW_POINTS |
        PLANE_CONSTRAINT_REJECT_LOW_PATCHES |
        PLANE_CONSTRAINT_REJECT_SMALL_RADIUS |
        PLANE_CONSTRAINT_REJECT_WEAK_PLANARITY;

    int best_flat_index = -1;
    double best_flat_score =
        -std::numeric_limits<double>::infinity();

    int best_slope_index = -1;
    double best_slope_score =
        -std::numeric_limits<double>::infinity();

    for (std::size_t plane_index = 0;
         plane_index < result.planes.size();
         ++plane_index)
    {
        PlaneObservation &plane =
            result.planes[plane_index];

        plane.constraint_eligible = false;
        plane.constraint_rejection_mask =
            PLANE_CONSTRAINT_REJECT_NONE;

        plane.constraint_quality_score =
            ComputeConstraintQualityScore(
                plane,
                config);

        plane.support_candidate = false;
        plane.active_support = false;
        plane.support_v4_confirmed = false;
        plane.support_mode =
            SupportSurfaceMode::None;
        plane.support_selection_score = 0.0;

        plane.support_temporal_normal_difference_deg =
            std::numeric_limits<double>::quiet_NaN();

        plane.support_temporal_plane_distance_difference_m =
            std::numeric_limits<double>::quiet_NaN();

        plane.ground_v4_normal_difference_deg =
            std::numeric_limits<double>::quiet_NaN();

        plane.ground_v4_plane_distance_difference_m =
            std::numeric_limits<double>::quiet_NaN();

        plane.local_ground_candidate = false;
        plane.active_ground = false;
        plane.wall_constraint_candidate = false;

        plane.local_ground_longitudinal_center_m =
            std::numeric_limits<double>::quiet_NaN();

        plane.local_ground_lateral_center_m =
            std::numeric_limits<double>::quiet_NaN();

        plane.local_ground_longitudinal_sigma_m =
            std::numeric_limits<double>::quiet_NaN();

        plane.local_ground_lateral_sigma_m =
            std::numeric_limits<double>::quiet_NaN();

        plane.local_ground_longitudinal_roi_gap_m =
            std::numeric_limits<double>::quiet_NaN();

        plane.local_ground_lateral_roi_gap_m =
            std::numeric_limits<double>::quiet_NaN();

        plane.local_ground_origin_footprint_gap_m =
            std::numeric_limits<double>::quiet_NaN();

        plane.local_ground_selection_score = 0.0;

        if (plane.point_count <
            config.constraint_minimum_points)
        {
            plane.constraint_rejection_mask |=
                PLANE_CONSTRAINT_REJECT_LOW_POINTS;
        }

        if (plane.merged_patch_count <
            config.constraint_minimum_merged_patches)
        {
            plane.constraint_rejection_mask |=
                PLANE_CONSTRAINT_REJECT_LOW_PATCHES;
        }

        // Ground / Horizontal / Oblique keep the original common
        // 0.80 m radius gate. Only semantic Vertical planes use
        // the Wall-specific radius.
        //
        // ComputeConstraintQualityScore() intentionally still uses
        // constraint_minimum_radius_m, so lowering the indoor Wall gate
        // does not artificially increase quality scores.
        const double minimum_constraint_radius_m =
            plane.semantic_type ==
                    PlaneSemanticType::Vertical
                ? config.wall_constraint_minimum_radius_m
                : config.constraint_minimum_radius_m;

        if (plane.radius_m <
            minimum_constraint_radius_m)
        {
            plane.constraint_rejection_mask |=
                PLANE_CONSTRAINT_REJECT_SMALL_RADIUS;
        }

        if (!std::isfinite(
                plane.minimum_eigenvalue) ||
            plane.minimum_eigenvalue >
                config.constraint_maximum_min_eigenvalue)
        {
            plane.constraint_rejection_mask |=
                PLANE_CONSTRAINT_REJECT_WEAK_PLANARITY;
        }

        const bool base_quality_ok =
            (plane.constraint_rejection_mask &
             base_quality_mask) == 0U;

        if (base_quality_ok &&
            std::isfinite(
                plane.angle_to_up_deg))
        {
            if (plane.angle_to_up_deg <=
                config.local_ground_maximum_angle_to_up_deg)
            {
                const double center_up_m =
                    plane.center_L.dot(
                        up_direction_L);

                if (!std::isfinite(
                        center_up_m) ||
                    center_up_m >
                        config.local_ground_maximum_center_up_m ||
                    center_up_m <
                        config.local_ground_minimum_center_up_m)
                {
                    plane.constraint_rejection_mask |=
                        PLANE_CONSTRAINT_REJECT_LOCAL_GROUND_OUTSIDE_ROI;
                }

                const LocalGroundFootprintMetrics metrics =
                    ComputeLocalGroundFootprintMetrics(
                        plane,
                        up_direction_L,
                        config);

                if (metrics.valid)
                {
                    plane.local_ground_longitudinal_center_m =
                        metrics.longitudinal_center_m;

                    plane.local_ground_lateral_center_m =
                        metrics.lateral_center_m;

                    plane.local_ground_longitudinal_sigma_m =
                        metrics.longitudinal_sigma_m;

                    plane.local_ground_lateral_sigma_m =
                        metrics.lateral_sigma_m;

                    plane.local_ground_longitudinal_roi_gap_m =
                        metrics.longitudinal_roi_gap_m;

                    plane.local_ground_lateral_roi_gap_m =
                        metrics.lateral_roi_gap_m;

                    plane.local_ground_origin_footprint_gap_m =
                        metrics.origin_footprint_gap_m;
                }

                double maximum_origin_gap_m =
                    config
                        .local_ground_maximum_origin_footprint_gap_m;

                if (plane.semantic_type ==
                    PlaneSemanticType::GroundCandidate)
                {
                    maximum_origin_gap_m =
                        config
                            .local_ground_ground_candidate_maximum_origin_gap_m;
                }

                if (!metrics.valid ||
                    metrics.longitudinal_roi_gap_m >
                        0.0 ||
                    metrics.lateral_roi_gap_m >
                        0.0 ||
                    metrics.origin_footprint_gap_m >
                        maximum_origin_gap_m)
                {
                    plane.constraint_rejection_mask |=
                        PLANE_CONSTRAINT_REJECT_LOCAL_GROUND_OUTSIDE_ROI;
                }

                plane.support_mode =
                    plane.angle_to_up_deg <=
                            config
                                .support_flat_maximum_angle_to_up_deg ?
                        SupportSurfaceMode::Flat :
                        SupportSurfaceMode::Slope;

                if (plane.support_mode ==
                    SupportSurfaceMode::Slope)
                {
                    const double plane_distance_m =
                        std::abs(
                            plane.d);

                    if (std::isfinite(
                            slope_tracker
                                .recent_active_ground_plane_distance_m) &&
                        std::isfinite(
                            plane_distance_m) &&
                        std::abs(
                            plane_distance_m -
                            slope_tracker
                                .recent_active_ground_plane_distance_m) >
                            config
                                .local_ground_slope_maximum_recent_support_distance_difference_m)
                    {
                        plane.constraint_rejection_mask |=
                            PLANE_CONSTRAINT_REJECT_LOCAL_GROUND_SLOPE_DISTANCE_JUMP;
                    }
                }

                const std::uint32_t ground_blocking_mask =
                    base_quality_mask |
                    PLANE_CONSTRAINT_REJECT_LOCAL_GROUND_OUTSIDE_ROI |
                    PLANE_CONSTRAINT_REJECT_LOCAL_GROUND_SLOPE_DISTANCE_JUMP;

                if ((plane.constraint_rejection_mask &
                     ground_blocking_mask) == 0U)
                {
                    plane.local_ground_candidate =
                        true;

                    plane.support_candidate =
                        true;

                    if (plane.support_mode ==
                        SupportSurfaceMode::Flat)
                    {
                        plane.support_v4_confirmed =
                            EvaluateGroundV4Confirmation(
                                plane,
                                ground_result,
                                config);
                    }

                    ++result.local_ground_candidate_count;
                    ++result.support_candidate_count;

                    const double longitudinal_span_score =
                        std::min(
                            1.0,
                            metrics.longitudinal_sigma_m /
                                1.00);

                    const double lateral_span_score =
                        std::min(
                            1.0,
                            metrics.lateral_sigma_m /
                                0.80);

                    const double local_overlap_score =
                        1.0 /
                        (1.0 +
                         metrics.origin_footprint_gap_m);

                    double selection_score =
                        plane.constraint_quality_score +
                        0.15 *
                            longitudinal_span_score +
                        0.20 *
                            lateral_span_score +
                        0.25 *
                            local_overlap_score;

                    if (plane.support_v4_confirmed)
                    {
                        selection_score +=
                            0.20;
                    }

                    plane.local_ground_selection_score =
                        selection_score;

                    plane.support_selection_score =
                        selection_score;

                    if (plane.support_mode ==
                        SupportSurfaceMode::Flat)
                    {
                        if (selection_score >
                            best_flat_score)
                        {
                            best_flat_score =
                                selection_score;

                            best_flat_index =
                                static_cast<int>(
                                    plane_index);
                        }
                    }
                    else
                    {
                        if (selection_score >
                            best_slope_score)
                        {
                            best_slope_score =
                                selection_score;

                            best_slope_index =
                                static_cast<int>(
                                    plane_index);
                        }
                    }
                }
            }
            else if (plane.angle_to_up_deg <
                     config
                         .refined_wall_minimum_angle_to_up_deg)
            {
                plane.constraint_rejection_mask |=
                    PLANE_CONSTRAINT_REJECT_LOCAL_GROUND_TOO_STEEP;
            }
        }

        // Wall branch unchanged from V1.5.
        if (plane.semantic_type ==
            PlaneSemanticType::Vertical)
        {
            if (plane.angle_to_up_deg <
                config.refined_wall_minimum_angle_to_up_deg)
            {
                plane.constraint_rejection_mask |=
                    PLANE_CONSTRAINT_REJECT_WALL_ANGLE;
            }

            if ((plane.constraint_rejection_mask &
                 (base_quality_mask |
                  PLANE_CONSTRAINT_REJECT_WALL_ANGLE)) ==
                0U)
            {
                plane.wall_constraint_candidate =
                    true;

                plane.constraint_eligible =
                    true;

                ++result.eligible_wall_plane_count;
                ++result.wall_constraint_candidate_count;
            }
        }
    }

    int selected_ground_index = -1;

    // Flat Ground has categorical priority.  Slope is only a fallback when
    // no valid flat Ground exists.
    if (best_flat_index >= 0)
    {
        selected_ground_index =
            best_flat_index;

        slope_tracker.pending_valid =
            false;

        slope_tracker.pending_count =
            0;

        result.support_pending_confirmation_count =
            0;
    }
    else if (best_slope_index >= 0)
    {
        PlaneObservation &slope =
            result.planes[
                static_cast<std::size_t>(
                    best_slope_index)];

        const double current_plane_distance_m =
            std::abs(
                slope.d);

        bool pending_consistent =
            false;

        double temporal_normal_difference_deg =
            std::numeric_limits<double>::quiet_NaN();

        double temporal_plane_distance_difference_m =
            std::numeric_limits<double>::quiet_NaN();

        double temporal_center_distance_m =
            std::numeric_limits<double>::quiet_NaN();

        if (slope_tracker.pending_valid)
        {
            temporal_normal_difference_deg =
                SignInvariantNormalAngleDeg(
                    slope_tracker.pending_normal_L,
                    slope.normal_L);

            temporal_plane_distance_difference_m =
                std::abs(
                    slope_tracker
                        .pending_plane_distance_m -
                    current_plane_distance_m);

            temporal_center_distance_m =
                (
                    slope.center_L -
                    slope_tracker.pending_center_L
                ).norm();

            pending_consistent =
                std::isfinite(
                    temporal_normal_difference_deg) &&
                std::isfinite(
                    temporal_plane_distance_difference_m) &&
                std::isfinite(
                    temporal_center_distance_m) &&
                temporal_normal_difference_deg <=
                    config
                        .local_ground_slope_temporal_maximum_normal_difference_deg &&
                temporal_plane_distance_difference_m <=
                    config
                        .local_ground_slope_temporal_maximum_plane_distance_difference_m &&
                temporal_center_distance_m <=
                    config
                        .local_ground_slope_temporal_maximum_center_distance_m;
        }

        slope.support_temporal_normal_difference_deg =
            temporal_normal_difference_deg;

        slope.support_temporal_plane_distance_difference_m =
            temporal_plane_distance_difference_m;

        if (pending_consistent)
        {
            ++slope_tracker.pending_count;
        }
        else
        {
            slope_tracker.pending_count =
                1;
        }

        slope_tracker.pending_valid =
            true;

        slope_tracker.pending_normal_L =
            slope.normal_L;

        slope_tracker.pending_center_L =
            slope.center_L;

        slope_tracker.pending_plane_distance_m =
            current_plane_distance_m;

        result.support_pending_confirmation_count =
            slope_tracker.pending_count;

        if (slope_tracker.pending_count >=
            config.local_ground_slope_confirmation_frames)
        {
            selected_ground_index =
                best_slope_index;
        }
    }
    else
    {
        slope_tracker.pending_valid =
            false;

        slope_tracker.pending_count =
            0;

        result.support_pending_confirmation_count =
            0;
    }

    if (selected_ground_index >= 0)
    {
        PlaneObservation &active_ground =
            result.planes[
                static_cast<std::size_t>(
                    selected_ground_index)];

        active_ground.active_ground =
            true;

        active_ground.active_support =
            true;

        active_ground.constraint_eligible =
            true;

        result.active_ground_plane_count =
            1;

        result.active_ground_index =
            selected_ground_index;

        result.active_support_plane_count =
            1;

        result.active_support_index =
            selected_ground_index;

        result.active_support_mode =
            active_ground.support_mode;

        result.eligible_ground_plane_count =
            1;

        const double active_plane_distance_m =
            std::abs(
                active_ground.d);

        if (std::isfinite(
                active_plane_distance_m))
        {
            slope_tracker
                .recent_active_ground_plane_distance_m =
                active_plane_distance_m;
        }
    }

    for (PlaneObservation &plane :
         result.planes)
    {
        if (plane.local_ground_candidate &&
            !plane.active_ground)
        {
            plane.constraint_rejection_mask |=
                PLANE_CONSTRAINT_REJECT_NOT_ACTIVE_SUPPORT;
        }
        else if (!plane.local_ground_candidate &&
                 !plane.wall_constraint_candidate)
        {
            plane.constraint_rejection_mask |=
                PLANE_CONSTRAINT_REJECT_UNSUPPORTED_SEMANTIC;
        }
    }
}





void StoreLatestMultiPlaneResult(
    const MultiPlaneExtractionResult &result)
{
    std::lock_guard<std::mutex> lock(
        g_multi_plane_snapshot_mutex);

    g_multi_plane_snapshot.result = result;
    ++g_multi_plane_snapshot.revision;
}

bool GetLatestMultiPlaneSnapshot(
    MultiPlaneSnapshot &snapshot)
{
    std::lock_guard<std::mutex> lock(
        g_multi_plane_snapshot_mutex);

    if (g_multi_plane_snapshot.revision == 0)
    {
        return false;
    }

    snapshot = g_multi_plane_snapshot;
    return true;
}

void ResetMultiPlaneSnapshot()
{
    std::lock_guard<std::mutex> lock(
        g_multi_plane_snapshot_mutex);

    g_multi_plane_snapshot =
        MultiPlaneSnapshot();
}

}  // namespace fr_slam
