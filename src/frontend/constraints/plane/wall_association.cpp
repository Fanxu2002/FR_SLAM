#include "fr_slam/frontend/wall_association.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace fr_slam
{

// ============================================================================
// Wall Association V3
//
// Diagnostics only.
//
// V3 keeps V2 fragmentation fixes and adds persistent static-wall identity:
//
//   1) Same-frame coplanar clustering:
//        multiple MultiPlane fragments that are very likely the same physical
//        wall are merged into ONE physical-wall candidate.
//
//   2) Short-horizon temporal association:
//        physical-wall candidates are associated one-to-one with recent wall
//        tracks.  Stable tracks keep a FROZEN reference plane for future
//        residual validation; identity matching itself uses the last observed
//        plane so frontend drift does not immediately destroy the track ID.
//
// IMPORTANT:
//     No pose / ICP / Ground / Loop / PGO feedback is applied here.
// ============================================================================

namespace
{
    struct WallAssociationV2Observation
    {
        std::size_t plane_index = 0;
        Eigen::Vector3d normal_A = Eigen::Vector3d::UnitX();
        double d_A = 0.0;
        Eigen::Vector3d center_A = Eigen::Vector3d::Zero();
        double radius_m = 0.0;
        double quality = 0.0;
        std::size_t point_count = 0;
    };

    struct WallAssociationV2PhysicalWall
    {
        std::size_t cluster_id = 0;
        std::vector<std::size_t> member_plane_indices;
        std::size_t representative_plane_index = 0;

        Eigen::Vector3d normal_A = Eigen::Vector3d::UnitX();
        double d_A = 0.0;
        Eigen::Vector3d center_A = Eigen::Vector3d::Zero();
        double radius_m = 0.0;
        double quality = 0.0;
        std::size_t total_points = 0;
    };

    struct WallAssociationV2Track
    {
        std::size_t id = 0;

        // Last observation: used only for short-horizon identity matching.
        Eigen::Vector3d last_normal_A = Eigen::Vector3d::UnitX();
        double last_d_A = 0.0;
        Eigen::Vector3d last_center_A = Eigen::Vector3d::Zero();
        double last_radius_m = 0.0;

        // Frozen reference: created only after consecutive confirmation.
        bool stable = false;
        Eigen::Vector3d reference_normal_A = Eigen::Vector3d::UnitX();
        double reference_d_A = 0.0;

        // Confirmation accumulator.  Reset when temporal continuity breaks.
        Eigen::Vector3d confirmation_normal_sum_A = Eigen::Vector3d::Zero();
        double confirmation_d_sum_A = 0.0;
        std::size_t confirmation_samples = 0;

        // V3 static-wall diagnostics.
        std::size_t first_seen_frame = 0;
        bool static_confirmed = false;
        std::size_t static_residual_samples = 0;
        double static_normal_squared_sum_deg2 = 0.0;
        double static_plane_d_squared_sum_m2 = 0.0;
        std::size_t persistent_wall_id =
            std::numeric_limits<std::size_t>::max();

        std::size_t total_hits = 0;
        std::size_t consecutive_hits = 0;
        std::size_t last_seen_frame = 0;
        std::size_t missed_frames = 0;
        bool matched_this_frame = false;
    };

    struct WallAssociationV3PersistentWall
    {
        std::size_t id = 0;

        // Frozen world-plane reference.  This is intentionally NOT updated
        // by EMA; otherwise frontend drift could drag the reference with it.
        Eigen::Vector3d reference_normal_A = Eigen::Vector3d::UnitX();
        double reference_d_A = 0.0;

        // Last observed support is used only for re-identification.
        Eigen::Vector3d last_center_A = Eigen::Vector3d::Zero();
        double last_radius_m = 0.0;

        std::size_t first_seen_frame = 0;
        std::size_t last_seen_frame = 0;
        std::size_t total_observations = 0;
        std::size_t reacquisition_count = 0;
        std::size_t fallback_count = 0;
        std::size_t rebound_count = 0;

        // V3.1 true long-gap re-identification confirmation.
        // A dormant persistent wall is not restored from a single frame.
        bool reid_pending_valid = false;
        std::size_t reid_pending_count = 0;
        std::size_t reid_pending_last_frame = 0;
        Eigen::Vector3d reid_pending_last_normal_A =
            Eigen::Vector3d::UnitX();
        double reid_pending_last_d_A = 0.0;
        Eigen::Vector3d reid_pending_last_center_A =
            Eigen::Vector3d::Zero();
        double reid_pending_last_radius_m = 0.0;

        bool active_this_frame = false;
        bool reacquired_this_frame = false;
        bool fallback_this_frame = false;
        bool rebound_this_frame = false;
    };

    struct WallAssociationV2CandidateDebug
    {
        std::size_t physical_wall_index = 0;
        std::size_t track_id = std::numeric_limits<std::size_t>::max();

        bool matched_existing_track = false;
        bool new_track = false;
        bool stable = false;
        bool constraint_ready = false;
        bool low_quality_untracked = false;

        // V3 persistent-static-wall diagnostics.
        bool static_confirmed = false;
        bool persistent_created = false;

        // V3.1 separates three different identity events that V3 used to
        // report together as "REACQUIRED".
        bool persistent_fallback = false;
        bool persistent_reacquired = false; // TRUE long-gap reacquisition.
        bool persistent_rebound = false;
        bool reid_pending = false;
        std::size_t reid_pending_count = 0;

        std::size_t persistent_wall_id =
            std::numeric_limits<std::size_t>::max();
        std::size_t observation_span_frames = 0;
        std::size_t static_residual_samples = 0;
        std::size_t persistent_total_observations = 0;
        double static_normal_rms_deg =
            std::numeric_limits<double>::quiet_NaN();
        double static_plane_d_rms_m =
            std::numeric_limits<double>::quiet_NaN();

        std::size_t consecutive_hits = 0;
        std::size_t total_hits = 0;
        std::size_t missed_frames = 0;

        double normal_difference_deg =
            std::numeric_limits<double>::quiet_NaN();

        double plane_distance_difference_m =
            std::numeric_limits<double>::quiet_NaN();

        double tangential_support_gap_m =
            std::numeric_limits<double>::quiet_NaN();

        double association_score =
            std::numeric_limits<double>::quiet_NaN();

        double reference_normal_difference_deg =
            std::numeric_limits<double>::quiet_NaN();

        double reference_plane_distance_difference_m =
            std::numeric_limits<double>::quiet_NaN();
    };

    struct WallAssociationV2FrameDebug
    {
        std::size_t raw_wall_candidates = 0;
        std::size_t physical_wall_candidates = 0;
        std::size_t fragments_merged = 0;
        std::size_t matched_tracks = 0;
        std::size_t new_tracks = 0;
        std::size_t stable_walls = 0;
        std::size_t active_walls = 0;
        std::size_t stable_tracks_alive = 0;
        std::size_t tracks_alive = 0;

        // V3 persistent registry summary.
        std::size_t static_confirmed_current = 0;
        std::size_t transient_current = 0;
        std::size_t persistent_walls = 0;
        std::size_t active_static_walls = 0;
        std::size_t dormant_walls = 0;

        // V3.1 identity-event summary.
        std::size_t fallback_this_frame = 0;
        std::size_t reacquired_this_frame = 0;
        std::size_t rebound_this_frame = 0;
        std::size_t reid_pending_current = 0;

        std::vector<WallAssociationV2PhysicalWall> physical_walls;
        std::vector<WallAssociationV2CandidateDebug> candidate_debug;
    };

    struct WallAssociationV2Runtime
    {
        std::vector<WallAssociationV2Track> tracks;
        std::vector<WallAssociationV3PersistentWall> persistent_walls;
        std::size_t next_track_id = 0;
        std::size_t next_persistent_wall_id = 0;
    };

    void WallAssociationV2CanonicalizePlane(
        Eigen::Vector3d &normal,
        double &d)
    {
        if (!normal.allFinite() ||
            !std::isfinite(d) ||
            normal.norm() < 1.0e-9)
        {
            return;
        }

        normal.normalize();

        Eigen::Index dominant_index = 0;
        normal.cwiseAbs().maxCoeff(&dominant_index);

        if (normal[dominant_index] < 0.0)
        {
            normal = -normal;
            d = -d;
        }
    }

    double WallAssociationV2NormalAngleDeg(
        const Eigen::Vector3d &a,
        const Eigen::Vector3d &b)
    {
        if (!a.allFinite() ||
            !b.allFinite() ||
            a.norm() < 1.0e-9 ||
            b.norm() < 1.0e-9)
        {
            return std::numeric_limits<double>::quiet_NaN();
        }

        const double cosine =
            std::abs(
                a.normalized().dot(
                    b.normalized()));

        const double clamped_cosine =
            std::max(
                -1.0,
                std::min(
                    1.0,
                    cosine));

        constexpr double kRadToDeg =
            57.2957795130823208768;

        return std::acos(clamped_cosine) * kRadToDeg;
    }

    // ------------------------------------------------------------------------
    // FR_WALL_HORIZONTAL_FRAGMENT_MERGE_V45
    //
    // Compare only the XY projection of two nearly-vertical wall normals.
    // Plane normals are sign-ambiguous, so use |dot| just like the existing
    // full-3D comparison.
    // ------------------------------------------------------------------------
    double WallAssociationV2HorizontalNormalAngleDeg(
        const Eigen::Vector3d &a,
        const Eigen::Vector3d &b)
    {
        if (!a.allFinite() ||
            !b.allFinite())
        {
            return std::numeric_limits<double>::quiet_NaN();
        }

        Eigen::Vector3d a_horizontal(
            a.x(),
            a.y(),
            0.0);

        Eigen::Vector3d b_horizontal(
            b.x(),
            b.y(),
            0.0);

        const double a_norm =
            a_horizontal.norm();

        const double b_norm =
            b_horizontal.norm();

        if (!std::isfinite(a_norm) ||
            !std::isfinite(b_norm) ||
            a_norm < 1.0e-9 ||
            b_norm < 1.0e-9)
        {
            return std::numeric_limits<double>::quiet_NaN();
        }

        a_horizontal /= a_norm;
        b_horizontal /= b_norm;

        const double cosine =
            std::abs(
                a_horizontal.dot(
                    b_horizontal));

        const double clamped_cosine =
            std::max(
                -1.0,
                std::min(
                    1.0,
                    cosine));

        constexpr double kRadToDeg =
            57.2957795130823208768;

        return std::acos(clamped_cosine) *
               kRadToDeg;
    }

    double WallAssociationV2TangentialDistance(
        const Eigen::Vector3d &center_a_A,
        const Eigen::Vector3d &center_b_A,
        const Eigen::Vector3d &normal_A)
    {
        if (!center_a_A.allFinite() ||
            !center_b_A.allFinite() ||
            !normal_A.allFinite() ||
            normal_A.norm() < 1.0e-9)
        {
            return std::numeric_limits<double>::quiet_NaN();
        }

        const Eigen::Vector3d unit_normal_A =
            normal_A.normalized();

        const Eigen::Vector3d delta_A =
            center_a_A - center_b_A;

        const Eigen::Vector3d tangential_delta_A =
            delta_A -
            unit_normal_A *
                unit_normal_A.dot(delta_A);

        return tangential_delta_A.norm();
    }

    double WallAssociationV2SupportGap(
        const Eigen::Vector3d &center_a_A,
        const double radius_a_m,
        const Eigen::Vector3d &center_b_A,
        const double radius_b_m,
        const Eigen::Vector3d &normal_A)
    {
        const double tangential_distance_m =
            WallAssociationV2TangentialDistance(
                center_a_A,
                center_b_A,
                normal_A);

        if (!std::isfinite(tangential_distance_m))
        {
            return std::numeric_limits<double>::quiet_NaN();
        }

        return std::max(
            0.0,
            tangential_distance_m -
                std::max(0.0, radius_a_m) -
                std::max(0.0, radius_b_m));
    }

    std::size_t WallAssociationV2FindRoot(
        std::vector<std::size_t> &parent,
        std::size_t index)
    {
        while (parent[index] != index)
        {
            parent[index] = parent[parent[index]];
            index = parent[index];
        }

        return index;
    }

    void WallAssociationV2Union(
        std::vector<std::size_t> &parent,
        const std::size_t a,
        const std::size_t b)
    {
        const std::size_t root_a =
            WallAssociationV2FindRoot(parent, a);

        const std::size_t root_b =
            WallAssociationV2FindRoot(parent, b);

        if (root_a != root_b)
        {
            parent[root_b] = root_a;
        }
    }

    std::vector<WallAssociationV2PhysicalWall>
    BuildWallAssociationV2PhysicalWalls(
        const ::fr_slam::MultiPlaneExtractionResult &result,
        const Eigen::Isometry3d &T_AL,
        std::size_t &raw_wall_candidates,
        const bool horizontal_fragment_merge_enabled)
    {
        constexpr double kClusterMaximumNormalDifferenceDeg = 5.0;
        constexpr double kClusterMaximumPlaneDistanceDifferenceM = 0.20;
        constexpr double kClusterMaximumSupportGapM = 1.50;

        // FR_WALL_HORIZONTAL_FRAGMENT_MERGE_V45
        //
        // Conservative fallback validated by the V4.4 shadow.
        constexpr double
            kFragmentMaximumHorizontalNormalDifferenceDeg = 0.75;
        constexpr double
            kFragmentMaximumPlaneDistanceDifferenceM = 0.08;
        constexpr double
            kFragmentMaximumSymmetricPlaneSeparationM = 0.08;
        constexpr double
            kFragmentMaximumSupportGapM = 0.50;

        std::vector<WallAssociationV2Observation> observations;

        const Eigen::Matrix3d R_AL =
            T_AL.rotation();

        const Eigen::Vector3d t_AL =
            T_AL.translation();

        for (std::size_t plane_index = 0;
             plane_index < result.planes.size();
             ++plane_index)
        {
            const ::fr_slam::PlaneObservation &plane =
                result.planes[plane_index];

            if (!plane.wall_constraint_candidate)
            {
                continue;
            }

            WallAssociationV2Observation observation;
            observation.plane_index = plane_index;
            observation.normal_A = R_AL * plane.normal_L;
            observation.d_A =
                plane.d -
                observation.normal_A.dot(t_AL);
            observation.center_A =
                R_AL * plane.center_L + t_AL;
            observation.radius_m =
                std::max(0.0, plane.radius_m);
            observation.quality =
                plane.constraint_quality_score;
            observation.point_count =
                plane.point_count;

            if (!observation.normal_A.allFinite() ||
                observation.normal_A.norm() < 1.0e-9 ||
                !std::isfinite(observation.d_A) ||
                !observation.center_A.allFinite())
            {
                continue;
            }

            WallAssociationV2CanonicalizePlane(
                observation.normal_A,
                observation.d_A);

            ++raw_wall_candidates;
            observations.push_back(observation);
        }

        if (observations.empty())
        {
            return {};
        }

        std::vector<std::size_t> parent(
            observations.size());

        for (std::size_t i = 0;
             i < parent.size();
             ++i)
        {
            parent[i] = i;
        }

        for (std::size_t i = 0;
             i < observations.size();
             ++i)
        {
            for (std::size_t j = i + 1;
                 j < observations.size();
                 ++j)
            {
                const double normal_difference_deg =
                    WallAssociationV2NormalAngleDeg(
                        observations[i].normal_A,
                        observations[j].normal_A);

                const bool legacy_normal_compatible =
                    std::isfinite(normal_difference_deg) &&
                    normal_difference_deg <=
                        kClusterMaximumNormalDifferenceDeg;

                double horizontal_normal_difference_deg =
                    std::numeric_limits<double>::quiet_NaN();

                bool horizontal_fragment_candidate = false;

                if (horizontal_fragment_merge_enabled &&
                    !legacy_normal_compatible)
                {
                    horizontal_normal_difference_deg =
                        WallAssociationV2HorizontalNormalAngleDeg(
                            observations[i].normal_A,
                            observations[j].normal_A);

                    horizontal_fragment_candidate =
                        std::isfinite(
                            horizontal_normal_difference_deg) &&
                        horizontal_normal_difference_deg <=
                            kFragmentMaximumHorizontalNormalDifferenceDeg;
                }

                if (!legacy_normal_compatible &&
                    !horizontal_fragment_candidate)
                {
                    continue;
                }

                Eigen::Vector3d normal_j_A =
                    observations[j].normal_A;

                double d_j_A =
                    observations[j].d_A;

                if (normal_j_A.dot(
                        observations[i].normal_A) < 0.0)
                {
                    normal_j_A = -normal_j_A;
                    d_j_A = -d_j_A;
                }

                const double plane_distance_difference_m =
                    std::abs(
                        observations[i].d_A -
                        d_j_A);

                const double maximum_plane_distance_difference_m =
                    legacy_normal_compatible
                        ? kClusterMaximumPlaneDistanceDifferenceM
                        : kFragmentMaximumPlaneDistanceDifferenceM;

                if (!std::isfinite(
                        plane_distance_difference_m) ||
                    plane_distance_difference_m >
                        maximum_plane_distance_difference_m)
                {
                    continue;
                }

                Eigen::Vector3d mean_normal_A =
                    observations[i].normal_A +
                    normal_j_A;

                if (!mean_normal_A.allFinite() ||
                    mean_normal_A.norm() < 1.0e-9)
                {
                    continue;
                }

                mean_normal_A.normalize();

                const double support_gap_m =
                    WallAssociationV2SupportGap(
                        observations[i].center_A,
                        observations[i].radius_m,
                        observations[j].center_A,
                        observations[j].radius_m,
                        mean_normal_A);

                const double maximum_support_gap_m =
                    legacy_normal_compatible
                        ? kClusterMaximumSupportGapM
                        : kFragmentMaximumSupportGapM;

                if (!std::isfinite(support_gap_m) ||
                    support_gap_m >
                        maximum_support_gap_m)
                {
                    continue;
                }

                if (horizontal_fragment_candidate)
                {
                    Eigen::Vector3d normal_i_unit =
                        observations[i].normal_A;

                    Eigen::Vector3d normal_j_unit =
                        normal_j_A;

                    const double normal_i_norm =
                        normal_i_unit.norm();

                    const double normal_j_norm =
                        normal_j_unit.norm();

                    if (!std::isfinite(normal_i_norm) ||
                        !std::isfinite(normal_j_norm) ||
                        normal_i_norm < 1.0e-9 ||
                        normal_j_norm < 1.0e-9)
                    {
                        continue;
                    }

                    const double d_i_unit =
                        observations[i].d_A /
                        normal_i_norm;

                    const double d_j_unit =
                        d_j_A /
                        normal_j_norm;

                    normal_i_unit /=
                        normal_i_norm;

                    normal_j_unit /=
                        normal_j_norm;

                    const double i_plane_at_j_center_m =
                        std::abs(
                            normal_i_unit.dot(
                                observations[j].center_A) +
                            d_i_unit);

                    const double j_plane_at_i_center_m =
                        std::abs(
                            normal_j_unit.dot(
                                observations[i].center_A) +
                            d_j_unit);

                    const double symmetric_plane_separation_m =
                        0.5 *
                        (
                            i_plane_at_j_center_m +
                            j_plane_at_i_center_m
                        );

                    if (!std::isfinite(
                            symmetric_plane_separation_m) ||
                        symmetric_plane_separation_m >
                            kFragmentMaximumSymmetricPlaneSeparationM)
                    {
                        continue;
                    }
                }

                WallAssociationV2Union(
                    parent,
                    i,
                    j);
            }
        }

        std::vector<std::vector<std::size_t>> clusters;
        std::vector<std::size_t> roots;

        for (std::size_t i = 0;
             i < observations.size();
             ++i)
        {
            const std::size_t root =
                WallAssociationV2FindRoot(
                    parent,
                    i);

            auto root_it =
                std::find(
                    roots.begin(),
                    roots.end(),
                    root);

            if (root_it == roots.end())
            {
                roots.push_back(root);
                clusters.push_back({i});
            }
            else
            {
                const std::size_t cluster_index =
                    static_cast<std::size_t>(
                        root_it - roots.begin());

                clusters[cluster_index].push_back(i);
            }
        }

        std::vector<WallAssociationV2PhysicalWall> physical_walls;
        physical_walls.reserve(clusters.size());

        for (std::size_t cluster_index = 0;
             cluster_index < clusters.size();
             ++cluster_index)
        {
            const std::vector<std::size_t> &members =
                clusters[cluster_index];

            std::size_t representative_observation_index =
                members.front();

            for (const std::size_t observation_index : members)
            {
                const WallAssociationV2Observation &candidate =
                    observations[observation_index];

                const WallAssociationV2Observation &representative =
                    observations[representative_observation_index];

                if (candidate.quality > representative.quality ||
                    (candidate.quality == representative.quality &&
                     candidate.point_count > representative.point_count))
                {
                    representative_observation_index =
                        observation_index;
                }
            }

            const WallAssociationV2Observation &representative =
                observations[representative_observation_index];

            Eigen::Vector3d normal_sum_A =
                Eigen::Vector3d::Zero();

            Eigen::Vector3d center_sum_A =
                Eigen::Vector3d::Zero();

            double d_sum_A = 0.0;
            double weight_sum = 0.0;
            std::size_t total_points = 0;

            for (const std::size_t observation_index : members)
            {
                const WallAssociationV2Observation &observation =
                    observations[observation_index];

                Eigen::Vector3d aligned_normal_A =
                    observation.normal_A;

                double aligned_d_A =
                    observation.d_A;

                if (aligned_normal_A.dot(
                        representative.normal_A) < 0.0)
                {
                    aligned_normal_A =
                        -aligned_normal_A;

                    aligned_d_A =
                        -aligned_d_A;
                }

                const double weight =
                    std::max(
                        1.0,
                        static_cast<double>(
                            observation.point_count)) *
                    std::max(
                        0.10,
                        observation.quality);

                normal_sum_A +=
                    weight * aligned_normal_A;

                d_sum_A +=
                    weight * aligned_d_A;

                center_sum_A +=
                    weight * observation.center_A;

                weight_sum += weight;
                total_points += observation.point_count;
            }

            if (!normal_sum_A.allFinite() ||
                normal_sum_A.norm() < 1.0e-9 ||
                weight_sum <= 0.0)
            {
                continue;
            }

            WallAssociationV2PhysicalWall physical_wall;
            physical_wall.cluster_id = cluster_index;
            physical_wall.representative_plane_index =
                representative.plane_index;
            physical_wall.normal_A =
                normal_sum_A.normalized();
            physical_wall.d_A =
                d_sum_A / weight_sum;
            physical_wall.center_A =
                center_sum_A / weight_sum;
            physical_wall.quality =
                representative.quality;
            physical_wall.total_points =
                total_points;

            double aggregate_radius_m = 0.0;

            for (const std::size_t observation_index : members)
            {
                const WallAssociationV2Observation &observation =
                    observations[observation_index];

                physical_wall.member_plane_indices.push_back(
                    observation.plane_index);

                const double tangential_distance_m =
                    WallAssociationV2TangentialDistance(
                        observation.center_A,
                        physical_wall.center_A,
                        physical_wall.normal_A);

                if (std::isfinite(tangential_distance_m))
                {
                    aggregate_radius_m =
                        std::max(
                            aggregate_radius_m,
                            tangential_distance_m +
                                observation.radius_m);
                }
            }

            physical_wall.radius_m =
                std::max(
                    aggregate_radius_m,
                    representative.radius_m);

            WallAssociationV2CanonicalizePlane(
                physical_wall.normal_A,
                physical_wall.d_A);

            physical_walls.push_back(
                physical_wall);
        }

        return physical_walls;
    }

    WallAssociationV2FrameDebug RunWallAssociationV3(
        const ::fr_slam::MultiPlaneExtractionResult &result,
        const Eigen::Isometry3d &T_AL,
        const std::size_t frame_index,
        WallAssociationV2Runtime &runtime,
        const bool horizontal_fragment_merge_enabled)
    {
        // Same-frame fragment merging happens in
        // BuildWallAssociationV2PhysicalWalls().  V3 deliberately keeps the
        // validated V2 clustering and short-horizon live-track association,
        // then adds a static gate + persistent physical-wall registry.

        constexpr double kExistingTrackMinimumQuality = 0.70;
        constexpr double kNewTrackMinimumQuality = 0.80;

        constexpr double kMaximumNormalDifferenceDeg = 4.0;
        constexpr double kMaximumPlaneDistanceDifferenceM = 0.20;
        constexpr double kMaximumTangentialSupportGapM = 1.00;

        constexpr std::size_t kMaximumMissedFrames = 10;
        constexpr std::size_t kStableConsecutiveHits = 3;

        // A stable live track is only "constraint ready" relative to its own
        // frozen reference if it passes this stricter gate.
        constexpr double kReadyMaximumNormalDifferenceDeg = 3.0;
        constexpr double kReadyMaximumPlaneDistanceDifferenceM = 0.15;
        constexpr double kReadyMinimumQuality = 0.80;

        // --------------------------------------------------------------------
        // V3 static confirmation gate.
        //
        // A short-lived planar object (vehicle panel / railing / local clutter)
        // should normally disappear before satisfying all of these conditions.
        // A parked vehicle can still look geometrically static; that is a known
        // limitation of pure geometry and is why this stage remains diagnostic.
        // --------------------------------------------------------------------
        constexpr std::size_t kStaticMinimumTotalHits = 10;
        constexpr std::size_t kStaticMinimumObservationSpanFrames = 15;
        constexpr std::size_t kStaticMinimumResidualSamples = 6;
        constexpr double kStaticMaximumNormalRmsDeg = 2.0;
        constexpr double kStaticMaximumPlaneDRmsM = 0.10;
        constexpr double kStaticMinimumQuality = 0.80;

        // --------------------------------------------------------------------
        // V3 persistent re-identification gate.
        //
        // This is intentionally wider than short-horizon V2 association:
        // support can move substantially along a real wall after occlusion,
        // while the frozen world-plane normal/d should remain similar.
        // --------------------------------------------------------------------
        constexpr double kReidMinimumQuality = 0.80;
        constexpr double kReidMaximumNormalDifferenceDeg = 5.0;
        constexpr double kReidMaximumPlaneDistanceDifferenceM = 0.30;
        constexpr double kReidMaximumTangentialSupportGapM = 3.00;
        constexpr std::size_t kReidMaximumDormantFrames = 5000;

        // V3.1: a TRUE long-gap re-identification must persist for several
        // consecutive frames.  A short-horizon fallback to the SAME live track
        // is handled separately and does not use this confirmation counter.
        constexpr std::size_t kReidConfirmationFrames = 3;
        constexpr double kReidPendingMaximumNormalDifferenceDeg = 3.0;
        constexpr double kReidPendingMaximumPlaneDistanceDifferenceM = 0.15;
        constexpr double kReidPendingMaximumTangentialSupportGapM = 1.00;

        // V3.1 final duplicate-prevention check before STATIC_NEW.  This is
        // deliberately tighter than broad dormant re-ID.
        constexpr double kDuplicateMaximumNormalDifferenceDeg = 3.0;
        constexpr double kDuplicateMaximumPlaneDistanceDifferenceM = 0.15;
        constexpr double kDuplicateMaximumTangentialSupportGapM = 2.00;

        WallAssociationV2FrameDebug debug;

        debug.physical_walls =
            BuildWallAssociationV2PhysicalWalls(
                result,
                T_AL,
                debug.raw_wall_candidates,
                horizontal_fragment_merge_enabled);

        debug.physical_wall_candidates =
            debug.physical_walls.size();

        if (debug.raw_wall_candidates >=
            debug.physical_wall_candidates)
        {
            debug.fragments_merged =
                debug.raw_wall_candidates -
                debug.physical_wall_candidates;
        }

        debug.candidate_debug.resize(
            debug.physical_walls.size());

        for (std::size_t candidate_index = 0;
             candidate_index < debug.candidate_debug.size();
             ++candidate_index)
        {
            debug.candidate_debug[candidate_index].physical_wall_index =
                candidate_index;
        }

        if (frame_index == 1U)
        {
            runtime = WallAssociationV2Runtime();
        }

        for (WallAssociationV3PersistentWall &persistent :
             runtime.persistent_walls)
        {
            persistent.active_this_frame = false;
            persistent.reacquired_this_frame = false;
            persistent.fallback_this_frame = false;
            persistent.rebound_this_frame = false;

            // A pending long-gap re-ID must be consecutive.  If one complete
            // frame passes without another supporting observation, restart.
            if (persistent.reid_pending_valid &&
                frame_index >
                    (persistent.reid_pending_last_frame + 1U))
            {
                persistent.reid_pending_valid = false;
                persistent.reid_pending_count = 0U;
            }
        }

        runtime.tracks.erase(
            std::remove_if(
                runtime.tracks.begin(),
                runtime.tracks.end(),
                [frame_index](
                    const WallAssociationV2Track &track)
                {
                    if (frame_index < track.last_seen_frame)
                    {
                        return true;
                    }

                    return (frame_index - track.last_seen_frame) >
                           kMaximumMissedFrames;
                }),
            runtime.tracks.end());

        for (WallAssociationV2Track &track : runtime.tracks)
        {
            track.matched_this_frame = false;
        }

        const auto FindPersistentWallById =
            [&runtime](const std::size_t persistent_wall_id)
            -> WallAssociationV3PersistentWall *
        {
            for (WallAssociationV3PersistentWall &persistent :
                 runtime.persistent_walls)
            {
                if (persistent.id == persistent_wall_id)
                {
                    return &persistent;
                }
            }

            return nullptr;
        };

        const auto UpdateStaticResidualStatistics =
            [](WallAssociationV2Track &track,
               const double normal_difference_deg,
               const double plane_distance_difference_m)
        {
            if (!std::isfinite(normal_difference_deg) ||
                !std::isfinite(plane_distance_difference_m))
            {
                return;
            }

            ++track.static_residual_samples;
            track.static_normal_squared_sum_deg2 +=
                normal_difference_deg * normal_difference_deg;
            track.static_plane_d_squared_sum_m2 +=
                plane_distance_difference_m *
                plane_distance_difference_m;
        };

        const auto StaticNormalRmsDeg =
            [](const WallAssociationV2Track &track)
            -> double
        {
            if (track.static_residual_samples == 0U)
            {
                return std::numeric_limits<double>::quiet_NaN();
            }

            return std::sqrt(
                track.static_normal_squared_sum_deg2 /
                static_cast<double>(track.static_residual_samples));
        };

        const auto StaticPlaneDRmsM =
            [](const WallAssociationV2Track &track)
            -> double
        {
            if (track.static_residual_samples == 0U)
            {
                return std::numeric_limits<double>::quiet_NaN();
            }

            return std::sqrt(
                track.static_plane_d_squared_sum_m2 /
                static_cast<double>(track.static_residual_samples));
        };

        const auto ObservationSpanFrames =
            [frame_index](const WallAssociationV2Track &track)
            -> std::size_t
        {
            if (frame_index < track.first_seen_frame)
            {
                return 0U;
            }

            return frame_index - track.first_seen_frame + 1U;
        };

        const auto RefreshPersistentObservation =
            [&](WallAssociationV2Track &track,
                const WallAssociationV2PhysicalWall &candidate,
                const bool reacquired)
        {
            if (track.persistent_wall_id ==
                std::numeric_limits<std::size_t>::max())
            {
                return;
            }

            WallAssociationV3PersistentWall *persistent =
                FindPersistentWallById(
                    track.persistent_wall_id);

            if (persistent == nullptr)
            {
                return;
            }

            persistent->last_center_A = candidate.center_A;
            persistent->last_radius_m = candidate.radius_m;
            persistent->last_seen_frame = frame_index;
            persistent->active_this_frame = true;
            ++persistent->total_observations;

            if (reacquired)
            {
                persistent->reacquired_this_frame = true;
                ++persistent->reacquisition_count;
            }
        };

        const auto TryCreatePersistentWall =
            [&](WallAssociationV2Track &track,
                const WallAssociationV2PhysicalWall &candidate,
                WallAssociationV2CandidateDebug &candidate_debug)
        {
            if (track.persistent_wall_id !=
                std::numeric_limits<std::size_t>::max())
            {
                return;
            }

            const double normal_rms_deg =
                StaticNormalRmsDeg(track);

            const double plane_d_rms_m =
                StaticPlaneDRmsM(track);

            const std::size_t observation_span_frames =
                ObservationSpanFrames(track);

            const bool static_gate_passed =
                track.stable &&
                track.total_hits >= kStaticMinimumTotalHits &&
                observation_span_frames >=
                    kStaticMinimumObservationSpanFrames &&
                track.static_residual_samples >=
                    kStaticMinimumResidualSamples &&
                std::isfinite(normal_rms_deg) &&
                normal_rms_deg <=
                    kStaticMaximumNormalRmsDeg &&
                std::isfinite(plane_d_rms_m) &&
                plane_d_rms_m <=
                    kStaticMaximumPlaneDRmsM &&
                std::isfinite(candidate.quality) &&
                candidate.quality >= kStaticMinimumQuality;

            if (!static_gate_passed)
            {
                return;
            }

            track.static_confirmed = true;

            // ------------------------------------------------------------
            // V3.1 duplicate prevention.
            //
            // Before allocating a new persistent_wall_id, perform one final
            // tight search over dormant persistent walls.  This catches a
            // live track that escaped broad re-ID earlier, survived long
            // enough to pass the Static Gate, but is actually an already
            // known physical wall.
            // ------------------------------------------------------------
            WallAssociationV3PersistentWall *best_duplicate = nullptr;
            double best_duplicate_score =
                std::numeric_limits<double>::infinity();

            for (WallAssociationV3PersistentWall &persistent :
                 runtime.persistent_walls)
            {
                // One persistent physical wall may own at most one current
                // observation.  Same-frame fragmentation belongs in V2 merge.
                if (persistent.active_this_frame)
                {
                    continue;
                }

                if (frame_index < persistent.last_seen_frame ||
                    (frame_index - persistent.last_seen_frame) >
                        kReidMaximumDormantFrames)
                {
                    continue;
                }

                const double normal_difference_deg =
                    WallAssociationV2NormalAngleDeg(
                        candidate.normal_A,
                        persistent.reference_normal_A);

                if (!std::isfinite(normal_difference_deg) ||
                    normal_difference_deg >
                        kDuplicateMaximumNormalDifferenceDeg)
                {
                    continue;
                }

                Eigen::Vector3d aligned_normal_A =
                    candidate.normal_A;
                double aligned_d_A =
                    candidate.d_A;

                if (aligned_normal_A.dot(
                        persistent.reference_normal_A) < 0.0)
                {
                    aligned_normal_A = -aligned_normal_A;
                    aligned_d_A = -aligned_d_A;
                }

                const double plane_distance_difference_m =
                    std::abs(
                        aligned_d_A -
                        persistent.reference_d_A);

                if (plane_distance_difference_m >
                    kDuplicateMaximumPlaneDistanceDifferenceM)
                {
                    continue;
                }

                const double support_gap_m =
                    WallAssociationV2SupportGap(
                        candidate.center_A,
                        candidate.radius_m,
                        persistent.last_center_A,
                        persistent.last_radius_m,
                        persistent.reference_normal_A);

                if (!std::isfinite(support_gap_m) ||
                    support_gap_m >
                        kDuplicateMaximumTangentialSupportGapM)
                {
                    continue;
                }

                const double score =
                    normal_difference_deg /
                        kDuplicateMaximumNormalDifferenceDeg +
                    plane_distance_difference_m /
                        kDuplicateMaximumPlaneDistanceDifferenceM +
                    support_gap_m /
                        kDuplicateMaximumTangentialSupportGapM;

                if (score < best_duplicate_score)
                {
                    best_duplicate_score = score;
                    best_duplicate = &persistent;
                }
            }

            if (best_duplicate != nullptr)
            {
                // Detach any stale live-track alias.  The current validated
                // static track becomes the sole live owner of this persistent
                // identity.
                for (WallAssociationV2Track &other_track :
                     runtime.tracks)
                {
                    if (&other_track != &track &&
                        other_track.persistent_wall_id ==
                            best_duplicate->id)
                    {
                        other_track.persistent_wall_id =
                            std::numeric_limits<std::size_t>::max();
                    }
                }

                track.persistent_wall_id =
                    best_duplicate->id;

                track.reference_normal_A =
                    best_duplicate->reference_normal_A;

                track.reference_d_A =
                    best_duplicate->reference_d_A;

                candidate_debug.persistent_rebound = true;
                candidate_debug.persistent_wall_id =
                    best_duplicate->id;

                candidate_debug.reference_normal_difference_deg =
                    WallAssociationV2NormalAngleDeg(
                        candidate.normal_A,
                        best_duplicate->reference_normal_A);

                Eigen::Vector3d aligned_normal_A =
                    candidate.normal_A;

                double aligned_d_A =
                    candidate.d_A;

                if (aligned_normal_A.dot(
                        best_duplicate->reference_normal_A) < 0.0)
                {
                    aligned_normal_A =
                        -aligned_normal_A;

                    aligned_d_A =
                        -aligned_d_A;
                }

                candidate_debug.reference_plane_distance_difference_m =
                    std::abs(
                        aligned_d_A -
                        best_duplicate->reference_d_A);

                candidate_debug.constraint_ready =
                    std::isfinite(candidate.quality) &&
                    candidate.quality >=
                        kReadyMinimumQuality &&
                    std::isfinite(
                        candidate_debug
                            .reference_normal_difference_deg) &&
                    candidate_debug
                            .reference_normal_difference_deg <=
                        kReadyMaximumNormalDifferenceDeg &&
                    std::isfinite(
                        candidate_debug
                            .reference_plane_distance_difference_m) &&
                    candidate_debug
                            .reference_plane_distance_difference_m <=
                        kReadyMaximumPlaneDistanceDifferenceM;

                best_duplicate->last_center_A =
                    candidate.center_A;

                best_duplicate->last_radius_m =
                    candidate.radius_m;

                best_duplicate->last_seen_frame =
                    frame_index;

                best_duplicate->active_this_frame =
                    true;

                best_duplicate->rebound_this_frame =
                    true;

                best_duplicate->reid_pending_valid =
                    false;

                best_duplicate->reid_pending_count =
                    0U;

                best_duplicate->first_seen_frame =
                    std::min(
                        best_duplicate->first_seen_frame,
                        track.first_seen_frame);

                best_duplicate->total_observations +=
                    track.total_hits;

                ++best_duplicate->rebound_count;

                return;
            }

            // No duplicate exists: allocate a genuinely new persistent wall.
            WallAssociationV3PersistentWall persistent;
            persistent.id = runtime.next_persistent_wall_id++;
            persistent.reference_normal_A =
                track.reference_normal_A;
            persistent.reference_d_A =
                track.reference_d_A;
            persistent.last_center_A =
                candidate.center_A;
            persistent.last_radius_m =
                candidate.radius_m;
            persistent.first_seen_frame =
                track.first_seen_frame;
            persistent.last_seen_frame =
                frame_index;
            persistent.total_observations =
                track.total_hits;
            persistent.active_this_frame = true;

            track.persistent_wall_id = persistent.id;

            runtime.persistent_walls.push_back(
                persistent);

            candidate_debug.persistent_created = true;
        };

        struct MatchPair
        {
            std::size_t candidate_index = 0;
            std::size_t track_index = 0;
            double score = std::numeric_limits<double>::infinity();
            double normal_difference_deg =
                std::numeric_limits<double>::quiet_NaN();
            double plane_distance_difference_m =
                std::numeric_limits<double>::quiet_NaN();
            double support_gap_m =
                std::numeric_limits<double>::quiet_NaN();
        };

        std::vector<MatchPair> match_pairs;

        for (std::size_t candidate_index = 0;
             candidate_index < debug.physical_walls.size();
             ++candidate_index)
        {
            const WallAssociationV2PhysicalWall &candidate =
                debug.physical_walls[candidate_index];

            if (!std::isfinite(candidate.quality) ||
                candidate.quality < kExistingTrackMinimumQuality)
            {
                continue;
            }

            for (std::size_t track_index = 0;
                 track_index < runtime.tracks.size();
                 ++track_index)
            {
                const WallAssociationV2Track &track =
                    runtime.tracks[track_index];

                const double normal_difference_deg =
                    WallAssociationV2NormalAngleDeg(
                        candidate.normal_A,
                        track.last_normal_A);

                if (!std::isfinite(normal_difference_deg) ||
                    normal_difference_deg >
                        kMaximumNormalDifferenceDeg)
                {
                    continue;
                }

                Eigen::Vector3d aligned_normal_A =
                    candidate.normal_A;

                double aligned_d_A =
                    candidate.d_A;

                if (aligned_normal_A.dot(
                        track.last_normal_A) < 0.0)
                {
                    aligned_normal_A =
                        -aligned_normal_A;

                    aligned_d_A =
                        -aligned_d_A;
                }

                const double plane_distance_difference_m =
                    std::abs(
                        aligned_d_A -
                        track.last_d_A);

                if (plane_distance_difference_m >
                    kMaximumPlaneDistanceDifferenceM)
                {
                    continue;
                }

                const double support_gap_m =
                    WallAssociationV2SupportGap(
                        candidate.center_A,
                        candidate.radius_m,
                        track.last_center_A,
                        track.last_radius_m,
                        track.last_normal_A);

                if (!std::isfinite(support_gap_m) ||
                    support_gap_m >
                        kMaximumTangentialSupportGapM)
                {
                    continue;
                }

                MatchPair pair;
                pair.candidate_index = candidate_index;
                pair.track_index = track_index;
                pair.normal_difference_deg =
                    normal_difference_deg;
                pair.plane_distance_difference_m =
                    plane_distance_difference_m;
                pair.support_gap_m =
                    support_gap_m;
                pair.score =
                    normal_difference_deg /
                        kMaximumNormalDifferenceDeg +
                    plane_distance_difference_m /
                        kMaximumPlaneDistanceDifferenceM +
                    support_gap_m /
                        kMaximumTangentialSupportGapM;

                match_pairs.push_back(pair);
            }
        }

        std::sort(
            match_pairs.begin(),
            match_pairs.end(),
            [](const MatchPair &lhs,
               const MatchPair &rhs)
            {
                return lhs.score < rhs.score;
            });

        std::vector<bool> candidate_assigned(
            debug.physical_walls.size(),
            false);

        std::vector<bool> track_assigned(
            runtime.tracks.size(),
            false);

        for (const MatchPair &pair : match_pairs)
        {
            if (candidate_assigned[pair.candidate_index] ||
                track_assigned[pair.track_index])
            {
                continue;
            }

            WallAssociationV2PhysicalWall &candidate =
                debug.physical_walls[pair.candidate_index];

            WallAssociationV2Track &track =
                runtime.tracks[pair.track_index];

            WallAssociationV2CandidateDebug &candidate_debug =
                debug.candidate_debug[pair.candidate_index];

            const bool temporally_contiguous =
                frame_index ==
                (track.last_seen_frame + 1U);

            Eigen::Vector3d aligned_normal_A =
                candidate.normal_A;

            double aligned_d_A =
                candidate.d_A;

            if (aligned_normal_A.dot(
                    track.last_normal_A) < 0.0)
            {
                aligned_normal_A =
                    -aligned_normal_A;

                aligned_d_A =
                    -aligned_d_A;
            }

            if (!track.stable)
            {
                if (!temporally_contiguous ||
                    track.confirmation_samples == 0U)
                {
                    track.confirmation_normal_sum_A =
                        aligned_normal_A;

                    track.confirmation_d_sum_A =
                        aligned_d_A;

                    track.confirmation_samples = 1U;
                    track.consecutive_hits = 1U;
                }
                else
                {
                    Eigen::Vector3d confirmation_normal_A =
                        aligned_normal_A;

                    if (confirmation_normal_A.dot(
                            track.confirmation_normal_sum_A) < 0.0)
                    {
                        confirmation_normal_A =
                            -confirmation_normal_A;

                        aligned_d_A =
                            -aligned_d_A;
                    }

                    track.confirmation_normal_sum_A +=
                        confirmation_normal_A;

                    track.confirmation_d_sum_A +=
                        aligned_d_A;

                    ++track.confirmation_samples;
                    ++track.consecutive_hits;
                }

                if (track.consecutive_hits >=
                        kStableConsecutiveHits &&
                    track.confirmation_samples > 0U &&
                    track.confirmation_normal_sum_A.allFinite() &&
                    track.confirmation_normal_sum_A.norm() > 1.0e-9)
                {
                    track.reference_normal_A =
                        track.confirmation_normal_sum_A.normalized();

                    track.reference_d_A =
                        track.confirmation_d_sum_A /
                        static_cast<double>(
                            track.confirmation_samples);

                    WallAssociationV2CanonicalizePlane(
                        track.reference_normal_A,
                        track.reference_d_A);

                    track.stable = true;
                }
            }
            else
            {
                if (temporally_contiguous)
                {
                    ++track.consecutive_hits;
                }
                else
                {
                    track.consecutive_hits = 1U;
                }
            }

            track.last_normal_A =
                aligned_normal_A.normalized();
            track.last_d_A = aligned_d_A;
            track.last_center_A = candidate.center_A;
            track.last_radius_m = candidate.radius_m;
            track.last_seen_frame = frame_index;
            track.missed_frames = 0U;
            track.matched_this_frame = true;
            ++track.total_hits;

            candidate_assigned[pair.candidate_index] = true;
            track_assigned[pair.track_index] = true;

            candidate_debug.track_id = track.id;
            candidate_debug.matched_existing_track = true;
            candidate_debug.stable = track.stable;
            candidate_debug.consecutive_hits = track.consecutive_hits;
            candidate_debug.total_hits = track.total_hits;
            candidate_debug.missed_frames = track.missed_frames;
            candidate_debug.normal_difference_deg =
                pair.normal_difference_deg;
            candidate_debug.plane_distance_difference_m =
                pair.plane_distance_difference_m;
            candidate_debug.tangential_support_gap_m =
                pair.support_gap_m;
            candidate_debug.association_score =
                pair.score;

            if (track.stable)
            {
                candidate_debug.reference_normal_difference_deg =
                    WallAssociationV2NormalAngleDeg(
                        candidate.normal_A,
                        track.reference_normal_A);

                Eigen::Vector3d reference_aligned_normal_A =
                    candidate.normal_A;

                double reference_aligned_d_A =
                    candidate.d_A;

                if (reference_aligned_normal_A.dot(
                        track.reference_normal_A) < 0.0)
                {
                    reference_aligned_normal_A =
                        -reference_aligned_normal_A;

                    reference_aligned_d_A =
                        -reference_aligned_d_A;
                }

                candidate_debug.reference_plane_distance_difference_m =
                    std::abs(
                        reference_aligned_d_A -
                        track.reference_d_A);

                candidate_debug.constraint_ready =
                    std::isfinite(candidate.quality) &&
                    candidate.quality >= kReadyMinimumQuality &&
                    std::isfinite(
                        candidate_debug.reference_normal_difference_deg) &&
                    candidate_debug.reference_normal_difference_deg <=
                        kReadyMaximumNormalDifferenceDeg &&
                    std::isfinite(
                        candidate_debug.reference_plane_distance_difference_m) &&
                    candidate_debug.reference_plane_distance_difference_m <=
                        kReadyMaximumPlaneDistanceDifferenceM;

                UpdateStaticResidualStatistics(
                    track,
                    candidate_debug.reference_normal_difference_deg,
                    candidate_debug.reference_plane_distance_difference_m);

                TryCreatePersistentWall(
                    track,
                    candidate,
                    candidate_debug);
            }

            if (track.persistent_wall_id !=
                std::numeric_limits<std::size_t>::max())
            {
                // If the persistent wall was created on this observation, its
                // total_observations already includes track.total_hits.  Do not
                // increment it a second time on the creation frame.
                if (!candidate_debug.persistent_created)
                {
                    RefreshPersistentObservation(
                        track,
                        candidate,
                        false);
                }

                candidate_debug.static_confirmed =
                    track.static_confirmed;
                candidate_debug.persistent_wall_id =
                    track.persistent_wall_id;
            }

            candidate_debug.observation_span_frames =
                ObservationSpanFrames(track);
            candidate_debug.static_residual_samples =
                track.static_residual_samples;
            candidate_debug.static_normal_rms_deg =
                StaticNormalRmsDeg(track);
            candidate_debug.static_plane_d_rms_m =
                StaticPlaneDRmsM(track);

            if (track.persistent_wall_id !=
                std::numeric_limits<std::size_t>::max())
            {
                const WallAssociationV3PersistentWall *persistent =
                    FindPersistentWallById(
                        track.persistent_wall_id);

                if (persistent != nullptr)
                {
                    candidate_debug.persistent_total_observations =
                        persistent->total_observations;
                }
            }

            ++debug.matched_tracks;

            if (candidate_debug.stable)
            {
                ++debug.stable_walls;
            }

            if (candidate_debug.constraint_ready)
            {
                ++debug.active_walls;
            }
        }

        // --------------------------------------------------------------------
        // V3.1 persistent identity recovery.
        //
        // V3 mixed two different situations under one "REACQUIRED" label:
        //
        //   A) the SAME live track survived, V2's one-frame gate missed once,
        //      and the persistent registry merely bridged that gap.
        //
        //   B) the old live track was gone and a genuinely new live track
        //      rediscovered a dormant physical wall.
        //
        // V3.1 separates them:
        //
        //   PERSISTENT_FALLBACK
        //       same live_track_id, immediate bridge, no reacquisition count.
        //
        //   REID_PENDING -> TRUE_REACQUIRED
        //       no live owner exists; require 3 consecutive compatible
        //       observations before restoring the persistent identity.
        // --------------------------------------------------------------------
        struct PersistentMatchPair
        {
            std::size_t candidate_index = 0;
            std::size_t persistent_index = 0;
            double score = std::numeric_limits<double>::infinity();
            double normal_difference_deg =
                std::numeric_limits<double>::quiet_NaN();
            double plane_distance_difference_m =
                std::numeric_limits<double>::quiet_NaN();
            double support_gap_m =
                std::numeric_limits<double>::quiet_NaN();
        };

        std::vector<PersistentMatchPair> persistent_match_pairs;

        for (std::size_t candidate_index = 0;
             candidate_index < debug.physical_walls.size();
             ++candidate_index)
        {
            if (candidate_assigned[candidate_index])
            {
                continue;
            }

            const WallAssociationV2PhysicalWall &candidate =
                debug.physical_walls[candidate_index];

            if (!std::isfinite(candidate.quality) ||
                candidate.quality < kReidMinimumQuality)
            {
                continue;
            }

            for (std::size_t persistent_index = 0;
                 persistent_index < runtime.persistent_walls.size();
                 ++persistent_index)
            {
                const WallAssociationV3PersistentWall &persistent =
                    runtime.persistent_walls[persistent_index];

                if (persistent.active_this_frame)
                {
                    continue;
                }

                if (frame_index < persistent.last_seen_frame ||
                    (frame_index - persistent.last_seen_frame) >
                        kReidMaximumDormantFrames)
                {
                    continue;
                }

                const double normal_difference_deg =
                    WallAssociationV2NormalAngleDeg(
                        candidate.normal_A,
                        persistent.reference_normal_A);

                if (!std::isfinite(normal_difference_deg) ||
                    normal_difference_deg >
                        kReidMaximumNormalDifferenceDeg)
                {
                    continue;
                }

                Eigen::Vector3d aligned_normal_A =
                    candidate.normal_A;

                double aligned_d_A =
                    candidate.d_A;

                if (aligned_normal_A.dot(
                        persistent.reference_normal_A) < 0.0)
                {
                    aligned_normal_A =
                        -aligned_normal_A;

                    aligned_d_A =
                        -aligned_d_A;
                }

                const double plane_distance_difference_m =
                    std::abs(
                        aligned_d_A -
                        persistent.reference_d_A);

                if (plane_distance_difference_m >
                    kReidMaximumPlaneDistanceDifferenceM)
                {
                    continue;
                }

                const double support_gap_m =
                    WallAssociationV2SupportGap(
                        candidate.center_A,
                        candidate.radius_m,
                        persistent.last_center_A,
                        persistent.last_radius_m,
                        persistent.reference_normal_A);

                if (!std::isfinite(support_gap_m) ||
                    support_gap_m >
                        kReidMaximumTangentialSupportGapM)
                {
                    continue;
                }

                PersistentMatchPair pair;
                pair.candidate_index = candidate_index;
                pair.persistent_index = persistent_index;
                pair.normal_difference_deg =
                    normal_difference_deg;
                pair.plane_distance_difference_m =
                    plane_distance_difference_m;
                pair.support_gap_m =
                    support_gap_m;
                pair.score =
                    normal_difference_deg /
                        kReidMaximumNormalDifferenceDeg +
                    plane_distance_difference_m /
                        kReidMaximumPlaneDistanceDifferenceM +
                    support_gap_m /
                        kReidMaximumTangentialSupportGapM;

                persistent_match_pairs.push_back(pair);
            }
        }

        std::sort(
            persistent_match_pairs.begin(),
            persistent_match_pairs.end(),
            [](const PersistentMatchPair &lhs,
               const PersistentMatchPair &rhs)
            {
                return lhs.score < rhs.score;
            });

        std::vector<bool> persistent_assigned(
            runtime.persistent_walls.size(),
            false);

        for (const PersistentMatchPair &pair :
             persistent_match_pairs)
        {
            if (candidate_assigned[pair.candidate_index] ||
                persistent_assigned[pair.persistent_index])
            {
                continue;
            }

            WallAssociationV2PhysicalWall &candidate =
                debug.physical_walls[pair.candidate_index];

            WallAssociationV3PersistentWall &persistent =
                runtime.persistent_walls[pair.persistent_index];

            WallAssociationV2CandidateDebug &candidate_debug =
                debug.candidate_debug[pair.candidate_index];

            Eigen::Vector3d aligned_normal_A =
                candidate.normal_A;

            double aligned_d_A =
                candidate.d_A;

            if (aligned_normal_A.dot(
                    persistent.reference_normal_A) < 0.0)
            {
                aligned_normal_A =
                    -aligned_normal_A;

                aligned_d_A =
                    -aligned_d_A;
            }

            // ------------------------------------------------------------
            // Case A: the SAME live track still exists.
            //
            // This is not a real reacquisition.  It is a persistent fallback
            // across a short V2 gate miss.  Keep the same live_track_id.
            // ------------------------------------------------------------
            WallAssociationV2Track *linked_live_track = nullptr;

            for (WallAssociationV2Track &track :
                 runtime.tracks)
            {
                if (!track.matched_this_frame &&
                    track.persistent_wall_id ==
                        persistent.id)
                {
                    linked_live_track = &track;
                    break;
                }
            }

            if (linked_live_track != nullptr)
            {
                const bool temporally_contiguous =
                    frame_index ==
                    (linked_live_track->last_seen_frame + 1U);

                linked_live_track->last_normal_A =
                    aligned_normal_A.normalized();

                linked_live_track->last_d_A =
                    aligned_d_A;

                linked_live_track->last_center_A =
                    candidate.center_A;

                linked_live_track->last_radius_m =
                    candidate.radius_m;

                linked_live_track->stable =
                    true;

                linked_live_track->static_confirmed =
                    true;

                linked_live_track->reference_normal_A =
                    persistent.reference_normal_A;

                linked_live_track->reference_d_A =
                    persistent.reference_d_A;

                linked_live_track->persistent_wall_id =
                    persistent.id;

                if (temporally_contiguous)
                {
                    ++linked_live_track->consecutive_hits;
                }
                else
                {
                    linked_live_track->consecutive_hits = 1U;
                }

                linked_live_track->last_seen_frame =
                    frame_index;

                linked_live_track->missed_frames =
                    0U;

                linked_live_track->matched_this_frame =
                    true;

                ++linked_live_track->total_hits;

                candidate_debug.track_id =
                    linked_live_track->id;

                candidate_debug.stable =
                    true;

                candidate_debug.static_confirmed =
                    true;

                candidate_debug.persistent_fallback =
                    true;

                candidate_debug.persistent_wall_id =
                    persistent.id;

                candidate_debug.consecutive_hits =
                    linked_live_track->consecutive_hits;

                candidate_debug.total_hits =
                    linked_live_track->total_hits;

                candidate_debug.normal_difference_deg =
                    pair.normal_difference_deg;

                candidate_debug.plane_distance_difference_m =
                    pair.plane_distance_difference_m;

                candidate_debug.tangential_support_gap_m =
                    pair.support_gap_m;

                candidate_debug.association_score =
                    pair.score;

                candidate_debug.reference_normal_difference_deg =
                    pair.normal_difference_deg;

                candidate_debug.reference_plane_distance_difference_m =
                    pair.plane_distance_difference_m;

                candidate_debug.constraint_ready =
                    candidate.quality >=
                        kReadyMinimumQuality &&
                    pair.normal_difference_deg <=
                        kReadyMaximumNormalDifferenceDeg &&
                    pair.plane_distance_difference_m <=
                        kReadyMaximumPlaneDistanceDifferenceM;

                UpdateStaticResidualStatistics(
                    *linked_live_track,
                    pair.normal_difference_deg,
                    pair.plane_distance_difference_m);

                candidate_debug.observation_span_frames =
                    ObservationSpanFrames(
                        *linked_live_track);

                candidate_debug.static_residual_samples =
                    linked_live_track->static_residual_samples;

                candidate_debug.static_normal_rms_deg =
                    StaticNormalRmsDeg(
                        *linked_live_track);

                candidate_debug.static_plane_d_rms_m =
                    StaticPlaneDRmsM(
                        *linked_live_track);

                persistent.last_center_A =
                    candidate.center_A;

                persistent.last_radius_m =
                    candidate.radius_m;

                persistent.last_seen_frame =
                    frame_index;

                persistent.active_this_frame =
                    true;

                persistent.fallback_this_frame =
                    true;

                persistent.reid_pending_valid =
                    false;

                persistent.reid_pending_count =
                    0U;

                ++persistent.total_observations;
                ++persistent.fallback_count;

                candidate_debug.persistent_total_observations =
                    persistent.total_observations;

                candidate_assigned[pair.candidate_index] =
                    true;

                persistent_assigned[pair.persistent_index] =
                    true;

                ++debug.stable_walls;

                if (candidate_debug.constraint_ready)
                {
                    ++debug.active_walls;
                }

                continue;
            }

            // ------------------------------------------------------------
            // Case B: no live owner exists.
            //
            // This is a possible TRUE long-gap re-identification.  Require
            // three consecutive compatible observations before restoring the
            // persistent identity and spawning a new live track.
            // ------------------------------------------------------------
            bool pending_consistent = false;

            if (persistent.reid_pending_valid &&
                frame_index ==
                    (persistent.reid_pending_last_frame + 1U))
            {
                const double pending_normal_difference_deg =
                    WallAssociationV2NormalAngleDeg(
                        aligned_normal_A,
                        persistent.reid_pending_last_normal_A);

                const double pending_plane_distance_difference_m =
                    std::abs(
                        aligned_d_A -
                        persistent.reid_pending_last_d_A);

                const double pending_support_gap_m =
                    WallAssociationV2SupportGap(
                        candidate.center_A,
                        candidate.radius_m,
                        persistent.reid_pending_last_center_A,
                        persistent.reid_pending_last_radius_m,
                        persistent.reference_normal_A);

                pending_consistent =
                    std::isfinite(
                        pending_normal_difference_deg) &&
                    pending_normal_difference_deg <=
                        kReidPendingMaximumNormalDifferenceDeg &&
                    std::isfinite(
                        pending_plane_distance_difference_m) &&
                    pending_plane_distance_difference_m <=
                        kReidPendingMaximumPlaneDistanceDifferenceM &&
                    std::isfinite(
                        pending_support_gap_m) &&
                    pending_support_gap_m <=
                        kReidPendingMaximumTangentialSupportGapM;
            }

            if (pending_consistent)
            {
                ++persistent.reid_pending_count;
            }
            else
            {
                persistent.reid_pending_count = 1U;
            }

            persistent.reid_pending_valid =
                true;

            persistent.reid_pending_last_frame =
                frame_index;

            persistent.reid_pending_last_normal_A =
                aligned_normal_A.normalized();

            persistent.reid_pending_last_d_A =
                aligned_d_A;

            persistent.reid_pending_last_center_A =
                candidate.center_A;

            persistent.reid_pending_last_radius_m =
                candidate.radius_m;

            candidate_debug.persistent_wall_id =
                persistent.id;

            candidate_debug.reid_pending =
                true;

            candidate_debug.reid_pending_count =
                persistent.reid_pending_count;

            candidate_debug.normal_difference_deg =
                pair.normal_difference_deg;

            candidate_debug.plane_distance_difference_m =
                pair.plane_distance_difference_m;

            candidate_debug.tangential_support_gap_m =
                pair.support_gap_m;

            candidate_debug.association_score =
                pair.score;

            candidate_debug.reference_normal_difference_deg =
                pair.normal_difference_deg;

            candidate_debug.reference_plane_distance_difference_m =
                pair.plane_distance_difference_m;

            candidate_assigned[pair.candidate_index] =
                true;

            persistent_assigned[pair.persistent_index] =
                true;

            if (persistent.reid_pending_count <
                kReidConfirmationFrames)
            {
                continue;
            }

            // Three consecutive observations confirmed the dormant wall.
            WallAssociationV2Track track;
            track.id = runtime.next_track_id++;

            if (frame_index + 1U >=
                kReidConfirmationFrames)
            {
                track.first_seen_frame =
                    frame_index + 1U -
                    kReidConfirmationFrames;
            }
            else
            {
                track.first_seen_frame =
                    frame_index;
            }

            track.last_normal_A =
                aligned_normal_A.normalized();

            track.last_d_A =
                aligned_d_A;

            track.last_center_A =
                candidate.center_A;

            track.last_radius_m =
                candidate.radius_m;

            track.stable =
                true;

            track.static_confirmed =
                true;

            track.reference_normal_A =
                persistent.reference_normal_A;

            track.reference_d_A =
                persistent.reference_d_A;

            track.persistent_wall_id =
                persistent.id;

            track.total_hits =
                persistent.reid_pending_count;

            track.consecutive_hits =
                persistent.reid_pending_count;

            track.last_seen_frame =
                frame_index;

            track.missed_frames =
                0U;

            track.matched_this_frame =
                true;

            // Seed new live-track residual diagnostics with the confirming
            // observation.  Persistent identity confidence comes from the
            // persistent wall itself, not from these fresh live statistics.
            track.static_residual_samples =
                1U;

            track.static_normal_squared_sum_deg2 =
                pair.normal_difference_deg *
                pair.normal_difference_deg;

            track.static_plane_d_squared_sum_m2 =
                pair.plane_distance_difference_m *
                pair.plane_distance_difference_m;

            runtime.tracks.push_back(
                track);

            WallAssociationV2Track &reacquired_track =
                runtime.tracks.back();

            candidate_debug.track_id =
                reacquired_track.id;

            candidate_debug.new_track =
                true;

            candidate_debug.stable =
                true;

            candidate_debug.static_confirmed =
                true;

            candidate_debug.persistent_reacquired =
                true;

            candidate_debug.reid_pending =
                false;

            candidate_debug.reid_pending_count =
                0U;

            candidate_debug.persistent_wall_id =
                persistent.id;

            candidate_debug.consecutive_hits =
                reacquired_track.consecutive_hits;

            candidate_debug.total_hits =
                reacquired_track.total_hits;

            candidate_debug.constraint_ready =
                candidate.quality >=
                    kReadyMinimumQuality &&
                pair.normal_difference_deg <=
                    kReadyMaximumNormalDifferenceDeg &&
                pair.plane_distance_difference_m <=
                    kReadyMaximumPlaneDistanceDifferenceM;

            candidate_debug.observation_span_frames =
                ObservationSpanFrames(
                    reacquired_track);

            candidate_debug.static_residual_samples =
                reacquired_track.static_residual_samples;

            candidate_debug.static_normal_rms_deg =
                StaticNormalRmsDeg(
                    reacquired_track);

            candidate_debug.static_plane_d_rms_m =
                StaticPlaneDRmsM(
                    reacquired_track);

            persistent.last_center_A =
                candidate.center_A;

            persistent.last_radius_m =
                candidate.radius_m;

            persistent.last_seen_frame =
                frame_index;

            persistent.active_this_frame =
                true;

            persistent.reacquired_this_frame =
                true;

            persistent.total_observations +=
                persistent.reid_pending_count;

            ++persistent.reacquisition_count;

            persistent.reid_pending_valid =
                false;

            persistent.reid_pending_count =
                0U;

            candidate_debug.persistent_total_observations =
                persistent.total_observations;

            ++debug.new_tracks;
            ++debug.stable_walls;

            if (candidate_debug.constraint_ready)
            {
                ++debug.active_walls;
            }
        }

        // Spawn ordinary transient live tracks only after persistent re-ID had
        // a chance to reclaim an old physical wall identity.
        for (std::size_t candidate_index = 0;
             candidate_index < debug.physical_walls.size();
             ++candidate_index)
        {
            if (candidate_assigned[candidate_index])
            {
                continue;
            }

            const WallAssociationV2PhysicalWall &candidate =
                debug.physical_walls[candidate_index];

            WallAssociationV2CandidateDebug &candidate_debug =
                debug.candidate_debug[candidate_index];

            if (!std::isfinite(candidate.quality) ||
                candidate.quality < kNewTrackMinimumQuality)
            {
                candidate_debug.low_quality_untracked = true;
                continue;
            }

            WallAssociationV2Track track;
            track.id = runtime.next_track_id++;
            track.first_seen_frame = frame_index;
            track.last_normal_A = candidate.normal_A;
            track.last_d_A = candidate.d_A;
            track.last_center_A = candidate.center_A;
            track.last_radius_m = candidate.radius_m;
            track.confirmation_normal_sum_A = candidate.normal_A;
            track.confirmation_d_sum_A = candidate.d_A;
            track.confirmation_samples = 1U;
            track.total_hits = 1U;
            track.consecutive_hits = 1U;
            track.last_seen_frame = frame_index;
            track.missed_frames = 0U;
            track.matched_this_frame = true;

            runtime.tracks.push_back(track);

            candidate_debug.track_id = track.id;
            candidate_debug.new_track = true;
            candidate_debug.consecutive_hits = 1U;
            candidate_debug.total_hits = 1U;
            candidate_debug.observation_span_frames = 1U;

            ++debug.new_tracks;
        }

        for (WallAssociationV2Track &track : runtime.tracks)
        {
            if (!track.matched_this_frame)
            {
                ++track.missed_frames;
                track.consecutive_hits = 0U;

                if (!track.stable)
                {
                    track.confirmation_normal_sum_A =
                        Eigen::Vector3d::Zero();
                    track.confirmation_d_sum_A = 0.0;
                    track.confirmation_samples = 0U;
                }
            }

            if (track.stable)
            {
                ++debug.stable_tracks_alive;
            }
        }

        debug.tracks_alive = runtime.tracks.size();
        debug.persistent_walls = runtime.persistent_walls.size();

        for (const WallAssociationV2CandidateDebug &candidate_debug :
             debug.candidate_debug)
        {
            if (candidate_debug.track_id !=
                    std::numeric_limits<std::size_t>::max() &&
                candidate_debug.static_confirmed)
            {
                ++debug.static_confirmed_current;
            }
            else if (candidate_debug.track_id !=
                         std::numeric_limits<std::size_t>::max() &&
                     !candidate_debug.low_quality_untracked)
            {
                ++debug.transient_current;
            }

            if (candidate_debug.reid_pending)
            {
                ++debug.reid_pending_current;
            }
        }

        for (const WallAssociationV3PersistentWall &persistent :
             runtime.persistent_walls)
        {
            if (persistent.active_this_frame)
            {
                bool ready_this_frame = false;

                for (const WallAssociationV2CandidateDebug &candidate_debug :
                     debug.candidate_debug)
                {
                    if (candidate_debug.persistent_wall_id == persistent.id &&
                        candidate_debug.constraint_ready)
                    {
                        ready_this_frame = true;
                        break;
                    }
                }

                if (ready_this_frame)
                {
                    ++debug.active_static_walls;
                }
            }
            else
            {
                ++debug.dormant_walls;
            }

            if (persistent.fallback_this_frame)
            {
                ++debug.fallback_this_frame;
            }

            if (persistent.reacquired_this_frame)
            {
                ++debug.reacquired_this_frame;
            }

            if (persistent.rebound_this_frame)
            {
                ++debug.rebound_this_frame;
            }
        }

        return debug;
    }

} // namespace



struct WallAssociation::Impl
{
    WallAssociationV2Runtime runtime;

    // Default false preserves existing Local Wall V1 behavior.
    bool horizontal_fragment_merge_enabled = false;
};

WallAssociation::WallAssociation()
    : impl_(std::make_unique<Impl>())
{
}

WallAssociation::~WallAssociation() = default;

WallAssociation::WallAssociation(
    WallAssociation &&other) noexcept = default;

WallAssociation &WallAssociation::operator=(
    WallAssociation &&other) noexcept = default;

void WallAssociation::Reset()
{
    impl_->runtime =
        WallAssociationV2Runtime();
}

void WallAssociation::SetHorizontalFragmentMergeEnabled(
    const bool enabled)
{
    if (!impl_)
    {
        return;
    }

    impl_->horizontal_fragment_merge_enabled =
        enabled;
}

WallAssociationResult WallAssociation::Update(
    const MultiPlaneExtractionResult &planes,
    const Eigen::Isometry3d &T_AL,
    std::size_t frame_index)
{
    WallAssociationResult result;

    if (!impl_ ||
        !T_AL.matrix().allFinite())
    {
        return result;
    }

    const WallAssociationV2FrameDebug debug =
        RunWallAssociationV3(
            planes,
            T_AL,
            frame_index,
            impl_->runtime,
            impl_->horizontal_fragment_merge_enabled);

    result.raw_wall_candidates =
        debug.raw_wall_candidates;

    result.physical_wall_candidates =
        debug.physical_wall_candidates;

    result.persistent_walls =
        debug.persistent_walls;

    for (std::size_t candidate_index = 0;
         candidate_index < debug.candidate_debug.size();
         ++candidate_index)
    {
        if (candidate_index >=
            debug.physical_walls.size())
        {
            continue;
        }

        const WallAssociationV2CandidateDebug &association =
            debug.candidate_debug[candidate_index];

        const bool has_persistent_wall =
            association.persistent_wall_id !=
            std::numeric_limits<std::size_t>::max();

        const bool active_static =
            has_persistent_wall &&
            association.constraint_ready &&
            !association.reid_pending;

        if (!active_static)
        {
            continue;
        }

        const WallAssociationV3PersistentWall *persistent =
            nullptr;

        for (const WallAssociationV3PersistentWall &candidate :
             impl_->runtime.persistent_walls)
        {
            if (candidate.id ==
                association.persistent_wall_id)
            {
                persistent = &candidate;
                break;
            }
        }

        if (persistent == nullptr ||
            !persistent->active_this_frame)
        {
            continue;
        }

        const WallAssociationV2PhysicalWall &observed =
            debug.physical_walls[candidate_index];

        if (!persistent->reference_normal_A.allFinite() ||
            persistent->reference_normal_A.norm() < 1.0e-9 ||
            !std::isfinite(persistent->reference_d_A) ||
            !observed.normal_A.allFinite() ||
            observed.normal_A.norm() < 1.0e-9 ||
            !std::isfinite(observed.d_A))
        {
            continue;
        }

        Eigen::Vector3d observed_normal_A =
            observed.normal_A;

        double observed_d_A =
            observed.d_A;

        if (observed_normal_A.dot(
                persistent->reference_normal_A) < 0.0)
        {
            observed_normal_A =
                -observed_normal_A;

            observed_d_A =
                -observed_d_A;
        }

        ActiveWallAssociation output;

        output.persistent_wall_id =
            persistent->id;

        output.reference_normal_A =
            persistent->reference_normal_A.normalized();

        output.reference_d_A =
            persistent->reference_d_A;

        output.observed_normal_A =
            observed_normal_A.normalized();

        output.observed_d_A =
            observed_d_A;

        output.observed_center_A =
            observed.center_A;

        output.observed_radius_m =
            observed.radius_m;

        output.quality =
            observed.quality;

        output.normal_difference_deg =
            association.reference_normal_difference_deg;

        output.plane_distance_difference_m =
            association.reference_plane_distance_difference_m;

        result.active_static_walls.push_back(
            output);
    }

    result.active_static_wall_count =
        result.active_static_walls.size();

    return result;
}

} // namespace fr_slam
