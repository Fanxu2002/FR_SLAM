#include "fr_slam/frontend/fr_lidar_frontend.hpp"
#include "fr_slam/frontend/fr_ground_segmenter.hpp"
#include "fr_slam/frontend/fr_ground_input_bridge.hpp"
#include "fr_slam/loop/fr_loop_retrieval_debug.hpp"
#include "fr_slam/loop/fr_loop_decision_debug.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <cstdint>
#include <cstddef>
#include <exception>
#include <utility>
#include <vector>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>
#include <pcl/search/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <Eigen/Eigenvalues>

#include <sophus/so3.hpp>

#include <rclcpp/rclcpp.hpp>
namespace
{
    const rclcpp::Logger kTimingLogger =
        rclcpp::get_logger("scan2local_map.timing");

    std::filesystem::path FrSlamOutputDirectory()
    {
        const char *configured_directory =
            std::getenv("FR_SLAM_OUTPUT_DIR");

        if (configured_directory != nullptr &&
            configured_directory[0] != '\0')
        {
            return std::filesystem::path(
                configured_directory);
        }

        const char *home_directory =
            std::getenv("HOME");

        if (home_directory != nullptr &&
            home_directory[0] != '\0')
        {
            return std::filesystem::path(
                       home_directory) /
                   "ros2_ws" /
                   "src" /
                   "fr_slam" /
                   "output";
        }

        return std::filesystem::path(
            "/tmp/fr_slam_output");
    }

    std::filesystem::path FrontendLoopDirectory()
    {
        return FrSlamOutputDirectory() /
               "loop";
    }

    Eigen::Vector3d FrontendRotationToRpy(
        const Eigen::Matrix3d &R)
    {
        const double sy =
            std::sqrt(
                R(0, 0) * R(0, 0) +
                R(1, 0) * R(1, 0));

        const bool singular =
            sy < 1.0e-8;

        double roll = 0.0;
        double pitch = 0.0;
        double yaw = 0.0;

        if (!singular)
        {
            roll =
                std::atan2(
                    R(2, 1),
                    R(2, 2));

            pitch =
                std::atan2(
                    -R(2, 0),
                    sy);

            yaw =
                std::atan2(
                    R(1, 0),
                    R(0, 0));
        }
        else
        {
            roll =
                std::atan2(
                    -R(1, 2),
                    R(1, 1));

            pitch =
                std::atan2(
                    -R(2, 0),
                    sy);

            yaw = 0.0;
        }

        return Eigen::Vector3d(
            roll,
            pitch,
            yaw);
    }

    double ElapsedMilliseconds(
        const std::chrono::steady_clock::time_point &start,
        const std::chrono::steady_clock::time_point &end)
    {
        return std::chrono::duration<double, std::milli>(
                   end - start)
            .count();
    }

    struct LoopTimingDiagnostics
    {
        std::chrono::steady_clock::time_point start =
            std::chrono::steady_clock::now();

        std::size_t current_keyframe_id = 0;
        std::size_t current_submap_id = 0;
        std::size_t candidates = 0;
        std::size_t verifier_prescore_calls = 0;
        std::size_t verifier_calls = 0;
        std::size_t pose_graph_optimize_calls = 0;

        double scan_context_detect_ms = 0.0;
        double verifier_prescore_ms = 0.0;
        double verifier_ms = 0.0;
        double pose_graph_optimize_ms = 0.0;
        double map_odom_ms = 0.0;
        double global_map_rebuild_ms = 0.0;
        double refinement_ms = 0.0;

        bool geometry_accepted = false;
        bool loop_edge_accepted = false;
        bool optimization_accepted = false;
    };

    class LoopTimingReporter
    {
    public:
        explicit LoopTimingReporter(
            LoopTimingDiagnostics &diagnostics)
            : diagnostics_(diagnostics)
        {
        }

        ~LoopTimingReporter()
        {
            const double total_ms =
                ElapsedMilliseconds(
                    diagnostics_.start,
                    std::chrono::steady_clock::now());

            RCLCPP_INFO(
                kTimingLogger,
                "FR_TIMING LOOP_BACKEND"
                " | current_kf=%zu"
                " | current_submap=%zu"
                " | total=%.3f ms"
                " | scan_context=%.3f"
                " | candidates=%zu"
                " | prescore=%.3f"
                " | prescore_calls=%zu"
                " | verifier=%.3f"
                " | verifier_calls=%zu"
                " | pgo=%.3f"
                " | pgo_calls=%zu"
                " | map_odom=%.3f"
                " | global_map=%.3f"
                " | refinement=%.3f"
                " | geometry_accepted=%s"
                " | loop_edge_accepted=%s"
                " | optimization_accepted=%s",
                diagnostics_.current_keyframe_id,
                diagnostics_.current_submap_id,
                total_ms,
                diagnostics_.scan_context_detect_ms,
                diagnostics_.candidates,
                diagnostics_.verifier_prescore_ms,
                diagnostics_.verifier_prescore_calls,
                diagnostics_.verifier_ms,
                diagnostics_.verifier_calls,
                diagnostics_.pose_graph_optimize_ms,
                diagnostics_.pose_graph_optimize_calls,
                diagnostics_.map_odom_ms,
                diagnostics_.global_map_rebuild_ms,
                diagnostics_.refinement_ms,
                diagnostics_.geometry_accepted ? "true" : "false",
                diagnostics_.loop_edge_accepted ? "true" : "false",
                diagnostics_.optimization_accepted ? "true" : "false");
        }

    private:
        LoopTimingDiagnostics &diagnostics_;
    };

    // ========================================================================
    // V20.2 Reverse RAW-SC sequence seed.
    //
    // This state intentionally lives in this translation unit so the reverse
    // traversal experiment does not require a public-header / ABI change.
    //
    // It is PRE-verification evidence only:
    //   * raw Scan Context must keep retrieving a nearby historical Keyframe;
    //   * frontend positions must already be locally close;
    //   * frontend headings must be approximately opposite;
    //   * historical Keyframe IDs must make real negative progress while
    //     current Keyframe IDs advance.
    //
    // The state NEVER creates a loop edge and NEVER bypasses ICP / graph / PGO
    // guards.  It only enables a safer reverse-traversal verification target.
    // ========================================================================
    struct ReverseLoopSeedTrackState
    {
        bool valid = false;

        std::size_t last_current_keyframe_id = 0;
        std::size_t last_historical_keyframe_id = 0;

        std::size_t support = 0;
        std::size_t reverse_progress_events = 0;

        // V2.3 stable-neighborhood evidence.
        //
        // Scan Context top-1 can quantize between adjacent historical
        // Keyframes on repetitive straight rows.  Track a separate local
        // historical neighborhood so one genuine reverse progress event plus
        // several observations in the same +/-1 KF region can confirm the
        // reverse traversal without requiring two strictly negative ID steps.
        std::size_t stable_center_historical_keyframe_id = 0;
        std::size_t stable_neighborhood_support = 0;
    };

    // ========================================================================
    // V22.6 Segment-track revalidation state.
    //
    // TRACK_PREDICTED / KEYFRAME_CENTERED verification is allowed only as a
    // short-range continuation mechanism.  It must not refresh itself forever
    // in a repetitive agricultural row.  After the current trajectory has
    // advanced another ~3m, the track is forced back through independent
    // segment-level retrieval + 3m<->3m geometry before it can continue.
    //
    // Translation-unit state avoids a public-header / ABI change, consistent
    // with the existing reverse-sequence experiment state above.
    // ========================================================================
    struct LoopTrackSegmentRevalidationStateV226
    {
        bool valid = false;
        std::size_t checkpoint_current_keyframe_id = 0;
        std::size_t checkpoint_historical_keyframe_id = 0;
        bool checkpoint_was_segment_verified = false;
    };

    constexpr std::size_t kReverseLoopSeedMinSupport = 3;
    constexpr std::size_t kReverseLoopSeedMinProgressEvents = 2;

    // V2.3 alternative confirmation for quantized/repetitive SC retrieval.
    constexpr std::size_t kReverseLoopSeedStableMinSupport = 4;
    constexpr std::size_t kReverseLoopSeedStableMinProgressEvents = 1;
    constexpr std::size_t kReverseLoopSeedStableHistoricalRadius = 1;
    constexpr double kReverseLoopSeedStableMaxFrontendDistance = 2.0;
    constexpr std::size_t kReverseLoopSeedMaxCurrentGap = 2;
    constexpr std::size_t kReverseLoopSeedMaxHistoricalStep = 4;

    // A +1 historical-KF fluctuation is treated as SC representative jitter.
    // It is preserved but is NOT allowed to advance the reverse seed track.
    constexpr std::int64_t kReverseLoopSeedForwardJitter = 1;

    // This mode is deliberately local.  It must not reopen the old
    // long-distance false-loop failure mode.
    constexpr double kReverseLoopSeedMaxFrontendDistance = 3.5;
    constexpr double kReverseLoopSeedMinYawDeg = 150.0;

    // V2.5 Reverse Spatial Candidate Injection.
    //
    // Scan Context remains the primary global place-recognition route.
    // This supplemental route only proposes sufficiently old historical KFs
    // that the FRONTEND already places very close to the current KF while the
    // headings are approximately opposite.  The injected candidate still has
    // to pass the complete compact-target Point-to-Plane / Hessian / graph /
    // temporal / PGO pipeline.
    constexpr std::size_t kReverseSpatialMaxCandidates = 4;

    // ========================================================================
    // V21 Hierarchical 3m Scan Context retrieval.
    //
    // Design principle:
    //   1) Region-level retrieval is HIGH RECALL and never uses the old
    //      single-Keyframe SC distance gate as a hard reject.
    //   2) A rolling current 3m Retrieval Window is compared against historical
    //      3m forward/backward Retrieval Windows.
    //   3) Region hits vote for finished historical Submaps.
    //   4) Inside the best Submaps, current single-KF SC ranks historical KFs.
    //   5) Ranked KFs are expanded by +/-2 KFs before the existing prescore /
    //      ICP / graph / temporal pipeline.
    //
    // Mapping Submaps and PoseGraph topology are unchanged: the final factor is
    // still Keyframe-to-Keyframe.  These constants intentionally live in this
    // .cpp so V21 can be tested without changing the public header / ABI.
    // ========================================================================
    constexpr double kHierarchicalRetrievalArcLengthM = 3.0;
    constexpr std::size_t kHierarchicalRetrievalMinKeyframes = 3;
    constexpr std::size_t kHierarchicalRetrievalMaxKeyframes = 30;

    // Compare against every valid historical 3m window, then aggregate by
    // historical Submap.  The first hard truncation happens only AFTER region
    // aggregation, which avoids losing a true Submap merely because none of
    // its individual windows landed in a global Top-N list.
    constexpr std::size_t kHierarchicalCandidateSubmaps = 5;
    constexpr std::size_t kHierarchicalRegionScoreBestWindows = 3;

    // Second-level KF retrieval inside each selected Submap.
    constexpr std::size_t kHierarchicalKeyframeSeedsPerSubmap = 3;
    constexpr std::size_t kHierarchicalKeyframeNeighborhoodRadius = 2;

    // The existing Candidate Manager normally verifies one KF per historical
    // Submap.  V21 deliberately allows a few KF hypotheses from a region that
    // was independently retrieved at Region-SC level.
    constexpr std::size_t kHierarchicalSameSubmapVerifyBudget = 3;
    constexpr std::size_t kHierarchicalExtraVerifyBudget = 0;

    // V22.6: keep discovery / verification units consistent.  Hierarchical
    // retrieval is built from 3m trajectory segments, therefore an independent
    // hierarchical candidate must also be verified with a 3m target segment.
    // A mature window may overshoot 3m by one Keyframe; 90% tolerates the
    // opposite discretization case without falling back to KF-centered ICP.
    constexpr double kHierarchicalSegmentGeometryMinArcRatio = 0.90;

    // V22.6: a track may use cheap KF-centered continuation only until the
    // current trajectory advances this physical distance.  Then old-track
    // injection / priority is disabled until independent segment geometry
    // succeeds again.
    constexpr double kLoopTrackSegmentRevalidationArcM = 3.0;

    // V22.7 post-loop SAME-TRACK graph guard.  The generic graph gate must
    // remain permissive enough to discover a genuinely new loop cluster after
    // accumulated drift, but an already-established local loop track should
    // not be allowed to wander several metres while still producing yellow
    // TRACK_ONLY observations.
    constexpr double kPostLoopSameTrackGraphMaxTranslationM = 1.5;
    constexpr double kPostLoopSameTrackGraphMaxRotationDeg = 10.0;

    // V22.7 representative-KF selection.  Segment geometry is still solved in
    // its anchor frame, but the PoseGraph/debug endpoint is re-anchored to the
    // historical member whose frontend position is closest to the loop-implied
    // current pose.  This prevents a 3m segment anchor (e.g. KF44) from being
    // visualized as if every observation physically corresponded to KF44.
    constexpr bool kUseSegmentRepresentativeKeyframeV227 = true;

    // V22.8 segment-continuation revalidation.
    //
    // Independent retrieval is the strongest way to renew a 3m-old track, but
    // it is intentionally high-recall rather than temporally dense.  Therefore
    // a mature, graph-consistent track may renew itself at the 3m deadline when
    // a FULL segment geometry observation also shows real physical progression
    // on the historical trajectory.  This is deliberately much stricter than
    // ordinary one-frame tracking and prevents a sticky representative KF from
    // buying another 3m of trust.
    constexpr double kLoopTrackContinuationMinHistoricalArcRatio = 0.25;
    constexpr double kLoopTrackContinuationMaxHistoricalArcRatio = 2.50;
    constexpr double kLoopTrackContinuationMaxRepresentativeDistanceM = 2.0;

    // V22.8 separates TRACK visualization confidence from PGO-factor
    // confidence.  PGO keeps the existing strict cycle gate (typically 3 deg),
    // while an already established SAME_CLUSTER track may remain visible as a
    // yellow observation under this wider *track-only* cycle bound.  Such an
    // observation can never enter the graph.
    constexpr double kLoopTrackRelaxedCycleMaxTranslationM = 0.50;
    constexpr double kLoopTrackRelaxedCycleMaxRotationDeg = 8.0;

    // V22 performance guard.  Candidate generation and cheap prescore remain
    // high-recall, but the expensive full LoopVerifier/ICP stage has a strict
    // per-current-Keyframe budget.  This prevents V21 from expanding into
    // 50~70 full ICP calls in repetitive agricultural rows.
    constexpr std::size_t kFullVerifierCallBudgetPerCurrentKeyframe = 5;
    constexpr std::size_t kFullVerifierCallsPerCandidate = 2;

    // V2.5 Trusted Reverse Temporal Continuity.
    //
    // The ordinary temporal track keeps its existing 10 deg bound.  A
    // confirmed reverse-sequence observation that also passes the strict
    // trusted reverse geometry gates may use a slightly wider local rotation
    // disagreement bound.  This is intentionally below the verifier's
    // trusted reverse single-observation cap (15 deg).
    constexpr double kTrustedReverseTemporalRotationErrorDeg = 12.0;

    // ========================================================================
    // V2.4 Post-loop cluster continuity.
    //
    // The old V10 cycle gate compared EVERY future loop against the most
    // recently inserted loop edge.  That is correct only while both edges
    // belong to the same local revisit cluster.  Once the vehicle reaches a
    // physically different revisit region, the previous loop must no longer be
    // a permanent SE(3) cycle anchor.
    //
    // V2.4 separates:
    //
    //   SAME_CLUSTER:
    //       keep the existing strict loop-to-loop cycle gate.
    //
    //   NEW_CLUSTER:
    //       do not compare against the remote old loop.  Require several
    //       consecutive, locally consistent verified observations before one
    //       factor is allowed to establish the new cluster.
    // ========================================================================
    struct PostLoopClusterTrackState
    {
        bool valid = false;
        std::size_t last_current_keyframe_id = 0;
        std::size_t last_historical_keyframe_id = 0;
        std::size_t support = 0;
        Eigen::Isometry3d T_loop_correction =
            Eigen::Isometry3d::Identity();
    };

    PostLoopClusterTrackState g_post_loop_new_cluster_track;

    // ~0.5 m/keyframe in the present setup: keep same-cluster cycle checks local.
    constexpr std::size_t kPostLoopSameClusterMaxCurrentGap = 20;
    constexpr std::size_t kPostLoopSameClusterMaxHistoricalGap = 12;

    // Independent new-cluster confirmation.
    constexpr std::size_t kPostLoopNewClusterMinSupport = 3;
    constexpr std::size_t kPostLoopNewClusterMaxCurrentGap = 2;
    constexpr std::size_t kPostLoopNewClusterMaxHistoricalGap = 4;
    constexpr double kPostLoopNewClusterMaxTranslationError = 1.0;
    constexpr double kPostLoopNewClusterMaxRotationErrorDeg = 8.0;

    double RelativeRotationDeg(
        const Eigen::Isometry3d &T_A,
        const Eigen::Isometry3d &T_B)
    {
        if (!T_A.matrix().allFinite() ||
            !T_B.matrix().allFinite())
        {
            return std::numeric_limits<double>::quiet_NaN();
        }

        const Eigen::Matrix3d R_AB =
            T_A.rotation().transpose() *
            T_B.rotation();

        Eigen::Quaterniond q_AB(R_AB);

        if (!q_AB.coeffs().allFinite() ||
            q_AB.norm() < 1.0e-12)
        {
            return std::numeric_limits<double>::quiet_NaN();
        }

        q_AB.normalize();

        // q and -q represent the same rotation.
        const double w =
            std::clamp(
                std::abs(q_AB.w()),
                0.0,
                1.0);

        return 2.0 *
               std::acos(w) *
               180.0 /
               M_PI;
    }

    // ============================================================================
    // ConvertToXYZ()
    //
    // Convert the project's LIDAR_POINT cloud into pcl::PointXYZ.
    //
    // Why:
    //     The main point-to-plane registration can keep using the project's custom
    //     point type, but this recovery experiment deliberately uses the standard
    //     PCL PointXYZ type for coarse point-to-point ICP.
    //
    // This also avoids unnecessary template / registration complications around
    // custom point types inside PCL's ICP implementation.
    // ============================================================================
    pcl::PointCloud<pcl::PointXYZ>::Ptr ConvertToXYZ(
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &cloud)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr xyz_cloud(
            new pcl::PointCloud<pcl::PointXYZ>);

        if (!cloud)
        {
            return xyz_cloud;
        }

        xyz_cloud->reserve(cloud->size());

        for (const LIDAR_POINT &point : cloud->points)
        {
            if (!std::isfinite(point.x) ||
                !std::isfinite(point.y) ||
                !std::isfinite(point.z))
            {
                continue;
            }

            pcl::PointXYZ xyz;
            xyz.x = point.x;
            xyz.y = point.y;
            xyz.z = point.z;

            xyz_cloud->push_back(xyz);
        }

        xyz_cloud->width =
            static_cast<std::uint32_t>(xyz_cloud->size());

        xyz_cloud->height = 1;
        xyz_cloud->is_dense = true;

        return xyz_cloud;
    }

    // ========================================================================
    // Loop Shadow Point-to-Plane -> Full 6x6 information.
    //
    // This production helper deliberately lives in this .cpp so the public
    // LoopVerifier / RegistrationScan2LocalMap headers do not need new ABI.
    // P2P ICP still owns the loop pose.  Only an edge that has survived the
    // existing loop gates uses this shadow geometry to build its information.
    //
    // Output convention:
    //     order = [tx ty tz rx ry rz]
    //     frame = current/source LiDAR (g2o to-node frame)
    // ========================================================================
    constexpr int kLoopInformationPlaneKnn = 5;
    constexpr double kLoopInformationMaxPlaneFitError = 0.15;
    constexpr double kLoopInformationMinimumScaleRange = 1.0;
    constexpr double kLoopInformationMaximumScaleRange = 50.0;
    constexpr double kLoopInformationRelativeEigenvalueFloor = 0.01;
    constexpr double kLoopInformationMinimumDirectionalConfidence = 0.01;
    constexpr std::size_t kLoopInformationMinimumCorrespondences = 50;

    pcl::PointCloud<pcl::PointXYZ>::Ptr VoxelFilterLoopInformation(
        const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud,
        double leaf_size)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(
            new pcl::PointCloud<pcl::PointXYZ>);

        if (!cloud ||
            cloud->empty() ||
            !std::isfinite(leaf_size) ||
            leaf_size <= 0.0)
        {
            return filtered;
        }

        pcl::VoxelGrid<pcl::PointXYZ> voxel;
        voxel.setInputCloud(cloud);

        const float leaf =
            static_cast<float>(leaf_size);

        voxel.setLeafSize(
            leaf,
            leaf,
            leaf);

        voxel.filter(*filtered);

        return filtered;
    }

    // ========================================================================
    // V22.4 3D visibility-aware overlap for trusted reverse segment geometry.
    //
    // Why V22.4:
    //   V22.3 only estimated a horizontal azimuth FOV.  That helps with a
    //   laptop/body blind wedge, but it still cannot distinguish:
    //     * a direction that is physically outside the effective LiDAR view,
    //     * a point hidden behind a nearer return,
    //     * a point that should genuinely have been visible.
    //
    // V22.4 keeps a spherical range image for every member Keyframe scan and
    // evaluates visibility in the ORIGINAL LiDAR frame of each scan:
    //
    //   source point in target-anchor frame
    //       -> transform into each target member LiDAR frame
    //       -> sensor azimuth / elevation check
    //       -> spherical z-buffer lookup
    //       -> if a nearer return exists: OCCLUDED -> ignore from denominator
    //       -> otherwise: EXPECTED_VISIBLE -> include in denominator
    //
    // This is the natural LiDAR equivalent of image z-buffer visibility and is
    // much closer to full ray visibility than a fused-cloud nearest-neighbor
    // overlap.  The wide persistent azimuth blind sector is still learned from
    // the raw member scans so a laptop/body self-occlusion does not become a
    // false mismatch after close-range points have been filtered out.
    //
    // IMPORTANT:
    //   Geometry correspondence is STILL checked against the fused 3m target
    //   with the existing verification_inlier_distance.  Visibility only
    //   decides whether a source point belongs in the denominator.
    // ========================================================================
    constexpr std::size_t kVisibilityAzimuthBins = 360;  // 1 deg / bin
    constexpr std::size_t kVisibilityElevationBins = 90; // 2 deg / bin
    constexpr double kVisibilityMinElevationRad = -0.5 * M_PI;
    constexpr double kVisibilityMaxElevationRad = 0.5 * M_PI;
    constexpr std::size_t kVisibilityMinReturnsPerAzimuthBinPerScan = 2;
    constexpr double kVisibilityMinAzimuthScanSupportRatio = 0.30;
    constexpr std::size_t kVisibilityMinAzimuthScanSupport = 2;
    constexpr std::size_t kVisibilityAzimuthDilationBins = 2;
    constexpr int kVisibilityRangeImageAzimuthSearchRadius = 1;
    constexpr int kVisibilityRangeImageElevationSearchRadius = 1;
    constexpr double kVisibilityOcclusionMarginM = 0.35;
    constexpr double kVisibilityElevationMarginRad = 3.0 * M_PI / 180.0;

    struct SphericalVisibilityImage
    {
        bool valid = false;
        std::size_t point_count = 0;
        double min_elevation_rad = std::numeric_limits<double>::infinity();
        double max_elevation_rad = -std::numeric_limits<double>::infinity();
        std::vector<float> minimum_range;
        std::vector<std::uint8_t> valid_cell;
    };

    struct SphericalVisibilityWindowScan
    {
        std::size_t keyframe_id = 0;

        // Transform a point from the 3m-window anchor frame A into this
        // member-scan LiDAR frame S:
        //
        //     p_S = T_S_A * p_A
        Eigen::Isometry3d T_scan_anchor =
            Eigen::Isometry3d::Identity();

        std::shared_ptr<const SphericalVisibilityImage>
            image;
    };

    struct SphericalVisibilityWindowModel
    {
        bool valid = false;
        std::size_t valid_scans = 0;
        std::size_t required_azimuth_support = 0;
        std::size_t observable_azimuth_bins = 0;

        double min_elevation_rad =
            std::numeric_limits<double>::infinity();

        double max_elevation_rad =
            -std::numeric_limits<double>::infinity();

        std::vector<std::size_t> azimuth_scan_support;
        std::vector<std::uint8_t> observable_azimuth;
        std::vector<SphericalVisibilityWindowScan> scans;
    };

    struct VisibilityAwareOverlapMetrics
    {
        bool valid = false;

        std::size_t source_points = 0;
        std::size_t target_points = 0;

        std::size_t eligible_L_to_K = 0;
        std::size_t eligible_K_to_L = 0;

        std::size_t occluded_L_to_K = 0;
        std::size_t occluded_K_to_L = 0;

        std::size_t unobservable_L_to_K = 0;
        std::size_t unobservable_K_to_L = 0;

        std::size_t inliers_L_to_K = 0;
        std::size_t inliers_K_to_L = 0;

        double eligible_fraction_L_to_K = 0.0;
        double eligible_fraction_K_to_L = 0.0;

        double raw_overlap_L_to_K = 0.0;
        double raw_overlap_K_to_L = 0.0;

        double visible_overlap_L_to_K = 0.0;
        double visible_overlap_K_to_L = 0.0;

        double arithmetic_mean = 0.0;
        double harmonic_mean = 0.0;
    };

    bool VisibilityAzimuthBin(
        double azimuth_rad,
        std::size_t &bin_index)
    {
        if (!std::isfinite(azimuth_rad))
        {
            return false;
        }

        const double wrapped =
            std::atan2(
                std::sin(azimuth_rad),
                std::cos(azimuth_rad));

        const double normalized =
            (wrapped + M_PI) /
            (2.0 * M_PI);

        const double scaled =
            normalized *
            static_cast<double>(
                kVisibilityAzimuthBins);

        std::size_t index =
            static_cast<std::size_t>(
                std::floor(scaled));

        if (index >= kVisibilityAzimuthBins)
        {
            index = kVisibilityAzimuthBins - 1;
        }

        bin_index = index;
        return true;
    }

    bool VisibilityElevationBin(
        double elevation_rad,
        std::size_t &bin_index)
    {
        if (!std::isfinite(elevation_rad) ||
            elevation_rad < kVisibilityMinElevationRad ||
            elevation_rad > kVisibilityMaxElevationRad)
        {
            return false;
        }

        const double normalized =
            (elevation_rad -
             kVisibilityMinElevationRad) /
            (kVisibilityMaxElevationRad -
             kVisibilityMinElevationRad);

        const double scaled =
            normalized *
            static_cast<double>(
                kVisibilityElevationBins);

        std::size_t index =
            static_cast<std::size_t>(
                std::floor(scaled));

        if (index >= kVisibilityElevationBins)
        {
            index = kVisibilityElevationBins - 1;
        }

        bin_index = index;
        return true;
    }

    bool VisibilitySphericalBin(
        const Eigen::Vector3d &point,
        std::size_t &azimuth_bin,
        std::size_t &elevation_bin,
        double &range_m,
        double &elevation_rad)
    {
        if (!point.allFinite())
        {
            return false;
        }

        range_m = point.norm();

        if (!std::isfinite(range_m) ||
            range_m <= 1.0e-3)
        {
            return false;
        }

        const double horizontal_range =
            std::hypot(
                point.x(),
                point.y());

        const double azimuth_rad =
            std::atan2(
                point.y(),
                point.x());

        elevation_rad =
            std::atan2(
                point.z(),
                horizontal_range);

        return VisibilityAzimuthBin(
                   azimuth_rad,
                   azimuth_bin) &&
               VisibilityElevationBin(
                   elevation_rad,
                   elevation_bin);
    }

    bool BuildSphericalVisibilityImage(
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &cloud,
        SphericalVisibilityImage &image)
    {
        image = SphericalVisibilityImage();

        if (!cloud || cloud->empty())
        {
            return false;
        }

        const std::size_t cell_count =
            kVisibilityAzimuthBins *
            kVisibilityElevationBins;

        image.minimum_range.assign(
            cell_count,
            std::numeric_limits<float>::infinity());

        image.valid_cell.assign(
            cell_count,
            0);

        for (const LIDAR_POINT &point :
             cloud->points)
        {
            const Eigen::Vector3d p(
                static_cast<double>(point.x),
                static_cast<double>(point.y),
                static_cast<double>(point.z));

            std::size_t azimuth_bin = 0;
            std::size_t elevation_bin = 0;
            double range_m = 0.0;
            double elevation_rad = 0.0;

            if (!VisibilitySphericalBin(
                    p,
                    azimuth_bin,
                    elevation_bin,
                    range_m,
                    elevation_rad))
            {
                continue;
            }

            const std::size_t index =
                elevation_bin *
                    kVisibilityAzimuthBins +
                azimuth_bin;

            const float range_f =
                static_cast<float>(range_m);

            if (image.valid_cell[index] == 0 ||
                range_f < image.minimum_range[index])
            {
                image.minimum_range[index] =
                    range_f;

                image.valid_cell[index] = 1;
            }

            ++image.point_count;

            image.min_elevation_rad =
                std::min(
                    image.min_elevation_rad,
                    elevation_rad);

            image.max_elevation_rad =
                std::max(
                    image.max_elevation_rad,
                    elevation_rad);
        }

        image.valid =
            image.point_count >= 20 &&
            std::isfinite(image.min_elevation_rad) &&
            std::isfinite(image.max_elevation_rad) &&
            image.min_elevation_rad <=
                image.max_elevation_rad;

        return image.valid;
    }

    bool QuerySphericalVisibilityRange(
        const SphericalVisibilityImage &image,
        const Eigen::Vector3d &point_in_scan,
        double &measured_range_m)
    {
        measured_range_m =
            std::numeric_limits<double>::infinity();

        if (!image.valid)
        {
            return false;
        }

        std::size_t azimuth_bin = 0;
        std::size_t elevation_bin = 0;
        double query_range_m = 0.0;
        double elevation_rad = 0.0;

        if (!VisibilitySphericalBin(
                point_in_scan,
                azimuth_bin,
                elevation_bin,
                query_range_m,
                elevation_rad))
        {
            return false;
        }

        bool found = false;

        for (int de =
                 -kVisibilityRangeImageElevationSearchRadius;
             de <=
             kVisibilityRangeImageElevationSearchRadius;
             ++de)
        {
            const int elevation_index =
                static_cast<int>(elevation_bin) +
                de;

            if (elevation_index < 0 ||
                elevation_index >=
                    static_cast<int>(
                        kVisibilityElevationBins))
            {
                continue;
            }

            for (int da =
                     -kVisibilityRangeImageAzimuthSearchRadius;
                 da <=
                 kVisibilityRangeImageAzimuthSearchRadius;
                 ++da)
            {
                const int wrapped_azimuth =
                    (static_cast<int>(azimuth_bin) +
                     da +
                     static_cast<int>(
                         kVisibilityAzimuthBins)) %
                    static_cast<int>(
                        kVisibilityAzimuthBins);

                const std::size_t index =
                    static_cast<std::size_t>(
                        elevation_index) *
                        kVisibilityAzimuthBins +
                    static_cast<std::size_t>(
                        wrapped_azimuth);

                if (image.valid_cell[index] == 0)
                {
                    continue;
                }

                const double candidate_range =
                    static_cast<double>(
                        image.minimum_range[index]);

                if (!std::isfinite(candidate_range))
                {
                    continue;
                }

                measured_range_m =
                    std::min(
                        measured_range_m,
                        candidate_range);

                found = true;
            }
        }

        return found;
    }

    std::size_t CountDirectionalLoopOverlapInliers(
        const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &source,
        const pcl::search::KdTree<pcl::PointXYZ>::Ptr &target_kdtree,
        const Eigen::Isometry3d &T_target_source,
        double inlier_distance)
    {
        if (!source ||
            !target_kdtree ||
            source->empty() ||
            !T_target_source.matrix().allFinite() ||
            !std::isfinite(inlier_distance) ||
            inlier_distance <= 0.0)
        {
            return 0;
        }

        const double maximum_squared_distance =
            inlier_distance *
            inlier_distance;

        std::vector<int> nearest_index(1);
        std::vector<float> nearest_squared_distance(1);

        std::size_t inliers = 0;

        for (const pcl::PointXYZ &point_source :
             source->points)
        {
            const Eigen::Vector3d p_source(
                static_cast<double>(point_source.x),
                static_cast<double>(point_source.y),
                static_cast<double>(point_source.z));

            const Eigen::Vector3d p_target =
                T_target_source *
                p_source;

            if (!p_target.allFinite())
            {
                continue;
            }

            pcl::PointXYZ query;
            query.x =
                static_cast<float>(p_target.x());
            query.y =
                static_cast<float>(p_target.y());
            query.z =
                static_cast<float>(p_target.z());

            if (target_kdtree->nearestKSearch(
                    query,
                    1,
                    nearest_index,
                    nearest_squared_distance) <= 0)
            {
                continue;
            }

            const double squared_distance =
                static_cast<double>(
                    nearest_squared_distance[0]);

            if (!std::isfinite(squared_distance) ||
                squared_distance >
                    maximum_squared_distance)
            {
                continue;
            }

            ++inliers;
        }

        return inliers;
    }

    enum class WindowPointVisibility
    {
        Unobservable = 0,
        Occluded = 1,
        ExpectedVisible = 2
    };

    WindowPointVisibility EvaluatePointVisibilityInWindow(
        const Eigen::Vector3d &point_in_anchor,
        const SphericalVisibilityWindowModel &window_model)
    {
        if (!point_in_anchor.allFinite() ||
            !window_model.valid ||
            window_model.observable_azimuth.size() !=
                kVisibilityAzimuthBins)
        {
            return WindowPointVisibility::Unobservable;
        }

        bool observed_sensor_direction = false;
        bool saw_occluding_return = false;

        for (const SphericalVisibilityWindowScan &scan :
             window_model.scans)
        {
            if (!scan.image ||
                !scan.image->valid ||
                !scan.T_scan_anchor.matrix().allFinite())
            {
                continue;
            }

            const Eigen::Vector3d p_scan =
                scan.T_scan_anchor *
                point_in_anchor;

            std::size_t azimuth_bin = 0;
            std::size_t elevation_bin = 0;
            double point_range_m = 0.0;
            double elevation_rad = 0.0;

            if (!VisibilitySphericalBin(
                    p_scan,
                    azimuth_bin,
                    elevation_bin,
                    point_range_m,
                    elevation_rad))
            {
                continue;
            }

            if (window_model.observable_azimuth[azimuth_bin] == 0)
            {
                continue;
            }

            if (elevation_rad <
                    window_model.min_elevation_rad -
                        kVisibilityElevationMarginRad ||
                elevation_rad >
                    window_model.max_elevation_rad +
                        kVisibilityElevationMarginRad)
            {
                continue;
            }

            observed_sensor_direction = true;

            double measured_range_m =
                std::numeric_limits<double>::infinity();

            const bool has_return =
                QuerySphericalVisibilityRange(
                    *scan.image,
                    p_scan,
                    measured_range_m);

            if (has_return &&
                measured_range_m +
                        kVisibilityOcclusionMarginM <
                    point_range_m)
            {
                saw_occluding_return = true;
                continue;
            }

            // Either the ray has a return at/behind the query point, or it has
            // no return at all.  Once the direction belongs to the learned
            // sensor FOV, both cases mean that no nearer occluder prevents this
            // point from being observable in this member scan.
            return WindowPointVisibility::ExpectedVisible;
        }

        if (saw_occluding_return)
        {
            return WindowPointVisibility::Occluded;
        }

        if (observed_sensor_direction)
        {
            return WindowPointVisibility::ExpectedVisible;
        }

        return WindowPointVisibility::Unobservable;
    }

    bool EvaluateDirectionalVisibilityAwareOverlap(
        const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &source,
        const pcl::search::KdTree<pcl::PointXYZ>::Ptr &target_kdtree,
        const Eigen::Isometry3d &T_target_source,
        const SphericalVisibilityWindowModel &target_visibility,
        double inlier_distance,
        std::size_t &eligible_points,
        std::size_t &occluded_points,
        std::size_t &unobservable_points,
        std::size_t &inliers,
        double &overlap)
    {
        eligible_points = 0;
        occluded_points = 0;
        unobservable_points = 0;
        inliers = 0;
        overlap = 0.0;

        if (!source ||
            !target_kdtree ||
            source->empty() ||
            !T_target_source.matrix().allFinite() ||
            !target_visibility.valid ||
            !std::isfinite(inlier_distance) ||
            inlier_distance <= 0.0)
        {
            return false;
        }

        const double maximum_squared_distance =
            inlier_distance *
            inlier_distance;

        std::vector<int> nearest_index(1);
        std::vector<float> nearest_squared_distance(1);

        for (const pcl::PointXYZ &point_source :
             source->points)
        {
            const Eigen::Vector3d p_source(
                static_cast<double>(point_source.x),
                static_cast<double>(point_source.y),
                static_cast<double>(point_source.z));

            const Eigen::Vector3d p_target =
                T_target_source *
                p_source;

            if (!p_target.allFinite())
            {
                continue;
            }

            const WindowPointVisibility visibility =
                EvaluatePointVisibilityInWindow(
                    p_target,
                    target_visibility);

            if (visibility ==
                WindowPointVisibility::Unobservable)
            {
                ++unobservable_points;
                continue;
            }

            if (visibility ==
                WindowPointVisibility::Occluded)
            {
                ++occluded_points;
                continue;
            }

            ++eligible_points;

            pcl::PointXYZ query;
            query.x =
                static_cast<float>(p_target.x());
            query.y =
                static_cast<float>(p_target.y());
            query.z =
                static_cast<float>(p_target.z());

            if (target_kdtree->nearestKSearch(
                    query,
                    1,
                    nearest_index,
                    nearest_squared_distance) <= 0)
            {
                continue;
            }

            const double squared_distance =
                static_cast<double>(
                    nearest_squared_distance[0]);

            if (!std::isfinite(squared_distance) ||
                squared_distance >
                    maximum_squared_distance)
            {
                continue;
            }

            ++inliers;
        }

        if (eligible_points == 0)
        {
            return false;
        }

        overlap =
            static_cast<double>(inliers) /
            static_cast<double>(eligible_points);

        return std::isfinite(overlap);
    }

    bool EvaluateVisibilityAwareLoopOverlap(
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &source_current_L,
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &target_historical_K,
        const Eigen::Isometry3d &T_K_L,
        const LoopVerifierConfig &verifier_config,
        const SphericalVisibilityWindowModel &source_visibility_L,
        const SphericalVisibilityWindowModel &target_visibility_K,
        VisibilityAwareOverlapMetrics &metrics)
    {
        metrics = VisibilityAwareOverlapMetrics();

        if (!source_current_L ||
            !target_historical_K ||
            source_current_L->empty() ||
            target_historical_K->empty() ||
            !T_K_L.matrix().allFinite() ||
            !source_visibility_L.valid ||
            !target_visibility_K.valid ||
            !std::isfinite(
                verifier_config.voxel_leaf_size) ||
            verifier_config.voxel_leaf_size <= 0.0 ||
            !std::isfinite(
                verifier_config.verification_inlier_distance) ||
            verifier_config.verification_inlier_distance <= 0.0)
        {
            return false;
        }

        const pcl::PointCloud<pcl::PointXYZ>::Ptr source_xyz =
            ConvertToXYZ(source_current_L);

        const pcl::PointCloud<pcl::PointXYZ>::Ptr target_xyz =
            ConvertToXYZ(target_historical_K);

        const pcl::PointCloud<pcl::PointXYZ>::Ptr source_filtered =
            VoxelFilterLoopInformation(
                source_xyz,
                verifier_config.voxel_leaf_size);

        const pcl::PointCloud<pcl::PointXYZ>::Ptr target_filtered =
            VoxelFilterLoopInformation(
                target_xyz,
                verifier_config.voxel_leaf_size);

        if (!source_filtered ||
            !target_filtered ||
            source_filtered->empty() ||
            target_filtered->empty())
        {
            return false;
        }

        pcl::search::KdTree<pcl::PointXYZ>::Ptr source_kdtree(
            new pcl::search::KdTree<pcl::PointXYZ>());

        pcl::search::KdTree<pcl::PointXYZ>::Ptr target_kdtree(
            new pcl::search::KdTree<pcl::PointXYZ>());

        source_kdtree->setInputCloud(source_filtered);
        target_kdtree->setInputCloud(target_filtered);

        metrics.source_points = source_filtered->size();
        metrics.target_points = target_filtered->size();

        const std::size_t raw_inliers_L_to_K =
            CountDirectionalLoopOverlapInliers(
                source_filtered,
                target_kdtree,
                T_K_L,
                verifier_config.verification_inlier_distance);

        const Eigen::Isometry3d T_L_K =
            T_K_L.inverse();

        if (!T_L_K.matrix().allFinite())
        {
            return false;
        }

        const std::size_t raw_inliers_K_to_L =
            CountDirectionalLoopOverlapInliers(
                target_filtered,
                source_kdtree,
                T_L_K,
                verifier_config.verification_inlier_distance);

        metrics.raw_overlap_L_to_K =
            static_cast<double>(raw_inliers_L_to_K) /
            static_cast<double>(metrics.source_points);

        metrics.raw_overlap_K_to_L =
            static_cast<double>(raw_inliers_K_to_L) /
            static_cast<double>(metrics.target_points);

        const bool L_to_K_ok =
            EvaluateDirectionalVisibilityAwareOverlap(
                source_filtered,
                target_kdtree,
                T_K_L,
                target_visibility_K,
                verifier_config.verification_inlier_distance,
                metrics.eligible_L_to_K,
                metrics.occluded_L_to_K,
                metrics.unobservable_L_to_K,
                metrics.inliers_L_to_K,
                metrics.visible_overlap_L_to_K);

        const bool K_to_L_ok =
            EvaluateDirectionalVisibilityAwareOverlap(
                target_filtered,
                source_kdtree,
                T_L_K,
                source_visibility_L,
                verifier_config.verification_inlier_distance,
                metrics.eligible_K_to_L,
                metrics.occluded_K_to_L,
                metrics.unobservable_K_to_L,
                metrics.inliers_K_to_L,
                metrics.visible_overlap_K_to_L);

        if (!L_to_K_ok ||
            !K_to_L_ok)
        {
            return false;
        }

        metrics.eligible_fraction_L_to_K =
            static_cast<double>(
                metrics.eligible_L_to_K) /
            static_cast<double>(
                metrics.source_points);

        metrics.eligible_fraction_K_to_L =
            static_cast<double>(
                metrics.eligible_K_to_L) /
            static_cast<double>(
                metrics.target_points);

        metrics.arithmetic_mean =
            0.5 *
            (metrics.visible_overlap_L_to_K +
             metrics.visible_overlap_K_to_L);

        const double overlap_sum =
            metrics.visible_overlap_L_to_K +
            metrics.visible_overlap_K_to_L;

        if (overlap_sum > 1.0e-12)
        {
            metrics.harmonic_mean =
                2.0 *
                metrics.visible_overlap_L_to_K *
                metrics.visible_overlap_K_to_L /
                overlap_sum;
        }

        metrics.valid =
            std::isfinite(metrics.raw_overlap_L_to_K) &&
            std::isfinite(metrics.raw_overlap_K_to_L) &&
            std::isfinite(metrics.visible_overlap_L_to_K) &&
            std::isfinite(metrics.visible_overlap_K_to_L) &&
            std::isfinite(metrics.eligible_fraction_L_to_K) &&
            std::isfinite(metrics.eligible_fraction_K_to_L) &&
            std::isfinite(metrics.arithmetic_mean) &&
            std::isfinite(metrics.harmonic_mean);

        return metrics.valid;
    }

    bool FitLoopInformationPlane(
        const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &target,
        const std::vector<int> &neighbor_indices,
        Eigen::Vector3d &plane_point,
        Eigen::Vector3d &plane_normal)
    {
        if (!target ||
            neighbor_indices.size() < 3)
        {
            return false;
        }

        Eigen::Vector3d centroid =
            Eigen::Vector3d::Zero();

        for (const int index : neighbor_indices)
        {
            if (index < 0 ||
                static_cast<std::size_t>(index) >= target->size())
            {
                return false;
            }

            const pcl::PointXYZ &point =
                target->points[static_cast<std::size_t>(index)];

            centroid +=
                Eigen::Vector3d(
                    static_cast<double>(point.x),
                    static_cast<double>(point.y),
                    static_cast<double>(point.z));
        }

        centroid /=
            static_cast<double>(
                neighbor_indices.size());

        Eigen::Matrix3d covariance =
            Eigen::Matrix3d::Zero();

        for (const int index : neighbor_indices)
        {
            const pcl::PointXYZ &point =
                target->points[static_cast<std::size_t>(index)];

            const Eigen::Vector3d p_target(
                static_cast<double>(point.x),
                static_cast<double>(point.y),
                static_cast<double>(point.z));

            const Eigen::Vector3d delta =
                p_target - centroid;

            covariance.noalias() +=
                delta * delta.transpose();
        }

        covariance /=
            static_cast<double>(
                neighbor_indices.size());

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>
            eigen_solver(
                covariance,
                Eigen::ComputeEigenvectors);

        if (eigen_solver.info() != Eigen::Success)
        {
            return false;
        }

        Eigen::Vector3d normal =
            eigen_solver.eigenvectors().col(0);

        const double normal_norm =
            normal.norm();

        if (!std::isfinite(normal_norm) ||
            normal_norm < 1.0e-12)
        {
            return false;
        }

        normal /= normal_norm;

        for (const int index : neighbor_indices)
        {
            const pcl::PointXYZ &point =
                target->points[static_cast<std::size_t>(index)];

            const Eigen::Vector3d p_target(
                static_cast<double>(point.x),
                static_cast<double>(point.y),
                static_cast<double>(point.z));

            const double distance =
                std::abs(
                    normal.dot(
                        p_target - centroid));

            if (!std::isfinite(distance) ||
                distance > kLoopInformationMaxPlaneFitError)
            {
                return false;
            }
        }

        plane_point = centroid;
        plane_normal = normal;

        return true;
    }

    double LoopInformationMedian(
        std::vector<double> values)
    {
        if (values.empty())
        {
            return std::numeric_limits<double>::quiet_NaN();
        }

        const std::size_t middle =
            values.size() / 2;

        std::nth_element(
            values.begin(),
            values.begin() +
                static_cast<std::ptrdiff_t>(middle),
            values.end());

        double median =
            values[middle];

        if (values.size() % 2 == 0)
        {
            const double lower =
                *std::max_element(
                    values.begin(),
                    values.begin() +
                        static_cast<std::ptrdiff_t>(middle));

            median =
                0.5 *
                (lower + median);
        }

        return median;
    }

    bool BuildLoopShadowInformationFull6x6(
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &source_current,
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &target_historical,
        const Eigen::Isometry3d &T_target_source,
        const LoopVerifierConfig &verifier_config,
        Eigen::Matrix<double, 6, 6> &information,
        std::size_t &shadow_correspondences,
        double &median_range,
        double &minimum_relative_eigenvalue)
    {
        information =
            Eigen::Matrix<double, 6, 6>::Identity();

        shadow_correspondences = 0;
        median_range =
            std::numeric_limits<double>::quiet_NaN();
        minimum_relative_eigenvalue =
            std::numeric_limits<double>::quiet_NaN();

        if (!source_current ||
            !target_historical ||
            source_current->empty() ||
            target_historical->empty() ||
            !T_target_source.matrix().allFinite() ||
            !std::isfinite(verifier_config.voxel_leaf_size) ||
            verifier_config.voxel_leaf_size <= 0.0 ||
            !std::isfinite(verifier_config.verification_inlier_distance) ||
            verifier_config.verification_inlier_distance <= 0.0)
        {
            return false;
        }

        const pcl::PointCloud<pcl::PointXYZ>::Ptr source_xyz =
            ConvertToXYZ(source_current);

        const pcl::PointCloud<pcl::PointXYZ>::Ptr target_xyz =
            ConvertToXYZ(target_historical);

        const pcl::PointCloud<pcl::PointXYZ>::Ptr source_filtered =
            VoxelFilterLoopInformation(
                source_xyz,
                verifier_config.voxel_leaf_size);

        const pcl::PointCloud<pcl::PointXYZ>::Ptr target_filtered =
            VoxelFilterLoopInformation(
                target_xyz,
                verifier_config.voxel_leaf_size);

        if (!source_filtered ||
            !target_filtered ||
            source_filtered->size() < verifier_config.min_cloud_points ||
            target_filtered->size() < verifier_config.min_cloud_points)
        {
            return false;
        }

        pcl::search::KdTree<pcl::PointXYZ>::Ptr target_kdtree(
            new pcl::search::KdTree<pcl::PointXYZ>());

        target_kdtree->setInputCloud(
            target_filtered);

        Eigen::Matrix<double, 6, 6> H_raw =
            Eigen::Matrix<double, 6, 6>::Zero();

        std::vector<double> correspondence_ranges;
        correspondence_ranges.reserve(
            source_filtered->size());

        std::vector<int> neighbor_indices(
            static_cast<std::size_t>(
                kLoopInformationPlaneKnn));

        std::vector<float> neighbor_squared_distances(
            static_cast<std::size_t>(
                kLoopInformationPlaneKnn));

        const double inlier_distance =
            verifier_config.verification_inlier_distance;

        const double maximum_squared_distance =
            inlier_distance *
            inlier_distance;

        const Eigen::Vector3d sensor_origin_target =
            T_target_source.translation();

        for (const pcl::PointXYZ &source_point :
             source_filtered->points)
        {
            const Eigen::Vector3d p_source(
                static_cast<double>(source_point.x),
                static_cast<double>(source_point.y),
                static_cast<double>(source_point.z));

            const Eigen::Vector3d p_target =
                T_target_source *
                p_source;

            if (!p_target.allFinite())
            {
                continue;
            }

            pcl::PointXYZ query;
            query.x =
                static_cast<float>(p_target.x());
            query.y =
                static_cast<float>(p_target.y());
            query.z =
                static_cast<float>(p_target.z());

            const int found =
                target_kdtree->nearestKSearch(
                    query,
                    kLoopInformationPlaneKnn,
                    neighbor_indices,
                    neighbor_squared_distances);

            if (found < kLoopInformationPlaneKnn)
            {
                continue;
            }

            const double nearest_squared_distance =
                static_cast<double>(
                    neighbor_squared_distances[0]);

            if (!std::isfinite(nearest_squared_distance) ||
                nearest_squared_distance > maximum_squared_distance)
            {
                continue;
            }

            Eigen::Vector3d plane_point;
            Eigen::Vector3d plane_normal;

            if (!FitLoopInformationPlane(
                    target_filtered,
                    neighbor_indices,
                    plane_point,
                    plane_normal))
            {
                continue;
            }

            const double residual =
                plane_normal.dot(
                    p_target -
                    plane_point);

            if (!std::isfinite(residual) ||
                std::abs(residual) > inlier_distance)
            {
                continue;
            }

            const Eigen::Vector3d lever_arm_target =
                p_target -
                sensor_origin_target;

            if (!lever_arm_target.allFinite())
            {
                continue;
            }

            Eigen::Matrix<double, 1, 6> J =
                Eigen::Matrix<double, 1, 6>::Zero();

            J.block<1, 3>(0, 0) =
                lever_arm_target.cross(
                                    plane_normal)
                    .transpose();

            J.block<1, 3>(0, 3) =
                plane_normal.transpose();

            H_raw.noalias() +=
                J.transpose() *
                J;

            const double range =
                lever_arm_target.norm();

            if (std::isfinite(range) &&
                range > 1.0e-9)
            {
                correspondence_ranges.push_back(
                    range);
            }

            ++shadow_correspondences;
        }

        if (shadow_correspondences <
                kLoopInformationMinimumCorrespondences ||
            correspondence_ranges.empty() ||
            !H_raw.allFinite())
        {
            return false;
        }

        median_range =
            LoopInformationMedian(
                correspondence_ranges);

        if (!std::isfinite(median_range) ||
            median_range <= 0.0)
        {
            return false;
        }

        const double scale_L =
            std::clamp(
                median_range,
                kLoopInformationMinimumScaleRange,
                kLoopInformationMaximumScaleRange);

        Eigen::Matrix<double, 6, 6> parameter_unscale =
            Eigen::Matrix<double, 6, 6>::Identity();

        const double inverse_scale =
            1.0 /
            scale_L;

        parameter_unscale(0, 0) = inverse_scale;
        parameter_unscale(1, 1) = inverse_scale;
        parameter_unscale(2, 2) = inverse_scale;

        Eigen::Matrix<double, 6, 6> H_analysis =
            parameter_unscale.transpose() *
            H_raw *
            parameter_unscale;

        H_analysis =
            0.5 *
            (H_analysis +
             H_analysis.transpose());

        if (!H_analysis.allFinite())
        {
            return false;
        }

        Eigen::SelfAdjointEigenSolver<
            Eigen::Matrix<double, 6, 6>>
            hessian_solver(
                H_analysis);

        if (hessian_solver.info() != Eigen::Success ||
            !hessian_solver.eigenvalues().allFinite())
        {
            return false;
        }

        const Eigen::Matrix<double, 6, 1> eigenvalues =
            hessian_solver.eigenvalues();

        const double lambda_max =
            eigenvalues(5);

        if (!std::isfinite(lambda_max) ||
            lambda_max <= 1.0e-12)
        {
            return false;
        }

        const Eigen::Matrix<double, 6, 1> relative_eigenvalues =
            eigenvalues /
            lambda_max;

        if (!relative_eigenvalues.allFinite())
        {
            return false;
        }

        minimum_relative_eigenvalue =
            relative_eigenvalues.minCoeff();

        Eigen::Matrix<double, 6, 6> inverse_relative_eigenvalues =
            Eigen::Matrix<double, 6, 6>::Zero();

        for (int i = 0;
             i < 6;
             ++i)
        {
            const double safe_relative =
                std::clamp(
                    relative_eigenvalues(i),
                    kLoopInformationRelativeEigenvalueFloor,
                    1.0);

            inverse_relative_eigenvalues(i, i) =
                1.0 /
                safe_relative;
        }

        Eigen::Matrix<double, 6, 6> covariance_target_rt =
            hessian_solver.eigenvectors() *
            inverse_relative_eigenvalues *
            hessian_solver.eigenvectors().transpose();

        covariance_target_rt =
            0.5 *
            (covariance_target_rt +
             covariance_target_rt.transpose());

        if (!covariance_target_rt.allFinite())
        {
            return false;
        }

        const Eigen::Matrix3d R_source_target =
            T_target_source.rotation().transpose();

        Eigen::Matrix<double, 6, 6> target_to_source =
            Eigen::Matrix<double, 6, 6>::Zero();

        target_to_source.block<3, 3>(0, 0) =
            R_source_target;

        target_to_source.block<3, 3>(3, 3) =
            R_source_target;

        Eigen::Matrix<double, 6, 6> covariance_source_rt =
            target_to_source *
            covariance_target_rt *
            target_to_source.transpose();

        Eigen::Matrix<double, 6, 6> rt_to_tr =
            Eigen::Matrix<double, 6, 6>::Zero();

        rt_to_tr.block<3, 3>(0, 3) =
            Eigen::Matrix3d::Identity();

        rt_to_tr.block<3, 3>(3, 0) =
            Eigen::Matrix3d::Identity();

        Eigen::Matrix<double, 6, 6> covariance_tr =
            rt_to_tr *
            covariance_source_rt *
            rt_to_tr.transpose();

        covariance_tr =
            0.5 *
            (covariance_tr +
             covariance_tr.transpose());

        if (!covariance_tr.allFinite())
        {
            return false;
        }

        Eigen::Matrix<double, 6, 1> confidence_tr =
            Eigen::Matrix<double, 6, 1>::Ones();

        for (int i = 0;
             i < 6;
             ++i)
        {
            const double variance =
                covariance_tr(i, i);

            if (!std::isfinite(variance) ||
                variance <= 0.0)
            {
                return false;
            }

            confidence_tr(i) =
                std::clamp(
                    1.0 / variance,
                    kLoopInformationMinimumDirectionalConfidence,
                    1.0);
        }

        const double maximum_confidence =
            confidence_tr.maxCoeff();

        if (!std::isfinite(maximum_confidence) ||
            maximum_confidence <= 0.0)
        {
            return false;
        }

        confidence_tr /=
            maximum_confidence;

        for (int i = 0;
             i < 6;
             ++i)
        {
            confidence_tr(i) =
                std::clamp(
                    confidence_tr(i),
                    kLoopInformationMinimumDirectionalConfidence,
                    1.0);
        }

        Eigen::SelfAdjointEigenSolver<
            Eigen::Matrix<double, 6, 6>>
            covariance_solver(
                covariance_tr);

        if (covariance_solver.info() != Eigen::Success ||
            !covariance_solver.eigenvalues().allFinite() ||
            covariance_solver.eigenvalues().minCoeff() <= 1.0e-12)
        {
            return false;
        }

        Eigen::Matrix<double, 6, 6> inverse_covariance_eigenvalues =
            Eigen::Matrix<double, 6, 6>::Zero();

        for (int i = 0;
             i < 6;
             ++i)
        {
            inverse_covariance_eigenvalues(i, i) =
                1.0 /
                covariance_solver.eigenvalues()(i);
        }

        Eigen::Matrix<double, 6, 6> precision_shape =
            covariance_solver.eigenvectors() *
            inverse_covariance_eigenvalues *
            covariance_solver.eigenvectors().transpose();

        precision_shape =
            0.5 *
            (precision_shape +
             precision_shape.transpose());

        if (!precision_shape.allFinite())
        {
            return false;
        }

        Eigen::Matrix<double, 6, 6> standardized_precision =
            Eigen::Matrix<double, 6, 6>::Zero();

        for (int i = 0;
             i < 6;
             ++i)
        {
            if (!std::isfinite(precision_shape(i, i)) ||
                precision_shape(i, i) <= 0.0)
            {
                return false;
            }

            for (int j = 0;
                 j < 6;
                 ++j)
            {
                if (!std::isfinite(precision_shape(j, j)) ||
                    precision_shape(j, j) <= 0.0)
                {
                    return false;
                }

                const double denominator =
                    std::sqrt(
                        precision_shape(i, i) *
                        precision_shape(j, j));

                if (!std::isfinite(denominator) ||
                    denominator <= 0.0)
                {
                    return false;
                }

                standardized_precision(i, j) =
                    precision_shape(i, j) /
                    denominator;
            }
        }

        standardized_precision =
            0.5 *
            (standardized_precision +
             standardized_precision.transpose());

        Eigen::Matrix<double, 6, 6> confidence_scale =
            Eigen::Matrix<double, 6, 6>::Zero();

        for (int i = 0;
             i < 6;
             ++i)
        {
            confidence_scale(i, i) =
                std::sqrt(
                    confidence_tr(i));
        }

        information =
            confidence_scale *
            standardized_precision *
            confidence_scale;

        information =
            0.5 *
            (information +
             information.transpose());

        if (!information.allFinite())
        {
            information =
                Eigen::Matrix<double, 6, 6>::Identity();
            return false;
        }

        Eigen::SelfAdjointEigenSolver<
            Eigen::Matrix<double, 6, 6>>
            information_solver(
                information,
                Eigen::EigenvaluesOnly);

        if (information_solver.info() != Eigen::Success ||
            !information_solver.eigenvalues().allFinite() ||
            information_solver.eigenvalues().minCoeff() <= 1.0e-9)
        {
            information =
                Eigen::Matrix<double, 6, 6>::Identity();
            return false;
        }

        return true;
    }

    // ========================================================================
    // Loop Full 6x6 information helpers.
    //
    // Input/output convention:
    //     order = [tx ty tz rx ry rz]
    //     frame = current/to-node LiDAR frame
    //
    // The matrix is a relative information SHAPE, not a physical covariance.
    // Global translation/rotation calibration (currently 1:30) remains in the
    // PoseGraphOptimizer and is applied there by congruence scaling.
    // ========================================================================
    void ComputeLoopInformationStats(
        const Eigen::Matrix<double, 6, 6> &information,
        double &maximum_absolute_off_diagonal,
        double &maximum_translation_rotation_coupling)
    {
        maximum_absolute_off_diagonal = 0.0;
        maximum_translation_rotation_coupling = 0.0;

        for (int i = 0;
             i < 6;
             ++i)
        {
            for (int j = 0;
                 j < 6;
                 ++j)
            {
                if (i == j)
                {
                    continue;
                }

                const double absolute_value =
                    std::abs(
                        information(i, j));

                maximum_absolute_off_diagonal =
                    std::max(
                        maximum_absolute_off_diagonal,
                        absolute_value);

                const bool translation_rotation_pair =
                    (i < 3 && j >= 3) ||
                    (i >= 3 && j < 3);

                if (translation_rotation_pair)
                {
                    maximum_translation_rotation_coupling =
                        std::max(
                            maximum_translation_rotation_coupling,
                            absolute_value);
                }
            }
        }
    }
} // namespace

RegistrationScan2LocalMap::LoopIcpDebugSnapshot
RegistrationScan2LocalMap::GetLoopIcpDebugSnapshot() const
{
    std::lock_guard<std::mutex> lock(
        backend_output_mutex_);

    LoopIcpDebugSnapshot snapshot;

    snapshot.historical_target_world =
        backend_loop_icp_historical_target_snapshot_;

    snapshot.initial_aligned_world =
        backend_loop_icp_initial_aligned_snapshot_;

    snapshot.final_aligned_world =
        backend_loop_icp_final_aligned_snapshot_;

    snapshot.revision =
        backend_loop_icp_debug_revision_snapshot_;

    snapshot.current_keyframe_id =
        backend_loop_icp_debug_current_kf_snapshot_;

    snapshot.historical_keyframe_id =
        backend_loop_icp_debug_historical_kf_snapshot_;

    snapshot.initial_guess_name =
        backend_loop_icp_debug_guess_name_snapshot_;

    snapshot.correction_translation =
        backend_loop_icp_debug_correction_translation_snapshot_;

    snapshot.correction_rotation_deg =
        backend_loop_icp_debug_correction_rotation_snapshot_;

    return snapshot;
}

bool RegistrationScan2LocalMap::BuildCandidateCenteredHistoricalTarget(
    std::size_t historical_keyframe_id,
    pcl::PointCloud<LIDAR_POINT>::Ptr &target_K,
    std::vector<std::size_t> *included_keyframe_ids) const
{
    target_K =
        pcl::make_shared<
            pcl::PointCloud<LIDAR_POINT>>();

    if (included_keyframe_ids != nullptr)
    {
        included_keyframe_ids->clear();
    }

    const Keyframe *anchor =
        FindBackendKeyframeById(
            historical_keyframe_id);

    if (anchor == nullptr ||
        !anchor->cloud ||
        anchor->cloud->empty() ||
        !anchor->T_WL.matrix().allFinite())
    {
        return false;
    }

    std::vector<const Keyframe *> window_keyframes;
    window_keyframes.reserve(
        2 * online_loop_candidate_target_half_window_ + 1);

    std::size_t estimated_points = 0;

    for (const Keyframe &keyframe :
         backend_keyframes_)
    {
        if (!keyframe.cloud ||
            keyframe.cloud->empty() ||
            !keyframe.T_WL.matrix().allFinite())
        {
            continue;
        }

        const std::size_t id_gap =
            keyframe.id >= historical_keyframe_id
                ? keyframe.id - historical_keyframe_id
                : historical_keyframe_id - keyframe.id;

        if (id_gap >
            online_loop_candidate_target_half_window_)
        {
            continue;
        }

        window_keyframes.push_back(
            &keyframe);

        estimated_points +=
            keyframe.cloud->size();
    }

    if (window_keyframes.size() <
        online_loop_candidate_target_min_keyframes_)
    {
        return false;
    }

    pcl::PointCloud<LIDAR_POINT>::Ptr accumulated_K =
        pcl::make_shared<
            pcl::PointCloud<LIDAR_POINT>>();

    accumulated_K->reserve(
        estimated_points);

    const Eigen::Isometry3d T_K_W =
        anchor->T_WL.inverse();

    if (!T_K_W.matrix().allFinite())
    {
        return false;
    }

    for (const Keyframe *keyframe :
         window_keyframes)
    {
        if (keyframe == nullptr)
        {
            continue;
        }

        const Eigen::Isometry3d T_K_N =
            T_K_W *
            keyframe->T_WL;

        if (!T_K_N.matrix().allFinite())
        {
            continue;
        }

        pcl::PointCloud<LIDAR_POINT>
            cloud_K;

        pcl::transformPointCloud(
            *keyframe->cloud,
            cloud_K,
            T_K_N.matrix().cast<float>());

        *accumulated_K +=
            cloud_K;

        if (included_keyframe_ids != nullptr)
        {
            included_keyframe_ids->push_back(
                keyframe->id);
        }
    }

    if (accumulated_K->empty())
    {
        return false;
    }

    pcl::VoxelGrid<LIDAR_POINT>
        voxel;

    voxel.setLeafSize(
        static_cast<float>(
            online_loop_candidate_target_voxel_leaf_size_),
        static_cast<float>(
            online_loop_candidate_target_voxel_leaf_size_),
        static_cast<float>(
            online_loop_candidate_target_voxel_leaf_size_));

    voxel.setInputCloud(
        accumulated_K);

    voxel.filter(
        *target_K);

    if (!target_K ||
        target_K->empty())
    {
        return false;
    }

    return true;
}

// ============================================================================
// AddKeyframeToPoseGraph()
//
// V6 backend bridge:
//
//     New Keyframe KF_i
//           |
//           +--> PoseGraph Vertex i, X_i = T_WK_i
//           |
//           +--> sequential Keyframe odometry edge from KF_(i-1)
//
// Measurement convention:
//
//     Z_(i-1,i)
//       = T_K(i-1)_Ki
//       = T_WK(i-1)^-1 * T_WKi
//
// Submap finishing is deliberately NOT involved here.
// ============================================================================

const RegistrationScan2LocalMap::BackendSubmapSnapshot *
RegistrationScan2LocalMap::FindBackendSubmapById(
    std::size_t submap_id) const
{
    for (const BackendSubmapSnapshot &submap :
         backend_finished_submaps_)
    {
        if (submap.id == submap_id)
        {
            return &submap;
        }
    }

    return nullptr;
}

const Keyframe *
RegistrationScan2LocalMap::FindBackendKeyframeById(
    std::size_t keyframe_id) const
{
    for (const Keyframe &keyframe :
         backend_keyframes_)
    {
        if (keyframe.id == keyframe_id)
        {
            return &keyframe;
        }
    }

    return nullptr;
}

// ============================================================================
// FindBestFinishedSubmapForKeyframe()
//
// The backend never reads frontend SubmapManager storage.  It searches only
// immutable finished-submap snapshots transferred by BackendKeyframeJob.
// ============================================================================
const RegistrationScan2LocalMap::BackendSubmapSnapshot *
RegistrationScan2LocalMap::FindBestFinishedSubmapForKeyframe(
    std::size_t keyframe_id) const
{
    const BackendSubmapSnapshot *best = nullptr;

    std::size_t best_center_distance =
        std::numeric_limits<std::size_t>::max();

    for (const BackendSubmapSnapshot &submap :
         backend_finished_submaps_)
    {
        if (!submap.cloud_S ||
            submap.cloud_S->empty() ||
            !submap.T_WS.matrix().allFinite())
        {
            continue;
        }

        for (std::size_t index = 0;
             index < submap.keyframe_ids.size();
             ++index)
        {
            if (submap.keyframe_ids[index] != keyframe_id)
            {
                continue;
            }

            const std::size_t center =
                submap.keyframe_ids.size() / 2;

            const std::size_t center_distance =
                index > center
                    ? index - center
                    : center - index;

            if (best == nullptr ||
                center_distance < best_center_distance)
            {
                best = &submap;
                best_center_distance = center_distance;
            }

            break;
        }
    }

    return best;
}

// ============================================================================
// DetectAndVerifyLoopFromKeyframe()
//
// Candidate retrieval:
//     Current Keyframe SC -> Historical Keyframe SC database
//
// Geometry verification:
//     Current Keyframe cloud (frame Lcurrent)
//              ->
//     Candidate-centered historical FINISHED Submap cloud_S (frame H)
//
// LoopVerifier returns:
//     T_H_Lcurrent
//
// Historical Submap H is ONLY an ICP target. It is not a graph vertex.
// If K is the historical candidate Keyframe:
//
//     T_H_K = T_WH^-1 * T_WK
//
// therefore the actual Keyframe-PoseGraph loop measurement is:
//
//     T_K_Lcurrent = T_H_K^-1 * T_H_Lcurrent
//
// and the final edge is:
//
//     historical KF K  --------  current KF L
// ============================================================================

void RegistrationScan2LocalMap::DetectAndVerifyLoopFromKeyframe(
    const Keyframe &current_keyframe,
    std::size_t current_submap_id)
{
    LoopTimingDiagnostics loop_timing;
    loop_timing.current_keyframe_id =
        current_keyframe.id;
    loop_timing.current_submap_id =
        current_submap_id;

    LoopTimingReporter loop_timing_reporter(
        loop_timing);

    if (!current_keyframe.cloud ||
        current_keyframe.cloud->empty() ||
        !current_keyframe.T_WL.matrix().allFinite())
    {
        return;
    }

    // V9 Multi-Loop Sequence:
    // Do NOT stop loop detection just because this frontend Submap already
    // contributed one loop factor.  Every new Keyframe is still allowed to
    // run Scan Context -> ICP -> temporal consistency.  A later spacing gate
    // decides whether the verified observation becomes a new PoseGraph edge
    // or is used only to extend the loop track.

    // ---------------------------------------------------------------------
    // Startup / local-neighborhood protection is KEYFRAME-based.
    //
    // Loop retrieval, loop confirmation, and PoseGraph nodes are all defined
    // at Keyframe granularity.  Therefore do not impose an additional
    // Submap-ID separation here (e.g. 5 Submaps ~= 50 new Keyframes with the
    // current overlapping Submap layout).  Reuse the LoopDetector's own
    // Keyframe separation so every candidate source follows the same topology
    // unit.
    // ---------------------------------------------------------------------
    const LoopDetectorConfig &active_loop_detector_config =
        loop_detector_.GetConfig();

    const std::size_t min_loop_keyframe_separation =
        active_loop_detector_config.min_keyframe_id_separation;

    if (current_keyframe.id < min_loop_keyframe_separation)
    {
        std::cout
            << "Keyframe loop detection skipped"
            << " | current_kf=" << current_keyframe.id
            << " | current_submap=" << current_submap_id
            << " | min_keyframe_gap="
            << min_loop_keyframe_separation
            << " | reason=INSUFFICIENT_KEYFRAME_HISTORY"
            << std::endl;
        return;
    }

    const std::chrono::steady_clock::time_point
        scan_context_detect_start =
            std::chrono::steady_clock::now();

    LoopDetectionDiagnostics detection_diagnostics;

    const std::vector<LoopCandidate> candidates =
        loop_detector_.Detect(
            current_keyframe.id,
            &detection_diagnostics);

    loop_timing.scan_context_detect_ms +=
        ElapsedMilliseconds(
            scan_context_detect_start,
            std::chrono::steady_clock::now());

    loop_timing.candidates =
        candidates.size();

    std::cout
        << "Keyframe loop detection"
        << " | mode=SCAN_CONTEXT_KEYFRAME"
        << " | current_keyframe=" << current_keyframe.id
        << " | current_submap=" << current_submap_id
        << " | descriptors=" << loop_detector_.DescriptorCount()
        << " | candidates=" << candidates.size()
        << " | eligible=" << detection_diagnostics.separation_eligible
        << " | valid_sc="
        << detection_diagnostics.valid_scan_context_matches
        << " | sc_gate="
        << detection_diagnostics.max_scan_context_distance
        << std::endl;

    std::cout
        << "FR_LOOP_TRACE"
        << " | stage=SC_SUMMARY"
        << " | current_kf=" << current_keyframe.id
        << " | current_submap=" << current_submap_id
        << " | descriptors=" << loop_detector_.DescriptorCount()
        << " | eligible=" << detection_diagnostics.separation_eligible
        << " | valid_matches="
        << detection_diagnostics.valid_scan_context_matches
        << " | returned_candidates=" << candidates.size()
        << " | sc_gate="
        << detection_diagnostics.max_scan_context_distance
        << std::endl;

    for (std::size_t index = 0;
         index < candidates.size();
         ++index)
    {
        const LoopCandidate &candidate =
            candidates[index];

        const std::size_t rank =
            index + 1;

        std::cout
            << "Keyframe loop candidate"
            << " | rank=" << rank
            << " | current_kf=" << candidate.current_id
            << " | historical_kf=" << candidate.candidate_id
            << " | sc_distance=" << candidate.scan_context_distance
            << " | sc_similarity=" << candidate.scan_context_similarity
            << " | sc_raw_cosine="
            << candidate.scan_context_raw_cosine_similarity
            << " | sc_sector_coverage="
            << candidate.scan_context_sector_coverage_ratio
            << " | sc_cell_coverage="
            << candidate.scan_context_cell_coverage_ratio
            << " | sc_compared_sectors="
            << candidate.scan_context_compared_sectors
            << " | sector_shift=" << candidate.sector_shift
            << " | yaw_shift=" << candidate.yaw_shift_deg
            << " deg"
            << " | time_separation="
            << candidate.time_separation_sec
            << " s"
            << " | pose_distance="
            << candidate.distance
            << " m"
            << std::endl;

        std::cout
            << "FR_LOOP_TRACE"
            << " | stage=SC_CANDIDATE"
            << " | current_kf=" << candidate.current_id
            << " | rank=" << rank
            << " | historical_kf=" << candidate.candidate_id
            << " | sc_distance=" << candidate.scan_context_distance
            << " | sc_similarity=" << candidate.scan_context_similarity
            << " | sector_coverage="
            << candidate.scan_context_sector_coverage_ratio
            << " | cell_coverage="
            << candidate.scan_context_cell_coverage_ratio
            << " | compared_sectors="
            << candidate.scan_context_compared_sectors
            << " | yaw_shift=" << candidate.yaw_shift_deg
            << " deg"
            << std::endl;
    }

    // ====================================================================
    // V21 Hierarchical 3m Scan Context Retrieval.
    //
    // Goal:
    //   Do not require the TRUE loop Keyframe to survive the original
    //   single-frame Scan Context hard gate.  First retrieve a historical
    //   REGION using a 3m multi-Keyframe descriptor, then search Keyframes only
    //   inside the best historical Submaps, then expand +/-2 KFs before the
    //   existing geometric verification.
    //
    // Current query:
    //   current KF + previous KFs until accumulated frontend trajectory length
    //   reaches approximately 3m.
    //
    // Historical database:
    //   for each eligible historical anchor KF K, build both:
    //       BACKWARD_3M : K, K-1, K-2, ...
    //       FORWARD_3M  : K, K+1, K+2, ...
    //   with every cloud transformed into anchor frame K before SC creation.
    //
    // IMPORTANT:
    //   Region SC is a RANKING / RECALL signal.  There is deliberately NO
    //   max_scan_context_distance hard reject in this V21 stage.
    // ====================================================================
    struct HierarchicalRetrievalWindow
    {
        pcl::PointCloud<LIDAR_POINT>::Ptr cloud =
            pcl::make_shared<pcl::PointCloud<LIDAR_POINT>>();
        std::vector<std::size_t> keyframe_ids;
        double arc_length_m = 0.0;
    };

    struct HierarchicalCachedRegionDescriptor
    {
        ScanContextDescriptor descriptor;
        std::size_t first_keyframe_id = 0;
        std::size_t last_keyframe_id = 0;
        std::size_t keyframe_count = 0;
        double arc_length_m = 0.0;
    };

    struct HierarchicalRegionHit
    {
        std::size_t submap_id = 0;
        std::size_t anchor_keyframe_id = 0;
        int direction = 0; // -1 = BACKWARD, +1 = FORWARD
        ScanContextMatch match;
    };

    struct HierarchicalSubmapScore
    {
        std::size_t submap_id = 0;
        std::size_t recall_support = 0;
        double score = std::numeric_limits<double>::infinity();
        double best_distance = std::numeric_limits<double>::infinity();
    };

    struct HierarchicalKeyframeMatch
    {
        std::size_t keyframe_id = 0;
        ScanContextMatch match;
    };

    // Function-local caches deliberately avoid a public-header / ABI change.
    // Historical Keyframe frontend poses/clouds are immutable in this backend,
    // so descriptors can be reused after the window has matured to ~3m.
    static std::unordered_map<
        std::uint64_t,
        HierarchicalCachedRegionDescriptor>
        hierarchical_region_descriptor_cache;

    static std::unordered_map<
        std::size_t,
        ScanContextDescriptor>
        hierarchical_single_kf_descriptor_cache;

    // V22.4: one spherical range image per immutable raw Keyframe scan.
    // This cache is independent from the fused 3m windows and is reused by
    // every reverse segment candidate in this process.
    static std::unordered_map<
        std::size_t,
        std::shared_ptr<SphericalVisibilityImage>>
        spherical_visibility_image_cache;

    static std::size_t hierarchical_cache_last_current_kf = 0;

    static LoopTrackSegmentRevalidationStateV226
        loop_track_segment_revalidation_v226;

    if (current_keyframe.id <
        hierarchical_cache_last_current_kf)
    {
        hierarchical_region_descriptor_cache.clear();
        hierarchical_single_kf_descriptor_cache.clear();
        spherical_visibility_image_cache.clear();
        loop_track_segment_revalidation_v226 =
            LoopTrackSegmentRevalidationStateV226();
    }

    // If the backend temporal track was cleared by PGO / cluster reset, its
    // private revalidation checkpoint must be cleared as well.
    if (!online_loop_track_.valid)
    {
        loop_track_segment_revalidation_v226 =
            LoopTrackSegmentRevalidationStateV226();
    }

    hierarchical_cache_last_current_kf =
        current_keyframe.id;

    const auto find_retrieval_keyframe =
        [this, &current_keyframe](std::size_t keyframe_id)
        -> const Keyframe *
    {
        if (keyframe_id == current_keyframe.id)
        {
            return &current_keyframe;
        }

        return FindBackendKeyframeById(
            keyframe_id);
    };

    const auto build_retrieval_window =
        [&find_retrieval_keyframe](
            std::size_t anchor_keyframe_id,
            int direction,
            HierarchicalRetrievalWindow &window)
    {
        window = HierarchicalRetrievalWindow();

        if (direction != -1 && direction != +1)
        {
            return false;
        }

        const Keyframe *anchor =
            find_retrieval_keyframe(
                anchor_keyframe_id);

        if (anchor == nullptr ||
            !anchor->cloud ||
            anchor->cloud->empty() ||
            !anchor->T_WL.matrix().allFinite())
        {
            return false;
        }

        const Eigen::Isometry3d T_A_W =
            anchor->T_WL.inverse();

        if (!T_A_W.matrix().allFinite())
        {
            return false;
        }

        const Keyframe *previous = anchor;

        for (std::size_t step = 0;
             step < kHierarchicalRetrievalMaxKeyframes;
             ++step)
        {
            std::size_t keyframe_id =
                anchor_keyframe_id;

            if (step > 0)
            {
                if (direction < 0)
                {
                    if (anchor_keyframe_id < step)
                    {
                        break;
                    }

                    keyframe_id =
                        anchor_keyframe_id - step;
                }
                else
                {
                    if (anchor_keyframe_id >
                        std::numeric_limits<std::size_t>::max() - step)
                    {
                        break;
                    }

                    keyframe_id =
                        anchor_keyframe_id + step;
                }
            }

            const Keyframe *member =
                find_retrieval_keyframe(
                    keyframe_id);

            if (member == nullptr ||
                !member->cloud ||
                member->cloud->empty() ||
                !member->T_WL.matrix().allFinite())
            {
                break;
            }

            if (step > 0)
            {
                const Eigen::Vector3d segment =
                    member->T_WL.translation() -
                    previous->T_WL.translation();

                if (!segment.allFinite())
                {
                    break;
                }

                const double segment_length =
                    segment.norm();

                if (!std::isfinite(segment_length))
                {
                    break;
                }

                window.arc_length_m +=
                    segment_length;
            }

            const Eigen::Isometry3d T_A_M =
                T_A_W *
                member->T_WL;

            if (!T_A_M.matrix().allFinite())
            {
                break;
            }

            pcl::PointCloud<LIDAR_POINT> cloud_A;

            pcl::transformPointCloud(
                *member->cloud,
                cloud_A,
                T_A_M.matrix().cast<float>());

            *window.cloud +=
                cloud_A;

            window.keyframe_ids.push_back(
                member->id);

            previous = member;

            // Include the first KF that reaches/crosses 3m, then stop.  This
            // keeps physical coverage comparable even when KF spacing varies.
            if (window.arc_length_m >=
                kHierarchicalRetrievalArcLengthM)
            {
                break;
            }
        }

        return window.cloud &&
               !window.cloud->empty() &&
               window.keyframe_ids.size() >=
                   kHierarchicalRetrievalMinKeyframes;
    };

    // V22.4: build a physical 3D visibility model for one fused trajectory
    // window.  Each member scan keeps its own LiDAR-frame spherical range
    // image while T_scan_anchor lets a fused-window point be ray-tested from
    // the original sensor position.
    const auto build_visibility_window_model =
        [&find_retrieval_keyframe,
         &spherical_visibility_image_cache](
            std::size_t anchor_keyframe_id,
            const std::vector<std::size_t> &member_keyframe_ids,
            SphericalVisibilityWindowModel &model)
    {
        model = SphericalVisibilityWindowModel();
        model.azimuth_scan_support.assign(
            kVisibilityAzimuthBins,
            0);
        model.observable_azimuth.assign(
            kVisibilityAzimuthBins,
            0);

        const Keyframe *anchor =
            find_retrieval_keyframe(
                anchor_keyframe_id);

        if (anchor == nullptr ||
            !anchor->T_WL.matrix().allFinite())
        {
            return false;
        }

        for (const std::size_t member_id :
             member_keyframe_ids)
        {
            const Keyframe *member =
                find_retrieval_keyframe(
                    member_id);

            if (member == nullptr ||
                !member->cloud ||
                member->cloud->empty() ||
                !member->T_WL.matrix().allFinite())
            {
                continue;
            }

            std::shared_ptr<SphericalVisibilityImage>
                image;

            const auto cached =
                spherical_visibility_image_cache.find(
                    member_id);

            if (cached !=
                    spherical_visibility_image_cache.end() &&
                cached->second &&
                cached->second->valid)
            {
                image = cached->second;
            }
            else
            {
                image =
                    std::make_shared<
                        SphericalVisibilityImage>();

                if (!BuildSphericalVisibilityImage(
                        member->cloud,
                        *image))
                {
                    continue;
                }

                spherical_visibility_image_cache[member_id] = image;
            }

            const Eigen::Isometry3d T_scan_anchor =
                member->T_WL.inverse() *
                anchor->T_WL;

            if (!T_scan_anchor.matrix().allFinite())
            {
                continue;
            }

            SphericalVisibilityWindowScan scan;
            scan.keyframe_id = member_id;
            scan.T_scan_anchor = T_scan_anchor;
            scan.image = image;
            model.scans.push_back(scan);

            ++model.valid_scans;

            model.min_elevation_rad =
                std::min(
                    model.min_elevation_rad,
                    image->min_elevation_rad);

            model.max_elevation_rad =
                std::max(
                    model.max_elevation_rad,
                    image->max_elevation_rad);

            std::vector<std::size_t> returns_per_azimuth(
                kVisibilityAzimuthBins,
                0);

            for (const LIDAR_POINT &point :
                 member->cloud->points)
            {
                const double azimuth =
                    std::atan2(
                        static_cast<double>(point.y),
                        static_cast<double>(point.x));

                std::size_t azimuth_bin = 0;

                if (VisibilityAzimuthBin(
                        azimuth,
                        azimuth_bin))
                {
                    ++returns_per_azimuth[azimuth_bin];
                }
            }

            for (std::size_t azimuth_bin = 0;
                 azimuth_bin <
                 kVisibilityAzimuthBins;
                 ++azimuth_bin)
            {
                if (returns_per_azimuth[azimuth_bin] >=
                    kVisibilityMinReturnsPerAzimuthBinPerScan)
                {
                    ++model.azimuth_scan_support[azimuth_bin];
                }
            }
        }

        if (model.valid_scans == 0 ||
            !std::isfinite(
                model.min_elevation_rad) ||
            !std::isfinite(
                model.max_elevation_rad))
        {
            return false;
        }

        model.required_azimuth_support =
            std::max(
                kVisibilityMinAzimuthScanSupport,
                static_cast<std::size_t>(
                    std::ceil(
                        kVisibilityMinAzimuthScanSupportRatio *
                        static_cast<double>(
                            model.valid_scans))));

        if (model.required_azimuth_support >
            model.valid_scans)
        {
            model.required_azimuth_support =
                model.valid_scans;
        }

        std::vector<std::uint8_t> raw_observable(
            kVisibilityAzimuthBins,
            0);

        for (std::size_t azimuth_bin = 0;
             azimuth_bin <
             kVisibilityAzimuthBins;
             ++azimuth_bin)
        {
            if (model.azimuth_scan_support[azimuth_bin] >=
                model.required_azimuth_support)
            {
                raw_observable[azimuth_bin] = 1;
            }
        }

        for (std::size_t azimuth_bin = 0;
             azimuth_bin <
             kVisibilityAzimuthBins;
             ++azimuth_bin)
        {
            if (raw_observable[azimuth_bin] == 0)
            {
                continue;
            }

            for (int offset =
                     -static_cast<int>(
                         kVisibilityAzimuthDilationBins);
                 offset <=
                 static_cast<int>(
                     kVisibilityAzimuthDilationBins);
                 ++offset)
            {
                const int wrapped =
                    (static_cast<int>(
                         azimuth_bin) +
                     offset +
                     static_cast<int>(
                         kVisibilityAzimuthBins)) %
                    static_cast<int>(
                        kVisibilityAzimuthBins);

                model.observable_azimuth[static_cast<std::size_t>(
                    wrapped)] = 1;
            }
        }

        model.observable_azimuth_bins =
            static_cast<std::size_t>(
                std::count(
                    model.observable_azimuth.begin(),
                    model.observable_azimuth.end(),
                    static_cast<std::uint8_t>(1)));

        model.valid =
            model.valid_scans >=
                kHierarchicalRetrievalMinKeyframes &&
            model.observable_azimuth_bins >= 10 &&
            model.min_elevation_rad <=
                model.max_elevation_rad;

        return model.valid;
    };

    const auto make_loop_candidate_from_sc_match =
        [this, &current_keyframe](
            std::size_t historical_keyframe_id,
            const ScanContextMatch *match)
    {
        LoopCandidate candidate;
        candidate.current_id =
            current_keyframe.id;
        candidate.candidate_id =
            historical_keyframe_id;

        const Keyframe *historical_keyframe =
            FindBackendKeyframeById(
                historical_keyframe_id);

        if (historical_keyframe != nullptr &&
            historical_keyframe->T_WL.matrix().allFinite())
        {
            candidate.distance =
                (current_keyframe.T_WL.translation() -
                 historical_keyframe->T_WL.translation())
                    .norm();

            candidate.time_separation_sec =
                current_keyframe.timestamp -
                historical_keyframe->timestamp;
        }
        else
        {
            candidate.distance =
                std::numeric_limits<double>::infinity();
            candidate.time_separation_sec =
                std::numeric_limits<double>::infinity();
        }

        if (match != nullptr &&
            match->valid &&
            std::isfinite(match->distance))
        {
            candidate.scan_context_distance =
                match->distance;
            candidate.scan_context_similarity =
                match->similarity;
            candidate.scan_context_raw_cosine_similarity =
                match->raw_cosine_similarity;
            candidate.scan_context_sector_coverage_ratio =
                match->sector_coverage_ratio;
            candidate.scan_context_cell_coverage_ratio =
                match->cell_coverage_ratio;
            candidate.scan_context_compared_sectors =
                match->compared_sectors;
            candidate.sector_shift =
                match->sector_shift;
            candidate.yaw_shift_deg =
                match->yaw_shift_deg;
        }
        else
        {
            // Neighborhood expansion remains legal even if this exact KF has
            // a weak/invalid single-frame SC.  Geometry will decide later.
            candidate.scan_context_distance =
                std::numeric_limits<double>::infinity();
            candidate.scan_context_similarity = 0.0;
            candidate.scan_context_raw_cosine_similarity = 0.0;
            candidate.scan_context_sector_coverage_ratio = 0.0;
            candidate.scan_context_cell_coverage_ratio = 0.0;
            candidate.scan_context_compared_sectors = 0;
            candidate.sector_shift = 0;
            candidate.yaw_shift_deg =
                std::numeric_limits<double>::quiet_NaN();
        }

        return candidate;
    };

    std::vector<LoopCandidate>
        hierarchical_retrieval_candidates;

    std::vector<std::size_t>
        hierarchical_retrieval_candidate_ids;

    std::unordered_map<std::size_t, std::size_t>
        hierarchical_candidate_submap_rank;

    std::unordered_map<std::size_t, double>
        hierarchical_candidate_region_score;

    std::unordered_map<std::size_t, double>
        hierarchical_candidate_region_similarity;

    // V22.6: remember which historical 3m direction caused the parent Submap
    // to be recalled.  Current query is always causal BACKWARD_3M; therefore
    // the recalled historical direction tells geometry which physical side of
    // candidate K should be fused.
    std::unordered_map<std::size_t, int>
        hierarchical_candidate_preferred_direction;

    const std::chrono::steady_clock::time_point
        hierarchical_retrieval_start =
            std::chrono::steady_clock::now();

    HierarchicalRetrievalWindow current_region_window;

    const bool current_region_window_ready =
        build_retrieval_window(
            current_keyframe.id,
            -1,
            current_region_window);

    ScanContext hierarchical_scan_context(
        active_loop_detector_config.scan_context);

    ScanContextDescriptor current_region_descriptor;
    bool current_region_descriptor_valid = false;

    if (current_region_window_ready)
    {
        current_region_descriptor =
            hierarchical_scan_context.MakeDescriptor(
                current_region_window.cloud);

        current_region_descriptor_valid =
            current_region_descriptor.valid;
    }

    // ====================================================================
    // V21 RViz retrieval debug.
    //
    // Publish the EXACT 3m windows used by Region SC, not a second
    // reconstruction inside the ROS node.  Both clouds are transformed from
    // their anchor frames back into the raw frontend odom frame so RViz can
    // show the frontend drift that the loop detector is trying to close.
    // ====================================================================
    const auto update_hierarchical_retrieval_debug =
        [this,
         &current_keyframe,
         &current_region_window,
         &build_retrieval_window](
            const HierarchicalRegionHit *selected_hit,
            std::size_t selected_submap_rank)
    {
        fr_slam_debug::LoopRetrievalDebugSnapshot snapshot;

        snapshot.current_keyframe_id =
            current_keyframe.id;

        snapshot.current_keyframe_ids =
            current_region_window.keyframe_ids;

        snapshot.current_arc_length_m =
            current_region_window.arc_length_m;

        if (current_region_window.cloud &&
            !current_region_window.cloud->empty() &&
            current_keyframe.T_WL.matrix().allFinite())
        {
            pcl::PointCloud<LIDAR_POINT>::Ptr current_window_odom =
                pcl::make_shared<pcl::PointCloud<LIDAR_POINT>>();

            pcl::transformPointCloud(
                *current_region_window.cloud,
                *current_window_odom,
                current_keyframe.T_WL.matrix().cast<float>());

            snapshot.current_window_odom =
                current_window_odom;
        }

        if (selected_hit != nullptr)
        {
            snapshot.historical_anchor_keyframe_id =
                selected_hit->anchor_keyframe_id;

            snapshot.historical_submap_id =
                selected_hit->submap_id;

            snapshot.historical_direction =
                selected_hit->direction;

            snapshot.submap_rank =
                selected_submap_rank;

            snapshot.region_sc_distance =
                selected_hit->match.distance;

            snapshot.region_sc_similarity =
                selected_hit->match.similarity;

            HierarchicalRetrievalWindow historical_window;

            if (build_retrieval_window(
                    selected_hit->anchor_keyframe_id,
                    selected_hit->direction,
                    historical_window))
            {
                snapshot.historical_keyframe_ids =
                    historical_window.keyframe_ids;

                snapshot.historical_arc_length_m =
                    historical_window.arc_length_m;

                const Keyframe *historical_anchor =
                    FindBackendKeyframeById(
                        selected_hit->anchor_keyframe_id);

                if (historical_anchor != nullptr &&
                    historical_anchor->T_WL.matrix().allFinite() &&
                    historical_window.cloud &&
                    !historical_window.cloud->empty())
                {
                    pcl::PointCloud<LIDAR_POINT>::Ptr historical_window_odom =
                        pcl::make_shared<pcl::PointCloud<LIDAR_POINT>>();

                    pcl::transformPointCloud(
                        *historical_window.cloud,
                        *historical_window_odom,
                        historical_anchor->T_WL.matrix().cast<float>());

                    snapshot.historical_window_odom =
                        historical_window_odom;
                }
            }
        }

        fr_slam_debug::UpdateLoopRetrievalDebugSnapshot(
            std::move(snapshot));
    };

    if (current_region_descriptor_valid)
    {
        std::vector<HierarchicalRegionHit>
            region_hits;

        region_hits.reserve(
            backend_keyframes_.size() * 2);

        const std::size_t max_historical_member_id =
            current_keyframe.id >=
                    min_loop_keyframe_separation
                ? current_keyframe.id -
                      min_loop_keyframe_separation
                : 0;

        for (const Keyframe &historical_anchor :
             backend_keyframes_)
        {
            if (historical_anchor.id >=
                    current_keyframe.id ||
                !historical_anchor.cloud ||
                historical_anchor.cloud->empty() ||
                !historical_anchor.T_WL.matrix().allFinite())
            {
                continue;
            }

            const std::size_t keyframe_gap =
                current_keyframe.id -
                historical_anchor.id;

            if (keyframe_gap <
                min_loop_keyframe_separation)
            {
                continue;
            }

            const double time_separation =
                current_keyframe.timestamp -
                historical_anchor.timestamp;

            if (!std::isfinite(time_separation) ||
                time_separation <
                    active_loop_detector_config
                        .min_time_separation_sec)
            {
                continue;
            }

            const BackendSubmapSnapshot *anchor_submap =
                FindBestFinishedSubmapForKeyframe(
                    historical_anchor.id);

            if (anchor_submap == nullptr)
            {
                continue;
            }

            for (const int direction : {-1, +1})
            {
                const std::uint64_t cache_key =
                    static_cast<std::uint64_t>(historical_anchor.id) * 2ULL +
                    (direction > 0 ? 1ULL : 0ULL);

                const HierarchicalCachedRegionDescriptor *cached = nullptr;

                const auto cache_iterator =
                    hierarchical_region_descriptor_cache.find(
                        cache_key);

                if (cache_iterator !=
                    hierarchical_region_descriptor_cache.end())
                {
                    cached =
                        &cache_iterator->second;
                }
                else
                {
                    HierarchicalRetrievalWindow historical_window;

                    if (!build_retrieval_window(
                            historical_anchor.id,
                            direction,
                            historical_window))
                    {
                        continue;
                    }

                    const ScanContextDescriptor descriptor =
                        hierarchical_scan_context.MakeDescriptor(
                            historical_window.cloud);

                    if (!descriptor.valid ||
                        historical_window.keyframe_ids.empty())
                    {
                        continue;
                    }

                    HierarchicalCachedRegionDescriptor entry;
                    entry.descriptor = descriptor;
                    entry.keyframe_count =
                        historical_window.keyframe_ids.size();
                    entry.arc_length_m =
                        historical_window.arc_length_m;

                    entry.first_keyframe_id =
                        *std::min_element(
                            historical_window.keyframe_ids.begin(),
                            historical_window.keyframe_ids.end());

                    entry.last_keyframe_id =
                        *std::max_element(
                            historical_window.keyframe_ids.begin(),
                            historical_window.keyframe_ids.end());

                    // Cache only a mature ~3m window (or a very dense window
                    // that hit the KF cap).  Forward windows near the current
                    // trajectory end are rebuilt until they mature.
                    const bool mature_window =
                        historical_window.arc_length_m >=
                            kHierarchicalRetrievalArcLengthM ||
                        historical_window.keyframe_ids.size() >=
                            kHierarchicalRetrievalMaxKeyframes;

                    if (mature_window)
                    {
                        const auto insertion =
                            hierarchical_region_descriptor_cache.emplace(
                                cache_key,
                                entry);

                        cached =
                            &insertion.first->second;
                    }
                    else
                    {
                        // Use this descriptor for the current query, but do not
                        // cache an incomplete forward window permanently.
                        const ScanContextMatch match =
                            hierarchical_scan_context.Compare(
                                descriptor,
                                current_region_descriptor);

                        if (entry.last_keyframe_id <=
                                max_historical_member_id &&
                            match.valid &&
                            std::isfinite(match.distance))
                        {
                            HierarchicalRegionHit hit;
                            hit.submap_id =
                                anchor_submap->id;
                            hit.anchor_keyframe_id =
                                historical_anchor.id;
                            hit.direction = direction;
                            hit.match = match;
                            region_hits.push_back(hit);
                        }

                        continue;
                    }
                }

                if (cached == nullptr ||
                    !cached->descriptor.valid ||
                    cached->last_keyframe_id >
                        max_historical_member_id)
                {
                    continue;
                }

                const ScanContextMatch match =
                    hierarchical_scan_context.Compare(
                        cached->descriptor,
                        current_region_descriptor);

                if (!match.valid ||
                    !std::isfinite(match.distance))
                {
                    continue;
                }

                HierarchicalRegionHit hit;
                hit.submap_id =
                    anchor_submap->id;
                hit.anchor_keyframe_id =
                    historical_anchor.id;
                hit.direction = direction;
                hit.match = match;
                region_hits.push_back(hit);
            }
        }

        std::sort(
            region_hits.begin(),
            region_hits.end(),
            [](const HierarchicalRegionHit &lhs,
               const HierarchicalRegionHit &rhs)
            {
                if (lhs.match.distance !=
                    rhs.match.distance)
                {
                    return lhs.match.distance <
                           rhs.match.distance;
                }

                return lhs.anchor_keyframe_id <
                       rhs.anchor_keyframe_id;
            });

        const std::size_t region_recall_count =
            region_hits.size();

        std::unordered_map<
            std::size_t,
            std::vector<double>>
            submap_region_distances;

        std::unordered_map<std::size_t, std::size_t>
            submap_recall_support;

        for (std::size_t hit_index = 0;
             hit_index < region_recall_count;
             ++hit_index)
        {
            const HierarchicalRegionHit &hit =
                region_hits[hit_index];

            submap_region_distances[hit.submap_id]
                .push_back(hit.match.distance);

            ++submap_recall_support[hit.submap_id];
        }

        std::vector<HierarchicalSubmapScore>
            submap_scores;

        submap_scores.reserve(
            submap_region_distances.size());

        for (auto &entry :
             submap_region_distances)
        {
            std::vector<double> &distances =
                entry.second;

            std::sort(
                distances.begin(),
                distances.end());

            const std::size_t score_count =
                std::min(
                    kHierarchicalRegionScoreBestWindows,
                    distances.size());

            if (score_count == 0)
            {
                continue;
            }

            double score_sum = 0.0;

            for (std::size_t score_index = 0;
                 score_index < score_count;
                 ++score_index)
            {
                score_sum +=
                    distances[score_index];
            }

            HierarchicalSubmapScore score;
            score.submap_id = entry.first;
            score.recall_support =
                submap_recall_support[entry.first];
            score.score =
                score_sum /
                static_cast<double>(score_count);
            score.best_distance =
                distances.front();

            submap_scores.push_back(score);
        }

        std::sort(
            submap_scores.begin(),
            submap_scores.end(),
            [](const HierarchicalSubmapScore &lhs,
               const HierarchicalSubmapScore &rhs)
            {
                if (std::abs(lhs.score - rhs.score) >
                    1.0e-12)
                {
                    return lhs.score < rhs.score;
                }

                if (lhs.recall_support !=
                    rhs.recall_support)
                {
                    return lhs.recall_support >
                           rhs.recall_support;
                }

                return lhs.submap_id <
                       rhs.submap_id;
            });

        const std::size_t selected_submap_count =
            std::min(
                kHierarchicalCandidateSubmaps,
                submap_scores.size());

        // V22.6: each selected historical Submap inherits the direction of its
        // best individual 3m Region-SC hit.  region_hits is globally sorted by
        // distance, so the first hit seen for a Submap is its strongest
        // direction hypothesis.
        std::unordered_map<std::size_t, int>
            hierarchical_submap_preferred_direction;

        for (const HierarchicalRegionHit &hit :
             region_hits)
        {
            if (hierarchical_submap_preferred_direction.find(
                    hit.submap_id) ==
                hierarchical_submap_preferred_direction.end())
            {
                hierarchical_submap_preferred_direction.emplace(
                    hit.submap_id,
                    hit.direction);
            }
        }

        // ---------------------------------------------------------------
        // RViz Region Retrieval snapshot.
        //
        // The Submap score is aggregated from several Region windows.  For
        // visualization choose the BEST individual Region hit that belongs to
        // the top-ranked Submap.  This is the historical 3m window that most
        // clearly explains why that Submap was recalled.
        // ---------------------------------------------------------------
        const HierarchicalRegionHit *debug_selected_region_hit =
            nullptr;

        if (selected_submap_count > 0)
        {
            const std::size_t debug_top_submap_id =
                submap_scores.front().submap_id;

            for (const HierarchicalRegionHit &hit :
                 region_hits)
            {
                if (hit.submap_id ==
                    debug_top_submap_id)
                {
                    debug_selected_region_hit =
                        &hit;
                    break;
                }
            }
        }

        update_hierarchical_retrieval_debug(
            debug_selected_region_hit,
            debug_selected_region_hit != nullptr
                ? 1
                : 0);

        if (debug_selected_region_hit != nullptr)
        {
            std::cout
                << "Hierarchical SC V21"
                << " | stage=RVIZ_RETRIEVAL_DEBUG"
                << " | current_kf=" << current_keyframe.id
                << " | historical_anchor="
                << debug_selected_region_hit->anchor_keyframe_id
                << " | historical_submap="
                << debug_selected_region_hit->submap_id
                << " | direction="
                << (debug_selected_region_hit->direction < 0
                        ? "BACKWARD_3M"
                        : "FORWARD_3M")
                << " | region_distance="
                << debug_selected_region_hit->match.distance
                << " | current_kfs="
                << current_region_window.keyframe_ids.size()
                << std::endl;
        }

        const ScanContextDescriptor current_single_descriptor =
            hierarchical_scan_context.MakeDescriptor(
                current_keyframe.cloud);

        for (std::size_t submap_rank = 0;
             submap_rank < selected_submap_count;
             ++submap_rank)
        {
            const HierarchicalSubmapScore &submap_score =
                submap_scores[submap_rank];

            const BackendSubmapSnapshot *submap =
                FindBackendSubmapById(
                    submap_score.submap_id);

            if (submap == nullptr ||
                !current_single_descriptor.valid)
            {
                continue;
            }

            std::vector<HierarchicalKeyframeMatch>
                keyframe_matches;

            keyframe_matches.reserve(
                submap->keyframe_ids.size());

            std::unordered_map<
                std::size_t,
                ScanContextMatch>
                match_by_keyframe_id;

            for (const std::size_t historical_kf_id :
                 submap->keyframe_ids)
            {
                if (historical_kf_id >=
                    current_keyframe.id)
                {
                    continue;
                }

                const std::size_t keyframe_gap =
                    current_keyframe.id -
                    historical_kf_id;

                if (keyframe_gap <
                    min_loop_keyframe_separation)
                {
                    continue;
                }

                const Keyframe *historical_kf =
                    FindBackendKeyframeById(
                        historical_kf_id);

                if (historical_kf == nullptr ||
                    !historical_kf->cloud ||
                    historical_kf->cloud->empty() ||
                    !historical_kf->T_WL.matrix().allFinite())
                {
                    continue;
                }

                const double time_separation =
                    current_keyframe.timestamp -
                    historical_kf->timestamp;

                if (!std::isfinite(time_separation) ||
                    time_separation <
                        active_loop_detector_config
                            .min_time_separation_sec)
                {
                    continue;
                }

                const ScanContextDescriptor *historical_descriptor =
                    nullptr;

                const auto descriptor_iterator =
                    hierarchical_single_kf_descriptor_cache.find(
                        historical_kf_id);

                if (descriptor_iterator !=
                    hierarchical_single_kf_descriptor_cache.end())
                {
                    historical_descriptor =
                        &descriptor_iterator->second;
                }
                else
                {
                    const ScanContextDescriptor descriptor =
                        hierarchical_scan_context.MakeDescriptor(
                            historical_kf->cloud);

                    if (!descriptor.valid)
                    {
                        continue;
                    }

                    const auto insertion =
                        hierarchical_single_kf_descriptor_cache.emplace(
                            historical_kf_id,
                            descriptor);

                    historical_descriptor =
                        &insertion.first->second;
                }

                if (historical_descriptor == nullptr ||
                    !historical_descriptor->valid)
                {
                    continue;
                }

                const ScanContextMatch match =
                    hierarchical_scan_context.Compare(
                        *historical_descriptor,
                        current_single_descriptor);

                if (!match.valid ||
                    !std::isfinite(match.distance))
                {
                    continue;
                }

                HierarchicalKeyframeMatch keyframe_match;
                keyframe_match.keyframe_id =
                    historical_kf_id;
                keyframe_match.match =
                    match;

                keyframe_matches.push_back(
                    keyframe_match);

                match_by_keyframe_id.emplace(
                    historical_kf_id,
                    match);
            }

            std::sort(
                keyframe_matches.begin(),
                keyframe_matches.end(),
                [](const HierarchicalKeyframeMatch &lhs,
                   const HierarchicalKeyframeMatch &rhs)
                {
                    if (lhs.match.distance !=
                        rhs.match.distance)
                    {
                        return lhs.match.distance <
                               rhs.match.distance;
                    }

                    return lhs.keyframe_id <
                           rhs.keyframe_id;
                });

            const std::size_t seed_count =
                std::min(
                    kHierarchicalKeyframeSeedsPerSubmap,
                    keyframe_matches.size());

            // Seed-first ordering: [seed, -1, +1, -2, +2, ...].  This gives
            // the true adjacent KF a verification chance even when its own SC
            // score is weak or invalid.
            std::vector<std::int64_t> neighborhood_offsets;
            neighborhood_offsets.reserve(
                1 +
                2 * kHierarchicalKeyframeNeighborhoodRadius);
            neighborhood_offsets.push_back(0);

            for (std::size_t neighborhood_step = 1;
                 neighborhood_step <=
                 kHierarchicalKeyframeNeighborhoodRadius;
                 ++neighborhood_step)
            {
                const std::int64_t signed_step =
                    static_cast<std::int64_t>(
                        neighborhood_step);

                neighborhood_offsets.push_back(-signed_step);
                neighborhood_offsets.push_back(+signed_step);
            }

            for (std::size_t seed_index = 0;
                 seed_index < seed_count;
                 ++seed_index)
            {
                const std::size_t seed_id =
                    keyframe_matches[seed_index].keyframe_id;

                for (const std::int64_t offset :
                     neighborhood_offsets)
                {
                    const std::int64_t candidate_id_signed =
                        static_cast<std::int64_t>(seed_id) +
                        offset;

                    if (candidate_id_signed < 0)
                    {
                        continue;
                    }

                    const std::size_t candidate_id =
                        static_cast<std::size_t>(
                            candidate_id_signed);

                    if (candidate_id >=
                        current_keyframe.id)
                    {
                        continue;
                    }

                    const std::size_t candidate_gap =
                        current_keyframe.id -
                        candidate_id;

                    if (candidate_gap <
                        min_loop_keyframe_separation)
                    {
                        continue;
                    }

                    const Keyframe *candidate_kf =
                        FindBackendKeyframeById(
                            candidate_id);

                    if (candidate_kf == nullptr ||
                        !candidate_kf->cloud ||
                        candidate_kf->cloud->empty() ||
                        !candidate_kf->T_WL.matrix().allFinite())
                    {
                        continue;
                    }

                    const bool already_added =
                        std::find(
                            hierarchical_retrieval_candidate_ids.begin(),
                            hierarchical_retrieval_candidate_ids.end(),
                            candidate_id) !=
                        hierarchical_retrieval_candidate_ids.end();

                    if (already_added)
                    {
                        continue;
                    }

                    const ScanContextMatch *candidate_match =
                        nullptr;

                    const auto match_iterator =
                        match_by_keyframe_id.find(
                            candidate_id);

                    if (match_iterator !=
                        match_by_keyframe_id.end())
                    {
                        candidate_match =
                            &match_iterator->second;
                    }

                    hierarchical_retrieval_candidates.push_back(
                        make_loop_candidate_from_sc_match(
                            candidate_id,
                            candidate_match));

                    hierarchical_retrieval_candidate_ids.push_back(
                        candidate_id);

                    hierarchical_candidate_submap_rank[candidate_id] =
                        submap_rank + 1;

                    hierarchical_candidate_region_score[candidate_id] =
                        submap_score.score;

                    hierarchical_candidate_region_similarity[candidate_id] =
                        candidate_match != nullptr
                            ? candidate_match->similarity
                            : 0.0;

                    const auto preferred_direction_iterator =
                        hierarchical_submap_preferred_direction.find(
                            submap_score.submap_id);

                    if (preferred_direction_iterator !=
                        hierarchical_submap_preferred_direction.end())
                    {
                        hierarchical_candidate_preferred_direction[candidate_id] =
                            preferred_direction_iterator->second;
                    }
                }
            }

            std::cout
                << "Hierarchical SC V21"
                << " | stage=SUBMAP_SELECTED"
                << " | current_kf=" << current_keyframe.id
                << " | submap_rank=" << (submap_rank + 1)
                << " | submap_id=" << submap_score.submap_id
                << " | region_score=" << submap_score.score
                << " | region_best_distance="
                << submap_score.best_distance
                << " | region_support="
                << submap_score.recall_support
                << " | kf_matches="
                << keyframe_matches.size()
                << " | kf_seeds=" << seed_count
                << std::endl;
        }

        std::cout
            << "Hierarchical SC V21"
            << " | stage=REGION_SUMMARY"
            << " | current_kf=" << current_keyframe.id
            << " | query_kfs="
            << current_region_window.keyframe_ids.size()
            << " | query_arc="
            << current_region_window.arc_length_m << " m"
            << " | valid_region_hits="
            << region_hits.size()
            << " | recall_windows="
            << region_recall_count
            << " | candidate_submaps="
            << selected_submap_count
            << " | expanded_kf_candidates="
            << hierarchical_retrieval_candidates.size()
            << " | old_sc_gate_NOT_used=true"
            << std::endl;
    }
    else
    {
        // Advance the debug revision even when Region SC is unavailable.
        // The ROS node will publish an empty historical cloud, preventing a
        // stale old candidate from remaining visible in RViz.
        update_hierarchical_retrieval_debug(
            nullptr,
            0);

        std::cout
            << "Hierarchical SC V21"
            << " | stage=REGION_SUMMARY"
            << " | current_kf=" << current_keyframe.id
            << " | query_kfs="
            << current_region_window.keyframe_ids.size()
            << " | query_arc="
            << current_region_window.arc_length_m << " m"
            << " | action=SKIP"
            << " | reason="
            << (current_region_window_ready
                    ? "INVALID_REGION_DESCRIPTOR"
                    : "INSUFFICIENT_3M_QUERY_WINDOW")
            << std::endl;
    }

    loop_timing.scan_context_detect_ms +=
        ElapsedMilliseconds(
            hierarchical_retrieval_start,
            std::chrono::steady_clock::now());

    loop_timing.candidates +=
        hierarchical_retrieval_candidates.size();

    // ====================================================================
    // V2.5 Reverse Spatial Candidate Injection.
    //
    // Motivation:
    //   In the reverse revisit corridor Scan Context can temporarily miss the
    //   physically correct historical KF completely (for example a current
    //   KF near 168 while the opposite historical pass is near 126).
    //
    // Policy:
    //   * Keyframe gap must satisfy the normal LoopDetector exclusion.
    //   * Time gap must satisfy the normal LoopDetector exclusion.
    //   * Frontend relative translation <= 3.5 m.
    //   * Frontend relative |yaw| >= 150 deg.
    //   * Keep only the nearest few historical KFs.
    //   * Do NOT duplicate a KF already returned by Scan Context.
    //
    // This is DISCOVERY ONLY.  It does not create a loop edge and does not
    // bypass prescore, Point-to-Plane V2, trusted reverse geometry, graph
    // consistency, temporal consistency, cluster continuity, or PGO guards.
    // ====================================================================
    std::vector<LoopCandidate> reverse_spatial_candidates;
    reverse_spatial_candidates.reserve(
        kReverseSpatialMaxCandidates);

    const LoopDetectorConfig &reverse_spatial_detector_config =
        loop_detector_.GetConfig();

    for (const Keyframe &historical_keyframe :
         backend_keyframes_)
    {
        if (historical_keyframe.id >= current_keyframe.id ||
            !historical_keyframe.T_WL.matrix().allFinite())
        {
            continue;
        }

        const std::size_t keyframe_gap =
            current_keyframe.id -
            historical_keyframe.id;

        if (keyframe_gap <
            reverse_spatial_detector_config
                .min_keyframe_id_separation)
        {
            continue;
        }

        const double time_separation =
            current_keyframe.timestamp -
            historical_keyframe.timestamp;

        if (!std::isfinite(time_separation) ||
            time_separation <
                reverse_spatial_detector_config
                    .min_time_separation_sec)
        {
            continue;
        }

        const Eigen::Isometry3d T_K_L_frontend_spatial =
            historical_keyframe.T_WL.inverse() *
            current_keyframe.T_WL;

        if (!T_K_L_frontend_spatial.matrix().allFinite())
        {
            continue;
        }

        const double frontend_distance =
            T_K_L_frontend_spatial.translation().norm();

        const Eigen::Matrix3d &R_K_L_frontend_spatial =
            T_K_L_frontend_spatial.rotation();

        const double frontend_yaw_rad =
            std::atan2(
                R_K_L_frontend_spatial(1, 0),
                R_K_L_frontend_spatial(0, 0));

        const double frontend_abs_yaw_deg =
            std::abs(
                std::atan2(
                    std::sin(frontend_yaw_rad),
                    std::cos(frontend_yaw_rad))) *
            180.0 / M_PI;

        if (!std::isfinite(frontend_distance) ||
            frontend_distance >
                kReverseLoopSeedMaxFrontendDistance ||
            !std::isfinite(frontend_abs_yaw_deg) ||
            frontend_abs_yaw_deg <
                kReverseLoopSeedMinYawDeg)
        {
            continue;
        }

        const bool already_returned_by_scan_context =
            std::any_of(
                candidates.begin(),
                candidates.end(),
                [&historical_keyframe](
                    const LoopCandidate &candidate)
                {
                    return candidate.candidate_id ==
                           historical_keyframe.id;
                });

        if (already_returned_by_scan_context)
        {
            continue;
        }

        LoopCandidate injected;
        injected.current_id =
            current_keyframe.id;
        injected.candidate_id =
            historical_keyframe.id;
        injected.distance =
            frontend_distance;
        injected.time_separation_sec =
            time_separation;

        // No Scan Context measurement exists for this injected candidate.
        injected.scan_context_distance =
            std::numeric_limits<double>::infinity();
        injected.scan_context_similarity = 0.0;
        injected.scan_context_raw_cosine_similarity = 0.0;
        injected.scan_context_sector_coverage_ratio = 0.0;
        injected.scan_context_cell_coverage_ratio = 0.0;
        injected.scan_context_compared_sectors = 0;
        injected.sector_shift = 0;
        injected.yaw_shift_deg =
            std::numeric_limits<double>::quiet_NaN();

        reverse_spatial_candidates.push_back(
            injected);
    }

    std::sort(
        reverse_spatial_candidates.begin(),
        reverse_spatial_candidates.end(),
        [](const LoopCandidate &lhs,
           const LoopCandidate &rhs)
        {
            if (lhs.distance != rhs.distance)
            {
                return lhs.distance < rhs.distance;
            }

            return lhs.candidate_id < rhs.candidate_id;
        });

    if (reverse_spatial_candidates.size() >
        kReverseSpatialMaxCandidates)
    {
        reverse_spatial_candidates.resize(
            kReverseSpatialMaxCandidates);
    }

    for (std::size_t index = 0;
         index < reverse_spatial_candidates.size();
         ++index)
    {
        const LoopCandidate &candidate =
            reverse_spatial_candidates[index];

        const Keyframe *historical_keyframe =
            FindBackendKeyframeById(
                candidate.candidate_id);

        double frontend_abs_yaw_deg =
            std::numeric_limits<double>::quiet_NaN();

        if (historical_keyframe != nullptr &&
            historical_keyframe->T_WL.matrix().allFinite())
        {
            const Eigen::Isometry3d T_K_L_frontend_spatial =
                historical_keyframe->T_WL.inverse() *
                current_keyframe.T_WL;

            if (T_K_L_frontend_spatial.matrix().allFinite())
            {
                const Eigen::Matrix3d &R_K_L_frontend_spatial =
                    T_K_L_frontend_spatial.rotation();

                const double frontend_yaw_rad =
                    std::atan2(
                        R_K_L_frontend_spatial(1, 0),
                        R_K_L_frontend_spatial(0, 0));

                frontend_abs_yaw_deg =
                    std::abs(
                        std::atan2(
                            std::sin(frontend_yaw_rad),
                            std::cos(frontend_yaw_rad))) *
                    180.0 / M_PI;
            }
        }

        std::cout
            << "Reverse Spatial Candidate V2.5"
            << " | current_kf=" << current_keyframe.id
            << " | rank=" << (index + 1)
            << " | historical_kf="
            << candidate.candidate_id
            << " | frontend_dt="
            << candidate.distance << " m"
            << " | frontend_abs_yaw="
            << frontend_abs_yaw_deg << " deg"
            << " | keyframe_gap="
            << (current_keyframe.id -
                candidate.candidate_id)
            << " | source=FRONTEND_SPATIAL_REVERSE"
            << " | action=INJECT"
            << std::endl;
    }

    // ====================================================================
    // V20.2 / V2.5 reverse-sequence seed track.
    //
    // Problem:
    //   In a long straight agricultural row, one frame can be geometrically
    //   compatible with both the 0-deg and 180-deg basins.  The normal loop
    //   track cannot help yet because it only exists AFTER geometry/graph
    //   verification succeeds.
    //
    // Solution:
    //   Build a tiny PRE-verification temporal track using RAW Scan Context
    //   candidates only.  A reverse sequence is confirmed only when:
    //
    //     current KF increases,
    //     and either:
    //       A) historical KF makes repeated negative progress, or
    //       B) after >=1 negative progress event, RAW-SC stays inside a
    //          stable +/-1 historical-KF neighborhood for >=4 observations,
    //     while frontend position is locally close and relative yaw is
    //     approximately opposite.
    //
    // This does NOT accept a loop.  It only enables
    // CANDIDATE_REVERSE_SEQUENCE later in the verifier.
    // ====================================================================
    static ReverseLoopSeedTrackState reverse_loop_seed_track;

    bool reverse_sequence_confirmed = false;

    // V2.2:
    // A +1 historical-KF jump is allowed as one-frame Scan Context jitter.
    // We intentionally do NOT move the persistent track anchor for that jitter,
    // but we must preserve an already-established reverse-sequence confirmation
    // for the current frame.
    bool reverse_sequence_preserved_forward_jitter = false;

    bool reverse_seed_candidate_found = false;

    std::size_t reverse_sequence_historical_kf =
        std::numeric_limits<std::size_t>::max();

    double reverse_seed_best_frontend_distance =
        std::numeric_limits<double>::infinity();

    double reverse_seed_best_yaw_deg =
        std::numeric_limits<double>::quiet_NaN();

    const char *reverse_seed_best_source =
        "NONE";

    const auto consider_reverse_seed_candidate =
        [this,
         &current_keyframe,
         &reverse_seed_candidate_found,
         &reverse_sequence_historical_kf,
         &reverse_seed_best_frontend_distance,
         &reverse_seed_best_yaw_deg,
         &reverse_seed_best_source](
            const LoopCandidate &candidate,
            const char *source_name)
    {
        const Keyframe *historical_keyframe =
            FindBackendKeyframeById(
                candidate.candidate_id);

        if (historical_keyframe == nullptr ||
            !historical_keyframe->T_WL.matrix().allFinite())
        {
            return;
        }

        const Eigen::Isometry3d T_K_L_frontend_seed =
            historical_keyframe->T_WL.inverse() *
            current_keyframe.T_WL;

        if (!T_K_L_frontend_seed.matrix().allFinite())
        {
            return;
        }

        const double frontend_distance =
            T_K_L_frontend_seed.translation().norm();

        const Eigen::Matrix3d R_K_L_frontend_seed =
            T_K_L_frontend_seed.rotation();

        const double frontend_yaw_rad =
            std::atan2(
                R_K_L_frontend_seed(1, 0),
                R_K_L_frontend_seed(0, 0));

        const double frontend_abs_yaw_deg =
            std::abs(
                std::atan2(
                    std::sin(frontend_yaw_rad),
                    std::cos(frontend_yaw_rad))) *
            180.0 / M_PI;

        const bool locally_close =
            std::isfinite(frontend_distance) &&
            frontend_distance <=
                kReverseLoopSeedMaxFrontendDistance;

        const bool opposite_heading =
            std::isfinite(frontend_abs_yaw_deg) &&
            frontend_abs_yaw_deg >=
                kReverseLoopSeedMinYawDeg;

        if (!locally_close ||
            !opposite_heading)
        {
            return;
        }

        // V2.5:
        // Prefer the physically closest reverse candidate from the UNION of
        // normal Scan Context discovery and the conservative frontend-spatial
        // reverse injection route.
        if (!reverse_seed_candidate_found ||
            frontend_distance <
                reverse_seed_best_frontend_distance)
        {
            reverse_seed_candidate_found = true;
            reverse_sequence_historical_kf =
                candidate.candidate_id;
            reverse_seed_best_frontend_distance =
                frontend_distance;
            reverse_seed_best_yaw_deg =
                frontend_abs_yaw_deg;
            reverse_seed_best_source =
                source_name;
        }
    };

    for (const LoopCandidate &candidate :
         candidates)
    {
        consider_reverse_seed_candidate(
            candidate,
            "RAW_SC");
    }

    for (const LoopCandidate &candidate :
         hierarchical_retrieval_candidates)
    {
        consider_reverse_seed_candidate(
            candidate,
            "HIERARCHICAL_3M_SC");
    }

    for (const LoopCandidate &candidate :
         reverse_spatial_candidates)
    {
        consider_reverse_seed_candidate(
            candidate,
            "REVERSE_SPATIAL");
    }

    if (reverse_seed_candidate_found)
    {
        const std::size_t current_id =
            current_keyframe.id;

        const std::size_t historical_id =
            reverse_sequence_historical_kf;

        if (!reverse_loop_seed_track.valid ||
            current_id <=
                reverse_loop_seed_track
                    .last_current_keyframe_id)
        {
            reverse_loop_seed_track =
                ReverseLoopSeedTrackState();

            reverse_loop_seed_track.valid = true;
            reverse_loop_seed_track.last_current_keyframe_id =
                current_id;
            reverse_loop_seed_track.last_historical_keyframe_id =
                historical_id;
            reverse_loop_seed_track.support = 1;
            reverse_loop_seed_track
                .stable_center_historical_keyframe_id =
                historical_id;
            reverse_loop_seed_track.stable_neighborhood_support = 1;

            std::cout
                << "Reverse SC seed track V20.2"
                << " | current_kf=" << current_id
                << " | historical_kf=" << historical_id
                << " | support=1/"
                << kReverseLoopSeedMinSupport
                << " | progress=0/"
                << kReverseLoopSeedMinProgressEvents
                << " | frontend_dt="
                << reverse_seed_best_frontend_distance
                << " m"
                << " | frontend_abs_yaw="
                << reverse_seed_best_yaw_deg
                << " deg"
                << " | action=START"
                << std::endl;
        }
        else
        {
            const std::size_t current_gap =
                current_id -
                reverse_loop_seed_track
                    .last_current_keyframe_id;

            const std::int64_t historical_delta =
                static_cast<std::int64_t>(
                    historical_id) -
                static_cast<std::int64_t>(
                    reverse_loop_seed_track
                        .last_historical_keyframe_id);

            const std::uint64_t historical_step =
                static_cast<std::uint64_t>(
                    historical_delta >= 0
                        ? historical_delta
                        : -historical_delta);

            const bool current_gap_ok =
                current_gap >= 1 &&
                current_gap <=
                    kReverseLoopSeedMaxCurrentGap;

            const bool historical_step_ok =
                historical_step <=
                kReverseLoopSeedMaxHistoricalStep;

            const bool reverse_direction_ok =
                historical_delta <=
                kReverseLoopSeedForwardJitter;

            if (current_gap_ok &&
                historical_step_ok &&
                reverse_direction_ok)
            {
                // --------------------------------------------------------
                // V2.3 stable historical neighborhood.
                //
                // Keep a separate center that is NOT the monotonic progress
                // anchor.  If raw SC keeps returning the same historical
                // Keyframe or an adjacent one (+/-1 KF), accumulate stable
                // neighborhood support.  If it leaves that neighborhood,
                // start a new stable cluster at the current historical KF.
                // --------------------------------------------------------
                const std::int64_t stable_historical_delta =
                    static_cast<std::int64_t>(historical_id) -
                    static_cast<std::int64_t>(
                        reverse_loop_seed_track
                            .stable_center_historical_keyframe_id);

                const std::uint64_t stable_historical_step =
                    static_cast<std::uint64_t>(
                        stable_historical_delta >= 0
                            ? stable_historical_delta
                            : -stable_historical_delta);

                if (stable_historical_step <=
                    kReverseLoopSeedStableHistoricalRadius)
                {
                    ++reverse_loop_seed_track
                          .stable_neighborhood_support;
                }
                else
                {
                    reverse_loop_seed_track
                        .stable_center_historical_keyframe_id =
                        historical_id;
                    reverse_loop_seed_track
                        .stable_neighborhood_support = 1;
                }

                if (historical_delta > 0)
                {
                    // Opposite-sign +1 jitter is tolerated, but it must not
                    // move the persistent sequence anchor forward and
                    // self-reinforce.
                    //
                    // V2.2:
                    // Preserve the already accumulated reverse-sequence
                    // evidence for THIS frame.  Do not increment support,
                    // do not increment progress, and do not update either
                    // persistent anchor.
                    reverse_sequence_preserved_forward_jitter = true;

                    std::cout
                        << "Reverse SC seed track V20.2"
                        << " | current_kf=" << current_id
                        << " | historical_kf=" << historical_id
                        << " | historical_delta="
                        << historical_delta
                        << " | support="
                        << reverse_loop_seed_track.support
                        << "/"
                        << kReverseLoopSeedMinSupport
                        << " | progress="
                        << reverse_loop_seed_track
                               .reverse_progress_events
                        << "/"
                        << kReverseLoopSeedMinProgressEvents
                        << " | action=PRESERVE_FORWARD_JITTER"
                        << std::endl;
                }
                else
                {
                    ++reverse_loop_seed_track.support;

                    if (historical_delta < 0)
                    {
                        ++reverse_loop_seed_track
                              .reverse_progress_events;
                    }

                    reverse_loop_seed_track
                        .last_current_keyframe_id =
                        current_id;

                    reverse_loop_seed_track
                        .last_historical_keyframe_id =
                        historical_id;

                    reverse_sequence_confirmed =
                        reverse_loop_seed_track.support >=
                            kReverseLoopSeedMinSupport &&
                        reverse_loop_seed_track
                                .reverse_progress_events >=
                            kReverseLoopSeedMinProgressEvents;

                    std::cout
                        << "Reverse SC seed track V20.2"
                        << " | current_kf=" << current_id
                        << " | historical_kf=" << historical_id
                        << " | historical_delta="
                        << historical_delta
                        << " | support="
                        << reverse_loop_seed_track.support
                        << "/"
                        << kReverseLoopSeedMinSupport
                        << " | progress="
                        << reverse_loop_seed_track
                               .reverse_progress_events
                        << "/"
                        << kReverseLoopSeedMinProgressEvents
                        << " | confirmed="
                        << (reverse_sequence_confirmed
                                ? "true"
                                : "false")
                        << " | frontend_dt="
                        << reverse_seed_best_frontend_distance
                        << " m"
                        << " | frontend_abs_yaw="
                        << reverse_seed_best_yaw_deg
                        << " deg"
                        << " | action=EXTEND"
                        << std::endl;
                }
            }
            else
            {
                reverse_loop_seed_track =
                    ReverseLoopSeedTrackState();

                reverse_loop_seed_track.valid = true;
                reverse_loop_seed_track
                    .last_current_keyframe_id =
                    current_id;
                reverse_loop_seed_track
                    .last_historical_keyframe_id =
                    historical_id;
                reverse_loop_seed_track.support = 1;
                reverse_loop_seed_track
                    .stable_center_historical_keyframe_id =
                    historical_id;
                reverse_loop_seed_track
                    .stable_neighborhood_support = 1;

                std::cout
                    << "Reverse SC seed track V20.2"
                    << " | current_kf=" << current_id
                    << " | historical_kf=" << historical_id
                    << " | current_gap=" << current_gap
                    << " | historical_delta="
                    << historical_delta
                    << " | historical_step="
                    << historical_step
                    << " | action=RESTART"
                    << std::endl;
            }
        }

        // V2.3 confirmation rule.
        //
        // Mode A -- STRICT_PROGRESS:
        //   Keep the original requirement:
        //     support >= 3 AND reverse progress events >= 2.
        //
        // Mode B -- STABLE_NEIGHBORHOOD:
        //   Repetitive straight rows often quantize SC top-1 into the same
        //   historical KF or its immediate neighbor.  Accept the sequence as
        //   confirmed when:
        //     stable neighborhood support >= 4,
        //     at least one genuine negative historical progress event exists,
        //     frontend translation is locally close (<= 2 m),
        //     frontend relative yaw is still opposite (>= 150 deg).
        //
        // This only enables CANDIDATE_REVERSE_SEQUENCE.  It still does NOT
        // bypass Point-to-Plane V2, trusted reverse geometry, graph gates,
        // temporal consistency, the two-frame first-loop batch, or PGO.
        const bool reverse_sequence_strict_evidence_ready =
            reverse_loop_seed_track.valid &&
            reverse_loop_seed_track.support >=
                kReverseLoopSeedMinSupport &&
            reverse_loop_seed_track.reverse_progress_events >=
                kReverseLoopSeedMinProgressEvents;

        const bool reverse_sequence_anchor_matches =
            reverse_loop_seed_track.last_current_keyframe_id ==
                current_keyframe.id &&
            reverse_loop_seed_track.last_historical_keyframe_id ==
                reverse_sequence_historical_kf;

        const bool reverse_sequence_strict_confirmed =
            reverse_sequence_strict_evidence_ready &&
            (reverse_sequence_anchor_matches ||
             reverse_sequence_preserved_forward_jitter);

        const bool reverse_sequence_stable_frontend_ok =
            std::isfinite(reverse_seed_best_frontend_distance) &&
            reverse_seed_best_frontend_distance <=
                kReverseLoopSeedStableMaxFrontendDistance &&
            std::isfinite(reverse_seed_best_yaw_deg) &&
            reverse_seed_best_yaw_deg >=
                kReverseLoopSeedMinYawDeg;

        const bool reverse_sequence_stable_confirmed =
            reverse_loop_seed_track.valid &&
            reverse_loop_seed_track.stable_neighborhood_support >=
                kReverseLoopSeedStableMinSupport &&
            reverse_loop_seed_track.reverse_progress_events >=
                kReverseLoopSeedStableMinProgressEvents &&
            reverse_sequence_stable_frontend_ok;

        reverse_sequence_confirmed =
            reverse_sequence_strict_confirmed ||
            reverse_sequence_stable_confirmed;

        const char *reverse_sequence_confirmation_mode =
            reverse_sequence_strict_confirmed
                ? "STRICT_PROGRESS"
                : (reverse_sequence_stable_confirmed
                       ? "STABLE_NEIGHBORHOOD"
                       : "NONE");

        std::cout
            << "FR_LOOP_TRACE"
            << " | stage=REVERSE_SEED"
            << " | current_kf=" << current_keyframe.id
            << " | historical_kf="
            << reverse_sequence_historical_kf
            << " | support="
            << reverse_loop_seed_track.support
            << " | progress="
            << reverse_loop_seed_track.reverse_progress_events
            << " | stable_support="
            << reverse_loop_seed_track.stable_neighborhood_support
            << "/"
            << kReverseLoopSeedStableMinSupport
            << " | stable_center_kf="
            << reverse_loop_seed_track
                   .stable_center_historical_keyframe_id
            << " | frontend_dt="
            << reverse_seed_best_frontend_distance
            << " m"
            << " | frontend_yaw="
            << reverse_seed_best_yaw_deg
            << " deg"
            << " | confirmed="
            << (reverse_sequence_confirmed
                    ? "true"
                    : "false")
            << " | confirmation_mode="
            << reverse_sequence_confirmation_mode
            << " | seed_source="
            << reverse_seed_best_source
            << std::endl;
    }
    else if (reverse_loop_seed_track.valid &&
             current_keyframe.id >
                 reverse_loop_seed_track
                         .last_current_keyframe_id +
                     kReverseLoopSeedMaxCurrentGap)
    {
        std::cout
            << "Reverse SC seed track V20.2"
            << " | current_kf=" << current_keyframe.id
            << " | last_current_kf="
            << reverse_loop_seed_track
                   .last_current_keyframe_id
            << " | action=RESET_STALE"
            << std::endl;

        reverse_loop_seed_track =
            ReverseLoopSeedTrackState();
    }

    // ====================================================================
    // V19.4-SC-DIAG -- PURE Scan Context ranking diagnostics.
    //
    // Why:
    //   LoopDetector returns only the normal gated Top-K candidates.  For the
    //   fold-back experiment we need to know whether the expected historical
    //   KF (e.g. 95/96/97) is:
    //
    //     A) valid but ranked below Top5,
    //     B) valid but just outside the SC distance gate,
    //     C) invalid because coverage/support failed.
    //
    // This block intentionally RECOMPUTES descriptors only for KF130..170.
    // It does NOT modify the real candidate list or any loop acceptance state.
    // ====================================================================
    constexpr bool kScDiagnosticEnabled = false;
    constexpr std::size_t kScDiagnosticBeginKf = 130;
    constexpr std::size_t kScDiagnosticEndKf = 170;
    constexpr std::size_t kScDiagnosticTopK = 30;
    constexpr std::size_t kScTrackNeighborhoodRadius = 3;

    if (kScDiagnosticEnabled &&
        current_keyframe.id >= kScDiagnosticBeginKf &&
        current_keyframe.id <= kScDiagnosticEndKf)
    {
        struct ScanContextDiagnosticEntry
        {
            std::size_t historical_keyframe_id = 0;
            double time_separation_sec =
                std::numeric_limits<double>::infinity();
            double pose_distance =
                std::numeric_limits<double>::infinity();

            bool descriptor_valid = false;
            bool match_valid = false;

            ScanContextMatch match;

            bool passed_sc_gate = false;
            bool returned_by_detector = false;
        };

        const LoopDetectorConfig &loop_detector_config =
            loop_detector_.GetConfig();

        ScanContext diagnostic_scan_context(
            loop_detector_config.scan_context);

        const ScanContextDescriptor current_descriptor =
            diagnostic_scan_context.MakeDescriptor(
                current_keyframe.cloud);

        std::vector<ScanContextDiagnosticEntry>
            sc_diagnostic_entries;

        std::vector<ScanContextDiagnosticEntry>
            sc_track_neighborhood_entries;

        if (!current_descriptor.valid)
        {
            std::cout
                << "SC_DIAGNOSTIC_TOP30"
                << " | current_kf=" << current_keyframe.id
                << " | status=CURRENT_DESCRIPTOR_INVALID"
                << std::endl;
        }
        else
        {
            sc_diagnostic_entries.reserve(
                backend_keyframes_.size());

            sc_track_neighborhood_entries.reserve(
                2 * kScTrackNeighborhoodRadius + 1);

            const bool has_expected_track_center =
                online_loop_track_.valid;

            const std::size_t expected_track_center =
                has_expected_track_center
                    ? online_loop_track_.last_historical_keyframe_id
                    : std::numeric_limits<std::size_t>::max();

            for (const Keyframe &history_keyframe :
                 backend_keyframes_)
            {
                if (history_keyframe.id >= current_keyframe.id)
                {
                    continue;
                }

                const std::size_t id_gap =
                    current_keyframe.id -
                    history_keyframe.id;

                if (id_gap <
                    loop_detector_config.min_keyframe_id_separation)
                {
                    continue;
                }

                const double time_separation =
                    current_keyframe.timestamp -
                    history_keyframe.timestamp;

                if (!std::isfinite(time_separation) ||
                    time_separation <
                        loop_detector_config.min_time_separation_sec)
                {
                    continue;
                }

                const bool in_track_neighborhood =
                    has_expected_track_center &&
                    (history_keyframe.id >= expected_track_center
                         ? history_keyframe.id - expected_track_center
                         : expected_track_center - history_keyframe.id) <=
                        kScTrackNeighborhoodRadius;

                ScanContextDiagnosticEntry entry;
                entry.historical_keyframe_id =
                    history_keyframe.id;
                entry.time_separation_sec =
                    time_separation;

                if (history_keyframe.T_WL.matrix().allFinite())
                {
                    entry.pose_distance =
                        (current_keyframe.T_WL.translation() -
                         history_keyframe.T_WL.translation())
                            .norm();
                }

                if (!history_keyframe.cloud ||
                    history_keyframe.cloud->empty())
                {
                    if (in_track_neighborhood)
                    {
                        sc_track_neighborhood_entries.push_back(entry);
                    }
                    continue;
                }

                const ScanContextDescriptor historical_descriptor =
                    diagnostic_scan_context.MakeDescriptor(
                        history_keyframe.cloud);

                entry.descriptor_valid =
                    historical_descriptor.valid;

                if (!historical_descriptor.valid)
                {
                    if (in_track_neighborhood)
                    {
                        sc_track_neighborhood_entries.push_back(entry);
                    }
                    continue;
                }

                entry.match =
                    diagnostic_scan_context.Compare(
                        historical_descriptor,
                        current_descriptor);

                entry.match_valid =
                    entry.match.valid &&
                    std::isfinite(entry.match.distance);

                if (entry.match_valid)
                {
                    entry.passed_sc_gate =
                        entry.match.distance <=
                        loop_detector_config
                            .max_scan_context_distance;

                    entry.returned_by_detector =
                        std::any_of(
                            candidates.begin(),
                            candidates.end(),
                            [&entry](const LoopCandidate &candidate)
                            {
                                return candidate.candidate_id ==
                                       entry.historical_keyframe_id;
                            });

                    sc_diagnostic_entries.push_back(entry);
                }

                if (in_track_neighborhood)
                {
                    sc_track_neighborhood_entries.push_back(entry);
                }
            }

            std::sort(
                sc_diagnostic_entries.begin(),
                sc_diagnostic_entries.end(),
                [](const ScanContextDiagnosticEntry &lhs,
                   const ScanContextDiagnosticEntry &rhs)
                {
                    if (lhs.match.distance != rhs.match.distance)
                    {
                        return lhs.match.distance <
                               rhs.match.distance;
                    }

                    return lhs.pose_distance <
                           rhs.pose_distance;
                });

            const std::size_t top_count =
                std::min(
                    kScDiagnosticTopK,
                    sc_diagnostic_entries.size());

            std::cout
                << "SC_DIAGNOSTIC_TOP30_BEGIN"
                << " | current_kf=" << current_keyframe.id
                << " | valid_matches="
                << sc_diagnostic_entries.size()
                << " | printed=" << top_count
                << " | real_sc_gate="
                << loop_detector_config.max_scan_context_distance
                << " | real_topk="
                << loop_detector_config.max_candidates
                << " | BEHAVIOR_UNCHANGED=true"
                << std::endl;

            for (std::size_t rank = 0;
                 rank < top_count;
                 ++rank)
            {
                const ScanContextDiagnosticEntry &entry =
                    sc_diagnostic_entries[rank];

                std::cout
                    << "SC_DIAGNOSTIC_TOP30"
                    << " | current_kf=" << current_keyframe.id
                    << " | rank=" << (rank + 1)
                    << " | historical_kf="
                    << entry.historical_keyframe_id
                    << " | sc_distance="
                    << entry.match.distance
                    << " | sc_similarity="
                    << entry.match.similarity
                    << " | sc_raw_cosine="
                    << entry.match.raw_cosine_similarity
                    << " | sc_sector_coverage="
                    << entry.match.sector_coverage_ratio
                    << " | sc_cell_coverage="
                    << entry.match.cell_coverage_ratio
                    << " | sc_compared_sectors="
                    << entry.match.compared_sectors
                    << " | sector_shift="
                    << entry.match.sector_shift
                    << " | yaw_shift="
                    << entry.match.yaw_shift_deg
                    << " deg"
                    << " | pose_distance="
                    << entry.pose_distance << " m"
                    << " | time_separation="
                    << entry.time_separation_sec << " s"
                    << " | passed_sc_gate="
                    << (entry.passed_sc_gate ? "true" : "false")
                    << " | returned_by_detector="
                    << (entry.returned_by_detector
                            ? "true"
                            : "false")
                    << std::endl;
            }

            std::cout
                << "SC_DIAGNOSTIC_TOP30_END"
                << " | current_kf=" << current_keyframe.id
                << std::endl;

            // ------------------------------------------------------------
            // Track-neighborhood diagnosis:
            //
            // Once a loop seed exists, inspect exactly +/-3 historical KFs
            // around the previous geometrically verified historical anchor.
            // These entries are printed even if their descriptor/match is
            // invalid, which tells us whether the correct continuation
            // disappeared because of ranking, distance gate, or SC validity.
            // ------------------------------------------------------------
            if (has_expected_track_center)
            {
                std::sort(
                    sc_track_neighborhood_entries.begin(),
                    sc_track_neighborhood_entries.end(),
                    [](const ScanContextDiagnosticEntry &lhs,
                       const ScanContextDiagnosticEntry &rhs)
                    {
                        return lhs.historical_keyframe_id <
                               rhs.historical_keyframe_id;
                    });

                std::cout
                    << "SC_TRACK_NEIGHBORHOOD_BEGIN"
                    << " | current_kf=" << current_keyframe.id
                    << " | expected_center="
                    << expected_track_center
                    << " | radius=+/-"
                    << kScTrackNeighborhoodRadius
                    << std::endl;

                for (const ScanContextDiagnosticEntry &entry :
                     sc_track_neighborhood_entries)
                {
                    std::size_t global_rank = 0;

                    if (entry.match_valid)
                    {
                        const auto rank_iterator =
                            std::find_if(
                                sc_diagnostic_entries.begin(),
                                sc_diagnostic_entries.end(),
                                [&entry](
                                    const ScanContextDiagnosticEntry &candidate)
                                {
                                    return candidate.historical_keyframe_id ==
                                           entry.historical_keyframe_id;
                                });

                        if (rank_iterator !=
                            sc_diagnostic_entries.end())
                        {
                            global_rank =
                                static_cast<std::size_t>(
                                    std::distance(
                                        sc_diagnostic_entries.begin(),
                                        rank_iterator)) +
                                1;
                        }
                    }

                    std::cout
                        << "SC_TRACK_NEIGHBORHOOD"
                        << " | current_kf=" << current_keyframe.id
                        << " | expected_center="
                        << expected_track_center
                        << " | historical_kf="
                        << entry.historical_keyframe_id
                        << " | descriptor_valid="
                        << (entry.descriptor_valid ? "true" : "false")
                        << " | match_valid="
                        << (entry.match_valid ? "true" : "false")
                        << " | global_rank=" << global_rank;

                    if (entry.match_valid)
                    {
                        std::cout
                            << " | sc_distance="
                            << entry.match.distance
                            << " | sc_similarity="
                            << entry.match.similarity
                            << " | sc_raw_cosine="
                            << entry.match.raw_cosine_similarity
                            << " | sc_sector_coverage="
                            << entry.match.sector_coverage_ratio
                            << " | sc_cell_coverage="
                            << entry.match.cell_coverage_ratio
                            << " | sc_compared_sectors="
                            << entry.match.compared_sectors
                            << " | sector_shift="
                            << entry.match.sector_shift
                            << " | yaw_shift="
                            << entry.match.yaw_shift_deg
                            << " deg"
                            << " | passed_sc_gate="
                            << (entry.passed_sc_gate ? "true" : "false")
                            << " | returned_by_detector="
                            << (entry.returned_by_detector
                                    ? "true"
                                    : "false");
                    }

                    std::cout
                        << " | pose_distance="
                        << entry.pose_distance << " m"
                        << " | time_separation="
                        << entry.time_separation_sec << " s"
                        << std::endl;
                }

                std::cout
                    << "SC_TRACK_NEIGHBORHOOD_END"
                    << " | current_kf=" << current_keyframe.id
                    << std::endl;
            }
        }
    }

    // ====================================================================
    // V22.6 3m Track Revalidation deadline.
    //
    // Measure physical FRONTEND trajectory arc from the last track checkpoint
    // to the current KF.  Once >=3m, the old track is no longer allowed to
    // inject / prioritize a historical KF.  The current frame must regain
    // independent segment evidence first.
    // ====================================================================
    double track_segment_revalidation_arc_m = 0.0;
    bool track_segment_revalidation_arc_valid = true;
    bool track_segment_revalidation_required = false;

    if (online_loop_track_.valid &&
        loop_track_segment_revalidation_v226.valid)
    {
        const std::size_t checkpoint_kf =
            loop_track_segment_revalidation_v226
                .checkpoint_current_keyframe_id;

        if (current_keyframe.id < checkpoint_kf)
        {
            track_segment_revalidation_arc_valid = false;
        }
        else
        {
            const Keyframe *previous_checkpoint_member =
                find_retrieval_keyframe(checkpoint_kf);

            if (previous_checkpoint_member == nullptr ||
                !previous_checkpoint_member->T_WL.matrix().allFinite())
            {
                track_segment_revalidation_arc_valid = false;
            }
            else
            {
                for (std::size_t keyframe_id = checkpoint_kf + 1;
                     keyframe_id <= current_keyframe.id;
                     ++keyframe_id)
                {
                    const Keyframe *member =
                        find_retrieval_keyframe(keyframe_id);

                    if (member == nullptr ||
                        !member->T_WL.matrix().allFinite())
                    {
                        track_segment_revalidation_arc_valid = false;
                        break;
                    }

                    const Eigen::Vector3d delta =
                        member->T_WL.translation() -
                        previous_checkpoint_member->T_WL.translation();

                    if (!delta.allFinite())
                    {
                        track_segment_revalidation_arc_valid = false;
                        break;
                    }

                    track_segment_revalidation_arc_m += delta.norm();
                    previous_checkpoint_member = member;

                    if (track_segment_revalidation_arc_m >=
                        kLoopTrackSegmentRevalidationArcM)
                    {
                        break;
                    }
                }
            }
        }

        track_segment_revalidation_required =
            !track_segment_revalidation_arc_valid ||
            track_segment_revalidation_arc_m >=
                kLoopTrackSegmentRevalidationArcM;

        std::cout
            << "LOOP_TRACK_3M_REVALIDATION_V22_8"
            << " | current_kf=" << current_keyframe.id
            << " | checkpoint_current_kf=" << checkpoint_kf
            << " | checkpoint_historical_kf="
            << loop_track_segment_revalidation_v226
                   .checkpoint_historical_keyframe_id
            << " | checkpoint_segment_verified="
            << (loop_track_segment_revalidation_v226
                        .checkpoint_was_segment_verified
                    ? "true"
                    : "false")
            << " | current_arc="
            << track_segment_revalidation_arc_m << " m"
            << " | arc_valid="
            << (track_segment_revalidation_arc_valid
                    ? "true"
                    : "false")
            << " | threshold="
            << kLoopTrackSegmentRevalidationArcM << " m"
            << " | required="
            << (track_segment_revalidation_required
                    ? "true"
                    : "false")
            << std::endl;
    }

    // ====================================================================
    // V14 Active-Track Continuation eligibility.
    //
    // Scan Context is the GLOBAL discovery mechanism.  Once a geometrically
    // verified temporal loop track already exists, however, the next frame
    // must not depend on Scan Context returning the same historical place
    // again.  In repetitive agricultural rows/corridors SC can momentarily
    // omit the correct historical KF even though the previous two loop
    // measurements were mutually consistent.
    //
    // A live track is therefore allowed to reach the Candidate Manager even
    // when the current Scan Context candidate list is empty.  The actual
    // continuation still has to pass full ICP, ICP-correction, graph, and
    // temporal-consistency gates below.
    // ====================================================================
    bool active_track_continuation_eligible = false;
    std::size_t active_track_current_gap = 0;

    if (!track_segment_revalidation_required &&
        online_loop_track_.valid &&
        current_keyframe.id >=
            online_loop_track_.last_current_keyframe_id)
    {
        active_track_current_gap =
            current_keyframe.id -
            online_loop_track_.last_current_keyframe_id;

        // V15 tentative continuation:
        //
        // A single graph-consistent verified loop observation is allowed a
        // short grace window to request ICP continuation even if Scan Context
        // misses the next few Keyframes.  This does NOT submit a loop edge and
        // does NOT bypass any geometry / ICP / graph / temporal gate below.
        // Once support reaches 2, fall back to the stricter active-track gap.
        const std::size_t continuation_gap_limit =
            online_loop_track_.support >= 2
                ? online_loop_max_current_keyframe_gap_
                : online_loop_tentative_max_current_keyframe_gap_;

        active_track_continuation_eligible =
            online_loop_track_.support >= 1 &&
            active_track_current_gap <=
                continuation_gap_limit;
    }

    if (candidates.empty())
    {
        if (detection_diagnostics.has_best_scan_context_match)
        {
            const LoopCandidate &best =
                detection_diagnostics.best_scan_context_match;

            std::cout
                << "Keyframe loop best rejected Scan Context match"
                << " | current_kf=" << best.current_id
                << " | historical_kf=" << best.candidate_id
                << " | sc_distance=" << best.scan_context_distance
                << " | sc_gate="
                << detection_diagnostics.max_scan_context_distance
                << " | sc_similarity=" << best.scan_context_similarity
                << " | sc_raw_cosine="
                << best.scan_context_raw_cosine_similarity
                << " | sc_sector_coverage="
                << best.scan_context_sector_coverage_ratio
                << " | sc_cell_coverage="
                << best.scan_context_cell_coverage_ratio
                << " | sc_compared_sectors="
                << best.scan_context_compared_sectors
                << " | reason=SCAN_CONTEXT_DISTANCE_GATE"
                << std::endl;
        }
        else
        {
            std::cout
                << "Keyframe loop has no valid Scan Context comparison"
                << " | current_kf=" << current_keyframe.id
                << " | eligible="
                << detection_diagnostics.separation_eligible
                << " | reason=SPARSE_OR_INSUFFICIENT_COVERAGE"
                << std::endl;
        }

        const bool reverse_spatial_continuation_eligible =
            !reverse_spatial_candidates.empty();

        const bool hierarchical_retrieval_eligible =
            !hierarchical_retrieval_candidates.empty();

        if (!active_track_continuation_eligible &&
            !reverse_spatial_continuation_eligible &&
            !hierarchical_retrieval_eligible)
        {
            return;
        }

        if (hierarchical_retrieval_eligible)
        {
            std::cout
                << "Hierarchical SC V21"
                << " | current_kf=" << current_keyframe.id
                << " | injected="
                << hierarchical_retrieval_candidates.size()
                << " | action=BYPASS_EMPTY_SINGLE_KF_SCAN_CONTEXT"
                << std::endl;
        }

        if (reverse_spatial_continuation_eligible)
        {
            std::cout
                << "Reverse Spatial Candidate V2.5"
                << " | current_kf=" << current_keyframe.id
                << " | injected="
                << reverse_spatial_candidates.size()
                << " | action=BYPASS_EMPTY_SCAN_CONTEXT"
                << std::endl;
        }

        if (active_track_continuation_eligible)
        {
            std::cout
                << "Loop Track Continuation V20"
                << " | current_kf=" << current_keyframe.id
                << " | action=BYPASS_EMPTY_SCAN_CONTEXT"
                << " | last_historical_kf="
                << online_loop_track_.last_historical_keyframe_id
                << " | current_gap=" << active_track_current_gap
                << std::endl;
        }
    }

    const LoopCandidate *best_candidate = nullptr;
    const BackendSubmapSnapshot *best_historical_submap = nullptr;
    LoopVerificationResult best_verification;
    bool best_candidate_used_reverse_sequence_target = false;
    bool best_candidate_used_hierarchical_segment_target = false;
    std::vector<std::size_t> best_candidate_target_keyframes;

    // ====================================================================
    // V14 Candidate Manager + Active Track Continuation
    //
    // 1. Keep the Scan Context candidate set, but if a temporal loop track
    //    already exists, promote candidates that continue the same historical
    //    Keyframe neighborhood.  This turns temporal consistency from a pure
    //    post-hoc rejector into a lightweight continuation prior.
    //
    // 2. Apply historical-Submap de-duplication BEFORE consuming the ICP
    //    verification budget.  The old implementation first truncated the
    //    raw Scan Context list to top-3 and only then de-duplicated; therefore
    //    several adjacent KFs from one wrong region could starve a true rank-5
    //    revisit completely.
    //
    // 3. Verify up to max_loop_candidates_to_verify_ UNIQUE historical
    //    Submaps (default V14 = 5).
    // ====================================================================
    LoopCandidate track_continuation_candidate;
    bool track_continuation_candidate_injected = false;

    // V18 injected historical-neighborhood candidates live in this vector so
    // candidate_order can safely store pointers to them for the duration of
    // this function call.
    std::vector<LoopCandidate> large_drift_neighborhood_candidates;
    large_drift_neighborhood_candidates.reserve(
        2 * online_loop_large_drift_neighborhood_radius_);

    std::vector<std::size_t> large_drift_injected_candidate_ids;
    large_drift_injected_candidate_ids.reserve(
        2 * online_loop_large_drift_neighborhood_radius_ +
        online_loop_drift_aware_corridor_max_candidates_);

    // V19.4 candidates found around the CORRECTED predicted current pose.
    std::vector<LoopCandidate> drift_aware_corridor_candidates;
    drift_aware_corridor_candidates.reserve(
        online_loop_drift_aware_corridor_max_candidates_);

    std::vector<std::size_t> drift_aware_corridor_candidate_ids;
    drift_aware_corridor_candidate_ids.reserve(
        online_loop_drift_aware_corridor_max_candidates_);

    // V22.4 reverse segment sliding anchors.  A confirmed reverse sequence
    // identifies the historical TRAJECTORY SEGMENT, not an exact LiDAR sample
    // position.  We therefore explicitly guarantee K-2..K+2 anchors are
    // available for 3m<->3m verification.
    constexpr std::size_t kReverseSegmentSlidingRadius = 2;
    std::vector<LoopCandidate> reverse_segment_sliding_candidates;
    reverse_segment_sliding_candidates.reserve(
        2 * kReverseSegmentSlidingRadius + 1);

    const double track_correction_translation =
        online_loop_track_.valid
            ? online_loop_track_.T_loop_correction.translation().norm()
            : 0.0;

    const double track_correction_rotation =
        online_loop_track_.valid
            ? RelativeRotationDeg(
                  Eigen::Isometry3d::Identity(),
                  online_loop_track_.T_loop_correction)
            : 0.0;

    const bool first_loop_large_drift_track =
        !has_last_online_loop_edge_ &&
        online_loop_track_.valid &&
        online_loop_track_.support >= 1 &&
        ((std::isfinite(track_correction_translation) &&
          track_correction_translation >
              online_loop_large_drift_trigger_translation_) ||
         (std::isfinite(track_correction_rotation) &&
          track_correction_rotation >
              online_loop_large_drift_trigger_rotation_deg_));

    std::vector<const LoopCandidate *> candidate_order;
    candidate_order.reserve(
        hierarchical_retrieval_candidates.size() +
        reverse_spatial_candidates.size() +
        candidates.size() +
        1 +
        2 * online_loop_large_drift_neighborhood_radius_ +
        online_loop_drift_aware_corridor_max_candidates_ +
        2 * kReverseSegmentSlidingRadius + 1);

    const auto append_candidate_order_unique =
        [&candidate_order](const LoopCandidate *candidate)
    {
        if (candidate == nullptr)
        {
            return;
        }

        const bool duplicate =
            std::any_of(
                candidate_order.begin(),
                candidate_order.end(),
                [candidate](const LoopCandidate *existing)
                {
                    return existing != nullptr &&
                           existing->candidate_id ==
                               candidate->candidate_id;
                });

        if (!duplicate)
        {
            candidate_order.push_back(candidate);
        }
    };

    // V21 region retrieval gets an early chance because its purpose is to
    // recover the true historical area when the old single-KF SC hard gate
    // returns zero or ranks the true KF too low.
    for (const LoopCandidate &candidate :
         hierarchical_retrieval_candidates)
    {
        append_candidate_order_unique(&candidate);
    }

    // V2.5 reverse-spatial discovery remains an independent supplemental
    // route.  Duplicate historical KFs are removed before verification.
    for (const LoopCandidate &candidate :
         reverse_spatial_candidates)
    {
        append_candidate_order_unique(&candidate);
    }

    for (const LoopCandidate &candidate : candidates)
    {
        append_candidate_order_unique(&candidate);
    }

    bool track_priority_active = false;
    std::size_t track_priority_radius = 0;

    if (!track_segment_revalidation_required &&
        online_loop_track_.valid &&
        current_keyframe.id >=
            online_loop_track_.last_current_keyframe_id)
    {
        const std::size_t current_gap =
            current_keyframe.id -
            online_loop_track_.last_current_keyframe_id;

        const std::size_t track_priority_gap_limit =
            online_loop_track_.support >= 2
                ? online_loop_max_current_keyframe_gap_
                : online_loop_tentative_max_current_keyframe_gap_;

        if (current_gap <=
            track_priority_gap_limit)
        {
            track_priority_active = true;

            track_priority_radius =
                online_loop_max_historical_progression_step_ *
                std::max<std::size_t>(1, current_gap);

            const std::int64_t last_historical =
                static_cast<std::int64_t>(
                    online_loop_track_.last_historical_keyframe_id);

            const std::int64_t backtrack_tolerance =
                static_cast<std::int64_t>(
                    online_loop_historical_backtrack_tolerance_);

            const auto track_compatible =
                [this,
                 last_historical,
                 track_priority_radius,
                 backtrack_tolerance](const LoopCandidate *candidate)
            {
                if (candidate == nullptr)
                {
                    return false;
                }

                const std::int64_t delta =
                    static_cast<std::int64_t>(
                        candidate->candidate_id) -
                    last_historical;

                const std::uint64_t absolute_delta =
                    static_cast<std::uint64_t>(
                        delta >= 0 ? delta : -delta);

                if (absolute_delta > track_priority_radius)
                {
                    return false;
                }

                if (online_loop_track_.historical_direction > 0 &&
                    delta < -backtrack_tolerance)
                {
                    return false;
                }

                if (online_loop_track_.historical_direction < 0 &&
                    delta > backtrack_tolerance)
                {
                    return false;
                }

                return true;
            };

            std::stable_sort(
                candidate_order.begin(),
                candidate_order.end(),
                [&track_compatible, last_historical](
                    const LoopCandidate *lhs,
                    const LoopCandidate *rhs)
                {
                    const bool lhs_track =
                        track_compatible(lhs);
                    const bool rhs_track =
                        track_compatible(rhs);

                    if (lhs_track != rhs_track)
                    {
                        return lhs_track;
                    }

                    if (!lhs_track ||
                        lhs == nullptr ||
                        rhs == nullptr)
                    {
                        return false;
                    }

                    const std::int64_t lhs_delta =
                        static_cast<std::int64_t>(lhs->candidate_id) -
                        last_historical;
                    const std::int64_t rhs_delta =
                        static_cast<std::int64_t>(rhs->candidate_id) -
                        last_historical;

                    const std::uint64_t lhs_distance =
                        static_cast<std::uint64_t>(
                            lhs_delta >= 0 ? lhs_delta : -lhs_delta);
                    const std::uint64_t rhs_distance =
                        static_cast<std::uint64_t>(
                            rhs_delta >= 0 ? rhs_delta : -rhs_delta);

                    return lhs_distance < rhs_distance;
                });

            const bool has_scan_context_track_candidate =
                std::any_of(
                    candidate_order.begin(),
                    candidate_order.end(),
                    [&track_compatible](const LoopCandidate *candidate)
                    {
                        return track_compatible(candidate);
                    });

            // If Scan Context did not return anything near the active
            // historical track, explicitly inject the last verified
            // historical KF.  Its ICP initial guess will be generated from
            // the previous loop-implied world correction (TRACK_PREDICTED),
            // not from Scan Context yaw.
            if (!has_scan_context_track_candidate &&
                online_loop_track_.support >= 1)
            {
                track_continuation_candidate = LoopCandidate();
                track_continuation_candidate.current_id =
                    current_keyframe.id;
                track_continuation_candidate.candidate_id =
                    online_loop_track_.last_historical_keyframe_id;
                track_continuation_candidate.distance =
                    std::numeric_limits<double>::infinity();
                track_continuation_candidate.scan_context_distance =
                    std::numeric_limits<double>::infinity();
                track_continuation_candidate.scan_context_similarity = 0.0;
                track_continuation_candidate.yaw_shift_deg =
                    std::numeric_limits<double>::quiet_NaN();

                candidate_order.insert(
                    candidate_order.begin(),
                    &track_continuation_candidate);

                track_continuation_candidate_injected = true;
            }
        }
    }

    // --------------------------------------------------------------------
    // V18 large-drift historical-neighborhood expansion.
    //
    // If the first verified loop says that the accumulated frontend drift is
    // already large, do not wait for Scan Context to independently return
    // history 95/97/98 on subsequent frames.  Probe nearby historical KFs
    // explicitly.  Direction, once known, is tried first; otherwise +/- are
    // alternated.  These are only CANDIDATES and still undergo the full ICP,
    // graph, temporal and batch-consensus gates.
    // --------------------------------------------------------------------
    if (first_loop_large_drift_track &&
        track_priority_active)
    {
        const std::int64_t center =
            static_cast<std::int64_t>(
                online_loop_track_.last_historical_keyframe_id);

        const auto inject_large_drift_candidate =
            [this,
             &current_keyframe,
             &large_drift_neighborhood_candidates,
             &large_drift_injected_candidate_ids,
             &candidate_order](std::int64_t historical_id)
        {
            if (historical_id < 0)
            {
                return;
            }

            const std::size_t candidate_id =
                static_cast<std::size_t>(historical_id);

            if (candidate_id == current_keyframe.id ||
                FindBackendKeyframeById(candidate_id) == nullptr)
            {
                return;
            }

            if (std::find(
                    large_drift_injected_candidate_ids.begin(),
                    large_drift_injected_candidate_ids.end(),
                    candidate_id) !=
                large_drift_injected_candidate_ids.end())
            {
                return;
            }

            for (const LoopCandidate *existing :
                 candidate_order)
            {
                if (existing != nullptr &&
                    existing->candidate_id == candidate_id)
                {
                    // Already supplied by Scan Context or TRACK_PREDICTED.
                    return;
                }
            }

            LoopCandidate injected;
            injected.current_id =
                current_keyframe.id;
            injected.candidate_id =
                candidate_id;
            injected.distance =
                std::numeric_limits<double>::infinity();
            injected.scan_context_distance =
                std::numeric_limits<double>::infinity();
            injected.scan_context_similarity = 0.0;
            injected.scan_context_raw_cosine_similarity = 0.0;
            injected.scan_context_sector_coverage_ratio = 0.0;
            injected.scan_context_cell_coverage_ratio = 0.0;
            injected.scan_context_compared_sectors = 0;
            injected.sector_shift = 0;
            injected.yaw_shift_deg =
                std::numeric_limits<double>::quiet_NaN();

            large_drift_neighborhood_candidates.push_back(
                injected);

            large_drift_injected_candidate_ids.push_back(
                candidate_id);
        };

        for (std::size_t step = 1;
             step <= online_loop_large_drift_neighborhood_radius_;
             ++step)
        {
            const std::int64_t signed_step =
                static_cast<std::int64_t>(step);

            if (online_loop_track_.historical_direction > 0)
            {
                inject_large_drift_candidate(
                    center + signed_step);
                inject_large_drift_candidate(
                    center - signed_step);
            }
            else if (online_loop_track_.historical_direction < 0)
            {
                inject_large_drift_candidate(
                    center - signed_step);
                inject_large_drift_candidate(
                    center + signed_step);
            }
            else
            {
                inject_large_drift_candidate(
                    center - signed_step);
                inject_large_drift_candidate(
                    center + signed_step);
            }
        }

        // Put the injected neighbors at the FRONT.  This is essential when the
        // same historical Submap also contains the previously used anchor:
        // the new independent KFs must get an ICP chance before the repeated
        // anchor consumes the normal same-Submap slot.
        for (auto iterator =
                 large_drift_neighborhood_candidates.rbegin();
             iterator != large_drift_neighborhood_candidates.rend();
             ++iterator)
        {
            candidate_order.insert(
                candidate_order.begin(),
                &(*iterator));
        }

        std::cout
            << "Loop Large-Drift Neighborhood V20"
            << " | current_kf=" << current_keyframe.id
            << " | center_historical_kf="
            << online_loop_track_.last_historical_keyframe_id
            << " | injected="
            << large_drift_neighborhood_candidates.size()
            << " | radius="
            << online_loop_large_drift_neighborhood_radius_
            << " | track_correction_translation="
            << track_correction_translation << " m"
            << " | track_correction_rotation="
            << track_correction_rotation << " deg"
            << " | direction="
            << online_loop_track_.historical_direction
            << std::endl;
    }

    // --------------------------------------------------------------------
    // V19.4 DRIFT-AWARE REVISIT CORRIDOR.
    //
    // The raw frontend path is allowed to drift. Therefore a Euclidean search
    // around current_keyframe.T_WL would be conceptually wrong.
    //
    // After one trusted seed exists, map the current frontend pose through the
    // loop correction and search the OLD historical trajectory around that
    // corrected pose:
    //
    //     T_WL_pred = C_track * T_WL_frontend(current)
    //
    // This is exactly the situation visible in RViz: the new pass may be
    // drawn several metres to the left/right, but the seed correction predicts
    // where it belongs on the historical pass.
    // --------------------------------------------------------------------
    double drift_aware_predicted_pose_nearest_distance =
        std::numeric_limits<double>::infinity();

    if (online_loop_drift_aware_corridor_enabled_ &&
        first_loop_large_drift_track &&
        track_priority_active)
    {
        const Eigen::Isometry3d T_W_L_corridor_prediction =
            online_loop_track_.T_loop_correction *
            current_keyframe.T_WL;

        if (T_W_L_corridor_prediction.matrix().allFinite())
        {
            struct CorridorCandidateDistance
            {
                std::size_t keyframe_id = 0;
                double distance = std::numeric_limits<double>::infinity();
            };

            std::vector<CorridorCandidateDistance> nearby_historical;
            nearby_historical.reserve(backend_keyframes_.size());

            for (const Keyframe &historical_kf :
                 backend_keyframes_)
            {
                if (historical_kf.id == current_keyframe.id ||
                    !historical_kf.T_WL.matrix().allFinite())
                {
                    continue;
                }

                // Only finished historical geometry is useful for a loop
                // target. This also naturally excludes the active recent area.
                if (FindBestFinishedSubmapForKeyframe(
                        historical_kf.id) == nullptr)
                {
                    continue;
                }

                const double corrected_space_distance =
                    (historical_kf.T_WL.translation() -
                     T_W_L_corridor_prediction.translation())
                        .norm();

                if (!std::isfinite(corrected_space_distance) ||
                    corrected_space_distance >
                        online_loop_drift_aware_corridor_radius_)
                {
                    continue;
                }

                CorridorCandidateDistance entry;
                entry.keyframe_id = historical_kf.id;
                entry.distance = corrected_space_distance;
                nearby_historical.push_back(entry);
            }

            std::sort(
                nearby_historical.begin(),
                nearby_historical.end(),
                [](const CorridorCandidateDistance &lhs,
                   const CorridorCandidateDistance &rhs)
                {
                    return lhs.distance < rhs.distance;
                });

            if (!nearby_historical.empty())
            {
                drift_aware_predicted_pose_nearest_distance =
                    nearby_historical.front().distance;
            }

            const std::size_t corridor_count =
                std::min(
                    online_loop_drift_aware_corridor_max_candidates_,
                    nearby_historical.size());

            for (std::size_t corridor_index = 0;
                 corridor_index < corridor_count;
                 ++corridor_index)
            {
                const std::size_t candidate_id =
                    nearby_historical[corridor_index].keyframe_id;

                bool already_present = false;

                for (const LoopCandidate *existing :
                     candidate_order)
                {
                    if (existing != nullptr &&
                        existing->candidate_id == candidate_id)
                    {
                        already_present = true;
                        break;
                    }
                }

                if (already_present)
                {
                    drift_aware_corridor_candidate_ids.push_back(
                        candidate_id);
                    continue;
                }

                LoopCandidate injected;
                injected.current_id = current_keyframe.id;
                injected.candidate_id = candidate_id;
                injected.distance =
                    nearby_historical[corridor_index].distance;
                injected.scan_context_distance =
                    std::numeric_limits<double>::infinity();
                injected.scan_context_similarity = 0.0;
                injected.scan_context_raw_cosine_similarity = 0.0;
                injected.scan_context_sector_coverage_ratio = 0.0;
                injected.scan_context_cell_coverage_ratio = 0.0;
                injected.scan_context_compared_sectors = 0;
                injected.sector_shift = 0;
                injected.yaw_shift_deg =
                    std::numeric_limits<double>::quiet_NaN();

                drift_aware_corridor_candidates.push_back(injected);
                drift_aware_corridor_candidate_ids.push_back(candidate_id);

                if (std::find(
                        large_drift_injected_candidate_ids.begin(),
                        large_drift_injected_candidate_ids.end(),
                        candidate_id) ==
                    large_drift_injected_candidate_ids.end())
                {
                    large_drift_injected_candidate_ids.push_back(
                        candidate_id);
                }
            }

            // Corrected-space candidates are the strongest continuation prior
            // after the seed, so verify them before pure KF-ID neighbors.
            for (auto iterator =
                     drift_aware_corridor_candidates.rbegin();
                 iterator != drift_aware_corridor_candidates.rend();
                 ++iterator)
            {
                candidate_order.insert(
                    candidate_order.begin(),
                    &(*iterator));
            }

            std::cout
                << "Loop Drift-Aware Corridor V20"
                << " | current_kf=" << current_keyframe.id
                << " | predicted_xyz=["
                << T_W_L_corridor_prediction.translation().x() << " "
                << T_W_L_corridor_prediction.translation().y() << " "
                << T_W_L_corridor_prediction.translation().z() << "]"
                << " | radius="
                << online_loop_drift_aware_corridor_radius_ << " m"
                << " | nearby="
                << nearby_historical.size()
                << " | injected="
                << drift_aware_corridor_candidates.size()
                << " | nearest_distance="
                << drift_aware_predicted_pose_nearest_distance << " m"
                << std::endl;
        }
    }

    // --------------------------------------------------------------------
    // V22.4 reverse SEGMENT sliding.
    //
    // A confirmed reverse sequence identifies a historical trajectory segment,
    // but the independently selected Keyframes on the two traversals are not
    // sampled at exactly the same physical positions.  Treating the predicted
    // historical KF as an exact correspondence creates artificial overlap loss.
    //
    // Guarantee K-2..K+2 are present as candidate anchors, then verify them in
    // |offset| order.  Each anchor builds its OWN historical forward 3m window,
    // and the winning anchor becomes the actual PoseGraph loop endpoint.
    // --------------------------------------------------------------------
    if (reverse_sequence_confirmed &&
        reverse_sequence_historical_kf !=
            std::numeric_limits<std::size_t>::max())
    {
        const std::int64_t reverse_center =
            static_cast<std::int64_t>(
                reverse_sequence_historical_kf);

        for (int offset =
                 -static_cast<int>(
                     kReverseSegmentSlidingRadius);
             offset <=
             static_cast<int>(
                 kReverseSegmentSlidingRadius);
             ++offset)
        {
            const std::int64_t historical_id =
                reverse_center +
                static_cast<std::int64_t>(offset);

            if (historical_id < 0)
            {
                continue;
            }

            const std::size_t candidate_id =
                static_cast<std::size_t>(
                    historical_id);

            if (candidate_id >=
                    current_keyframe.id ||
                FindBackendKeyframeById(
                    candidate_id) == nullptr)
            {
                continue;
            }

            const bool already_present =
                std::any_of(
                    candidate_order.begin(),
                    candidate_order.end(),
                    [candidate_id](
                        const LoopCandidate *candidate)
                    {
                        return candidate != nullptr &&
                               candidate->candidate_id ==
                                   candidate_id;
                    });

            if (already_present)
            {
                continue;
            }

            LoopCandidate injected;
            injected.current_id =
                current_keyframe.id;
            injected.candidate_id =
                candidate_id;

            const Keyframe *historical_keyframe =
                FindBackendKeyframeById(
                    candidate_id);

            if (historical_keyframe != nullptr &&
                historical_keyframe->T_WL.matrix().allFinite())
            {
                injected.distance =
                    (current_keyframe.T_WL.translation() -
                     historical_keyframe->T_WL.translation())
                        .norm();

                injected.time_separation_sec =
                    current_keyframe.timestamp -
                    historical_keyframe->timestamp;
            }
            else
            {
                injected.distance =
                    std::numeric_limits<double>::infinity();

                injected.time_separation_sec =
                    std::numeric_limits<double>::infinity();
            }

            injected.scan_context_distance =
                std::numeric_limits<double>::infinity();
            injected.scan_context_similarity = 0.0;
            injected.scan_context_raw_cosine_similarity = 0.0;
            injected.scan_context_sector_coverage_ratio = 0.0;
            injected.scan_context_cell_coverage_ratio = 0.0;
            injected.scan_context_compared_sectors = 0;
            injected.sector_shift = 0;
            injected.yaw_shift_deg =
                std::numeric_limits<double>::quiet_NaN();

            reverse_segment_sliding_candidates.push_back(
                injected);

            candidate_order.push_back(
                &reverse_segment_sliding_candidates.back());
        }

        std::stable_sort(
            candidate_order.begin(),
            candidate_order.end(),
            [reverse_sequence_historical_kf,
             kReverseSegmentSlidingRadius](
                const LoopCandidate *lhs,
                const LoopCandidate *rhs)
            {
                const auto sliding_distance =
                    [reverse_sequence_historical_kf](
                        const LoopCandidate *candidate)
                    -> std::size_t
                {
                    if (candidate == nullptr)
                    {
                        return std::numeric_limits<
                            std::size_t>::max();
                    }

                    const std::size_t id =
                        candidate->candidate_id;

                    return id >=
                                   reverse_sequence_historical_kf
                               ? id -
                                     reverse_sequence_historical_kf
                               : reverse_sequence_historical_kf -
                                     id;
                };

                const std::size_t lhs_distance =
                    sliding_distance(lhs);

                const std::size_t rhs_distance =
                    sliding_distance(rhs);

                const bool lhs_sliding =
                    lhs_distance <=
                    kReverseSegmentSlidingRadius;

                const bool rhs_sliding =
                    rhs_distance <=
                    kReverseSegmentSlidingRadius;

                if (lhs_sliding !=
                    rhs_sliding)
                {
                    return lhs_sliding;
                }

                if (lhs_sliding &&
                    lhs_distance !=
                        rhs_distance)
                {
                    return lhs_distance <
                           rhs_distance;
                }

                return false;
            });

        std::cout
            << "REVERSE_SEGMENT_SLIDING_V22_4"
            << " | current_kf="
            << current_keyframe.id
            << " | center_historical_kf="
            << reverse_sequence_historical_kf
            << " | radius="
            << kReverseSegmentSlidingRadius
            << " | injected="
            << reverse_segment_sliding_candidates.size()
            << " | action=PRIORITIZE_K_PLUS_MINUS_2"
            << std::endl;
    }

    std::cout
        << "Loop Candidate Manager V20"
        << " | current_kf=" << current_keyframe.id
        << " | raw_candidates=" << candidates.size()
        << " | hierarchical_3m_candidates="
        << hierarchical_retrieval_candidates.size()
        << " | reverse_spatial_candidates="
        << reverse_spatial_candidates.size()
        << " | max_unique_submaps="
        << max_loop_candidates_to_verify_
        << " | track_priority="
        << (track_priority_active ? "true" : "false")
        << " | track_candidate_injected="
        << (track_continuation_candidate_injected ? "true" : "false")
        << " | large_drift_track="
        << (first_loop_large_drift_track ? "true" : "false")
        << " | large_drift_neighbors="
        << large_drift_neighborhood_candidates.size()
        << " | drift_aware_corridor_candidates="
        << drift_aware_corridor_candidate_ids.size()
        << " | corridor_nearest="
        << drift_aware_predicted_pose_nearest_distance
        << "m";

    if (track_priority_active)
    {
        std::cout
            << " | track_last_historical_kf="
            << online_loop_track_.last_historical_keyframe_id
            << " | track_direction="
            << online_loop_track_.historical_direction
            << " | track_radius="
            << track_priority_radius;
    }

    std::cout << std::endl;

    // Normal discovery verifies one KF per historical Submap.  During the V18
    // large-drift neighborhood expansion we deliberately allow several nearby
    // KFs from the SAME Submap, because independent anchors such as 96/97/98
    // often live in one frozen Submap.
    std::unordered_map<std::size_t, std::size_t>
        verified_historical_submap_counts;

    std::size_t verified_candidate_count = 0;

    const std::size_t hierarchical_extra_verify_budget =
        std::min(
            kHierarchicalExtraVerifyBudget,
            hierarchical_retrieval_candidates.size());

    const std::size_t verification_budget =
        max_loop_candidates_to_verify_ +
        hierarchical_extra_verify_budget +
        (first_loop_large_drift_track
             ? online_loop_large_drift_extra_verify_budget_
             : 0);

    // ====================================================================
    // REVERSE_POSE_AUDIT V1
    //
    // Temporary coordinate-frame audit for the fold-back section.
    // It is intentionally restricted to a small current-KF interval and
    // only to spatially-close, approximately opposite-heading candidates so
    // the diagnostic does not explode the log size again.
    //
    // For current KF L and historical KF K we audit:
    //
    //   T_K_L_frontend = T_WK^-1 * T_WL
    //   T_K_L_initial  = ICP hypothesis seed
    //   T_K_L_final    = LoopVerifier result
    //   T_WL_loop      = T_WK * T_K_L_final
    //
    // The frontend reconstruction identity
    //
    //   T_WK * T_K_L_frontend == T_WL
    //
    // must be numerically zero-error.  For a valid reverse revisit the
    // relative yaw T_K_L may be near 180 deg while the WORLD correction
    // between T_WL_frontend and T_WL_loop should remain small.
    // ====================================================================
    constexpr bool kReversePoseAuditEnabled = false;
    constexpr std::size_t kReversePoseAuditBeginKf = 210;
    constexpr std::size_t kReversePoseAuditEndKf = 230;
    constexpr double kReversePoseAuditMinAbsYawDeg = 120.0;
    constexpr double kReversePoseAuditMaxTranslationM = 3.5;

    std::size_t full_verifier_calls_this_keyframe = 0;

    for (std::size_t index = 0;
         index < candidate_order.size();
         ++index)
    {
        if (full_verifier_calls_this_keyframe >=
            kFullVerifierCallBudgetPerCurrentKeyframe)
        {
            std::cout
                << "Loop Full Verifier Budget V22"
                << " | current_kf=" << current_keyframe.id
                << " | calls=" << full_verifier_calls_this_keyframe
                << " | budget="
                << kFullVerifierCallBudgetPerCurrentKeyframe
                << " | action=STOP_CANDIDATE_VERIFICATION"
                << std::endl;
            break;
        }

        if (verified_candidate_count >=
            verification_budget)
        {
            break;
        }

        if (candidate_order[index] == nullptr)
        {
            continue;
        }

        const LoopCandidate &candidate =
            *candidate_order[index];

        std::int64_t reverse_segment_anchor_offset = 0;

        const bool candidate_matches_reverse_sequence =
            reverse_sequence_confirmed &&
            reverse_sequence_historical_kf !=
                std::numeric_limits<std::size_t>::max() &&
            [&]()
        {
            const std::int64_t delta =
                static_cast<std::int64_t>(
                    candidate.candidate_id) -
                static_cast<std::int64_t>(
                    reverse_sequence_historical_kf);

            reverse_segment_anchor_offset =
                delta;

            return std::llabs(delta) <=
                   static_cast<std::int64_t>(
                       kReverseSegmentSlidingRadius);
        }();

        const BackendSubmapSnapshot *historical_submap =
            FindBestFinishedSubmapForKeyframe(
                candidate.candidate_id);

        if (historical_submap == nullptr)
        {
            std::cout
                << "Keyframe loop verification skipped"
                << " | current_kf=" << current_keyframe.id
                << " | historical_kf=" << candidate.candidate_id
                << " | reason=NO_FINISHED_HISTORICAL_SUBMAP"
                << std::endl;
            continue;
        }

        // ----------------------------------------------------------------
        // Reject normal local-trajectory overlap before ICP using KEYFRAME
        // topology, not Submap topology.
        //
        // Scan Context retrieval, temporal confirmation, and PoseGraph nodes
        // are all Keyframe-based.  Injected candidates (track continuation,
        // neighborhood expansion, drift-aware corridor) can bypass the raw
        // LoopDetector candidate list, so the same Keyframe separation is
        // enforced once more at this unified candidate entry point.
        //
        // The historical Submap is still kept for geometry organization and
        // per-Submap verification de-duplication only; its ID no longer decides
        // whether two Keyframes are temporally far enough apart to form a loop.
        // ----------------------------------------------------------------
        if (candidate.candidate_id >= current_keyframe.id)
        {
            std::cout
                << "Keyframe loop candidate rejected"
                << " | current_kf=" << current_keyframe.id
                << " | historical_kf=" << candidate.candidate_id
                << " | current_submap=" << current_submap_id
                << " | historical_submap=" << historical_submap->id
                << " | reason=NOT_HISTORICAL_KEYFRAME"
                << std::endl;
            continue;
        }

        const std::size_t keyframe_gap =
            current_keyframe.id - candidate.candidate_id;

        if (keyframe_gap < min_loop_keyframe_separation)
        {
            std::cout
                << "Keyframe loop candidate rejected"
                << " | current_kf=" << current_keyframe.id
                << " | historical_kf=" << candidate.candidate_id
                << " | current_submap=" << current_submap_id
                << " | historical_submap=" << historical_submap->id
                << " | keyframe_gap=" << keyframe_gap
                << " | min_keyframe_gap="
                << min_loop_keyframe_separation
                << " | reason=LOCAL_KEYFRAME_NEIGHBORHOOD"
                << std::endl;
            continue;
        }

        const bool large_drift_injected_candidate =
            std::find(
                large_drift_injected_candidate_ids.begin(),
                large_drift_injected_candidate_ids.end(),
                candidate.candidate_id) !=
            large_drift_injected_candidate_ids.end();

        std::size_t &submap_verify_count =
            verified_historical_submap_counts[historical_submap->id];

        const bool hierarchical_retrieval_candidate =
            std::find(
                hierarchical_retrieval_candidate_ids.begin(),
                hierarchical_retrieval_candidate_ids.end(),
                candidate.candidate_id) !=
            hierarchical_retrieval_candidate_ids.end();

        const std::size_t same_submap_budget =
            candidate_matches_reverse_sequence
                ? (2 * kReverseSegmentSlidingRadius + 1)
                : (first_loop_large_drift_track &&
                           large_drift_injected_candidate
                       ? online_loop_large_drift_same_submap_verify_budget_
                       : (hierarchical_retrieval_candidate
                              ? kHierarchicalSameSubmapVerifyBudget
                              : 1));

        if (submap_verify_count >=
            same_submap_budget)
        {
            continue;
        }

        ++submap_verify_count;
        ++verified_candidate_count;

        // --------------------------------------------------------------------
        // Candidate-anchored initial guesses.
        //
        // DO NOT use:
        //
        //     historical.T_WS^-1 * current.T_WL
        //
        // as the only ICP translation guess. If odometry drift is large, that
        // translation can be tens of metres away from the true loop and ICP
        // never gets a chance to converge.
        //
        // Scan Context has already told us that current KF is likely at the
        // historical candidate KF place. Therefore anchor translation at the
        // candidate KF position inside historical Submap H, and test base /
        // +/- Scan Context yaw explicitly.
        // --------------------------------------------------------------------
        const Keyframe *historical_keyframe =
            FindBackendKeyframeById(candidate.candidate_id);

        if (historical_keyframe == nullptr ||
            !historical_keyframe->T_WL.matrix().allFinite())
        {
            continue;
        }

        // ------------------------------------------------------------------------
        // V20.2: raw frontend odometry arc length from historical KF to current KF.
        //
        // IMPORTANT:
        // T_WL here is the continuous frontend odometry pose, not the optimized
        // PoseGraph pose.  Therefore it can be used to estimate how much trajectory
        // was actually travelled between the two candidate Keyframes.
        // ------------------------------------------------------------------------
        double candidate_odom_arc_length = 0.0;
        bool candidate_odom_arc_valid = true;

        const Keyframe *arc_previous_keyframe =
            historical_keyframe;

        for (std::size_t keyframe_id =
                 candidate.candidate_id + 1;
             keyframe_id <= current_keyframe.id;
             ++keyframe_id)
        {
            const Keyframe *arc_current_keyframe =
                FindBackendKeyframeById(keyframe_id);

            if (arc_current_keyframe == nullptr ||
                !arc_current_keyframe->T_WL.matrix().allFinite() ||
                arc_previous_keyframe == nullptr ||
                !arc_previous_keyframe->T_WL.matrix().allFinite())
            {
                candidate_odom_arc_valid = false;
                break;
            }

            const Eigen::Vector3d delta =
                arc_current_keyframe->T_WL.translation() -
                arc_previous_keyframe->T_WL.translation();

            if (!delta.allFinite())
            {
                candidate_odom_arc_valid = false;
                break;
            }

            candidate_odom_arc_length += delta.norm();

            arc_previous_keyframe =
                arc_current_keyframe;
        }

        if (!std::isfinite(candidate_odom_arc_length))
        {
            candidate_odom_arc_valid = false;
        }

        // V19 geometry target: a small LocalMap centered on THIS candidate KF.
        // The target frame is K itself, not the owning finished Submap H.
        pcl::PointCloud<LIDAR_POINT>::Ptr
            historical_target_K;

        std::vector<std::size_t>
            historical_target_keyframes;

        const bool keyframe_centered_target_ready =
            BuildCandidateCenteredHistoricalTarget(
                candidate.candidate_id,
                historical_target_K,
                &historical_target_keyframes);

        if (!keyframe_centered_target_ready &&
            !hierarchical_retrieval_candidate)
        {
            std::cout
                << "Keyframe loop verification skipped"
                << " | current_kf=" << current_keyframe.id
                << " | historical_kf=" << candidate.candidate_id
                << " | reason=FAILED_KEYFRAME_CENTERED_TARGET"
                << std::endl;
            continue;
        }

        if (!keyframe_centered_target_ready &&
            hierarchical_retrieval_candidate)
        {
            std::cout
                << "HIERARCHICAL_3M_GEOMETRY_V22_6"
                << " | current_kf=" << current_keyframe.id
                << " | historical_kf=" << candidate.candidate_id
                << " | keyframe_centered_target=false"
                << " | action=CONTINUE_SEGMENT_ONLY"
                << std::endl;
        }

        // ------------------------------------------------------------------------
        // V22 reverse geometry: CURRENT 3m window <-> HISTORICAL FORWARD 3m window.
        //
        // The reverse-pose audit proved that T_K_L and the graph convention are
        // correct.  The failure mode was geometric support: a single current KF
        // often had only ~0.30 overlap with a historical local map when the same
        // row was traversed in the opposite direction.
        //
        // For a CONFIRMED reverse sequence we therefore verify the same physical
        // trajectory extent on both sides:
        //
        //   source_L = current causal 3m window: L, L-1, L-2, ...
        //   target_K = historical forward 3m window: K, K+1, K+2, ...
        //
        // Both clouds are already expressed in their anchor frames L and K, so
        // LoopVerifier still estimates exactly the same measurement T_K_L.
        // Mapping Submaps and PoseGraph topology remain unchanged.
        // ------------------------------------------------------------------------

        pcl::PointCloud<LIDAR_POINT>::ConstPtr
            reverse_sequence_source_L;

        pcl::PointCloud<LIDAR_POINT>::ConstPtr
            reverse_sequence_target_K;

        std::vector<std::size_t>
            reverse_sequence_source_keyframes;

        std::vector<std::size_t>
            reverse_sequence_target_keyframes;

        double reverse_sequence_source_arc_m = 0.0;
        double reverse_sequence_target_arc_m = 0.0;
        std::size_t reverse_sequence_source_filtered_points = 0;
        std::size_t reverse_sequence_target_filtered_points = 0;
        bool reverse_sequence_target_ready = false;

        SphericalVisibilityWindowModel
            reverse_sequence_source_visibility;

        SphericalVisibilityWindowModel
            reverse_sequence_target_visibility;

        bool reverse_sequence_visibility_ready = false;

        // V22.6 generic hierarchical 3m geometry.  Unlike the reverse-only
        // path above, this applies to ANY candidate that came from the V21
        // 3m Region-SC hierarchy.  The historical direction is inherited from
        // the best Region-SC window of the parent Submap.
        pcl::PointCloud<LIDAR_POINT>::ConstPtr
            hierarchical_segment_source_L;

        pcl::PointCloud<LIDAR_POINT>::ConstPtr
            hierarchical_segment_target_K;

        std::vector<std::size_t>
            hierarchical_segment_source_keyframes;

        std::vector<std::size_t>
            hierarchical_segment_target_keyframes;

        double hierarchical_segment_source_arc_m = 0.0;
        double hierarchical_segment_target_arc_m = 0.0;
        std::size_t hierarchical_segment_source_filtered_points = 0;
        std::size_t hierarchical_segment_target_filtered_points = 0;
        int hierarchical_segment_direction = 0;
        bool hierarchical_segment_target_ready = false;

        SphericalVisibilityWindowModel
            hierarchical_segment_source_visibility;

        SphericalVisibilityWindowModel
            hierarchical_segment_target_visibility;

        bool hierarchical_segment_visibility_ready = false;

        if (candidate_matches_reverse_sequence)
        {
            // Current query window was built once at the start of this loop
            // search and is anchored in the current Keyframe L.
            if (current_region_window_ready &&
                current_region_window.cloud &&
                !current_region_window.cloud->empty())
            {
                reverse_sequence_source_L =
                    current_region_window.cloud;

                reverse_sequence_source_keyframes =
                    current_region_window.keyframe_ids;

                reverse_sequence_source_arc_m =
                    current_region_window.arc_length_m;
            }

            // In a reverse revisit, the current causal window L,L-1,... maps
            // to the historical trajectory on the FORWARD side of anchor K:
            // K,K+1,K+2,... .  Using BACKWARD here would cover the wrong side
            // of the physical segment.
            HierarchicalRetrievalWindow historical_reverse_window;

            if (build_retrieval_window(
                    candidate.candidate_id,
                    +1,
                    historical_reverse_window))
            {
                bool historical_window_is_old_enough = true;

                const std::size_t newest_allowed_historical_kf =
                    current_keyframe.id >= min_loop_keyframe_separation
                        ? current_keyframe.id - min_loop_keyframe_separation
                        : 0;

                for (const std::size_t member_id :
                     historical_reverse_window.keyframe_ids)
                {
                    if (member_id > newest_allowed_historical_kf)
                    {
                        historical_window_is_old_enough = false;
                        break;
                    }
                }

                if (historical_window_is_old_enough)
                {
                    reverse_sequence_target_K =
                        historical_reverse_window.cloud;

                    reverse_sequence_target_keyframes =
                        historical_reverse_window.keyframe_ids;

                    reverse_sequence_target_arc_m =
                        historical_reverse_window.arc_length_m;
                }
            }

            const LoopVerifierConfig &reverse_verifier_config =
                loop_verifier_.GetConfig();

            const auto voxel_count =
                [&reverse_verifier_config](
                    const pcl::PointCloud<LIDAR_POINT>::ConstPtr &cloud)
                -> std::size_t
            {
                if (!cloud || cloud->empty() ||
                    !std::isfinite(reverse_verifier_config.voxel_leaf_size) ||
                    reverse_verifier_config.voxel_leaf_size <= 0.0)
                {
                    return 0;
                }

                pcl::PointCloud<LIDAR_POINT>::Ptr filtered =
                    pcl::make_shared<pcl::PointCloud<LIDAR_POINT>>();

                pcl::VoxelGrid<LIDAR_POINT> voxel;
                voxel.setInputCloud(cloud);

                const float leaf =
                    static_cast<float>(
                        reverse_verifier_config.voxel_leaf_size);

                voxel.setLeafSize(leaf, leaf, leaf);
                voxel.filter(*filtered);

                return filtered->size();
            };

            reverse_sequence_source_filtered_points =
                voxel_count(reverse_sequence_source_L);

            reverse_sequence_target_filtered_points =
                voxel_count(reverse_sequence_target_K);

            // 90% of the requested 3m is accepted to tolerate the final KF
            // spacing discretization.  The hard 30-KF cap is only an anomaly
            // guard now, not a normal termination condition.
            constexpr double kReverseGeometryMinArcRatio = 0.90;
            const double min_reverse_arc_m =
                kHierarchicalRetrievalArcLengthM *
                kReverseGeometryMinArcRatio;

            reverse_sequence_target_ready =
                reverse_sequence_source_L &&
                reverse_sequence_target_K &&
                !reverse_sequence_source_L->empty() &&
                !reverse_sequence_target_K->empty() &&
                reverse_sequence_source_keyframes.size() >=
                    kHierarchicalRetrievalMinKeyframes &&
                reverse_sequence_target_keyframes.size() >=
                    kHierarchicalRetrievalMinKeyframes &&
                reverse_sequence_source_arc_m >= min_reverse_arc_m &&
                reverse_sequence_target_arc_m >= min_reverse_arc_m &&
                reverse_sequence_source_filtered_points >=
                    reverse_verifier_config.min_cloud_points &&
                reverse_sequence_target_filtered_points >=
                    reverse_verifier_config.min_cloud_points;

            // V22.4: build the 3D spherical visibility model from the RAW
            // member scans and the member-to-anchor poses.  This retains the
            // original sensor origin for each ray instead of inferring FOV
            // from the fused 3m cloud.
            if (reverse_sequence_target_ready)
            {
                const bool source_visibility_ok =
                    build_visibility_window_model(
                        current_keyframe.id,
                        reverse_sequence_source_keyframes,
                        reverse_sequence_source_visibility);

                const bool target_visibility_ok =
                    build_visibility_window_model(
                        candidate.candidate_id,
                        reverse_sequence_target_keyframes,
                        reverse_sequence_target_visibility);

                reverse_sequence_visibility_ready =
                    source_visibility_ok &&
                    target_visibility_ok;

                const double source_visible_deg =
                    360.0 *
                    static_cast<double>(
                        reverse_sequence_source_visibility
                            .observable_azimuth_bins) /
                    static_cast<double>(
                        kVisibilityAzimuthBins);

                const double target_visible_deg =
                    360.0 *
                    static_cast<double>(
                        reverse_sequence_target_visibility
                            .observable_azimuth_bins) /
                    static_cast<double>(
                        kVisibilityAzimuthBins);

                std::cout
                    << "LOOP_3D_VISIBILITY_PROFILE_V22_4"
                    << " | current_kf="
                    << current_keyframe.id
                    << " | historical_kf="
                    << candidate.candidate_id
                    << " | reverse_center_kf="
                    << reverse_sequence_historical_kf
                    << " | anchor_offset="
                    << reverse_segment_anchor_offset
                    << " | source_scans="
                    << reverse_sequence_source_visibility
                           .valid_scans
                    << " | source_required_az_support="
                    << reverse_sequence_source_visibility
                           .required_azimuth_support
                    << " | source_observable_az_bins="
                    << reverse_sequence_source_visibility
                           .observable_azimuth_bins
                    << "/" << kVisibilityAzimuthBins
                    << " | source_observable_az_deg="
                    << source_visible_deg
                    << " | source_elevation_deg=["
                    << reverse_sequence_source_visibility
                               .min_elevation_rad *
                           180.0 / M_PI
                    << ","
                    << reverse_sequence_source_visibility
                               .max_elevation_rad *
                           180.0 / M_PI
                    << "]"
                    << " | target_scans="
                    << reverse_sequence_target_visibility
                           .valid_scans
                    << " | target_required_az_support="
                    << reverse_sequence_target_visibility
                           .required_azimuth_support
                    << " | target_observable_az_bins="
                    << reverse_sequence_target_visibility
                           .observable_azimuth_bins
                    << "/" << kVisibilityAzimuthBins
                    << " | target_observable_az_deg="
                    << target_visible_deg
                    << " | target_elevation_deg=["
                    << reverse_sequence_target_visibility
                               .min_elevation_rad *
                           180.0 / M_PI
                    << ","
                    << reverse_sequence_target_visibility
                               .max_elevation_rad *
                           180.0 / M_PI
                    << "]"
                    << " | ready="
                    << (reverse_sequence_visibility_ready
                            ? "true"
                            : "false")
                    << std::endl;
            }

            std::cout
                << "Reverse segment 3m geometry V22.4"
                << " | current_kf=" << current_keyframe.id
                << " | historical_kf=" << candidate.candidate_id
                << " | reverse_center_kf="
                << reverse_sequence_historical_kf
                << " | anchor_offset="
                << reverse_segment_anchor_offset
                << " | source_kfs=[";

            for (std::size_t i = 0;
                 i < reverse_sequence_source_keyframes.size();
                 ++i)
            {
                if (i > 0)
                {
                    std::cout << ",";
                }
                std::cout << reverse_sequence_source_keyframes[i];
            }

            std::cout
                << "]"
                << " | source_arc=" << reverse_sequence_source_arc_m
                << " m"
                << " | source_filtered="
                << reverse_sequence_source_filtered_points
                << " | target_kfs=[";

            for (std::size_t i = 0;
                 i < reverse_sequence_target_keyframes.size();
                 ++i)
            {
                if (i > 0)
                {
                    std::cout << ",";
                }
                std::cout << reverse_sequence_target_keyframes[i];
            }

            std::cout
                << "]"
                << " | target_arc=" << reverse_sequence_target_arc_m
                << " m"
                << " | target_filtered="
                << reverse_sequence_target_filtered_points
                << " | min_cloud_points="
                << reverse_verifier_config.min_cloud_points
                << " | ready="
                << (reverse_sequence_target_ready
                        ? "true"
                        : "false")
                << std::endl;
        }

        // --------------------------------------------------------------------
        // V22.6 Hierarchical 3m<->3m geometry.
        //
        // IMPORTANT BUG FIX:
        //   V21 used 3m segments to recall a historical region, but a winning
        //   hierarchical candidate could still be verified with
        //   KEYFRAME_CENTERED geometry.  In a repetitive turn/row this lets a
        //   weak region recall become a strong-looking KF-centered ICP and can
        //   then self-perpetuate through TRACK_PREDICTED.
        //
        // From V22.6, every hierarchical candidate must stay segment-level:
        //   current causal 3m window <-> historical 3m window in the direction
        //   that produced the Region-SC recall.  No KF-centered fallback.
        // --------------------------------------------------------------------
        if (hierarchical_retrieval_candidate &&
            !candidate_matches_reverse_sequence)
        {
            if (current_region_window_ready &&
                current_region_window.cloud &&
                !current_region_window.cloud->empty())
            {
                hierarchical_segment_source_L =
                    current_region_window.cloud;
                hierarchical_segment_source_keyframes =
                    current_region_window.keyframe_ids;
                hierarchical_segment_source_arc_m =
                    current_region_window.arc_length_m;
            }

            const auto preferred_direction_iterator =
                hierarchical_candidate_preferred_direction.find(
                    candidate.candidate_id);

            if (preferred_direction_iterator !=
                hierarchical_candidate_preferred_direction.end())
            {
                hierarchical_segment_direction =
                    preferred_direction_iterator->second;
            }
            else if (online_loop_track_.valid &&
                     online_loop_track_.historical_direction != 0)
            {
                // Current query is causal BACKWARD_3M.
                // Same-direction historical progression (+1 IDs) maps to a
                // BACKWARD historical window; reverse progression (-1 IDs)
                // maps to a FORWARD historical window.
                hierarchical_segment_direction =
                    online_loop_track_.historical_direction > 0
                        ? -1
                        : +1;
            }
            else
            {
                hierarchical_segment_direction = -1;
            }

            HierarchicalRetrievalWindow
                historical_hierarchical_window;

            if (build_retrieval_window(
                    candidate.candidate_id,
                    hierarchical_segment_direction,
                    historical_hierarchical_window))
            {
                bool historical_window_is_old_enough = true;

                const std::size_t newest_allowed_historical_kf =
                    current_keyframe.id >= min_loop_keyframe_separation
                        ? current_keyframe.id - min_loop_keyframe_separation
                        : 0;

                for (const std::size_t member_id :
                     historical_hierarchical_window.keyframe_ids)
                {
                    if (member_id > newest_allowed_historical_kf)
                    {
                        historical_window_is_old_enough = false;
                        break;
                    }
                }

                if (historical_window_is_old_enough)
                {
                    hierarchical_segment_target_K =
                        historical_hierarchical_window.cloud;
                    hierarchical_segment_target_keyframes =
                        historical_hierarchical_window.keyframe_ids;
                    hierarchical_segment_target_arc_m =
                        historical_hierarchical_window.arc_length_m;
                }
            }

            const LoopVerifierConfig &segment_verifier_config =
                loop_verifier_.GetConfig();

            const auto segment_voxel_count =
                [&segment_verifier_config](
                    const pcl::PointCloud<LIDAR_POINT>::ConstPtr &cloud)
                -> std::size_t
            {
                if (!cloud || cloud->empty() ||
                    !std::isfinite(
                        segment_verifier_config.voxel_leaf_size) ||
                    segment_verifier_config.voxel_leaf_size <= 0.0)
                {
                    return 0;
                }

                pcl::PointCloud<LIDAR_POINT>::Ptr filtered =
                    pcl::make_shared<pcl::PointCloud<LIDAR_POINT>>();

                pcl::VoxelGrid<LIDAR_POINT> voxel;
                voxel.setInputCloud(cloud);

                const float leaf =
                    static_cast<float>(
                        segment_verifier_config.voxel_leaf_size);

                voxel.setLeafSize(leaf, leaf, leaf);
                voxel.filter(*filtered);

                return filtered->size();
            };

            hierarchical_segment_source_filtered_points =
                segment_voxel_count(
                    hierarchical_segment_source_L);

            hierarchical_segment_target_filtered_points =
                segment_voxel_count(
                    hierarchical_segment_target_K);

            const double min_segment_arc_m =
                kHierarchicalRetrievalArcLengthM *
                kHierarchicalSegmentGeometryMinArcRatio;

            hierarchical_segment_target_ready =
                hierarchical_segment_source_L &&
                hierarchical_segment_target_K &&
                !hierarchical_segment_source_L->empty() &&
                !hierarchical_segment_target_K->empty() &&
                hierarchical_segment_source_keyframes.size() >=
                    kHierarchicalRetrievalMinKeyframes &&
                hierarchical_segment_target_keyframes.size() >=
                    kHierarchicalRetrievalMinKeyframes &&
                hierarchical_segment_source_arc_m >=
                    min_segment_arc_m &&
                hierarchical_segment_target_arc_m >=
                    min_segment_arc_m &&
                hierarchical_segment_source_filtered_points >=
                    segment_verifier_config.min_cloud_points &&
                hierarchical_segment_target_filtered_points >=
                    segment_verifier_config.min_cloud_points;

            if (hierarchical_segment_target_ready)
            {
                const bool source_visibility_ok =
                    build_visibility_window_model(
                        current_keyframe.id,
                        hierarchical_segment_source_keyframes,
                        hierarchical_segment_source_visibility);

                const bool target_visibility_ok =
                    build_visibility_window_model(
                        candidate.candidate_id,
                        hierarchical_segment_target_keyframes,
                        hierarchical_segment_target_visibility);

                hierarchical_segment_visibility_ready =
                    source_visibility_ok &&
                    target_visibility_ok;
            }

            std::cout
                << "HIERARCHICAL_3M_GEOMETRY_V22_6"
                << " | current_kf=" << current_keyframe.id
                << " | historical_kf="
                << candidate.candidate_id
                << " | direction="
                << (hierarchical_segment_direction < 0
                        ? "BACKWARD_3M"
                        : "FORWARD_3M")
                << " | source_kfs=[";

            for (std::size_t i = 0;
                 i < hierarchical_segment_source_keyframes.size();
                 ++i)
            {
                if (i > 0)
                {
                    std::cout << ",";
                }
                std::cout
                    << hierarchical_segment_source_keyframes[i];
            }

            std::cout
                << "] | source_arc="
                << hierarchical_segment_source_arc_m
                << " m | source_filtered="
                << hierarchical_segment_source_filtered_points
                << " | target_kfs=[";

            for (std::size_t i = 0;
                 i < hierarchical_segment_target_keyframes.size();
                 ++i)
            {
                if (i > 0)
                {
                    std::cout << ",";
                }
                std::cout
                    << hierarchical_segment_target_keyframes[i];
            }

            std::cout
                << "] | target_arc="
                << hierarchical_segment_target_arc_m
                << " m | target_filtered="
                << hierarchical_segment_target_filtered_points
                << " | visibility_ready="
                << (hierarchical_segment_visibility_ready
                        ? "true"
                        : "false")
                << " | ready="
                << (hierarchical_segment_target_ready
                        ? "true"
                        : "false")
                << std::endl;

            if (!hierarchical_segment_target_ready)
            {
                std::cout
                    << "HIERARCHICAL_3M_GEOMETRY_V22_6"
                    << " | current_kf=" << current_keyframe.id
                    << " | historical_kf="
                    << candidate.candidate_id
                    << " | action=REJECT_NO_KF_CENTERED_FALLBACK"
                    << std::endl;
                continue;
            }
        }

        // Scan Context has already selected candidate K as the place hypothesis.
        // In K coordinates, the base same-place pose is identity.
        const Eigen::Isometry3d T_K_L_anchor =
            Eigen::Isometry3d::Identity();

        std::vector<std::pair<const char *, Eigen::Isometry3d>>
            initial_guesses;

        // V15: when this candidate is compatible with an existing temporal
        // loop track, propagate the previously verified world correction to
        // the current frontend pose and express that prediction in the
        // historical Submap frame:
        //
        //     T_WL_pred = C_prev * T_WL_frontend
        //     T_HL_pred = T_WH^-1 * T_WL_pred
        //
        // This is a continuation hypothesis only.  It receives no special
        // acceptance privilege and must pass the same full verifier / graph
        // gates as every Scan Context hypothesis.
        bool candidate_matches_active_track = false;

        // V14.1: candidate_delta is also used by the TRACK_PREDICTED
        // diagnostic below, so keep it in the candidate-loop scope rather
        // than declaring it inside the track_priority_active block.
        std::int64_t candidate_delta = 0;

        if (track_priority_active)
        {
            candidate_delta =
                static_cast<std::int64_t>(candidate.candidate_id) -
                static_cast<std::int64_t>(
                    online_loop_track_.last_historical_keyframe_id);

            const std::uint64_t candidate_absolute_delta =
                static_cast<std::uint64_t>(
                    candidate_delta >= 0
                        ? candidate_delta
                        : -candidate_delta);

            const std::int64_t backtrack_tolerance =
                static_cast<std::int64_t>(
                    online_loop_historical_backtrack_tolerance_);

            candidate_matches_active_track =
                candidate_absolute_delta <= track_priority_radius &&
                !(online_loop_track_.historical_direction > 0 &&
                  candidate_delta < -backtrack_tolerance) &&
                !(online_loop_track_.historical_direction < 0 &&
                  candidate_delta > backtrack_tolerance);
        }

        if (candidate_matches_active_track &&
            online_loop_track_.support >= 1)
        {
            const Eigen::Isometry3d T_W_L_track_pred =
                online_loop_track_.T_loop_correction *
                current_keyframe.T_WL;

            const Eigen::Isometry3d T_K_L_track_pred =
                historical_keyframe->T_WL.inverse() *
                T_W_L_track_pred;

            if (T_K_L_track_pred.matrix().allFinite())
            {
                const char *track_guess_name =
                    online_loop_track_.support >= 2
                        ? "TRACK_PREDICTED"
                        : "TENTATIVE_TRACK_PREDICTED";

                initial_guesses.emplace_back(
                    track_guess_name,
                    T_K_L_track_pred);

                std::cout
                    << "Loop Track Continuation V20"
                    << " | current_kf=" << current_keyframe.id
                    << " | historical_kf=" << candidate.candidate_id
                    << " | historical_submap=" << historical_submap->id
                    << " | mode=" << track_guess_name
                    << " | support=" << online_loop_track_.support
                    << " | candidate_delta=" << candidate_delta
                    << " | current_gap=" << active_track_current_gap
                    << std::endl;
            }
        }

        initial_guesses.emplace_back(
            "CANDIDATE_POSE",
            T_K_L_anchor);

        // ------------------------------------------------------------------------
        // Reverse-traversal frontend-pose hypothesis.
        //
        // Use this hypothesis only when:
        //
        //   1. Scan Context has already selected historical KF K.
        //   2. Frontend currently places K and L within a few metres.
        //   3. Their frontend headings are approximately opposite.
        //
        // This is deliberately NOT enabled for arbitrary long-range candidates.
        // It exists only to resolve the 0/180-degree ambiguity of reverse traversal.
        //
        // K = historical candidate Keyframe
        // L = current Keyframe
        //
        //     T_K_L_frontend = T_WK^-1 * T_WL
        //
        // Because the historical verification target is already expressed in K,
        // this transform can be used directly as an ICP initial guess.
        // ------------------------------------------------------------------------
        const auto reverse_pose_audit_yaw_deg =
            [](const Eigen::Isometry3d &T)
        {
            if (!T.matrix().allFinite())
            {
                return std::numeric_limits<double>::quiet_NaN();
            }

            const Eigen::Matrix3d &R = T.rotation();

            return std::atan2(
                       R(1, 0),
                       R(0, 0)) *
                   180.0 / M_PI;
        };

        bool reverse_pose_audit_candidate = false;
        double reverse_pose_audit_frontend_dt =
            std::numeric_limits<double>::infinity();
        double reverse_pose_audit_frontend_abs_yaw_deg =
            std::numeric_limits<double>::infinity();

        const Eigen::Isometry3d T_K_L_frontend =
            historical_keyframe->T_WL.inverse() *
            current_keyframe.T_WL;

        if (T_K_L_frontend.matrix().allFinite())
        {
            const double frontend_relative_translation =
                T_K_L_frontend.translation().norm();

            const Eigen::Matrix3d &R_K_L_frontend =
                T_K_L_frontend.rotation();

            const double frontend_relative_yaw_rad =
                std::atan2(
                    R_K_L_frontend(1, 0),
                    R_K_L_frontend(0, 0));

            const double frontend_relative_abs_yaw_deg =
                std::abs(frontend_relative_yaw_rad) *
                180.0 / M_PI;

            constexpr double
                kReverseFrontendMinYawDeg = 120.0;

            constexpr double
                kReverseFrontendMaxTranslation = 3.5;

            reverse_pose_audit_frontend_dt =
                frontend_relative_translation;

            reverse_pose_audit_frontend_abs_yaw_deg =
                frontend_relative_abs_yaw_deg;

            reverse_pose_audit_candidate =
                kReversePoseAuditEnabled &&
                current_keyframe.id >=
                    kReversePoseAuditBeginKf &&
                current_keyframe.id <=
                    kReversePoseAuditEndKf &&
                frontend_relative_translation <=
                    kReversePoseAuditMaxTranslationM &&
                frontend_relative_abs_yaw_deg >=
                    kReversePoseAuditMinAbsYawDeg;

            if (reverse_pose_audit_candidate)
            {
                const Eigen::Isometry3d
                    T_W_L_frontend_reconstructed =
                        historical_keyframe->T_WL *
                        T_K_L_frontend;

                const Eigen::Isometry3d
                    T_reconstruction_error =
                        T_W_L_frontend_reconstructed.inverse() *
                        current_keyframe.T_WL;

                const double reconstruction_dt =
                    T_reconstruction_error.matrix().allFinite()
                        ? T_reconstruction_error.translation().norm()
                        : std::numeric_limits<double>::infinity();

                const double reconstruction_dR =
                    T_reconstruction_error.matrix().allFinite()
                        ? RelativeRotationDeg(
                              Eigen::Isometry3d::Identity(),
                              T_reconstruction_error)
                        : std::numeric_limits<double>::infinity();

                std::cout
                    << "REVERSE_POSE_AUDIT"
                    << " | stage=BASE"
                    << " | current_kf=" << current_keyframe.id
                    << " | historical_kf=" << candidate.candidate_id
                    << " | yaw_WK="
                    << reverse_pose_audit_yaw_deg(
                           historical_keyframe->T_WL)
                    << " deg"
                    << " | yaw_WL="
                    << reverse_pose_audit_yaw_deg(
                           current_keyframe.T_WL)
                    << " deg"
                    << " | frontend_T_K_L_dt="
                    << frontend_relative_translation << " m"
                    << " | frontend_T_K_L_yaw="
                    << reverse_pose_audit_yaw_deg(
                           T_K_L_frontend)
                    << " deg"
                    << " | frontend_abs_yaw="
                    << frontend_relative_abs_yaw_deg << " deg"
                    << " | reconstruction_dt="
                    << reconstruction_dt << " m"
                    << " | reconstruction_dR="
                    << reconstruction_dR << " deg"
                    << std::endl;
            }

            const bool reverse_frontend_candidate =
                frontend_relative_translation <=
                    kReverseFrontendMaxTranslation &&
                frontend_relative_abs_yaw_deg >=
                    kReverseFrontendMinYawDeg;

            if (reverse_frontend_candidate)
            {
                initial_guesses.emplace_back(
                    "CANDIDATE_REVERSE_FRONTEND",
                    T_K_L_frontend);

                std::cout
                    << "Reverse traversal hypothesis"
                    << " | current_kf="
                    << current_keyframe.id
                    << " | historical_kf="
                    << candidate.candidate_id
                    << " | frontend_dt="
                    << frontend_relative_translation
                    << " m"
                    << " | frontend_abs_yaw="
                    << frontend_relative_abs_yaw_deg
                    << " deg"
                    << " | action=ADD_REVERSE_FRONTEND"
                    << std::endl;
            }
        }

        // ------------------------------------------------------------------------
        // V20.2 confirmed reverse-sequence hypothesis.
        //
        // The transform is the same physically meaningful frontend relative
        // pose as CANDIDATE_REVERSE_FRONTEND, but this hypothesis is only
        // exposed after THREE raw-SC observations with TWO real negative
        // historical-KF progress events.  Later, prescore/full ICP use the
        // 3m-to-3m direction-matched historical target for this hypothesis so the
        // target has enough support without becoming a long +/-2-KF corridor
        // that encourages longitudinal sliding.
        // ------------------------------------------------------------------------
        if (candidate_matches_reverse_sequence &&
            reverse_sequence_target_ready &&
            T_K_L_frontend.matrix().allFinite())
        {
            initial_guesses.emplace_back(
                "CANDIDATE_REVERSE_SEQUENCE",
                T_K_L_frontend);

            std::cout
                << "Reverse sequence hypothesis V20.2"
                << " | current_kf="
                << current_keyframe.id
                << " | historical_kf="
                << candidate.candidate_id
                << " | seed_support="
                << reverse_loop_seed_track.support
                << " | reverse_progress="
                << reverse_loop_seed_track
                       .reverse_progress_events
                << " | target=REVERSE_3M_X_3M"
                << " | action=ADD"
                << std::endl;
        }

        // ----------------------------------------------------------------
        // V15 explicit 180-degree complementary Scan Context hypotheses.
        //
        // In repetitive rows/corridors Scan Context can return yaw ~= 0 deg
        // even when the same place is being traversed in the opposite
        // direction.  Testing only +yaw and -yaw cannot recover that case:
        // for yaw=0 both hypotheses are still 0 deg.
        //
        // Therefore test the four SC modes:
        //     +yaw, -yaw, +yaw+180deg, -yaw+180deg
        // while de-duplicating equivalent angles (POSE already represents
        // 0 deg).  Every surviving mode still has to pass prescore, full ICP,
        // ICP-correction gate, graph gate and temporal consistency.
        // ----------------------------------------------------------------
        std::vector<double> candidate_yaw_modes_rad;
        candidate_yaw_modes_rad.reserve(5);
        candidate_yaw_modes_rad.push_back(0.0); // CANDIDATE_POSE

        const auto append_candidate_yaw =
            [&initial_guesses,
             &candidate_yaw_modes_rad,
             &T_K_L_anchor](const char *name, double yaw_rad)
        {
            const double normalized_yaw =
                std::atan2(
                    std::sin(yaw_rad),
                    std::cos(yaw_rad));

            constexpr double kYawDuplicateToleranceRad = 1.0e-3;

            for (const double existing_yaw :
                 candidate_yaw_modes_rad)
            {
                const double angular_difference =
                    std::atan2(
                        std::sin(normalized_yaw - existing_yaw),
                        std::cos(normalized_yaw - existing_yaw));

                if (std::abs(angular_difference) <=
                    kYawDuplicateToleranceRad)
                {
                    return;
                }
            }

            Eigen::Isometry3d guess = T_K_L_anchor;
            guess.linear() =
                T_K_L_anchor.rotation() *
                Eigen::AngleAxisd(
                    normalized_yaw,
                    Eigen::Vector3d::UnitZ())
                    .toRotationMatrix();

            initial_guesses.emplace_back(
                name,
                guess);

            candidate_yaw_modes_rad.push_back(
                normalized_yaw);
        };

        if (std::isfinite(candidate.yaw_shift_deg))
        {
            const double yaw_rad =
                candidate.yaw_shift_deg *
                M_PI / 180.0;

            append_candidate_yaw(
                "CANDIDATE_SC_POSITIVE",
                yaw_rad);

            append_candidate_yaw(
                "CANDIDATE_SC_NEGATIVE",
                -yaw_rad);

            append_candidate_yaw(
                "CANDIDATE_SC_POSITIVE_FLIP180",
                yaw_rad + M_PI);

            append_candidate_yaw(
                "CANDIDATE_SC_NEGATIVE_FLIP180",
                -yaw_rad + M_PI);
        }

        // --------------------------------------------------------------------
        // LoopVerifier V15: cheap pre-score -> ALL competitive full ICP hypotheses.
        //
        // Previous behavior ran one FULL ICP for every initial guess.  With
        // CANDIDATE_POSE / +/-SC / +/-SC+180 modes can create several ICP trials per
        // historical candidate even when the first hypothesis was already good.
        //
        // V3 first evaluates each hypothesis at its INITIAL pose using the
        // cached downsampled source + historical target KD-tree.  No ICP
        // iteration is performed during this stage.  The hypotheses are then
        // tried in descending geometric-overlap order.
        //
        // Safety behavior:
        //   * The pre-score gate is deliberately loose (default 3% overlap at
        //     2 m), far below the final 15% overlap gate.
        //   * If the best full ICP is rejected, the next plausible hypothesis
        //     is still tried.
        //   * V13 does NOT stop at the first accepted hypothesis. Every
        //     competitive 0/+SC/-SC/+180-complement mode is checked against graph consistency.
        // --------------------------------------------------------------------
        struct RankedInitialGuess
        {
            const char *name = "NONE";
            Eigen::Isometry3d transform =
                Eigen::Isometry3d::Identity();
            LoopVerifierInitialGuessScore score;
        };

        std::vector<RankedInitialGuess> ranked_guesses;
        ranked_guesses.reserve(initial_guesses.size());

        for (const auto &guess_entry : initial_guesses)
        {
            RankedInitialGuess ranked_guess;
            ranked_guess.name = guess_entry.first;
            ranked_guess.transform = guess_entry.second;

            const std::chrono::steady_clock::time_point
                prescore_start =
                    std::chrono::steady_clock::now();

            pcl::PointCloud<LIDAR_POINT>::ConstPtr
                prescore_source =
                    current_keyframe.cloud;

            pcl::PointCloud<LIDAR_POINT>::ConstPtr
                prescore_target =
                    historical_target_K;

            const bool reverse_sequence_guess =
                std::string(guess_entry.first) ==
                "CANDIDATE_REVERSE_SEQUENCE";

            if (reverse_sequence_guess &&
                reverse_sequence_target_ready &&
                reverse_sequence_source_L &&
                reverse_sequence_target_K &&
                !reverse_sequence_source_L->empty() &&
                !reverse_sequence_target_K->empty())
            {
                // V22: compare the same ~3m physical support on both sides.
                // source is in current anchor L; target is in historical K.
                prescore_source =
                    reverse_sequence_source_L;
                prescore_target =
                    reverse_sequence_target_K;
            }
            else if (hierarchical_retrieval_candidate &&
                     !candidate_matches_reverse_sequence &&
                     hierarchical_segment_target_ready)
            {
                // V22.6: hierarchical discovery is segment-level, therefore
                // EVERY seed for this candidate is scored on the same 3m<->3m
                // physical support.  TRACK_PREDICTED cannot win by looking only
                // at a compact KF-centered target anymore.
                prescore_source =
                    hierarchical_segment_source_L;
                prescore_target =
                    hierarchical_segment_target_K;
            }

            const bool trusted_reverse_sequence_gate =
                reverse_sequence_guess &&
                reverse_sequence_confirmed &&
                candidate_matches_reverse_sequence;

            const bool score_ok =
                loop_verifier_.ScoreInitialGuess(
                    prescore_source,
                    prescore_target,
                    ranked_guess.transform,
                    trusted_reverse_sequence_gate,
                    ranked_guess.score);

            loop_timing.verifier_prescore_ms +=
                ElapsedMilliseconds(
                    prescore_start,
                    std::chrono::steady_clock::now());

            ++loop_timing.verifier_prescore_calls;

            ranked_guess.score.valid =
                score_ok && ranked_guess.score.valid;

            ranked_guesses.push_back(
                ranked_guess);
        }

        std::sort(
            ranked_guesses.begin(),
            ranked_guesses.end(),
            [](const RankedInitialGuess &lhs,
               const RankedInitialGuess &rhs)
            {
                if (lhs.score.valid != rhs.score.valid)
                {
                    return lhs.score.valid;
                }

                if (std::abs(
                        lhs.score.overlap_ratio -
                        rhs.score.overlap_ratio) > 1.0e-12)
                {
                    return lhs.score.overlap_ratio >
                           rhs.score.overlap_ratio;
                }

                return lhs.score.rmse < rhs.score.rmse;
            });

        // V22 correctness guard: once the reverse sequence has been confirmed
        // and its 3m<->3m geometry is ready, the physically correct reverse
        // hypothesis must receive a full-ICP slot before the symmetric 0-deg
        // agricultural-row minimum can consume the per-candidate budget.
        if (candidate_matches_reverse_sequence &&
            reverse_sequence_target_ready)
        {
            std::stable_sort(
                ranked_guesses.begin(),
                ranked_guesses.end(),
                [](const RankedInitialGuess &lhs,
                   const RankedInitialGuess &rhs)
                {
                    const bool lhs_reverse =
                        std::string(lhs.name) ==
                        "CANDIDATE_REVERSE_SEQUENCE";

                    const bool rhs_reverse =
                        std::string(rhs.name) ==
                        "CANDIDATE_REVERSE_SEQUENCE";

                    if (lhs_reverse != rhs_reverse)
                    {
                        return lhs_reverse;
                    }

                    return false;
                });
        }

        std::cout
            << "LoopVerifier prescore"
            << " | current_kf=" << current_keyframe.id
            << " | historical_kf=" << candidate.candidate_id;

        for (const RankedInitialGuess &ranked_guess : ranked_guesses)
        {
            std::cout
                << " | " << ranked_guess.name
                << "=[valid:"
                << (ranked_guess.score.valid ? "true" : "false")
                << ",overlap:" << ranked_guess.score.overlap_ratio
                << ",rmse:" << ranked_guess.score.rmse
                << "]";
        }

        std::cout << std::endl;

        LoopVerificationResult verification;
        bool verification_success = false;
        bool any_trial_success = false;
        bool any_geometry_accepted = false;
        const char *best_initial_guess_name = "NONE";
        bool verification_used_reverse_sequence_target = false;
        bool verification_used_hierarchical_segment_target = false;

        double verification_graph_correction_translation =
            std::numeric_limits<double>::infinity();
        double verification_graph_correction_rotation =
            std::numeric_limits<double>::infinity();

        const LoopVerifierConfig &verifier_config =
            loop_verifier_.GetConfig();

        // ----------------------------------------------------------------
        // V15 yaw-ambiguity + tentative/active-track handling.
        //
        // Do NOT stop at the first geometrically accepted ICP hypothesis.
        // Repeated agricultural / corridor geometry can make both the 0-deg
        // and 180-deg modes look excellent locally.  Each competitive
        // hypothesis is therefore fully solved and immediately checked against
        // the graph-consistency envelope BEFORE it is allowed to compete for
        // the final loop measurement.
        //
        // This preserves the graph gate as a safety guard while using it to
        // disambiguate symmetric ICP minima instead of applying it only after
        // one possibly-wrong hypothesis has already won.
        // ----------------------------------------------------------------
        std::size_t full_verifier_calls_this_candidate = 0;

        // V22.4: when the reverse segment is confirmed, spend one full ICP on
        // each sliding anchor instead of spending two ICP hypotheses on the
        // first one or two anchors.  The reverse-sequence hypothesis is already
        // explicitly moved to the front below, so a 5-call global budget can
        // evaluate K-2..K+2 once each.
        const std::size_t
            full_verifier_budget_this_candidate =
                candidate_matches_reverse_sequence &&
                        reverse_sequence_target_ready
                    ? 1
                    : kFullVerifierCallsPerCandidate;

        for (const RankedInitialGuess &ranked_guess : ranked_guesses)
        {
            if (full_verifier_calls_this_keyframe >=
                    kFullVerifierCallBudgetPerCurrentKeyframe ||
                full_verifier_calls_this_candidate >=
                    full_verifier_budget_this_candidate)
            {
                break;
            }

            if (!ranked_guess.score.valid ||
                ranked_guess.score.overlap_ratio <
                    verifier_config.prescore_min_overlap_ratio)
            {
                continue;
            }

            LoopVerificationResult trial;

            // NaN disables LoopVerifier's own absolute-yaw replacement. We
            // already generated the candidate-centered yaw hypothesis above.
            const std::chrono::steady_clock::time_point
                verifier_start =
                    std::chrono::steady_clock::now();

            pcl::PointCloud<LIDAR_POINT>::ConstPtr
                verification_source =
                    current_keyframe.cloud;

            pcl::PointCloud<LIDAR_POINT>::ConstPtr
                verification_target =
                    historical_target_K;

            const bool reverse_sequence_guess =
                std::string(ranked_guess.name) ==
                "CANDIDATE_REVERSE_SEQUENCE";

            bool trial_used_hierarchical_segment_target = false;

            if (reverse_sequence_guess &&
                reverse_sequence_target_ready &&
                reverse_sequence_source_L &&
                reverse_sequence_target_K &&
                !reverse_sequence_source_L->empty() &&
                !reverse_sequence_target_K->empty())
            {
                verification_source =
                    reverse_sequence_source_L;
                verification_target =
                    reverse_sequence_target_K;
            }
            else if (hierarchical_retrieval_candidate &&
                     !candidate_matches_reverse_sequence &&
                     hierarchical_segment_target_ready)
            {
                verification_source =
                    hierarchical_segment_source_L;
                verification_target =
                    hierarchical_segment_target_K;
                trial_used_hierarchical_segment_target = true;
            }

            const std::string solver_seed_name(
                ranked_guess.name);

            const bool enable_loop_weak_seed_prior =
                solver_seed_name ==
                    "CANDIDATE_REVERSE_FRONTEND" ||
                solver_seed_name ==
                    "CANDIDATE_REVERSE_SEQUENCE";

            const bool enable_trusted_reverse_sequence_gate =
                reverse_sequence_guess &&
                reverse_sequence_confirmed &&
                candidate_matches_reverse_sequence;

            const bool trial_success =
                loop_verifier_.Verify(
                    verification_source,
                    verification_target,
                    ranked_guess.transform,
                    std::numeric_limits<double>::quiet_NaN(),
                    enable_loop_weak_seed_prior,
                    enable_trusted_reverse_sequence_gate,
                    trial);

            loop_timing.verifier_ms +=
                ElapsedMilliseconds(
                    verifier_start,
                    std::chrono::steady_clock::now());

            ++loop_timing.verifier_calls;
            ++full_verifier_calls_this_keyframe;
            ++full_verifier_calls_this_candidate;

            if (!trial_success)
            {
                if (reverse_pose_audit_candidate)
                {
                    std::cout
                        << "REVERSE_POSE_AUDIT"
                        << " | stage=VERIFY_FAILED"
                        << " | current_kf=" << current_keyframe.id
                        << " | historical_kf=" << candidate.candidate_id
                        << " | hypothesis=" << ranked_guess.name
                        << " | frontend_dt="
                        << reverse_pose_audit_frontend_dt << " m"
                        << " | frontend_abs_yaw="
                        << reverse_pose_audit_frontend_abs_yaw_deg
                        << " deg"
                        << " | initial_T_K_L_dt="
                        << ranked_guess.transform.translation().norm()
                        << " m"
                        << " | initial_T_K_L_yaw="
                        << reverse_pose_audit_yaw_deg(
                               ranked_guess.transform)
                        << " deg"
                        << " | verifier_success=false"
                        << std::endl;
                }

                std::cout
                    << "LoopVerifier hypothesis V20"
                    << " | current_kf=" << current_keyframe.id
                    << " | historical_kf=" << candidate.candidate_id
                    << " | initial_guess=" << ranked_guess.name
                    << " | weak_seed_prior="
                    << (enable_loop_weak_seed_prior ? "ON" : "OFF")
                    << " | trusted_reverse_sequence_gate="
                    << (enable_trusted_reverse_sequence_gate ? "ON" : "OFF")
                    << " | target="
                    << (reverse_sequence_guess
                            ? "REVERSE_3M_X_3M"
                            : (trial_used_hierarchical_segment_target
                                   ? "HIERARCHICAL_3M_X_3M"
                                   : "KEYFRAME_CENTERED_WINDOW"))
                    << " | success=false"
                    << std::endl;
                continue;
            }

            // ================================================================
            // V22.4 trusted reverse 3D VISIBILITY-AWARE overlap gate.
            //
            // For every fused source point, visibility is tested from the
            // ORIGINAL LiDAR origins of the opposite 3m window.  Points that
            // fall in a learned self-occlusion sector, outside vertical FOV, or
            // behind a nearer spherical-range return do NOT enter the overlap
            // denominator.  Only points that should physically have been
            // observable are judged as match / mismatch.
            // ================================================================
            if (reverse_sequence_guess &&
                enable_trusted_reverse_sequence_gate &&
                reverse_sequence_target_ready &&
                verification_source &&
                verification_target &&
                !verification_source->empty() &&
                !verification_target->empty())
            {
                const double verifier_one_way_overlap =
                    trial.overlap_ratio;

                const bool geometry_before_visibility =
                    trial.accepted;

                if (reverse_sequence_visibility_ready)
                {
                    VisibilityAwareOverlapMetrics
                        visibility_metrics;

                    const bool visibility_metrics_ok =
                        EvaluateVisibilityAwareLoopOverlap(
                            verification_source,
                            verification_target,
                            trial.T_target_source,
                            verifier_config,
                            reverse_sequence_source_visibility,
                            reverse_sequence_target_visibility,
                            visibility_metrics);

                    constexpr std::size_t
                        kVisibilityReverseMinCloudPoints = 150;

                    constexpr std::size_t
                        kVisibilityReverseMinEligiblePoints = 120;

                    constexpr std::size_t
                        kVisibilityReverseMinDirectionalInliers = 80;

                    constexpr double
                        kVisibilityReverseMinEligibleFraction = 0.20;

                    if (visibility_metrics_ok)
                    {
                        const bool visibility_geometry_accepted =
                            visibility_metrics.source_points >=
                                kVisibilityReverseMinCloudPoints &&
                            visibility_metrics.target_points >=
                                kVisibilityReverseMinCloudPoints &&
                            visibility_metrics.eligible_L_to_K >=
                                kVisibilityReverseMinEligiblePoints &&
                            visibility_metrics.eligible_K_to_L >=
                                kVisibilityReverseMinEligiblePoints &&
                            visibility_metrics.inliers_L_to_K >=
                                kVisibilityReverseMinDirectionalInliers &&
                            visibility_metrics.inliers_K_to_L >=
                                kVisibilityReverseMinDirectionalInliers &&
                            visibility_metrics
                                    .eligible_fraction_L_to_K >=
                                kVisibilityReverseMinEligibleFraction &&
                            visibility_metrics
                                    .eligible_fraction_K_to_L >=
                                kVisibilityReverseMinEligibleFraction &&
                            visibility_metrics.arithmetic_mean >=
                                verifier_config
                                    .trusted_reverse_min_overlap_ratio &&
                            std::isfinite(trial.rmse) &&
                            trial.rmse <=
                                verifier_config
                                    .trusted_reverse_max_rmse &&
                            std::isfinite(
                                trial.correction_translation) &&
                            trial.correction_translation <=
                                verifier_config
                                    .trusted_reverse_max_correction_translation &&
                            std::isfinite(
                                trial.correction_rotation_deg) &&
                            trial.correction_rotation_deg <=
                                verifier_config
                                    .trusted_reverse_max_correction_rotation_deg;

                        trial.overlap_ratio =
                            visibility_metrics.arithmetic_mean;

                        trial.accepted =
                            visibility_geometry_accepted;

                        std::cout
                            << "LOOP_3D_VISIBILITY_OVERLAP_V22_4"
                            << " | current_kf="
                            << current_keyframe.id
                            << " | historical_kf="
                            << candidate.candidate_id
                            << " | reverse_center_kf="
                            << reverse_sequence_historical_kf
                            << " | anchor_offset="
                            << reverse_segment_anchor_offset
                            << " | hypothesis="
                            << ranked_guess.name
                            << " | source_points="
                            << visibility_metrics.source_points
                            << " | target_points="
                            << visibility_metrics.target_points
                            << " | eligible_L_to_K="
                            << visibility_metrics.eligible_L_to_K
                            << " | eligible_K_to_L="
                            << visibility_metrics.eligible_K_to_L
                            << " | occluded_L_to_K="
                            << visibility_metrics.occluded_L_to_K
                            << " | occluded_K_to_L="
                            << visibility_metrics.occluded_K_to_L
                            << " | unobservable_L_to_K="
                            << visibility_metrics.unobservable_L_to_K
                            << " | unobservable_K_to_L="
                            << visibility_metrics.unobservable_K_to_L
                            << " | eligible_fraction_L_to_K="
                            << visibility_metrics
                                   .eligible_fraction_L_to_K
                            << " | eligible_fraction_K_to_L="
                            << visibility_metrics
                                   .eligible_fraction_K_to_L
                            << " | inliers_L_to_K="
                            << visibility_metrics.inliers_L_to_K
                            << " | inliers_K_to_L="
                            << visibility_metrics.inliers_K_to_L
                            << " | overlap_verifier_one_way="
                            << verifier_one_way_overlap
                            << " | raw_overlap_L_to_K="
                            << visibility_metrics.raw_overlap_L_to_K
                            << " | raw_overlap_K_to_L="
                            << visibility_metrics.raw_overlap_K_to_L
                            << " | visible_overlap_L_to_K="
                            << visibility_metrics.visible_overlap_L_to_K
                            << " | visible_overlap_K_to_L="
                            << visibility_metrics.visible_overlap_K_to_L
                            << " | visible_overlap_mean="
                            << visibility_metrics.arithmetic_mean
                            << " | visible_overlap_harmonic="
                            << visibility_metrics.harmonic_mean
                            << " | decision_overlap="
                            << trial.overlap_ratio
                            << " | min_overlap="
                            << verifier_config
                                   .trusted_reverse_min_overlap_ratio
                            << " | rmse="
                            << trial.rmse
                            << " | correction_dt="
                            << trial.correction_translation
                            << " | correction_dR="
                            << trial.correction_rotation_deg
                            << " | geometry_before="
                            << (geometry_before_visibility
                                    ? "PASS"
                                    : "REJECT")
                            << " | geometry_after="
                            << (trial.accepted
                                    ? "PASS"
                                    : "REJECT")
                            << " | decision_metric=SPHERICAL_RAY_VISIBILITY_MEAN"
                            << std::endl;
                    }
                    else
                    {
                        std::cout
                            << "LOOP_3D_VISIBILITY_OVERLAP_V22_4"
                            << " | current_kf="
                            << current_keyframe.id
                            << " | historical_kf="
                            << candidate.candidate_id
                            << " | anchor_offset="
                            << reverse_segment_anchor_offset
                            << " | hypothesis="
                            << ranked_guess.name
                            << " | valid=false"
                            << " | overlap_verifier_one_way="
                            << verifier_one_way_overlap
                            << " | action=KEEP_VERIFIER_DECISION"
                            << std::endl;
                    }
                }
                else
                {
                    std::cout
                        << "LOOP_3D_VISIBILITY_OVERLAP_V22_4"
                        << " | current_kf="
                        << current_keyframe.id
                        << " | historical_kf="
                        << candidate.candidate_id
                        << " | anchor_offset="
                        << reverse_segment_anchor_offset
                        << " | hypothesis="
                        << ranked_guess.name
                        << " | valid=false"
                        << " | reason=VISIBILITY_MODEL_NOT_READY"
                        << " | overlap_verifier_one_way="
                        << verifier_one_way_overlap
                        << " | action=KEEP_VERIFIER_DECISION"
                        << std::endl;
                }
            }

            // ================================================================
            // V22.6 visibility-aware overlap for generic hierarchical segments.
            //
            // The reverse-only V22.4 path above keeps its stricter trusted
            // reverse gates.  For generic V21 hierarchical candidates we keep
            // the normal LoopVerifier acceptance and add only the physically
            // visible 3m-overlap requirement.  This avoids making a compact
            // turn/corridor alias look strong merely because fused hidden points
            // happen to coincide.
            // ================================================================
            if (trial_used_hierarchical_segment_target &&
                hierarchical_segment_visibility_ready &&
                verification_source &&
                verification_target &&
                !verification_source->empty() &&
                !verification_target->empty())
            {
                const double verifier_one_way_overlap =
                    trial.overlap_ratio;

                const bool geometry_before_visibility =
                    trial.accepted;

                VisibilityAwareOverlapMetrics
                    visibility_metrics;

                const bool visibility_metrics_ok =
                    EvaluateVisibilityAwareLoopOverlap(
                        verification_source,
                        verification_target,
                        trial.T_target_source,
                        verifier_config,
                        hierarchical_segment_source_visibility,
                        hierarchical_segment_target_visibility,
                        visibility_metrics);

                constexpr std::size_t
                    kHierarchicalVisibilityMinCloudPoints = 150;

                constexpr std::size_t
                    kHierarchicalVisibilityMinEligiblePoints = 120;

                constexpr std::size_t
                    kHierarchicalVisibilityMinDirectionalInliers = 80;

                constexpr double
                    kHierarchicalVisibilityMinEligibleFraction = 0.20;

                if (visibility_metrics_ok)
                {
                    const bool visibility_support_ok =
                        visibility_metrics.source_points >=
                            kHierarchicalVisibilityMinCloudPoints &&
                        visibility_metrics.target_points >=
                            kHierarchicalVisibilityMinCloudPoints &&
                        visibility_metrics.eligible_L_to_K >=
                            kHierarchicalVisibilityMinEligiblePoints &&
                        visibility_metrics.eligible_K_to_L >=
                            kHierarchicalVisibilityMinEligiblePoints &&
                        visibility_metrics.inliers_L_to_K >=
                            kHierarchicalVisibilityMinDirectionalInliers &&
                        visibility_metrics.inliers_K_to_L >=
                            kHierarchicalVisibilityMinDirectionalInliers &&
                        visibility_metrics.eligible_fraction_L_to_K >=
                            kHierarchicalVisibilityMinEligibleFraction &&
                        visibility_metrics.eligible_fraction_K_to_L >=
                            kHierarchicalVisibilityMinEligibleFraction &&
                        visibility_metrics.arithmetic_mean >=
                            verifier_config
                                .trusted_reverse_min_overlap_ratio;

                    trial.overlap_ratio =
                        visibility_metrics.arithmetic_mean;

                    trial.accepted =
                        geometry_before_visibility &&
                        visibility_support_ok;

                    std::cout
                        << "HIERARCHICAL_3M_VISIBILITY_V22_6"
                        << " | current_kf="
                        << current_keyframe.id
                        << " | historical_kf="
                        << candidate.candidate_id
                        << " | direction="
                        << (hierarchical_segment_direction < 0
                                ? "BACKWARD_3M"
                                : "FORWARD_3M")
                        << " | hypothesis="
                        << ranked_guess.name
                        << " | eligible_L_to_K="
                        << visibility_metrics.eligible_L_to_K
                        << " | eligible_K_to_L="
                        << visibility_metrics.eligible_K_to_L
                        << " | inliers_L_to_K="
                        << visibility_metrics.inliers_L_to_K
                        << " | inliers_K_to_L="
                        << visibility_metrics.inliers_K_to_L
                        << " | visible_overlap_L_to_K="
                        << visibility_metrics.visible_overlap_L_to_K
                        << " | visible_overlap_K_to_L="
                        << visibility_metrics.visible_overlap_K_to_L
                        << " | visible_overlap_mean="
                        << visibility_metrics.arithmetic_mean
                        << " | overlap_verifier_one_way="
                        << verifier_one_way_overlap
                        << " | min_overlap="
                        << verifier_config
                               .trusted_reverse_min_overlap_ratio
                        << " | geometry_before="
                        << (geometry_before_visibility
                                ? "PASS"
                                : "REJECT")
                        << " | geometry_after="
                        << (trial.accepted
                                ? "PASS"
                                : "REJECT")
                        << std::endl;
                }
                else
                {
                    // A failed visibility model must not silently downgrade a
                    // required hierarchical segment back to KF-centered logic.
                    // Keep the segment ICP result, but log that visibility could
                    // not add support this frame.
                    std::cout
                        << "HIERARCHICAL_3M_VISIBILITY_V22_6"
                        << " | current_kf="
                        << current_keyframe.id
                        << " | historical_kf="
                        << candidate.candidate_id
                        << " | valid=false"
                        << " | action=KEEP_SEGMENT_VERIFIER_DECISION"
                        << std::endl;
                }
            }

            any_trial_success = true;

            if (trial.accepted)
            {
                any_geometry_accepted = true;
            }

            // ================================================================
            // Loop ICP RViz diagnostic.
            //
            // Capture ONLY reverse-traversal hypotheses.  The snapshot is
            // produced immediately after LoopVerifier returns, BEFORE ICP /
            // graph gates can reject it, so the exact failure mode remains
            // visible.
            //
            // The ICP target is expressed in historical candidate frame K.
            // Put all three clouds into the same PoseGraph/world frame:
            //
            //   Hist   = T_WK * target_K
            //   Init   = T_WK * T_KL_initial * current_cloud_L
            //   Final  = T_WK * T_KL_final   * current_cloud_L
            //
            // If the yellow initial cloud overlaps the red historical target
            // while the green final cloud slides away, the initial transform
            // is correct and ICP itself is leaving the correct basin.
            // ================================================================
            const std::string loop_icp_debug_guess_name =
                ranked_guess.name != nullptr
                    ? std::string(ranked_guess.name)
                    : std::string();

            const bool loop_icp_debug_reverse_frontend =
                loop_icp_debug_guess_name ==
                "CANDIDATE_REVERSE_FRONTEND";

            const bool loop_icp_debug_reverse_sequence =
                loop_icp_debug_guess_name ==
                "CANDIDATE_REVERSE_SEQUENCE";

            if ((loop_icp_debug_reverse_frontend ||
                 loop_icp_debug_reverse_sequence) &&
                verification_target &&
                !verification_target->empty() &&
                verification_source &&
                !verification_source->empty() &&
                ranked_guess.transform.matrix().allFinite() &&
                trial.T_target_source.matrix().allFinite())
            {
                Eigen::Isometry3d T_W_K_debug =
                    historical_keyframe->T_WL;

                // Prefer the current PoseGraph estimate if available.  Before
                // the first accepted loop it is normally identical to raw
                // frontend T_WL; after PGO it keeps the diagnostic aligned
                // with RViz's backend/world frame.
                const PoseGraphNode *historical_graph_node =
                    pose_graph_.GetNode(
                        historical_keyframe->id);

                if (historical_graph_node != nullptr &&
                    historical_graph_node->T_WK
                        .matrix()
                        .allFinite())
                {
                    T_W_K_debug =
                        historical_graph_node->T_WK;
                }

                if (T_W_K_debug.matrix().allFinite())
                {
                    const Eigen::Isometry3d T_W_L_initial =
                        T_W_K_debug *
                        ranked_guess.transform;

                    const Eigen::Isometry3d T_W_L_final =
                        T_W_K_debug *
                        trial.T_target_source;

                    if (T_W_L_initial.matrix().allFinite() &&
                        T_W_L_final.matrix().allFinite())
                    {
                        pcl::PointCloud<LIDAR_POINT>::Ptr
                            historical_target_world(
                                new pcl::PointCloud<LIDAR_POINT>());

                        pcl::PointCloud<LIDAR_POINT>::Ptr
                            initial_aligned_world(
                                new pcl::PointCloud<LIDAR_POINT>());

                        pcl::PointCloud<LIDAR_POINT>::Ptr
                            final_aligned_world(
                                new pcl::PointCloud<LIDAR_POINT>());

                        pcl::transformPointCloud(
                            *verification_target,
                            *historical_target_world,
                            T_W_K_debug.matrix().cast<float>());

                        pcl::transformPointCloud(
                            *verification_source,
                            *initial_aligned_world,
                            T_W_L_initial.matrix().cast<float>());

                        pcl::transformPointCloud(
                            *verification_source,
                            *final_aligned_world,
                            T_W_L_final.matrix().cast<float>());

                        if (!historical_target_world->empty() &&
                            !initial_aligned_world->empty() &&
                            !final_aligned_world->empty())
                        {
                            const int debug_priority =
                                loop_icp_debug_reverse_sequence
                                    ? 2
                                    : 1;

                            bool snapshot_replaced = false;
                            std::size_t snapshot_revision = 0;

                            {
                                std::lock_guard<std::mutex> lock(
                                    backend_output_mutex_);

                                const bool newer_current_kf =
                                    backend_loop_icp_debug_revision_snapshot_ == 0 ||
                                    current_keyframe.id >
                                        backend_loop_icp_debug_current_kf_snapshot_;

                                const bool same_current_kf =
                                    backend_loop_icp_debug_revision_snapshot_ != 0 &&
                                    current_keyframe.id ==
                                        backend_loop_icp_debug_current_kf_snapshot_;

                                const bool higher_priority =
                                    same_current_kf &&
                                    debug_priority >
                                        backend_loop_icp_debug_priority_snapshot_;

                                const bool same_priority_larger_slide =
                                    same_current_kf &&
                                    debug_priority ==
                                        backend_loop_icp_debug_priority_snapshot_ &&
                                    std::isfinite(
                                        trial.correction_translation) &&
                                    (!std::isfinite(
                                         backend_loop_icp_debug_correction_translation_snapshot_) ||
                                     trial.correction_translation >
                                         backend_loop_icp_debug_correction_translation_snapshot_);

                                if (newer_current_kf ||
                                    higher_priority ||
                                    same_priority_larger_slide)
                                {
                                    backend_loop_icp_historical_target_snapshot_ =
                                        historical_target_world;

                                    backend_loop_icp_initial_aligned_snapshot_ =
                                        initial_aligned_world;

                                    backend_loop_icp_final_aligned_snapshot_ =
                                        final_aligned_world;

                                    backend_loop_icp_debug_current_kf_snapshot_ =
                                        current_keyframe.id;

                                    backend_loop_icp_debug_historical_kf_snapshot_ =
                                        candidate.candidate_id;

                                    backend_loop_icp_debug_guess_name_snapshot_ =
                                        loop_icp_debug_guess_name;

                                    backend_loop_icp_debug_correction_translation_snapshot_ =
                                        trial.correction_translation;

                                    backend_loop_icp_debug_correction_rotation_snapshot_ =
                                        trial.correction_rotation_deg;

                                    backend_loop_icp_debug_priority_snapshot_ =
                                        debug_priority;

                                    ++backend_loop_icp_debug_revision_snapshot_;

                                    snapshot_revision =
                                        backend_loop_icp_debug_revision_snapshot_;

                                    snapshot_replaced = true;
                                }
                            }

                            if (snapshot_replaced)
                            {
                                std::cout
                                    << "LOOP_ICP_RVIZ_DEBUG"
                                    << " | current_kf="
                                    << current_keyframe.id
                                    << " | historical_kf="
                                    << candidate.candidate_id
                                    << " | initial_guess="
                                    << loop_icp_debug_guess_name
                                    << " | target="
                                    << (reverse_sequence_guess
                                            ? "REVERSE_3M_X_3M"
                                            : "KEYFRAME_CENTERED_WINDOW")
                                    << " | historical_points="
                                    << historical_target_world->size()
                                    << " | initial_points="
                                    << initial_aligned_world->size()
                                    << " | final_points="
                                    << final_aligned_world->size()
                                    << " | icp_dt="
                                    << trial.correction_translation
                                    << " m"
                                    << " | icp_dR="
                                    << trial.correction_rotation_deg
                                    << " deg"
                                    << " | revision="
                                    << snapshot_revision
                                    << std::endl;
                            }
                        }
                    }
                }
            }

            const bool trial_icp_gate_pass =
                trial.accepted &&
                std::isfinite(trial.correction_translation) &&
                std::isfinite(trial.correction_rotation_deg) &&
                trial.correction_translation <=
                    max_loop_icp_correction_translation_ &&
                trial.correction_rotation_deg <=
                    max_loop_icp_correction_rotation_deg_;

            double trial_graph_correction_translation =
                std::numeric_limits<double>::infinity();
            double trial_graph_correction_rotation =
                std::numeric_limits<double>::infinity();
            bool trial_graph_gate_pass = false;
            double trial_graph_correction_path_ratio =
                std::numeric_limits<double>::infinity();

            bool trial_graph_correction_path_ratio_ok = true;

            double trial_graph_translation_limit =
                max_loop_graph_correction_translation_;
            double trial_graph_rotation_limit =
                max_loop_graph_correction_rotation_deg_;
            bool trial_graph_followup_mode = false;
            double trial_graph_corridor_distance =
                std::numeric_limits<double>::infinity();
            double trial_graph_track_prediction_translation_error =
                std::numeric_limits<double>::infinity();
            double trial_graph_track_prediction_rotation_error =
                std::numeric_limits<double>::infinity();

            bool trial_graph_has_trusted_raw_seed = false;

            // V19 target frame IS candidate Keyframe K.
            const Eigen::Isometry3d trial_T_K_L =
                trial.T_target_source;

            if (trial_T_K_L.matrix().allFinite())
            {
                const Eigen::Isometry3d trial_T_W_L_loop =
                    historical_keyframe->T_WL *
                    trial_T_K_L;

                if (trial_T_W_L_loop.matrix().allFinite())
                {
                    trial_graph_correction_translation =
                        (trial_T_W_L_loop.translation() -
                         current_keyframe.T_WL.translation())
                            .norm();

                    trial_graph_correction_rotation =
                        RelativeRotationDeg(
                            current_keyframe.T_WL,
                            trial_T_W_L_loop);

                    // ----------------------------------------------------
                    // V19.6 adaptive graph gate.
                    //
                    // The fixed 20 m threshold is kept for an unknown seed.
                    // A trusted raw-SC pending seed is sufficient to enable
                    // continuation mode when the local/corridor checks pass.
                    // may use the larger continuation cap, and only when the
                    // candidate is spatially compatible with the corrected
                    // revisit corridor AND agrees with the previous track
                    // prediction in local pose space.
                    // ----------------------------------------------------
                    trial_graph_has_trusted_raw_seed =
                        !pending_first_loop_batch_.empty() &&
                        pending_first_loop_batch_.front()
                            .raw_scan_context_supported;

                    const bool trial_graph_has_confirmed_track =
                        !has_last_online_loop_edge_ &&
                        online_loop_track_.valid &&
                        online_loop_track_.support >=
                            online_loop_graph_followup_min_track_support_ &&
                        trial_graph_has_trusted_raw_seed;

                    if (trial_graph_has_confirmed_track)
                    {
                        const Eigen::Isometry3d
                            trial_T_W_L_track_prediction =
                                online_loop_track_.T_loop_correction *
                                current_keyframe.T_WL;

                        if (trial_T_W_L_track_prediction.matrix().allFinite())
                        {
                            trial_graph_corridor_distance =
                                (historical_keyframe->T_WL.translation() -
                                 trial_T_W_L_track_prediction.translation())
                                    .norm();

                            const Eigen::Isometry3d
                                trial_track_prediction_error =
                                    trial_T_W_L_track_prediction.inverse() *
                                    trial_T_W_L_loop;

                            if (trial_track_prediction_error.matrix().allFinite())
                            {
                                trial_graph_track_prediction_translation_error =
                                    trial_track_prediction_error
                                        .translation()
                                        .norm();

                                trial_graph_track_prediction_rotation_error =
                                    RelativeRotationDeg(
                                        Eigen::Isometry3d::Identity(),
                                        trial_track_prediction_error);
                            }

                            const bool trial_graph_corridor_ok =
                                std::isfinite(
                                    trial_graph_corridor_distance) &&
                                trial_graph_corridor_distance <=
                                    online_loop_drift_aware_corridor_accept_radius_;

                            const bool trial_graph_track_prediction_ok =
                                std::isfinite(
                                    trial_graph_track_prediction_translation_error) &&
                                std::isfinite(
                                    trial_graph_track_prediction_rotation_error) &&
                                trial_graph_track_prediction_translation_error <=
                                    online_loop_track_translation_error_ &&
                                trial_graph_track_prediction_rotation_error <=
                                    online_loop_track_rotation_error_deg_;

                            trial_graph_followup_mode =
                                trial_graph_corridor_ok &&
                                trial_graph_track_prediction_ok;

                            if (trial_graph_followup_mode)
                            {
                                trial_graph_translation_limit =
                                    online_loop_graph_followup_translation_cap_;
                            }
                        }
                    }

                    // V22.7: once a post-loop temporal track is already
                    // established, apply a tighter graph guard only to
                    // candidates that stay in the same local historical
                    // neighborhood.  Remote candidates keep the generic gate
                    // so a genuinely new loop cluster can still be discovered.
                    if (has_last_online_loop_edge_ &&
                        online_loop_track_.valid)
                    {
                        const std::size_t trial_hist_gap_to_track =
                            candidate.candidate_id >=
                                    online_loop_track_
                                        .last_historical_keyframe_id
                                ? candidate.candidate_id -
                                      online_loop_track_
                                          .last_historical_keyframe_id
                                : online_loop_track_
                                          .last_historical_keyframe_id -
                                      candidate.candidate_id;

                        const std::size_t trial_current_gap_to_track =
                            current_keyframe.id >=
                                    online_loop_track_
                                        .last_current_keyframe_id
                                ? current_keyframe.id -
                                      online_loop_track_
                                          .last_current_keyframe_id
                                : online_loop_track_
                                          .last_current_keyframe_id -
                                      current_keyframe.id;

                        const bool same_post_loop_track_neighborhood =
                            trial_hist_gap_to_track <=
                                kPostLoopSameClusterMaxHistoricalGap &&
                            trial_current_gap_to_track <=
                                kPostLoopSameClusterMaxCurrentGap;

                        if (same_post_loop_track_neighborhood)
                        {
                            trial_graph_translation_limit =
                                std::min(
                                    trial_graph_translation_limit,
                                    kPostLoopSameTrackGraphMaxTranslationM);
                            trial_graph_rotation_limit =
                                std::min(
                                    trial_graph_rotation_limit,
                                    kPostLoopSameTrackGraphMaxRotationDeg);
                        }
                    }

                    if (candidate_odom_arc_valid &&
                        candidate_odom_arc_length >=
                            online_loop_large_drift_min_arc_length_for_ratio_ &&
                        std::isfinite(trial_graph_correction_translation) &&
                        trial_graph_correction_translation >
                            online_loop_large_drift_trigger_translation_)
                    {
                        trial_graph_correction_path_ratio =
                            trial_graph_correction_translation /
                            candidate_odom_arc_length;

                        trial_graph_correction_path_ratio_ok =
                            std::isfinite(trial_graph_correction_path_ratio) &&
                            trial_graph_correction_path_ratio <=
                                online_loop_large_drift_max_correction_path_ratio_;
                    }

                    trial_graph_gate_pass =
                        std::isfinite(
                            trial_graph_correction_translation) &&
                        std::isfinite(
                            trial_graph_correction_rotation) &&
                        trial_graph_correction_translation <=
                            trial_graph_translation_limit &&
                        trial_graph_correction_rotation <=
                            trial_graph_rotation_limit &&
                        trial_graph_correction_path_ratio_ok;
                }
            }

            if (reverse_pose_audit_candidate &&
                trial.T_target_source.matrix().allFinite())
            {
                const Eigen::Isometry3d &T_K_L_final =
                    trial.T_target_source;

                const Eigen::Isometry3d T_W_L_loop_audit =
                    historical_keyframe->T_WL *
                    T_K_L_final;

                const Eigen::Isometry3d
                    T_frontend_to_loop =
                        current_keyframe.T_WL.inverse() *
                        T_W_L_loop_audit;

                const Eigen::Isometry3d
                    T_initial_to_final =
                        ranked_guess.transform.inverse() *
                        T_K_L_final;

                const double frontend_vs_loop_dt =
                    T_frontend_to_loop.matrix().allFinite()
                        ? T_frontend_to_loop.translation().norm()
                        : std::numeric_limits<double>::infinity();

                const double frontend_vs_loop_dR =
                    T_frontend_to_loop.matrix().allFinite()
                        ? RelativeRotationDeg(
                              Eigen::Isometry3d::Identity(),
                              T_frontend_to_loop)
                        : std::numeric_limits<double>::infinity();

                const double frontend_vs_loop_dyaw =
                    reverse_pose_audit_yaw_deg(
                        T_frontend_to_loop);

                const double initial_to_final_dt =
                    T_initial_to_final.matrix().allFinite()
                        ? T_initial_to_final.translation().norm()
                        : std::numeric_limits<double>::infinity();

                const double initial_to_final_dR =
                    T_initial_to_final.matrix().allFinite()
                        ? RelativeRotationDeg(
                              Eigen::Isometry3d::Identity(),
                              T_initial_to_final)
                        : std::numeric_limits<double>::infinity();

                std::cout
                    << "REVERSE_POSE_AUDIT"
                    << " | stage=TRIAL"
                    << " | current_kf=" << current_keyframe.id
                    << " | historical_kf=" << candidate.candidate_id
                    << " | hypothesis=" << ranked_guess.name
                    << " | yaw_WK="
                    << reverse_pose_audit_yaw_deg(
                           historical_keyframe->T_WL)
                    << " deg"
                    << " | yaw_WL_frontend="
                    << reverse_pose_audit_yaw_deg(
                           current_keyframe.T_WL)
                    << " deg"
                    << " | frontend_T_K_L_yaw="
                    << reverse_pose_audit_yaw_deg(
                           T_K_L_frontend)
                    << " deg"
                    << " | initial_T_K_L_yaw="
                    << reverse_pose_audit_yaw_deg(
                           ranked_guess.transform)
                    << " deg"
                    << " | final_T_K_L_yaw="
                    << reverse_pose_audit_yaw_deg(
                           T_K_L_final)
                    << " deg"
                    << " | yaw_WL_loop="
                    << reverse_pose_audit_yaw_deg(
                           T_W_L_loop_audit)
                    << " deg"
                    << " | initial_to_final_dt="
                    << initial_to_final_dt << " m"
                    << " | initial_to_final_dR="
                    << initial_to_final_dR << " deg"
                    << " | frontend_vs_loop_dt="
                    << frontend_vs_loop_dt << " m"
                    << " | frontend_vs_loop_dR="
                    << frontend_vs_loop_dR << " deg"
                    << " | frontend_vs_loop_dyaw="
                    << frontend_vs_loop_dyaw << " deg"
                    << " | overlap=" << trial.overlap_ratio
                    << " | rmse=" << trial.rmse << " m"
                    << " | geometry="
                    << (trial.accepted ? "PASS" : "REJECT")
                    << " | icp_gate="
                    << (trial_icp_gate_pass ? "PASS" : "REJECT")
                    << " | graph_dt="
                    << trial_graph_correction_translation << " m"
                    << " | graph_dR="
                    << trial_graph_correction_rotation << " deg"
                    << " | graph_gate="
                    << (trial_graph_gate_pass ? "PASS" : "REJECT")
                    << std::endl;
            }

            std::cout
                << "LoopVerifier hypothesis V20"
                << " | odom_arc_length="
                << candidate_odom_arc_length << " m"
                << " | graph_correction_path_ratio="
                << trial_graph_correction_path_ratio
                << " | path_ratio_gate="
                << (trial_graph_correction_path_ratio_ok
                        ? "PASS"
                        : "REJECT")
                << " | current_kf=" << current_keyframe.id
                << " | historical_kf=" << candidate.candidate_id
                << " | initial_guess=" << ranked_guess.name
                << " | weak_seed_prior="
                << (enable_loop_weak_seed_prior ? "ON" : "OFF")
                << " | trusted_reverse_sequence_gate="
                << (enable_trusted_reverse_sequence_gate ? "ON" : "OFF")
                << " | target="
                << (reverse_sequence_guess
                        ? "REVERSE_3M_X_3M"
                        : (trial_used_hierarchical_segment_target
                               ? "HIERARCHICAL_3M_X_3M"
                               : "KEYFRAME_CENTERED_WINDOW"))
                << " | success=true"
                << " | accepted_geometry="
                << (trial.accepted ? "true" : "false")
                << " | overlap=" << trial.overlap_ratio
                << " | rmse=" << trial.rmse << " m"
                << " | icp_correction_translation="
                << trial.correction_translation << " m"
                << " | icp_correction_rotation="
                << trial.correction_rotation_deg << " deg"
                << " | icp_gate="
                << (trial_icp_gate_pass ? "PASS" : "REJECT")
                << " | graph_correction_translation="
                << trial_graph_correction_translation << " m"
                << " | graph_correction_rotation="
                << trial_graph_correction_rotation << " deg"
                << " | graph_translation_limit="
                << trial_graph_translation_limit << " m"
                << " | graph_rotation_limit="
                << trial_graph_rotation_limit << " deg"
                << " | graph_gate_mode="
                << (trial_graph_followup_mode
                        ? "TRACK_ADAPTIVE"
                        : "SEED_FIXED")
                << " | trusted_raw_seed="
                << (trial_graph_has_trusted_raw_seed
                        ? "true"
                        : "false")
                << " | graph_corridor_distance="
                << trial_graph_corridor_distance << " m"
                << " | graph_track_pred_dt="
                << trial_graph_track_prediction_translation_error << " m"
                << " | graph_track_pred_dR="
                << trial_graph_track_prediction_rotation_error << " deg"
                << " | graph_gate="
                << (trial_graph_gate_pass ? "PASS" : "REJECT")
                << std::endl;

            if (!trial_icp_gate_pass ||
                !trial_graph_gate_pass)
            {
                continue;
            }

            bool trial_better =
                !verification_success;

            if (!trial_better &&
                trial.overlap_ratio >
                    verification.overlap_ratio + 1.0e-12)
            {
                trial_better = true;
            }
            else if (!trial_better &&
                     std::abs(
                         trial.overlap_ratio -
                         verification.overlap_ratio) <= 1.0e-12 &&
                     trial.rmse <
                         verification.rmse - 1.0e-12)
            {
                trial_better = true;
            }
            else if (!trial_better &&
                     std::abs(
                         trial.overlap_ratio -
                         verification.overlap_ratio) <= 1.0e-12 &&
                     std::abs(
                         trial.rmse -
                         verification.rmse) <= 1.0e-12 &&
                     trial_graph_correction_rotation <
                         verification_graph_correction_rotation)
            {
                trial_better = true;
            }

            if (trial_better)
            {
                verification = trial;
                verification_success = true;
                best_initial_guess_name = ranked_guess.name;
                verification_used_reverse_sequence_target =
                    reverse_sequence_guess;
                verification_used_hierarchical_segment_target =
                    trial_used_hierarchical_segment_target;
                verification_graph_correction_translation =
                    trial_graph_correction_translation;
                verification_graph_correction_rotation =
                    trial_graph_correction_rotation;
            }

            // V14 deliberately has NO early break here.  A locally excellent
            // 0-degree mode must not prevent the 180-degree alternative from
            // being solved and checked against the graph.
        }

        std::cout
            << "Keyframe loop verification"
            << " | current_kf=" << current_keyframe.id
            << " | current_submap=" << current_submap_id
            << " | historical_kf=" << candidate.candidate_id
            << " | historical_submap=" << historical_submap->id
            << " | geometry_target="
            << (verification_used_reverse_sequence_target
                    ? "REVERSE_3M_X_3M"
                    : (verification_used_hierarchical_segment_target
                           ? "HIERARCHICAL_3M_X_3M"
                           : "KEYFRAME_CENTERED"))
            << " | target_center_kf=" << candidate.candidate_id
            << " | target_window_kfs="
            << (verification_used_reverse_sequence_target
                    ? reverse_sequence_target_keyframes.size()
                    : (verification_used_hierarchical_segment_target
                           ? hierarchical_segment_target_keyframes.size()
                           : historical_target_keyframes.size()))
            << " | any_success="
            << (any_trial_success ? "true" : "false")
            << " | any_geometry_accepted="
            << (any_geometry_accepted ? "true" : "false")
            << " | graph_compatible="
            << (verification_success ? "true" : "false")
            << " | accepted="
            << (verification_success && verification.accepted
                    ? "true"
                    : "false")
            << " | initial_guess=" << best_initial_guess_name
            << " | source_points=" << verification.source_points
            << " | target_points=" << verification.target_points
            << " | inliers=" << verification.inliers
            << " | overlap=" << verification.overlap_ratio
            << " | rmse=" << verification.rmse << " m"
            << " | correction_translation="
            << verification.correction_translation << " m"
            << " | correction_rotation="
            << verification.correction_rotation_deg << " deg"
            << " | graph_correction_translation="
            << verification_graph_correction_translation << " m"
            << " | graph_correction_rotation="
            << verification_graph_correction_rotation << " deg"
            << std::endl;

        if (!verification_success ||
            !verification.accepted)
        {
            // Geometry may have converged beautifully in a symmetric but
            // graph-inconsistent 180-degree basin.  Reject only this candidate
            // and keep evaluating the remaining unique historical Submaps.
            continue;
        }

        // Candidate-centered initialization means a real match should already
        // start near the correct basin. Large ICP corrections here are a strong
        // false-loop signal and used to create long diagonal loop edges.
        if (!std::isfinite(verification.correction_translation) ||
            !std::isfinite(verification.correction_rotation_deg) ||
            verification.correction_translation >
                max_loop_icp_correction_translation_ ||
            verification.correction_rotation_deg >
                max_loop_icp_correction_rotation_deg_)
        {
            std::cout
                << "Keyframe loop verification rejected"
                << " | current_kf=" << current_keyframe.id
                << " | historical_kf=" << candidate.candidate_id
                << " | reason=ICP_CORRECTION_TOO_LARGE"
                << " | correction_translation="
                << verification.correction_translation << " m"
                << " | correction_rotation="
                << verification.correction_rotation_deg << " deg"
                << std::endl;
            continue;
        }

        bool repeated_pending_large_drift_anchor = false;

        if (!has_last_online_loop_edge_ &&
            !pending_first_loop_batch_.empty())
        {
            const Eigen::Isometry3d &anchor_correction =
                pending_first_loop_batch_.front()
                    .T_loop_correction;

            const double anchor_correction_translation =
                anchor_correction.translation().norm();

            const double anchor_correction_rotation =
                RelativeRotationDeg(
                    Eigen::Isometry3d::Identity(),
                    anchor_correction);

            const bool pending_large_drift =
                (std::isfinite(anchor_correction_translation) &&
                 anchor_correction_translation >
                     online_loop_large_drift_trigger_translation_) ||
                (std::isfinite(anchor_correction_rotation) &&
                 anchor_correction_rotation >
                     online_loop_large_drift_trigger_rotation_deg_);

            if (pending_large_drift)
            {
                for (const PendingLoopConstraint &stored_constraint :
                     pending_first_loop_batch_)
                {
                    if (stored_constraint.historical_keyframe_id ==
                        candidate.candidate_id)
                    {
                        repeated_pending_large_drift_anchor = true;
                        break;
                    }
                }
            }
        }

        if (repeated_pending_large_drift_anchor)
        {
            std::cout
                << "Loop Large-Drift Candidate V20"
                << " | current_kf=" << current_keyframe.id
                << " | historical_kf=" << candidate.candidate_id
                << " | action=SKIP_REPEATED_PENDING_ANCHOR"
                << " | reason=NEED_INDEPENDENT_HISTORICAL_KF"
                << std::endl;
            continue;
        }

        bool better =
            best_candidate == nullptr;

        if (!better)
        {
            constexpr double epsilon = 1.0e-12;

            if (verification.overlap_ratio >
                best_verification.overlap_ratio + epsilon)
            {
                better = true;
            }
            else if (std::abs(
                         verification.overlap_ratio -
                         best_verification.overlap_ratio) <= epsilon &&
                     verification.rmse <
                         best_verification.rmse - epsilon)
            {
                better = true;
            }
            else if (std::abs(
                         verification.overlap_ratio -
                         best_verification.overlap_ratio) <= epsilon &&
                     std::abs(
                         verification.rmse -
                         best_verification.rmse) <= epsilon &&
                     verification.correction_translation <
                         best_verification.correction_translation)
            {
                better = true;
            }
        }

        if (better)
        {
            best_candidate = &candidate;
            best_historical_submap = historical_submap;
            best_verification = verification;
            best_candidate_used_reverse_sequence_target =
                verification_used_reverse_sequence_target;
            best_candidate_used_hierarchical_segment_target =
                verification_used_hierarchical_segment_target;

            if (verification_used_reverse_sequence_target)
            {
                best_candidate_target_keyframes =
                    reverse_sequence_target_keyframes;
            }
            else if (verification_used_hierarchical_segment_target)
            {
                best_candidate_target_keyframes =
                    hierarchical_segment_target_keyframes;
            }
            else
            {
                best_candidate_target_keyframes =
                    historical_target_keyframes;
            }
        }
    }

    if (best_candidate == nullptr ||
        best_historical_submap == nullptr)
    {
        std::cout
            << "Keyframe loop geometry result"
            << " | current_kf=" << current_keyframe.id
            << " | accepted_geometry=false"
            << std::endl;
        return;
    }

    loop_timing.geometry_accepted = true;

    // ------------------------------------------------------------------------
    // V19: ICP was solved directly in historical candidate Keyframe frame K.
    //
    //     target frame = K
    //     source frame = L
    //     verifier result = T_K_L
    //
    // No Submap->Keyframe conversion is necessary.
    // ------------------------------------------------------------------------
    const Keyframe *best_historical_keyframe =
        FindBackendKeyframeById(best_candidate->candidate_id);

    if (best_historical_keyframe == nullptr ||
        !best_historical_keyframe->T_WL.matrix().allFinite())
    {
        return;
    }

    const Eigen::Isometry3d T_K_L =
        best_verification.T_target_source;

    if (!T_K_L.matrix().allFinite())
    {
        return;
    }

    // What world pose would this loop imply for the CURRENT KEYFRAME?
    const Eigen::Isometry3d T_W_L_loop =
        best_historical_keyframe->T_WL *
        T_K_L;

    // ========================================================================
    // V22.7 Segment representative Keyframe.
    //
    // Geometry is solved in the segment ANCHOR frame K, but K is only a
    // coordinate choice.  For graph/debug semantics choose the historical
    // member R whose frontend position is closest to the loop-implied current
    // position, then re-express the exact same rigid transform as T_R_L:
    //
    //     T_R_L = T_W_R^-1 * T_W_K * T_K_L
    //
    // Therefore T_W_R*T_R_L == T_W_K*T_K_L; graph correction is unchanged.
    // ========================================================================
    std::size_t representative_historical_keyframe_id =
        best_candidate->candidate_id;
    Eigen::Isometry3d T_representative_L =
        T_K_L;
    double representative_distance_m =
        std::numeric_limits<double>::infinity();

    const bool best_geometry_is_segment =
        best_candidate_used_reverse_sequence_target ||
        best_candidate_used_hierarchical_segment_target;

    if (kUseSegmentRepresentativeKeyframeV227 &&
        best_geometry_is_segment &&
        T_W_L_loop.matrix().allFinite() &&
        !best_candidate_target_keyframes.empty())
    {
        for (const std::size_t member_id :
             best_candidate_target_keyframes)
        {
            const Keyframe *member =
                FindBackendKeyframeById(member_id);

            if (member == nullptr ||
                !member->T_WL.matrix().allFinite())
            {
                continue;
            }

            const double distance_m =
                (member->T_WL.translation() -
                 T_W_L_loop.translation())
                    .norm();

            if (!std::isfinite(distance_m))
            {
                continue;
            }

            if (distance_m < representative_distance_m)
            {
                representative_distance_m = distance_m;
                representative_historical_keyframe_id = member_id;
            }
        }

        const Keyframe *representative_keyframe =
            FindBackendKeyframeById(
                representative_historical_keyframe_id);

        if (representative_keyframe != nullptr &&
            representative_keyframe->T_WL.matrix().allFinite())
        {
            T_representative_L =
                representative_keyframe->T_WL.inverse() *
                best_historical_keyframe->T_WL *
                T_K_L;

            if (!T_representative_L.matrix().allFinite())
            {
                representative_historical_keyframe_id =
                    best_candidate->candidate_id;
                representative_distance_m =
                    std::numeric_limits<double>::infinity();
                T_representative_L = T_K_L;
            }
        }
    }

    std::cout
        << "LOOP_SEGMENT_REPRESENTATIVE_V22_7"
        << " | current_kf=" << current_keyframe.id
        << " | segment_anchor_kf="
        << best_candidate->candidate_id
        << " | representative_kf="
        << representative_historical_keyframe_id
        << " | target_members="
        << best_candidate_target_keyframes.size()
        << " | representative_distance="
        << representative_distance_m << " m"
        << " | segment_geometry="
        << (best_geometry_is_segment ? "true" : "false")
        << std::endl;

    // ========================================================================
    // FR_FRONTEND_LOOP_DRIFT_DIAG_V1
    //
    // Pure PRE-PGO diagnostic.
    //
    // K = historical Keyframe
    // L = current Keyframe
    //
    // Frontend accumulated relative pose:
    //
    //     T_K_L_frontend
    //         =
    //     T_W_K_frontend^-1 * T_W_L_frontend
    //
    // Independent loop-verifier geometry:
    //
    //     T_K_L
    //
    // Loop-implied current world pose:
    //
    //     T_W_L_loop
    //         =
    //     T_W_K_frontend * T_K_L
    //
    // Therefore this block directly measures the discrepancy that already
    // exists BEFORE AddLoopEdge() and BEFORE g2o optimization.
    //
    // IMPORTANT:
    //   * No value below is fed back to the frontend.
    //   * No loop measurement is modified.
    //   * No information matrix is modified.
    //   * No graph state is modified.
    // ========================================================================

    const Eigen::Isometry3d T_K_L_frontend =
        best_historical_keyframe->T_WL.inverse() *
        current_keyframe.T_WL;

    if (!T_K_L_frontend.matrix().allFinite() ||
        !T_W_L_loop.matrix().allFinite())
    {
        return;
    }

    const Eigen::Vector3d world_error_W =
        current_keyframe.T_WL.translation() -
        T_W_L_loop.translation();

    const double world_error_norm_m =
        world_error_W.norm();

    const Eigen::Isometry3d T_K_L_error =
        T_K_L.inverse() *
        T_K_L_frontend;

    if (!T_K_L_error.matrix().allFinite())
    {
        return;
    }

    const Eigen::Vector3d frontend_relative_rpy =
        FrontendRotationToRpy(
            T_K_L_frontend.rotation());

    const Eigen::Vector3d loop_relative_rpy =
        FrontendRotationToRpy(
            T_K_L.rotation());

    const Eigen::Vector3d relative_error_rpy =
        FrontendRotationToRpy(
            T_K_L_error.rotation());

    constexpr double kFrontendLoopDriftRadToDeg =
        180.0 /
        3.14159265358979323846;

    const double relative_rotation_error_deg =
        RelativeRotationDeg(
            T_K_L,
            T_K_L_frontend);

    const double graph_correction_translation =
        (T_W_L_loop.translation() -
         current_keyframe.T_WL.translation())
            .norm();

    const double graph_correction_rotation =
        RelativeRotationDeg(
            current_keyframe.T_WL,
            T_W_L_loop);

    double frontend_graph_translation_limit =
        max_loop_graph_correction_translation_;
    double frontend_graph_rotation_limit =
        max_loop_graph_correction_rotation_deg_;

    bool frontend_graph_followup_mode = false;

    double frontend_graph_corridor_distance =
        std::numeric_limits<double>::infinity();

    double frontend_graph_track_prediction_translation_error =
        std::numeric_limits<double>::infinity();

    double frontend_graph_track_prediction_rotation_error =
        std::numeric_limits<double>::infinity();

    const bool frontend_graph_has_trusted_raw_seed =
        !pending_first_loop_batch_.empty() &&
        pending_first_loop_batch_.front()
            .raw_scan_context_supported;

    const bool frontend_graph_has_confirmed_track =
        !has_last_online_loop_edge_ &&
        online_loop_track_.valid &&
        online_loop_track_.support >=
            online_loop_graph_followup_min_track_support_ &&
        frontend_graph_has_trusted_raw_seed;

    if (frontend_graph_has_confirmed_track)
    {
        const Eigen::Isometry3d T_W_L_track_prediction_for_graph_gate =
            online_loop_track_.T_loop_correction *
            current_keyframe.T_WL;

        if (T_W_L_track_prediction_for_graph_gate.matrix().allFinite())
        {
            frontend_graph_corridor_distance =
                (best_historical_keyframe->T_WL.translation() -
                 T_W_L_track_prediction_for_graph_gate.translation())
                    .norm();

            const Eigen::Isometry3d
                frontend_graph_track_prediction_error =
                    T_W_L_track_prediction_for_graph_gate.inverse() *
                    T_W_L_loop;

            if (frontend_graph_track_prediction_error.matrix().allFinite())
            {
                frontend_graph_track_prediction_translation_error =
                    frontend_graph_track_prediction_error
                        .translation()
                        .norm();

                frontend_graph_track_prediction_rotation_error =
                    RelativeRotationDeg(
                        Eigen::Isometry3d::Identity(),
                        frontend_graph_track_prediction_error);
            }

            const bool frontend_graph_corridor_ok =
                std::isfinite(frontend_graph_corridor_distance) &&
                frontend_graph_corridor_distance <=
                    online_loop_drift_aware_corridor_accept_radius_;

            const bool frontend_graph_track_prediction_ok =
                std::isfinite(
                    frontend_graph_track_prediction_translation_error) &&
                std::isfinite(
                    frontend_graph_track_prediction_rotation_error) &&
                frontend_graph_track_prediction_translation_error <=
                    online_loop_track_translation_error_ &&
                frontend_graph_track_prediction_rotation_error <=
                    online_loop_track_rotation_error_deg_;

            frontend_graph_followup_mode =
                frontend_graph_corridor_ok &&
                frontend_graph_track_prediction_ok;

            if (frontend_graph_followup_mode)
            {
                frontend_graph_translation_limit =
                    online_loop_graph_followup_translation_cap_;
            }
        }
    }

    // V22.7 same-track hardening, mirrored from candidate-level gating.
    if (has_last_online_loop_edge_ &&
        online_loop_track_.valid)
    {
        const std::size_t final_hist_gap_to_track =
            best_candidate->candidate_id >=
                    online_loop_track_.last_historical_keyframe_id
                ? best_candidate->candidate_id -
                      online_loop_track_.last_historical_keyframe_id
                : online_loop_track_.last_historical_keyframe_id -
                      best_candidate->candidate_id;

        const std::size_t final_current_gap_to_track =
            current_keyframe.id >=
                    online_loop_track_.last_current_keyframe_id
                ? current_keyframe.id -
                      online_loop_track_.last_current_keyframe_id
                : online_loop_track_.last_current_keyframe_id -
                      current_keyframe.id;

        if (final_hist_gap_to_track <=
                kPostLoopSameClusterMaxHistoricalGap &&
            final_current_gap_to_track <=
                kPostLoopSameClusterMaxCurrentGap)
        {
            frontend_graph_translation_limit =
                std::min(
                    frontend_graph_translation_limit,
                    kPostLoopSameTrackGraphMaxTranslationM);
            frontend_graph_rotation_limit =
                std::min(
                    frontend_graph_rotation_limit,
                    kPostLoopSameTrackGraphMaxRotationDeg);
        }
    }

    const bool frontend_loop_drift_graph_gate_pass =
        std::isfinite(graph_correction_translation) &&
        std::isfinite(graph_correction_rotation) &&
        graph_correction_translation <=
            frontend_graph_translation_limit &&
        graph_correction_rotation <=
            frontend_graph_rotation_limit;

    RCLCPP_WARN(
        rclcpp::get_logger(
            "scan2local_map.frontend_loop_drift"),
        "FR_FRONTEND_LOOP_DRIFT"
        " | stage=PRE_PGO"
        " | current_kf=%zu"
        " | historical_kf=%zu"
        " | current_submap=%zu"
        " | historical_submap=%zu"
        " | overlap=%.6f"
        " | loop_rmse=%.6f"
        " | world_error=[%.6f %.6f %.6f]"
        " | world_error_norm=%.6f"
        " | relative_error_t=[%.6f %.6f %.6f]"
        " | relative_error_rpy_deg=[%.6f %.6f %.6f]"
        " | relative_translation_norm=%.6f"
        " | relative_rotation_deg=%.6f"
        " | graph_translation_limit=%.6f"
        " | graph_rotation_limit=%.6f"
        " | graph_gate_mode=%s"
        " | trusted_raw_seed=%s"
        " | graph_corridor_distance=%.6f"
        " | graph_track_pred_dt=%.6f"
        " | graph_track_pred_dR=%.6f"
        " | graph_gate=%s",
        current_keyframe.id,
        best_historical_keyframe->id,
        current_submap_id,
        best_historical_submap->id,
        best_verification.overlap_ratio,
        best_verification.rmse,
        world_error_W.x(),
        world_error_W.y(),
        world_error_W.z(),
        world_error_norm_m,
        T_K_L_error.translation().x(),
        T_K_L_error.translation().y(),
        T_K_L_error.translation().z(),
        relative_error_rpy.x() *
            kFrontendLoopDriftRadToDeg,
        relative_error_rpy.y() *
            kFrontendLoopDriftRadToDeg,
        relative_error_rpy.z() *
            kFrontendLoopDriftRadToDeg,
        T_K_L_error.translation().norm(),
        relative_rotation_error_deg,
        frontend_graph_translation_limit,
        frontend_graph_rotation_limit,
        frontend_graph_followup_mode
            ? "TRACK_ADAPTIVE"
            : "SEED_FIXED",
        frontend_graph_has_trusted_raw_seed
            ? "true"
            : "false",
        frontend_graph_corridor_distance,
        frontend_graph_track_prediction_translation_error,
        frontend_graph_track_prediction_rotation_error,
        frontend_loop_drift_graph_gate_pass
            ? "PASS"
            : "REJECT");

    try
    {
        static bool frontend_loop_drift_csv_initialized =
            false;

        const std::filesystem::path frontend_loop_drift_directory =
            FrontendLoopDirectory();

        std::filesystem::create_directories(
            frontend_loop_drift_directory);

        const std::filesystem::path frontend_loop_drift_csv_path =
            frontend_loop_drift_directory /
            "frontend_loop_drift.csv";

        std::ios_base::openmode frontend_loop_drift_mode =
            std::ios::out;

        if (!frontend_loop_drift_csv_initialized)
        {
            frontend_loop_drift_mode |=
                std::ios::trunc;
        }
        else
        {
            frontend_loop_drift_mode |=
                std::ios::app;
        }

        std::ofstream frontend_loop_drift_file(
            frontend_loop_drift_csv_path,
            frontend_loop_drift_mode);

        if (frontend_loop_drift_file.is_open())
        {
            frontend_loop_drift_file
                << std::fixed
                << std::setprecision(9);

            if (!frontend_loop_drift_csv_initialized)
            {
                frontend_loop_drift_file
                    << "current_kf,historical_kf,"
                    << "current_submap,historical_submap,"
                    << "overlap,loop_rmse,"
                    << "frontend_world_x,frontend_world_y,frontend_world_z,"
                    << "loop_world_x,loop_world_y,loop_world_z,"
                    << "world_error_dx,world_error_dy,world_error_dz,"
                    << "world_error_norm,"
                    << "frontend_rel_tx,frontend_rel_ty,frontend_rel_tz,"
                    << "frontend_rel_roll_deg,frontend_rel_pitch_deg,frontend_rel_yaw_deg,"
                    << "loop_rel_tx,loop_rel_ty,loop_rel_tz,"
                    << "loop_rel_roll_deg,loop_rel_pitch_deg,loop_rel_yaw_deg,"
                    << "relative_error_tx,relative_error_ty,relative_error_tz,"
                    << "relative_error_roll_deg,relative_error_pitch_deg,relative_error_yaw_deg,"
                    << "relative_translation_norm,relative_rotation_deg,"
                    << "graph_correction_translation,graph_correction_rotation_deg,"
                    << "graph_gate_pass\n";
            }

            frontend_loop_drift_file
                << current_keyframe.id << ","
                << best_historical_keyframe->id << ","
                << current_submap_id << ","
                << best_historical_submap->id << ","
                << best_verification.overlap_ratio << ","
                << best_verification.rmse << ","
                << current_keyframe.T_WL.translation().x() << ","
                << current_keyframe.T_WL.translation().y() << ","
                << current_keyframe.T_WL.translation().z() << ","
                << T_W_L_loop.translation().x() << ","
                << T_W_L_loop.translation().y() << ","
                << T_W_L_loop.translation().z() << ","
                << world_error_W.x() << ","
                << world_error_W.y() << ","
                << world_error_W.z() << ","
                << world_error_norm_m << ","
                << T_K_L_frontend.translation().x() << ","
                << T_K_L_frontend.translation().y() << ","
                << T_K_L_frontend.translation().z() << ","
                << frontend_relative_rpy.x() *
                       kFrontendLoopDriftRadToDeg
                << ","
                << frontend_relative_rpy.y() *
                       kFrontendLoopDriftRadToDeg
                << ","
                << frontend_relative_rpy.z() *
                       kFrontendLoopDriftRadToDeg
                << ","
                << T_K_L.translation().x() << ","
                << T_K_L.translation().y() << ","
                << T_K_L.translation().z() << ","
                << loop_relative_rpy.x() *
                       kFrontendLoopDriftRadToDeg
                << ","
                << loop_relative_rpy.y() *
                       kFrontendLoopDriftRadToDeg
                << ","
                << loop_relative_rpy.z() *
                       kFrontendLoopDriftRadToDeg
                << ","
                << T_K_L_error.translation().x() << ","
                << T_K_L_error.translation().y() << ","
                << T_K_L_error.translation().z() << ","
                << relative_error_rpy.x() *
                       kFrontendLoopDriftRadToDeg
                << ","
                << relative_error_rpy.y() *
                       kFrontendLoopDriftRadToDeg
                << ","
                << relative_error_rpy.z() *
                       kFrontendLoopDriftRadToDeg
                << ","
                << T_K_L_error.translation().norm() << ","
                << relative_rotation_error_deg << ","
                << graph_correction_translation << ","
                << graph_correction_rotation << ","
                << (frontend_loop_drift_graph_gate_pass ? 1 : 0)
                << "\n";

            frontend_loop_drift_file.flush();
            frontend_loop_drift_csv_initialized = true;
        }
    }
    catch (const std::exception &exception)
    {
        RCLCPP_WARN(
            rclcpp::get_logger(
                "scan2local_map.frontend_loop_drift"),
            "FR_FRONTEND_LOOP_DRIFT"
            " | stage=CSV"
            " | action=FAILED"
            " | what=%s",
            exception.what());
    }

    std::cout
        << "Keyframe loop geometry best"
        << " | current_kf=" << current_keyframe.id
        << " | current_submap=" << current_submap_id
        << " | historical_kf=" << best_candidate->candidate_id
        << " | historical_submap=" << best_historical_submap->id
        << " | geometry_target="
        << (best_candidate_used_reverse_sequence_target
                ? "REVERSE_3M_X_3M"
                : (best_candidate_used_hierarchical_segment_target
                       ? "HIERARCHICAL_3M_X_3M"
                       : "KEYFRAME_CENTERED"))
        << " | overlap=" << best_verification.overlap_ratio
        << " | rmse=" << best_verification.rmse << " m"
        << " | graph_correction_translation="
        << graph_correction_translation << " m"
        << " | graph_correction_rotation="
        << graph_correction_rotation << " deg"
        << std::endl;

    std::size_t loop_trace_selected_sc_rank = 0;

    for (std::size_t candidate_index = 0;
         candidate_index < candidates.size();
         ++candidate_index)
    {
        if (candidates[candidate_index].candidate_id ==
            best_candidate->candidate_id)
        {
            loop_trace_selected_sc_rank =
                candidate_index + 1;
            break;
        }
    }

    const bool loop_trace_selected_track_injected =
        track_continuation_candidate_injected &&
        best_candidate == &track_continuation_candidate;

    const bool loop_trace_selected_corridor =
        std::find(
            drift_aware_corridor_candidate_ids.begin(),
            drift_aware_corridor_candidate_ids.end(),
            best_candidate->candidate_id) !=
        drift_aware_corridor_candidate_ids.end();

    const bool loop_trace_selected_large_drift_neighbor =
        std::find(
            large_drift_injected_candidate_ids.begin(),
            large_drift_injected_candidate_ids.end(),
            best_candidate->candidate_id) !=
        large_drift_injected_candidate_ids.end();

    const bool loop_trace_selected_reverse_spatial =
        std::any_of(
            reverse_spatial_candidates.begin(),
            reverse_spatial_candidates.end(),
            [best_candidate](const LoopCandidate &candidate)
            {
                return candidate.candidate_id ==
                       best_candidate->candidate_id;
            });

    const bool loop_trace_selected_hierarchical_3m =
        std::find(
            hierarchical_retrieval_candidate_ids.begin(),
            hierarchical_retrieval_candidate_ids.end(),
            best_candidate->candidate_id) !=
        hierarchical_retrieval_candidate_ids.end();

    const char *loop_trace_selected_source =
        loop_trace_selected_track_injected
            ? "TRACK_INJECTED"
            : (loop_trace_selected_sc_rank > 0
                   ? "RAW_SC"
                   : (loop_trace_selected_hierarchical_3m
                          ? "HIERARCHICAL_3M_SC"
                          : (loop_trace_selected_reverse_spatial
                                 ? "REVERSE_SPATIAL"
                                 : (loop_trace_selected_corridor
                                        ? "DRIFT_CORRIDOR"
                                        : (loop_trace_selected_large_drift_neighbor
                                               ? "HISTORICAL_NEIGHBOR"
                                               : "OTHER_INJECTED")))));

    std::cout
        << "FR_LOOP_TRACE"
        << " | stage=GEOMETRY_SELECTED"
        << " | current_kf=" << current_keyframe.id
        << " | historical_kf=" << best_candidate->candidate_id
        << " | historical_submap=" << best_historical_submap->id
        << " | source=" << loop_trace_selected_source
        << " | sc_rank=" << loop_trace_selected_sc_rank
        << " | geometry_target="
        << (best_candidate_used_reverse_sequence_target
                ? "REVERSE_3M_X_3M"
                : (best_candidate_used_hierarchical_segment_target
                       ? "HIERARCHICAL_3M_X_3M"
                       : "KEYFRAME_CENTERED"))
        << " | overlap=" << best_verification.overlap_ratio
        << " | rmse=" << best_verification.rmse << " m"
        << " | icp_dt=" << best_verification.correction_translation << " m"
        << " | icp_dR=" << best_verification.correction_rotation_deg << " deg"
        << " | graph_dt=" << graph_correction_translation << " m"
        << " | graph_dR=" << graph_correction_rotation << " deg"
        << std::endl;

    // V13 redundant final safety check. Candidate-level graph gating already
    // removed graph-incompatible hypotheses before best-candidate selection,
    // but keep this independent guard immediately before temporal/batch logic.
    if (!frontend_loop_drift_graph_gate_pass)
    {
        std::cout
            << "Keyframe loop decision"
            << " | current_kf=" << current_keyframe.id
            << " | decision=REJECT"
            << " | reason=GRAPH_CORRECTION_TOO_LARGE"
            << " | graph_correction_translation="
            << graph_correction_translation << " m"
            << " | graph_translation_limit="
            << frontend_graph_translation_limit << " m"
            << " | graph_correction_rotation="
            << graph_correction_rotation << " deg"
            << " | graph_rotation_limit="
            << frontend_graph_rotation_limit << " deg"
            << " | graph_gate_mode="
            << (frontend_graph_followup_mode
                    ? "TRACK_ADAPTIVE"
                    : "SEED_FIXED")
            << " | corridor_distance="
            << frontend_graph_corridor_distance << " m"
            << " | track_pred_dt="
            << frontend_graph_track_prediction_translation_error << " m"
            << " | track_pred_dR="
            << frontend_graph_track_prediction_rotation_error << " deg"
            << std::endl;

        std::cout
            << "FR_LOOP_TRACE"
            << " | stage=DECISION"
            << " | current_kf=" << current_keyframe.id
            << " | historical_kf=" << best_candidate->candidate_id
            << " | decision=REJECT"
            << " | reason=GRAPH_CORRECTION_TOO_LARGE"
            << " | graph_dt=" << graph_correction_translation << " m"
            << " | graph_dt_limit=" << frontend_graph_translation_limit << " m"
            << " | graph_dR=" << graph_correction_rotation << " deg"
            << std::endl;
        return;
    }

    // ------------------------------------------------------------------------
    // V19.2 independent Scan-Context evidence.
    //
    // The V19.1 false loop exposed a self-confirmation failure:
    //
    //   KF148 -> Hist97   : raw Scan Context candidate
    //   KF149 -> Hist98   : injected TRACK/neighborhood candidate
    //   KF150 -> Hist99   : injected TRACK/neighborhood candidate
    //
    // The last two were geometrically consistent because a repetitive
    // agricultural corridor can keep matching while the track predictor walks
    // through adjacent historical KFs.  They were NOT independent place-
    // recognition observations.
    //
    // Raw Scan Context candidates are stored in `candidates`.  Injected
    // candidates live only in candidate_order / auxiliary vectors.  Therefore
    // check whether this current frame independently retrieved the same
    // historical neighborhood before allowing it to become a large-drift
    // first-batch factor.
    // ------------------------------------------------------------------------
    bool best_candidate_raw_sc_supported = false;
    std::size_t best_candidate_raw_sc_support_kf =
        std::numeric_limits<std::size_t>::max();
    double best_candidate_raw_sc_support_similarity = 0.0;
    std::size_t best_candidate_raw_sc_support_gap =
        std::numeric_limits<std::size_t>::max();

    for (const LoopCandidate &raw_candidate :
         candidates)
    {
        const std::size_t raw_gap =
            raw_candidate.candidate_id >= best_candidate->candidate_id
                ? raw_candidate.candidate_id - best_candidate->candidate_id
                : best_candidate->candidate_id - raw_candidate.candidate_id;

        if (raw_gap >
            online_loop_large_drift_raw_sc_support_radius_)
        {
            continue;
        }

        if (!best_candidate_raw_sc_supported ||
            raw_gap < best_candidate_raw_sc_support_gap ||
            (raw_gap == best_candidate_raw_sc_support_gap &&
             raw_candidate.scan_context_similarity >
                 best_candidate_raw_sc_support_similarity))
        {
            best_candidate_raw_sc_supported = true;
            best_candidate_raw_sc_support_kf =
                raw_candidate.candidate_id;
            best_candidate_raw_sc_support_similarity =
                raw_candidate.scan_context_similarity;
            best_candidate_raw_sc_support_gap =
                raw_gap;
        }
    }

    // V21: Region-SC retrieval is also independent GLOBAL place-recognition
    // evidence.  The legacy PendingLoopConstraint field is still named
    // raw_scan_context_supported for ABI compatibility, but from V21 onward it
    // means "independent global retrieval supported": raw single-KF SC OR the
    // hierarchical 3m Region-SC route.
    const bool best_candidate_hierarchical_sc_supported =
        std::find(
            hierarchical_retrieval_candidate_ids.begin(),
            hierarchical_retrieval_candidate_ids.end(),
            best_candidate->candidate_id) !=
        hierarchical_retrieval_candidate_ids.end();

    std::size_t best_candidate_hierarchical_submap_rank =
        std::numeric_limits<std::size_t>::max();

    double best_candidate_hierarchical_region_score =
        std::numeric_limits<double>::infinity();

    double best_candidate_hierarchical_similarity = 0.0;

    if (best_candidate_hierarchical_sc_supported)
    {
        const auto rank_iterator =
            hierarchical_candidate_submap_rank.find(
                best_candidate->candidate_id);

        if (rank_iterator !=
            hierarchical_candidate_submap_rank.end())
        {
            best_candidate_hierarchical_submap_rank =
                rank_iterator->second;
        }

        const auto score_iterator =
            hierarchical_candidate_region_score.find(
                best_candidate->candidate_id);

        if (score_iterator !=
            hierarchical_candidate_region_score.end())
        {
            best_candidate_hierarchical_region_score =
                score_iterator->second;
        }

        const auto similarity_iterator =
            hierarchical_candidate_region_similarity.find(
                best_candidate->candidate_id);

        if (similarity_iterator !=
            hierarchical_candidate_region_similarity.end())
        {
            best_candidate_hierarchical_similarity =
                similarity_iterator->second;
        }
    }

    const bool best_candidate_independent_retrieval_supported =
        best_candidate_raw_sc_supported ||
        best_candidate_hierarchical_sc_supported;

    const std::size_t best_candidate_independent_support_kf =
        best_candidate_raw_sc_supported
            ? best_candidate_raw_sc_support_kf
            : best_candidate->candidate_id;

    const double best_candidate_independent_support_similarity =
        best_candidate_raw_sc_supported
            ? best_candidate_raw_sc_support_similarity
            : best_candidate_hierarchical_similarity;

    bool best_candidate_corridor_supported = false;
    double best_candidate_corridor_distance =
        std::numeric_limits<double>::infinity();

    if (online_loop_drift_aware_corridor_enabled_ &&
        online_loop_track_.valid)
    {
        const Eigen::Isometry3d T_W_L_corridor_prediction =
            online_loop_track_.T_loop_correction *
            current_keyframe.T_WL;

        if (T_W_L_corridor_prediction.matrix().allFinite())
        {
            const Keyframe *best_historical_for_corridor =
                FindBackendKeyframeById(
                    best_candidate->candidate_id);

            if (best_historical_for_corridor != nullptr &&
                best_historical_for_corridor->T_WL.matrix().allFinite())
            {
                best_candidate_corridor_distance =
                    (best_historical_for_corridor->T_WL.translation() -
                     T_W_L_corridor_prediction.translation())
                        .norm();

                best_candidate_corridor_supported =
                    std::isfinite(best_candidate_corridor_distance) &&
                    best_candidate_corridor_distance <=
                        online_loop_drift_aware_corridor_accept_radius_;
            }
        }
    }

    const bool current_first_loop_large_drift_candidate =
        !has_last_online_loop_edge_ &&
        ((std::isfinite(graph_correction_translation) &&
          graph_correction_translation >
              online_loop_large_drift_trigger_translation_) ||
         (std::isfinite(graph_correction_rotation) &&
          graph_correction_rotation >
              online_loop_large_drift_trigger_rotation_deg_));

    std::cout
        << "Loop Independent Retrieval Evidence V21"
        << " | current_kf=" << current_keyframe.id
        << " | historical_kf=" << best_candidate->candidate_id
        << " | large_drift="
        << (current_first_loop_large_drift_candidate ? "true" : "false")
        << " | raw_sc_supported="
        << (best_candidate_raw_sc_supported ? "true" : "false")
        << " | raw_support_kf="
        << best_candidate_raw_sc_support_kf
        << " | raw_support_gap="
        << best_candidate_raw_sc_support_gap
        << " | raw_support_similarity="
        << best_candidate_raw_sc_support_similarity
        << " | hierarchical_3m_supported="
        << (best_candidate_hierarchical_sc_supported ? "true" : "false")
        << " | hierarchical_submap_rank="
        << best_candidate_hierarchical_submap_rank
        << " | hierarchical_region_score="
        << best_candidate_hierarchical_region_score
        << " | independent_global_retrieval="
        << (best_candidate_independent_retrieval_supported
                ? "true"
                : "false")
        << " | corridor_supported="
        << (best_candidate_corridor_supported ? "true" : "false")
        << " | corridor_distance="
        << best_candidate_corridor_distance << " m"
        << std::endl;

    // ========================================================================
    // V22.8 3m segment revalidation policy.
    //
    // Independent segment retrieval remains the strongest checkpoint refresh.
    // Unlike V22.7, however, failure to obtain an independent retrieval hit on
    // the exact deadline frame is NOT an immediate rejection.  We defer that
    // decision until the ordinary temporal-track checks below have run.  A
    // mature track may then use SEGMENT_CONTINUATION revalidation only if:
    //   * full segment geometry was used;
    //   * temporal pose/progression checks pass;
    //   * the representative KF has physically progressed along the historical
    //     trajectory by an amount compatible with the current ~3m motion.
    //
    // This keeps the anti-sticking property of V22.6/22.7 without creating
    // artificial holes whenever Scan Context misses one otherwise-good frame.
    // ========================================================================
    const bool best_candidate_used_segment_geometry =
        best_candidate_used_reverse_sequence_target ||
        best_candidate_used_hierarchical_segment_target;

    const bool best_candidate_independent_segment_evidence =
        best_candidate_independent_retrieval_supported &&
        best_candidate_used_segment_geometry;

    const bool track_segment_continuation_check_required =
        track_segment_revalidation_required &&
        !best_candidate_independent_segment_evidence;

    bool best_candidate_segment_continuation_evidence = false;
    double continuation_historical_arc_m =
        std::numeric_limits<double>::quiet_NaN();
    double continuation_historical_arc_ratio =
        std::numeric_limits<double>::quiet_NaN();
    std::int64_t continuation_representative_delta = 0;

    if (track_segment_continuation_check_required)
    {
        std::cout
            << "LOOP_TRACK_3M_REVALIDATION_V22_8"
            << " | current_kf=" << current_keyframe.id
            << " | historical_kf="
            << representative_historical_keyframe_id
            << " | segment_anchor_kf="
            << best_candidate->candidate_id
            << " | current_arc="
            << track_segment_revalidation_arc_m << " m"
            << " | independent_retrieval="
            << (best_candidate_independent_retrieval_supported
                    ? "true"
                    : "false")
            << " | segment_geometry="
            << (best_candidate_used_segment_geometry
                    ? "true"
                    : "false")
            << " | action=DEFER_TO_SEGMENT_CONTINUATION_CHECK"
            << std::endl;
    }

    // ------------------------------------------------------------------------
    // Keyframe-level temporal consistency.
    //
    // The loop track is expressed as a world correction on the current
    // Keyframe, so normal frontend Submap transitions do not reset it.
    //
    // V19 compares the CURRENT POSE predicted by the previous accepted loop
    // observation against the CURRENT POSE measured by this loop:
    //
    //     T_WL_loop = T_WH * T_HL
    //     T_correction = T_WL_loop * inverse(T_WL_frontend)
    //
    // For a true loop this correction should remain nearly constant for
    // consecutive Keyframes, even when the Active Submap changes.
    // ------------------------------------------------------------------------
    const Eigen::Isometry3d T_loop_correction =
        T_W_L_loop *
        current_keyframe.T_WL.inverse();

    if (!T_W_L_loop.matrix().allFinite() ||
        !T_loop_correction.matrix().allFinite())
    {
        return;
    }

    // ------------------------------------------------------------------------
    // V22.5: Historical 3m-segment continuity for FIRST-loop confirmation.
    //
    // V22.4 already verifies reverse revisits as physical 3m trajectory
    // segments, but the first-loop bootstrap still judged "same historical
    // place" using fixed Keyframe-ID gaps (for example +/-2 KF).  That is an
    // architectural mismatch: two 3m windows can strongly overlap in physical
    // trajectory space even when their representative anchor IDs differ by 5
    // or more because Keyframes are not sampled at identical positions.
    //
    // V22.5 therefore measures continuity in METRES along the historical
    // frontend trajectory and also reports overlap between the two forward-3m
    // member sets.  KF-ID distance is retained only as a diagnostic; it is no
    // longer the acceptance criterion for first-loop local continuity.
    // ------------------------------------------------------------------------
    struct HistoricalSegmentContinuityV225
    {
        bool valid = false;
        bool first_window_valid = false;
        bool second_window_valid = false;
        bool continuous = false;
        double trajectory_arc_gap_m =
            std::numeric_limits<double>::infinity();
        double anchor_euclidean_gap_m =
            std::numeric_limits<double>::infinity();
        std::size_t first_window_keyframes = 0;
        std::size_t second_window_keyframes = 0;
        std::size_t shared_keyframes = 0;
        double shared_over_min_ratio = 0.0;
    };

    const auto evaluate_historical_segment_continuity_v225 =
        [&find_retrieval_keyframe, &build_retrieval_window](
            std::size_t first_anchor_kf,
            std::size_t second_anchor_kf)
        -> HistoricalSegmentContinuityV225
    {
        HistoricalSegmentContinuityV225 result;

        const Keyframe *first_anchor =
            find_retrieval_keyframe(first_anchor_kf);
        const Keyframe *second_anchor =
            find_retrieval_keyframe(second_anchor_kf);

        if (first_anchor == nullptr ||
            second_anchor == nullptr ||
            !first_anchor->T_WL.matrix().allFinite() ||
            !second_anchor->T_WL.matrix().allFinite())
        {
            return result;
        }

        result.anchor_euclidean_gap_m =
            (first_anchor->T_WL.translation() -
             second_anchor->T_WL.translation())
                .norm();

        if (!std::isfinite(result.anchor_euclidean_gap_m))
        {
            return result;
        }

        // Measure distance ALONG the historical trajectory, not by KF count.
        // This remains meaningful when Keyframe spacing changes.
        const std::size_t low_kf =
            std::min(first_anchor_kf, second_anchor_kf);
        const std::size_t high_kf =
            std::max(first_anchor_kf, second_anchor_kf);

        double trajectory_arc_gap_m = 0.0;
        const Keyframe *previous =
            find_retrieval_keyframe(low_kf);

        if (previous == nullptr ||
            !previous->T_WL.matrix().allFinite())
        {
            return result;
        }

        for (std::size_t keyframe_id = low_kf + 1;
             keyframe_id <= high_kf;
             ++keyframe_id)
        {
            const Keyframe *member =
                find_retrieval_keyframe(keyframe_id);

            if (member == nullptr ||
                !member->T_WL.matrix().allFinite())
            {
                return result;
            }

            const double step_m =
                (member->T_WL.translation() -
                 previous->T_WL.translation())
                    .norm();

            if (!std::isfinite(step_m))
            {
                return result;
            }

            trajectory_arc_gap_m += step_m;
            previous = member;
        }

        result.trajectory_arc_gap_m =
            trajectory_arc_gap_m;
        result.valid =
            std::isfinite(result.trajectory_arc_gap_m);

        // Build the same FORWARD 3m historical windows used by reverse segment
        // verification.  Shared member KFs are a direct indicator that the
        // two representatives refer to the same physical trajectory segment.
        HierarchicalRetrievalWindow first_window;
        HierarchicalRetrievalWindow second_window;

        result.first_window_valid =
            build_retrieval_window(
                first_anchor_kf,
                +1,
                first_window);
        result.second_window_valid =
            build_retrieval_window(
                second_anchor_kf,
                +1,
                second_window);

        if (result.first_window_valid)
        {
            result.first_window_keyframes =
                first_window.keyframe_ids.size();
        }

        if (result.second_window_valid)
        {
            result.second_window_keyframes =
                second_window.keyframe_ids.size();
        }

        if (result.first_window_valid &&
            result.second_window_valid)
        {
            std::unordered_set<std::size_t> first_members;
            first_members.reserve(first_window.keyframe_ids.size());

            for (const std::size_t member_id :
                 first_window.keyframe_ids)
            {
                first_members.insert(member_id);
            }

            for (const std::size_t member_id :
                 second_window.keyframe_ids)
            {
                if (first_members.find(member_id) !=
                    first_members.end())
                {
                    ++result.shared_keyframes;
                }
            }

            const std::size_t min_window_size =
                std::min(
                    first_window.keyframe_ids.size(),
                    second_window.keyframe_ids.size());

            if (min_window_size > 0)
            {
                result.shared_over_min_ratio =
                    static_cast<double>(result.shared_keyframes) /
                    static_cast<double>(min_window_size);
            }
        }

        // Primary physical criterion: the two anchors lie within one 3m
        // historical trajectory segment.  The shared-member fallback handles
        // the discrete endpoint case where both windows demonstrably overlap
        // even if the summed anchor-to-anchor arc is just beyond 3m.
        const bool arc_continuous =
            result.valid &&
            result.trajectory_arc_gap_m <=
                kHierarchicalRetrievalArcLengthM;

        const bool member_overlap_continuous =
            result.first_window_valid &&
            result.second_window_valid &&
            result.shared_keyframes >= 2;

        result.continuous =
            arc_continuous ||
            member_overlap_continuous;

        return result;
    };

    // ------------------------------------------------------------------------
    // V17.1 LOCAL_STRONG classification.
    //
    // The critical distinction is between:
    //
    //   A) a physically nearby, already graph-compatible revisit, and
    //   B) a large-drift / symmetric false match that merely repeats with
    //      similar local geometry.
    //
    // Only A may use repeated matches to the SAME historical KF as temporal
    // confirmation.  This classification does NOT bypass ICP, graph,
    // correction-consistency, or PGO rollback gates.
    // ------------------------------------------------------------------------
    const LoopVerifierConfig &active_loop_verifier_config =
        loop_verifier_.GetConfig();

    const bool trusted_reverse_temporal_candidate =
        best_candidate_used_reverse_sequence_target &&
        reverse_sequence_confirmed &&
        best_candidate != nullptr &&
        (best_candidate->candidate_id >=
                 reverse_sequence_historical_kf
             ? best_candidate->candidate_id -
                   reverse_sequence_historical_kf
             : reverse_sequence_historical_kf -
                   best_candidate->candidate_id) <=
            kReverseSegmentSlidingRadius &&
        std::isfinite(best_verification.overlap_ratio) &&
        std::isfinite(best_verification.rmse) &&
        std::isfinite(best_verification.correction_translation) &&
        std::isfinite(best_verification.correction_rotation_deg) &&
        std::isfinite(graph_correction_translation) &&
        std::isfinite(graph_correction_rotation) &&
        best_verification.overlap_ratio >=
            active_loop_verifier_config
                .trusted_reverse_min_overlap_ratio &&
        best_verification.rmse <=
            active_loop_verifier_config
                .trusted_reverse_max_rmse &&
        best_verification.correction_translation <=
            active_loop_verifier_config
                .trusted_reverse_max_correction_translation &&
        best_verification.correction_rotation_deg <=
            active_loop_verifier_config
                .trusted_reverse_max_correction_rotation_deg &&
        graph_correction_translation <=
            online_loop_first_local_max_graph_translation_ &&
        graph_correction_rotation <=
            online_loop_first_local_max_graph_rotation_deg_;

    const bool trusted_reverse_sequence_geometry =
        !has_last_online_loop_edge_ &&
        trusted_reverse_temporal_candidate;

    const bool normal_first_loop_local_strong_candidate =
        !has_last_online_loop_edge_ &&
        std::isfinite(best_verification.overlap_ratio) &&
        std::isfinite(best_verification.rmse) &&
        std::isfinite(best_verification.correction_translation) &&
        std::isfinite(best_verification.correction_rotation_deg) &&
        std::isfinite(graph_correction_translation) &&
        std::isfinite(graph_correction_rotation) &&
        best_verification.overlap_ratio >=
            online_loop_first_local_min_overlap_ &&
        best_verification.rmse <=
            online_loop_first_local_max_rmse_ &&
        best_verification.correction_translation <=
            online_loop_first_edge_max_icp_translation_ &&
        best_verification.correction_rotation_deg <=
            online_loop_first_edge_max_icp_rotation_deg_ &&
        graph_correction_translation <=
            online_loop_first_local_max_graph_translation_ &&
        graph_correction_rotation <=
            online_loop_first_local_max_graph_rotation_deg_;

    const bool first_loop_local_strong_candidate =
        normal_first_loop_local_strong_candidate ||
        trusted_reverse_sequence_geometry;

    if (trusted_reverse_sequence_geometry)
    {
        std::cout
            << "Loop Reverse Sequence Geometry V2.1"
            << " | current_kf=" << current_keyframe.id
            << " | historical_kf=" << best_candidate->candidate_id
            << " | overlap=" << best_verification.overlap_ratio
            << " | rmse=" << best_verification.rmse << " m"
            << " | icp_dt="
            << best_verification.correction_translation << " m"
            << " | icp_dR="
            << best_verification.correction_rotation_deg << " deg"
            << " | graph_dt=" << graph_correction_translation << " m"
            << " | graph_dR=" << graph_correction_rotation << " deg"
            << " | action=TRUSTED_REVERSE_MULTI_FRAME_SUPPORT"
            << std::endl;
    }

    // ------------------------------------------------------------------------
    // V7: Historical Keyframe progression consistency.
    //
    // The important behavior change compared with V6 is:
    //
    //   A single bad candidate MUST NOT destroy an already plausible loop
    //   track.
    //
    // Example from the V6 log:
    //
    //   KF518 -> Hist1       good, support 1/3
    //   KF519 -> Hist20      outlier
    //   KF520 -> Hist4       good again
    //
    // V6 reset the track on KF519.  V7 rejects KF519 as an outlier and keeps
    // the Hist1 track alive.  Because current_gap=2 is still allowed, KF520
    // can continue the same sequence.
    // ------------------------------------------------------------------------
    const bool had_track =
        online_loop_track_.valid;

    const std::size_t loop_trace_support_before =
        had_track
            ? online_loop_track_.support
            : 0;

    const int loop_trace_direction_before =
        had_track
            ? online_loop_track_.historical_direction
            : 0;

    bool track_stale = false;
    bool extends_track = false;

    std::size_t current_gap = 0;
    std::size_t current_submap_gap = 0;
    std::size_t historical_gap = 0;
    std::size_t historical_submap_gap = 0;
    std::size_t allowed_historical_progression = 0;

    std::int64_t historical_delta = 0;

    bool current_gap_ok = true;
    bool current_submap_gap_ok = true;
    bool historical_gap_ok = true;
    bool historical_submap_gap_ok = true;
    bool historical_progression_step_ok = true;
    bool historical_progression_direction_ok = true;
    bool correction_ok = true;

    double track_translation_error =
        std::numeric_limits<double>::infinity();
    double track_rotation_error =
        std::numeric_limits<double>::infinity();

    if (had_track)
    {
        current_gap =
            current_keyframe.id >
                    online_loop_track_.last_current_keyframe_id
                ? current_keyframe.id -
                      online_loop_track_.last_current_keyframe_id
                : online_loop_track_.last_current_keyframe_id -
                      current_keyframe.id;

        // If the last accepted temporal observation is already too old,
        // that track is considered stale.  V15 gives a support==1 tentative
        // track a slightly longer grace window so a short Scan Context dropout
        // (for example KF146 -> KF150) does not erase a plausible revisit.
        // Once support>=2, retain the stricter active-track gap.
        const std::size_t temporal_current_gap_limit =
            online_loop_track_.support >= 2
                ? online_loop_max_current_keyframe_gap_
                : online_loop_tentative_max_current_keyframe_gap_;

        track_stale =
            current_gap > temporal_current_gap_limit;

        if (!track_stale)
        {
            current_submap_gap =
                current_submap_id >
                        online_loop_track_.last_current_submap_id
                    ? current_submap_id -
                          online_loop_track_.last_current_submap_id
                    : online_loop_track_.last_current_submap_id -
                          current_submap_id;

            historical_delta =
                static_cast<std::int64_t>(best_candidate->candidate_id) -
                static_cast<std::int64_t>(
                    online_loop_track_.last_historical_keyframe_id);

            historical_gap =
                historical_delta >= 0
                    ? static_cast<std::size_t>(historical_delta)
                    : static_cast<std::size_t>(-historical_delta);

            historical_submap_gap =
                best_historical_submap->id >
                        online_loop_track_.last_historical_submap_id
                    ? best_historical_submap->id -
                          online_loop_track_.last_historical_submap_id
                    : online_loop_track_.last_historical_submap_id -
                          best_historical_submap->id;

            // If one current KF was missed, allow proportionally more
            // historical progression.  For example:
            //
            //   current gap = 1 -> max historical step = 6
            //   current gap = 2 -> max historical step = 12
            //
            // The old broad <=15 gate is still applied below as a second cap.
            const std::size_t progression_scale =
                std::max<std::size_t>(1, current_gap);

            allowed_historical_progression =
                online_loop_max_historical_progression_step_ *
                progression_scale;

            current_gap_ok =
                current_gap <= temporal_current_gap_limit;

            current_submap_gap_ok =
                current_submap_gap <=
                online_loop_max_current_submap_gap_;

            historical_gap_ok =
                historical_gap <=
                online_loop_max_historical_keyframe_gap_;

            historical_submap_gap_ok =
                historical_submap_gap <=
                online_loop_max_historical_submap_gap_;

            historical_progression_step_ok =
                historical_gap <= allowed_historical_progression;

            // Direction rule:
            //
            //   direction = +1 : historical ids should mainly increase
            //   direction = -1 : historical ids should mainly decrease
            //   direction =  0 : not enough evidence yet, accept either sign
            //
            // A one-keyframe opposite jitter is tolerated.
            const std::int64_t backtrack_tolerance =
                static_cast<std::int64_t>(
                    online_loop_historical_backtrack_tolerance_);

            if (online_loop_track_.historical_direction > 0)
            {
                historical_progression_direction_ok =
                    historical_delta >= -backtrack_tolerance;
            }
            else if (online_loop_track_.historical_direction < 0)
            {
                historical_progression_direction_ok =
                    historical_delta <= backtrack_tolerance;
            }

            // V19 origin-invariant local consistency:
            //
            // Previous implementation directly compared translations of two
            // left-multiplicative WORLD corrections.  A few degrees of yaw
            // difference can create several metres of apparent translation
            // error when the trajectory is far from the world origin.
            //
            // Instead use the previous loop correction to PREDICT where this
            // current KF should be, then compare that predicted current pose
            // against this candidate's measured loop pose:
            //
            //   T_WL_pred = C_prev * T_WL_frontend(current)
            //   E_local   = T_WL_pred^-1 * T_WL_loop(measured)
            //
            // E_local is a local pose disagreement at the current KF and is
            // insensitive to the arbitrary world origin.
            const Eigen::Isometry3d T_W_L_track_prediction =
                online_loop_track_.T_loop_correction *
                current_keyframe.T_WL;

            const Eigen::Isometry3d track_local_error =
                T_W_L_track_prediction.inverse() *
                T_W_L_loop;

            track_translation_error =
                track_local_error.translation().norm();

            track_rotation_error =
                RelativeRotationDeg(
                    Eigen::Isometry3d::Identity(),
                    track_local_error);

            const double effective_track_rotation_error_limit_deg =
                trusted_reverse_temporal_candidate
                    ? std::max(
                          online_loop_track_rotation_error_deg_,
                          kTrustedReverseTemporalRotationErrorDeg)
                    : online_loop_track_rotation_error_deg_;

            correction_ok =
                std::isfinite(track_translation_error) &&
                std::isfinite(track_rotation_error) &&
                track_translation_error <=
                    online_loop_track_translation_error_ &&
                track_rotation_error <=
                    effective_track_rotation_error_limit_deg;

            extends_track =
                current_gap_ok &&
                current_submap_gap_ok &&
                historical_gap_ok &&
                historical_submap_gap_ok &&
                historical_progression_step_ok &&
                historical_progression_direction_ok &&
                correction_ok;

            if (!extends_track)
            {
                // IMPORTANT:
                // Do NOT reset online_loop_track_ here.
                // One isolated SC/ICP outlier should be ignored, not allowed
                // to erase the good evidence accumulated by previous KFs.
                std::cout
                    << "Keyframe loop temporal candidate rejected"
                    << " | current_kf=" << current_keyframe.id
                    << " | current_submap=" << current_submap_id
                    << " | historical_kf="
                    << best_candidate->candidate_id
                    << " | historical_submap="
                    << best_historical_submap->id
                    << " | preserved_support="
                    << online_loop_track_.support
                    << "/" << online_loop_min_support_
                    << " | last_current_kf="
                    << online_loop_track_.last_current_keyframe_id
                    << " | last_historical_kf="
                    << online_loop_track_.last_historical_keyframe_id
                    << " | current_gap=" << current_gap
                    << " | historical_delta=" << historical_delta
                    << " | historical_gap=" << historical_gap
                    << " | allowed_progression="
                    << allowed_historical_progression
                    << " | direction="
                    << online_loop_track_.historical_direction
                    << " | step_ok="
                    << (historical_progression_step_ok ? "true" : "false")
                    << " | direction_ok="
                    << (historical_progression_direction_ok ? "true" : "false")
                    << " | correction_ok="
                    << (correction_ok ? "true" : "false")
                    << " | track_translation_error="
                    << track_translation_error << " m"
                    << " | track_rotation_error="
                    << track_rotation_error << " deg"
                    << " | track_rotation_limit="
                    << effective_track_rotation_error_limit_deg
                    << " deg"
                    << " | trusted_reverse_temporal="
                    << (trusted_reverse_temporal_candidate
                            ? "true"
                            : "false")
                    << " | action=PRESERVE_TRACK"
                    << std::endl;

                std::cout
                    << "FR_LOOP_TRACE"
                    << " | stage=TRACK"
                    << " | current_kf=" << current_keyframe.id
                    << " | historical_kf=" << best_candidate->candidate_id
                    << " | decision=REJECT_PRESERVE"
                    << " | support=" << online_loop_track_.support
                    << " | current_gap=" << current_gap
                    << " | historical_delta=" << historical_delta
                    << " | historical_gap=" << historical_gap
                    << " | direction=" << online_loop_track_.historical_direction
                    << " | current_gap_ok=" << (current_gap_ok ? "true" : "false")
                    << " | current_submap_gap_ok=" << (current_submap_gap_ok ? "true" : "false")
                    << " | historical_gap_ok=" << (historical_gap_ok ? "true" : "false")
                    << " | historical_submap_gap_ok=" << (historical_submap_gap_ok ? "true" : "false")
                    << " | step_ok=" << (historical_progression_step_ok ? "true" : "false")
                    << " | direction_ok=" << (historical_progression_direction_ok ? "true" : "false")
                    << " | correction_ok=" << (correction_ok ? "true" : "false")
                    << " | track_dt=" << track_translation_error << " m"
                    << " | track_dR=" << track_rotation_error << " deg"
                    << " | track_dR_limit="
                    << effective_track_rotation_error_limit_deg
                    << " deg"
                    << " | trusted_reverse_temporal="
                    << (trusted_reverse_temporal_candidate
                            ? "true"
                            : "false")
                    << std::endl;

                return;
            }
        }
    }

    // ========================================================================
    // V22.8 SEGMENT_CONTINUATION revalidation fallback.
    //
    // At a >=3m checkpoint, V22.7 required an independent retrieval hit on the
    // exact frame.  That was too brittle: a perfectly continuous segment track
    // could disappear for several KFs despite good geometry and progression.
    //
    // Here we permit a non-independent refresh ONLY after the normal temporal
    // checks above have already proved that the observation extends the live
    // track.  We additionally demand physical progression of the representative
    // historical KF over the checkpoint interval.  A sticky anchor therefore
    // cannot refresh itself with historical_arc ~= 0.
    // ========================================================================
    if (track_segment_continuation_check_required)
    {
        bool historical_arc_valid = true;
        continuation_historical_arc_m = 0.0;

        const std::size_t checkpoint_historical_kf =
            loop_track_segment_revalidation_v226
                .checkpoint_historical_keyframe_id;
        const std::size_t current_representative_kf =
            representative_historical_keyframe_id;

        continuation_representative_delta =
            static_cast<std::int64_t>(current_representative_kf) -
            static_cast<std::int64_t>(checkpoint_historical_kf);

        if (checkpoint_historical_kf != current_representative_kf)
        {
            const int step =
                current_representative_kf > checkpoint_historical_kf
                    ? +1
                    : -1;

            std::size_t previous_id = checkpoint_historical_kf;
            const Keyframe *previous_member =
                FindBackendKeyframeById(previous_id);

            if (previous_member == nullptr ||
                !previous_member->T_WL.matrix().allFinite())
            {
                historical_arc_valid = false;
            }
            else
            {
                while (previous_id != current_representative_kf)
                {
                    const std::int64_t next_signed =
                        static_cast<std::int64_t>(previous_id) + step;

                    if (next_signed < 0)
                    {
                        historical_arc_valid = false;
                        break;
                    }

                    const std::size_t next_id =
                        static_cast<std::size_t>(next_signed);
                    const Keyframe *next_member =
                        FindBackendKeyframeById(next_id);

                    if (next_member == nullptr ||
                        !next_member->T_WL.matrix().allFinite())
                    {
                        historical_arc_valid = false;
                        break;
                    }

                    const Eigen::Vector3d delta =
                        next_member->T_WL.translation() -
                        previous_member->T_WL.translation();

                    if (!delta.allFinite())
                    {
                        historical_arc_valid = false;
                        break;
                    }

                    continuation_historical_arc_m += delta.norm();
                    previous_id = next_id;
                    previous_member = next_member;
                }
            }
        }

        if (historical_arc_valid &&
            std::isfinite(track_segment_revalidation_arc_m) &&
            track_segment_revalidation_arc_m > 1e-6)
        {
            continuation_historical_arc_ratio =
                continuation_historical_arc_m /
                track_segment_revalidation_arc_m;
        }

        bool representative_direction_ok = true;
        const std::int64_t backtrack_tolerance =
            static_cast<std::int64_t>(
                online_loop_historical_backtrack_tolerance_);

        if (online_loop_track_.historical_direction > 0)
        {
            representative_direction_ok =
                continuation_representative_delta >=
                -backtrack_tolerance;
        }
        else if (online_loop_track_.historical_direction < 0)
        {
            representative_direction_ok =
                continuation_representative_delta <=
                backtrack_tolerance;
        }

        const bool representative_distance_ok =
            std::isfinite(representative_distance_m) &&
            representative_distance_m <=
                kLoopTrackContinuationMaxRepresentativeDistanceM;

        const bool historical_arc_ratio_ok =
            historical_arc_valid &&
            std::isfinite(continuation_historical_arc_ratio) &&
            continuation_historical_arc_ratio >=
                kLoopTrackContinuationMinHistoricalArcRatio &&
            continuation_historical_arc_ratio <=
                kLoopTrackContinuationMaxHistoricalArcRatio;

        best_candidate_segment_continuation_evidence =
            had_track &&
            !track_stale &&
            extends_track &&
            best_candidate_used_segment_geometry &&
            representative_direction_ok &&
            representative_distance_ok &&
            historical_arc_ratio_ok;

        std::cout
            << "LOOP_TRACK_3M_REVALIDATION_V22_8"
            << " | current_kf=" << current_keyframe.id
            << " | checkpoint_current_kf="
            << loop_track_segment_revalidation_v226
                   .checkpoint_current_keyframe_id
            << " | checkpoint_historical_kf="
            << checkpoint_historical_kf
            << " | representative_kf="
            << current_representative_kf
            << " | representative_delta="
            << continuation_representative_delta
            << " | representative_distance="
            << representative_distance_m << " m"
            << " | current_arc="
            << track_segment_revalidation_arc_m << " m"
            << " | historical_arc="
            << continuation_historical_arc_m << " m"
            << " | historical_arc_ratio="
            << continuation_historical_arc_ratio
            << " | ratio_range=["
            << kLoopTrackContinuationMinHistoricalArcRatio
            << ","
            << kLoopTrackContinuationMaxHistoricalArcRatio
            << "]"
            << " | temporal_extends="
            << (extends_track ? "true" : "false")
            << " | representative_direction_ok="
            << (representative_direction_ok ? "true" : "false")
            << " | representative_distance_ok="
            << (representative_distance_ok ? "true" : "false")
            << " | decision="
            << (best_candidate_segment_continuation_evidence
                    ? "PASS"
                    : "REJECT")
            << std::endl;

        if (!best_candidate_segment_continuation_evidence)
        {
            std::cout
                << "Keyframe loop decision"
                << " | current_kf=" << current_keyframe.id
                << " | historical_kf="
                << representative_historical_keyframe_id
                << " | segment_anchor_kf="
                << best_candidate->candidate_id
                << " | decision=PENDING"
                << " | reason=TRACK_3M_REVALIDATION_REQUIRED"
                << " | geometry_accepted=true"
                << " | loop_edge_accepted=false"
                << std::endl;

            return;
        }
    }

    if (!had_track || track_stale)
    {
        // No usable previous track: start from the current verified loop.
        online_loop_track_ = OnlineLoopTrack();
        online_loop_track_.valid = true;
        online_loop_track_.support = 1;
        online_loop_track_.local_strong_support =
            first_loop_local_strong_candidate
                ? 1
                : 0;
        extends_track = false;

        // If no loop factor has been committed yet, a stale/restarted
        // temporal track also invalidates the delayed first-loop batch.
        // Hypotheses from two unrelated revisit sequences must never be mixed.
        if (!has_last_online_loop_edge_)
        {
            pending_first_loop_batch_.clear();
        }
    }
    else
    {
        // V16: before the FIRST real loop edge is committed, repeatedly
        // matching the exact same historical Keyframe is NOT independent
        // temporal evidence.  This prevents corridor/row symmetry from turning
        //
        //     149->96, 150->96, 151->96
        //
        // into an artificial support=3/3 sequence.
        const bool repeated_first_loop_historical_anchor =
            !has_last_online_loop_edge_ &&
            historical_delta == 0;

        HistoricalSegmentContinuityV225
            temporal_segment_continuity_v225;

        if (!has_last_online_loop_edge_ &&
            first_loop_local_strong_candidate)
        {
            temporal_segment_continuity_v225 =
                evaluate_historical_segment_continuity_v225(
                    online_loop_track_.last_historical_keyframe_id,
                    best_candidate->candidate_id);
        }

        const bool local_strong_cluster_continuation =
            !has_last_online_loop_edge_ &&
            first_loop_local_strong_candidate &&
            temporal_segment_continuity_v225.continuous;

        if (!has_last_online_loop_edge_ &&
            first_loop_local_strong_candidate)
        {
            std::cout
                << "FIRST_LOOP_3M_SEGMENT_CONTINUITY_V22_5"
                << " | stage=TEMPORAL"
                << " | current_kf=" << current_keyframe.id
                << " | previous_historical_kf="
                << online_loop_track_.last_historical_keyframe_id
                << " | current_historical_kf="
                << best_candidate->candidate_id
                << " | historical_kf_gap="
                << std::llabs(historical_delta)
                << " | trajectory_arc_gap="
                << temporal_segment_continuity_v225.trajectory_arc_gap_m
                << " m"
                << " | anchor_euclidean_gap="
                << temporal_segment_continuity_v225.anchor_euclidean_gap_m
                << " m"
                << " | shared_kfs="
                << temporal_segment_continuity_v225.shared_keyframes
                << " | shared_over_min="
                << temporal_segment_continuity_v225.shared_over_min_ratio
                << " | continuous="
                << (temporal_segment_continuity_v225.continuous
                        ? "true"
                        : "false")
                << std::endl;
        }

        if (repeated_first_loop_historical_anchor)
        {
            if (first_loop_local_strong_candidate)
            {
                // Same historical anchor may confirm a REAL local closure,
                // but it remains temporal evidence only: the unique-edge batch
                // below will not stage another factor for the same old KF.
                ++online_loop_track_.support;
                ++online_loop_track_.local_strong_support;

                std::cout
                    << "First loop temporal duplicate historical anchor"
                    << " | current_kf=" << current_keyframe.id
                    << " | historical_kf="
                    << best_candidate->candidate_id
                    << " | support="
                    << online_loop_track_.support
                    << "/" << online_loop_first_edge_min_support_
                    << " | local_strong_support="
                    << online_loop_track_.local_strong_support
                    << "/" << online_loop_first_edge_min_support_
                    << " | graph_dt="
                    << graph_correction_translation << " m"
                    << " | graph_dR="
                    << graph_correction_rotation << " deg"
                    << " | action=COUNT_LOCAL_STRONG_TEMPORAL_ONLY"
                    << std::endl;
            }
            else
            {
                // Large-drift repeated-anchor matches are exactly the failure
                // mode seen at KF149/150/151 -> KF96.  They must not mature a
                // first-loop transaction merely because the same wrong basin
                // is repeatable.
                online_loop_track_.local_strong_support = 0;

                std::cout
                    << "First loop temporal duplicate historical anchor"
                    << " | current_kf=" << current_keyframe.id
                    << " | historical_kf="
                    << best_candidate->candidate_id
                    << " | support_preserved="
                    << online_loop_track_.support
                    << "/" << online_loop_first_edge_min_support_
                    << " | local_strong_support=0/"
                    << online_loop_first_edge_min_support_
                    << " | graph_dt="
                    << graph_correction_translation << " m"
                    << " | graph_dR="
                    << graph_correction_rotation << " deg"
                    << " | action=DO_NOT_COUNT_NONLOCAL_DUPLICATE"
                    << std::endl;
            }
        }
        else
        {
            ++online_loop_track_.support;

            // V19.3:
            // A +/-1 or +/-2 KF change inside a LOCAL_STRONG historical
            // cluster is not a new physical place.  Keep accumulating local
            // strong temporal confidence instead of resetting it to 1.
            if (local_strong_cluster_continuation)
            {
                ++online_loop_track_.local_strong_support;

                std::cout
                    << "First loop local-strong cluster continuation V20"
                    << " | current_kf=" << current_keyframe.id
                    << " | historical_kf=" << best_candidate->candidate_id
                    << " | historical_delta=" << historical_delta
                    << " | local_strong_support="
                    << online_loop_track_.local_strong_support
                    << "/" << online_loop_first_edge_min_support_
                    << " | action=COUNT_CLUSTER_TEMPORAL_ONLY"
                    << std::endl;
            }
            else
            {
                online_loop_track_.local_strong_support =
                    first_loop_local_strong_candidate
                        ? 1
                        : 0;
            }

            // Lock historical traversal direction only after real progress.
            if (online_loop_track_.historical_direction == 0)
            {
                const std::int64_t lock_threshold =
                    static_cast<std::int64_t>(
                        online_loop_historical_backtrack_tolerance_);

                if (historical_delta > lock_threshold)
                {
                    online_loop_track_.historical_direction = +1;
                }
                else if (historical_delta < -lock_threshold)
                {
                    online_loop_track_.historical_direction = -1;
                }
            }
        }
    }

    online_loop_track_.last_current_submap_id =
        current_submap_id;
    online_loop_track_.last_current_keyframe_id =
        current_keyframe.id;
    online_loop_track_.last_historical_keyframe_id =
        best_candidate->candidate_id;
    online_loop_track_.last_historical_submap_id =
        best_historical_submap->id;
    online_loop_track_.T_loop_correction =
        T_loop_correction;

    // V22.8: refresh the physical checkpoint only AFTER this observation has
    // actually passed temporal track consistency and updated online_loop_track_.
    // A checkpoint can be renewed by either:
    //   A) independent segment retrieval, or
    //   B) strict segment-continuation revalidation at the >=3m deadline.
    // In both cases the stored historical checkpoint is the representative KF,
    // not the segment anchor, so physical progression is measured correctly.
    const bool best_candidate_checkpoint_revalidation_evidence =
        best_candidate_independent_segment_evidence ||
        best_candidate_segment_continuation_evidence;

    if (best_candidate_checkpoint_revalidation_evidence &&
        (!loop_track_segment_revalidation_v226.valid ||
         track_segment_revalidation_required))
    {
        loop_track_segment_revalidation_v226.valid = true;
        loop_track_segment_revalidation_v226
            .checkpoint_current_keyframe_id =
            current_keyframe.id;
        loop_track_segment_revalidation_v226
            .checkpoint_historical_keyframe_id =
            representative_historical_keyframe_id;
        loop_track_segment_revalidation_v226
            .checkpoint_was_segment_verified = true;

        std::cout
            << "LOOP_TRACK_3M_REVALIDATION_V22_8"
            << " | current_kf=" << current_keyframe.id
            << " | historical_kf="
            << representative_historical_keyframe_id
            << " | segment_anchor_kf="
            << best_candidate->candidate_id
            << " | geometry_target="
            << (best_candidate_used_reverse_sequence_target
                    ? "REVERSE_3M_X_3M"
                    : (best_candidate_used_hierarchical_segment_target
                           ? "HIERARCHICAL_3M_X_3M"
                           : "KEYFRAME_CENTERED"))
            << " | independent_retrieval="
            << (best_candidate_independent_segment_evidence
                    ? "true"
                    : "false")
            << " | revalidation_source="
            << (best_candidate_independent_segment_evidence
                    ? "INDEPENDENT_SEGMENT"
                    : "SEGMENT_CONTINUATION")
            << " | current_arc="
            << track_segment_revalidation_arc_m << " m"
            << " | historical_arc="
            << continuation_historical_arc_m << " m"
            << " | action=REFRESH_SEGMENT_CHECKPOINT"
            << std::endl;
    }
    else if (best_candidate_independent_segment_evidence &&
             loop_track_segment_revalidation_v226.valid)
    {
        // Overlapping 3m windows on adjacent KFs are NOT new checkpoints.  Keep
        // measuring from the old checkpoint until the physical 3m deadline is
        // actually reached.
        std::cout
            << "LOOP_TRACK_3M_REVALIDATION_V22_8"
            << " | current_kf=" << current_keyframe.id
            << " | historical_kf="
            << representative_historical_keyframe_id
            << " | segment_anchor_kf="
            << best_candidate->candidate_id
            << " | checkpoint_current_kf="
            << loop_track_segment_revalidation_v226
                   .checkpoint_current_keyframe_id
            << " | current_arc="
            << track_segment_revalidation_arc_m << " m"
            << " | action=KEEP_EXISTING_CHECKPOINT"
            << std::endl;
    }
    else if (!loop_track_segment_revalidation_v226.valid)
    {
        // A legacy KF-centered track may start, but it gets only one 3m grace
        // interval before segment revalidation becomes mandatory.
        loop_track_segment_revalidation_v226.valid = true;
        loop_track_segment_revalidation_v226
            .checkpoint_current_keyframe_id =
            current_keyframe.id;
        loop_track_segment_revalidation_v226
            .checkpoint_historical_keyframe_id =
            representative_historical_keyframe_id;
        loop_track_segment_revalidation_v226
            .checkpoint_was_segment_verified = false;

        std::cout
            << "LOOP_TRACK_3M_REVALIDATION_V22_8"
            << " | current_kf=" << current_keyframe.id
            << " | historical_kf="
            << representative_historical_keyframe_id
            << " | segment_anchor_kf="
            << best_candidate->candidate_id
            << " | segment_verified=false"
            << " | action=START_TRACK_CHECKPOINT"
            << std::endl;
    }

    std::cout
        << "Keyframe loop temporal"
        << " | current_kf=" << current_keyframe.id
        << " | current_submap=" << current_submap_id
        << " | historical_kf=" << best_candidate->candidate_id
        << " | historical_submap=" << best_historical_submap->id
        << " | support=" << online_loop_track_.support
        << "/" << online_loop_min_support_
        << " | local_strong_support="
        << online_loop_track_.local_strong_support
        << "/" << online_loop_first_edge_min_support_
        << " | local_strong_current="
        << (first_loop_local_strong_candidate ? "true" : "false")
        << " | extended=" << (extends_track ? "true" : "false")
        << " | track_stale=" << (track_stale ? "true" : "false")
        << " | historical_delta=" << historical_delta
        << " | historical_direction="
        << online_loop_track_.historical_direction
        << " | allowed_progression="
        << allowed_historical_progression
        << " | track_translation_error="
        << track_translation_error << " m"
        << " | track_rotation_error="
        << track_rotation_error << " deg"
        << " | trusted_reverse_temporal="
        << (trusted_reverse_temporal_candidate
                ? "true"
                : "false")
        << std::endl;

    std::cout
        << "FR_LOOP_TRACE"
        << " | stage=TRACK"
        << " | current_kf=" << current_keyframe.id
        << " | historical_kf=" << best_candidate->candidate_id
        << " | action="
        << (!had_track
                ? "START"
                : (track_stale
                       ? "RESTART_STALE"
                       : (extends_track ? "EXTEND" : "UPDATE")))
        << " | support_before=" << loop_trace_support_before
        << " | support_after=" << online_loop_track_.support
        << " | direction_before=" << loop_trace_direction_before
        << " | direction_after=" << online_loop_track_.historical_direction
        << " | current_gap=" << current_gap
        << " | historical_delta=" << historical_delta
        << " | allowed_progression=" << allowed_historical_progression
        << " | track_dt=" << track_translation_error << " m"
        << " | track_dR=" << track_rotation_error << " deg"
        << " | local_strong="
        << (first_loop_local_strong_candidate ? "true" : "false")
        << std::endl;

    // ------------------------------------------------------------------------
    // V11: first-loop BATCH confirmation.
    //
    // The first backend correction is no longer allowed to come from one
    // endpoint constraint.  We collect several independent strong loop
    // constraints and require them to support the same left-multiplicative
    // world correction before staging them together in PoseGraph.
    // ------------------------------------------------------------------------
    const bool first_online_loop_edge =
        !has_last_online_loop_edge_;

    if (first_online_loop_edge)
    {
        const bool geometry_finite =
            std::isfinite(best_verification.overlap_ratio) &&
            std::isfinite(best_verification.rmse) &&
            std::isfinite(best_verification.correction_translation) &&
            std::isfinite(best_verification.correction_rotation_deg);

        const bool normal_common_geometry_ok =
            geometry_finite &&
            best_verification.overlap_ratio >=
                online_loop_first_edge_min_overlap_ &&
            best_verification.rmse <=
                online_loop_first_edge_max_rmse_;

        const bool reverse_sequence_common_geometry_ok =
            geometry_finite &&
            trusted_reverse_sequence_geometry;

        const bool common_geometry_ok =
            normal_common_geometry_ok ||
            reverse_sequence_common_geometry_ok;

        // The first member is the anchor and remains deliberately strict.
        const bool first_anchor_geometry_ok =
            common_geometry_ok &&
            best_verification.correction_translation <=
                online_loop_first_edge_max_icp_translation_ &&
            best_verification.correction_rotation_deg <=
                online_loop_first_edge_max_icp_rotation_deg_;

        // Once one strong anchor exists, later members mainly need to support
        // the same T_loop_correction.  Their local ICP gate is intentionally
        // wider so a valid sequence is not lost merely because one frame needs
        // a larger local rotational correction.
        const bool followup_geometry_ok =
            common_geometry_ok &&
            best_verification.correction_translation <=
                online_loop_first_batch_followup_max_icp_translation_ &&
            best_verification.correction_rotation_deg <=
                online_loop_first_batch_followup_max_icp_rotation_deg_;

        const bool strong_first_geometry =
            pending_first_loop_batch_.empty()
                ? first_anchor_geometry_ok
                : followup_geometry_ok;

        bool spacing_ok = true;
        bool correction_consistent = true;
        bool historical_anchor_unique = true;
        bool historical_sequence_monotonic = true;
        bool historical_segment_continuous = true;
        bool temporal_track_continuous = true;
        HistoricalSegmentContinuityV225
            batch_segment_continuity_v225;
        int batch_historical_direction = 0;
        std::int64_t batch_historical_delta = 0;
        double batch_translation_error = 0.0;
        double batch_rotation_error_deg = 0.0;
        std::size_t batch_current_gap = 0;
        std::size_t batch_historical_gap = 0;

        if (!pending_first_loop_batch_.empty())
        {
            const PendingLoopConstraint &last_constraint =
                pending_first_loop_batch_.back();

            if (current_keyframe.id >=
                last_constraint.current_keyframe_id)
            {
                batch_current_gap =
                    current_keyframe.id -
                    last_constraint.current_keyframe_id;
            }

            batch_historical_gap =
                best_candidate->candidate_id >=
                        last_constraint.historical_keyframe_id
                    ? best_candidate->candidate_id -
                          last_constraint.historical_keyframe_id
                    : last_constraint.historical_keyframe_id -
                          best_candidate->candidate_id;

            // V16: repeated historical anchors are NOT independent loop
            // constraints.  Check the whole pending batch, not only the most
            // recent member.
            for (const PendingLoopConstraint &stored_constraint :
                 pending_first_loop_batch_)
            {
                if (stored_constraint.historical_keyframe_id ==
                    best_candidate->candidate_id)
                {
                    historical_anchor_unique = false;
                    break;
                }
            }

            batch_historical_delta =
                static_cast<std::int64_t>(
                    best_candidate->candidate_id) -
                static_cast<std::int64_t>(
                    last_constraint.historical_keyframe_id);

            // Every newly staged first-loop member must make real historical
            // progress.  A zero delta is a duplicate and cannot enter the batch.
            if (batch_historical_delta == 0)
            {
                historical_sequence_monotonic = false;
            }

            // V19.6 robust historical-direction logic.
            //
            // The historical geometry target is centered on one KF but uses a
            // +/-2 KF window.  Therefore +/-1 center-id motion is treated as
            // representative jitter rather than reliable direction evidence.
            const std::int64_t batch_direction_jitter =
                static_cast<std::int64_t>(
                    online_loop_first_batch_direction_jitter_tolerance_);

            bool stored_batch_direction_conflict = false;

            if (pending_first_loop_batch_.size() >= 2)
            {
                for (std::size_t batch_index = 1;
                     batch_index < pending_first_loop_batch_.size();
                     ++batch_index)
                {
                    const std::int64_t stored_delta =
                        static_cast<std::int64_t>(
                            pending_first_loop_batch_[batch_index]
                                .historical_keyframe_id) -
                        static_cast<std::int64_t>(
                            pending_first_loop_batch_[batch_index - 1]
                                .historical_keyframe_id);

                    if (std::llabs(stored_delta) <=
                        batch_direction_jitter)
                    {
                        continue;
                    }

                    const int stored_direction =
                        stored_delta > 0 ? +1 : -1;

                    if (batch_historical_direction == 0)
                    {
                        batch_historical_direction =
                            stored_direction;
                    }
                    else if (batch_historical_direction !=
                             stored_direction)
                    {
                        stored_batch_direction_conflict = true;
                        break;
                    }
                }
            }

            if (stored_batch_direction_conflict)
            {
                historical_sequence_monotonic = false;
            }
            else if (batch_historical_direction == 0)
            {
                if (batch_historical_delta >
                    batch_direction_jitter)
                {
                    batch_historical_direction = +1;
                }
                else if (batch_historical_delta <
                         -batch_direction_jitter)
                {
                    batch_historical_direction = -1;
                }
            }
            else if (batch_historical_direction > 0)
            {
                historical_sequence_monotonic =
                    historical_sequence_monotonic &&
                    batch_historical_delta >=
                        -batch_direction_jitter;
            }
            else
            {
                historical_sequence_monotonic =
                    historical_sequence_monotonic &&
                    batch_historical_delta <=
                        batch_direction_jitter;
            }

            // V22.5: first-loop spacing is now physical-segment based.
            // Keep the CURRENT-KF spacing so the follow-up is a genuinely new
            // observation in time, but replace the historical KF-ID min/max
            // gate with continuity of the two historical FORWARD-3m windows.
            batch_segment_continuity_v225 =
                evaluate_historical_segment_continuity_v225(
                    last_constraint.historical_keyframe_id,
                    best_candidate->candidate_id);

            historical_segment_continuous =
                batch_segment_continuity_v225.continuous;

            temporal_track_continuous =
                had_track &&
                !track_stale &&
                extends_track;

            spacing_ok =
                batch_current_gap >=
                    online_loop_first_batch_current_spacing_ &&
                historical_segment_continuous;

            // V22.5 keeps historical direction as a diagnostic only.  The
            // actual local-place continuity test is now 3m-segment continuity
            // plus the temporal-track and correction-consistency gates.
            historical_sequence_monotonic = true;

            std::cout
                << "FIRST_LOOP_3M_SEGMENT_CONTINUITY_V22_5"
                << " | stage=BATCH_FOLLOWUP"
                << " | current_kf=" << current_keyframe.id
                << " | anchor_historical_kf="
                << last_constraint.historical_keyframe_id
                << " | candidate_historical_kf="
                << best_candidate->candidate_id
                << " | historical_kf_gap="
                << batch_historical_gap
                << " | trajectory_arc_gap="
                << batch_segment_continuity_v225.trajectory_arc_gap_m
                << " m"
                << " | anchor_euclidean_gap="
                << batch_segment_continuity_v225.anchor_euclidean_gap_m
                << " m"
                << " | shared_kfs="
                << batch_segment_continuity_v225.shared_keyframes
                << " | shared_over_min="
                << batch_segment_continuity_v225.shared_over_min_ratio
                << " | historical_segment_continuous="
                << (historical_segment_continuous ? "true" : "false")
                << " | temporal_track_continuous="
                << (temporal_track_continuous ? "true" : "false")
                << std::endl;

            // V19.6 pairwise/local first-batch consistency.
            //
            // track_translation_error / track_rotation_error were computed
            // above BEFORE online_loop_track_.T_loop_correction was updated to
            // this current observation.  They therefore compare:
            //
            //   previous verified loop correction -> current frontend pose
            //                             versus
            //   current verified loop pose
            //
            // This is the desired local continuity test.  Do not compare every
            // later frame forever against the first anchor; accumulated
            // frontend drift is precisely what the loop is intended to repair.
            batch_translation_error =
                track_translation_error;

            batch_rotation_error_deg =
                track_rotation_error;

            correction_consistent =
                std::isfinite(batch_translation_error) &&
                std::isfinite(batch_rotation_error_deg) &&
                batch_translation_error <=
                    online_loop_first_batch_max_translation_error_ &&
                batch_rotation_error_deg <=
                    online_loop_first_batch_max_rotation_error_deg_;
        }

        // ----------------------------------------------------------------
        // V19.3 LARGE_DRIFT evidence fusion.
        //
        // Anchor:
        //   MUST come from independent global retrieval: raw single-KF SC or V21 hierarchical 3m Region-SC.
        //
        // Follow-up:
        //   may be independently retrieved OR may be injected, but an injected
        //   follow-up must have VERY strong geometry and tight correction
        //   agreement.  This avoids V19.1's "track proves itself" failure
        //   without V19.2's over-strict "every edge must be raw SC" deadlock.
        // ----------------------------------------------------------------
        const bool large_drift_anchor_evidence_ok =
            !current_first_loop_large_drift_candidate ||
            !pending_first_loop_batch_.empty() ||
            best_candidate_independent_retrieval_supported;

        const bool strict_injected_large_drift_followup =
            current_first_loop_large_drift_candidate &&
            !best_candidate_independent_retrieval_supported &&
            !pending_first_loop_batch_.empty() &&
            best_candidate_corridor_supported &&
            best_verification.overlap_ratio >=
                online_loop_large_drift_injected_min_overlap_ &&
            best_verification.rmse <=
                online_loop_large_drift_injected_max_rmse_ &&
            correction_consistent &&
            batch_translation_error <=
                online_loop_large_drift_injected_max_consistency_translation_ &&
            batch_rotation_error_deg <=
                online_loop_large_drift_injected_max_consistency_rotation_deg_;

        const bool large_drift_followup_evidence_ok =
            !current_first_loop_large_drift_candidate ||
            pending_first_loop_batch_.empty() ||
            best_candidate_independent_retrieval_supported ||
            strict_injected_large_drift_followup;

        const bool large_drift_evidence_ok =
            large_drift_anchor_evidence_ok &&
            large_drift_followup_evidence_ok;

        // ----------------------------------------------------------------
        // V22.5 LOCAL_STRONG_SEGMENT_CLUSTER.
        //
        // V22.4's +/-2 KF rule is removed.  A strong follow-up belongs to the
        // same local first-loop place when its historical FORWARD-3m segment
        // is physically continuous with the stored representative.  This
        // matches the segment-to-segment verifier and is invariant to uneven
        // Keyframe sampling density.
        // ----------------------------------------------------------------
        bool local_strong_cluster_member = false;
        std::size_t local_cluster_gap =
            std::numeric_limits<std::size_t>::max();
        HistoricalSegmentContinuityV225
            local_cluster_segment_continuity_v225;

        if (first_loop_local_strong_candidate &&
            !pending_first_loop_batch_.empty())
        {
            const std::size_t cluster_anchor =
                pending_first_loop_batch_.front()
                    .historical_keyframe_id;

            local_cluster_gap =
                best_candidate->candidate_id >= cluster_anchor
                    ? best_candidate->candidate_id - cluster_anchor
                    : cluster_anchor - best_candidate->candidate_id;

            local_cluster_segment_continuity_v225 =
                evaluate_historical_segment_continuity_v225(
                    cluster_anchor,
                    best_candidate->candidate_id);

            local_strong_cluster_member =
                local_cluster_segment_continuity_v225.continuous &&
                temporal_track_continuous;

            std::cout
                << "FIRST_LOOP_3M_SEGMENT_CONTINUITY_V22_5"
                << " | stage=LOCAL_CLUSTER"
                << " | current_kf=" << current_keyframe.id
                << " | cluster_anchor_kf=" << cluster_anchor
                << " | candidate_historical_kf="
                << best_candidate->candidate_id
                << " | historical_kf_gap=" << local_cluster_gap
                << " | trajectory_arc_gap="
                << local_cluster_segment_continuity_v225
                       .trajectory_arc_gap_m
                << " m"
                << " | anchor_euclidean_gap="
                << local_cluster_segment_continuity_v225
                       .anchor_euclidean_gap_m
                << " m"
                << " | shared_kfs="
                << local_cluster_segment_continuity_v225.shared_keyframes
                << " | shared_over_min="
                << local_cluster_segment_continuity_v225
                       .shared_over_min_ratio
                << " | temporal_track_continuous="
                << (temporal_track_continuous ? "true" : "false")
                << " | segment_cluster_member="
                << (local_strong_cluster_member ? "true" : "false")
                << std::endl;
        }

        bool local_cluster_representative_refreshed = false;

        if (local_strong_cluster_member &&
            correction_consistent)
        {
            PendingLoopConstraint &representative =
                pending_first_loop_batch_.front();

            const bool representative_not_local_strong =
                representative.overlap <
                    online_loop_first_local_min_overlap_ ||
                representative.rmse >
                    online_loop_first_local_max_rmse_;

            const bool current_geometry_better =
                representative_not_local_strong ||
                best_verification.overlap_ratio >
                    representative.overlap + 1e-6 ||
                (std::abs(
                     best_verification.overlap_ratio -
                     representative.overlap) <= 1e-6 &&
                 best_verification.rmse <
                     representative.rmse);

            if (current_geometry_better)
            {
                representative.historical_keyframe_id =
                    best_candidate->candidate_id;
                representative.current_keyframe_id =
                    current_keyframe.id;
                representative.historical_submap_id =
                    best_historical_submap->id;
                representative.current_submap_id =
                    current_submap_id;
                representative.T_historical_current =
                    T_K_L;
                representative.T_loop_correction =
                    T_loop_correction;
                representative.overlap =
                    best_verification.overlap_ratio;
                representative.rmse =
                    best_verification.rmse;
                representative.correction_translation =
                    best_verification.correction_translation;
                representative.correction_rotation_deg =
                    best_verification.correction_rotation_deg;
                representative.raw_scan_context_supported =
                    best_candidate_independent_retrieval_supported;
                representative.raw_scan_context_support_kf =
                    best_candidate_independent_support_kf;
                representative.raw_scan_context_support_similarity =
                    best_candidate_independent_support_similarity;

                local_cluster_representative_refreshed = true;
            }

            std::cout
                << "Local-strong cluster observation V20"
                << " | current_kf=" << current_keyframe.id
                << " | historical_kf=" << best_candidate->candidate_id
                << " | cluster_gap=" << local_cluster_gap
                << " | batch_kept=1"
                << " | representative_refreshed="
                << (local_cluster_representative_refreshed
                        ? "true"
                        : "false")
                << " | local_strong_support="
                << online_loop_track_.local_strong_support
                << "/" << online_loop_first_edge_min_support_
                << std::endl;
        }

        if (!local_strong_cluster_member &&
            strong_first_geometry &&
            spacing_ok &&
            temporal_track_continuous &&
            correction_consistent &&
            historical_anchor_unique &&
            historical_sequence_monotonic &&
            large_drift_evidence_ok)
        {
            PendingLoopConstraint constraint;

            constraint.historical_keyframe_id =
                best_candidate->candidate_id;
            constraint.current_keyframe_id =
                current_keyframe.id;
            constraint.historical_submap_id =
                best_historical_submap->id;
            constraint.current_submap_id =
                current_submap_id;
            constraint.T_historical_current =
                T_K_L;
            constraint.T_loop_correction =
                T_loop_correction;
            constraint.overlap =
                best_verification.overlap_ratio;
            constraint.rmse =
                best_verification.rmse;
            constraint.correction_translation =
                best_verification.correction_translation;
            constraint.correction_rotation_deg =
                best_verification.correction_rotation_deg;
            constraint.raw_scan_context_supported =
                best_candidate_independent_retrieval_supported;
            constraint.raw_scan_context_support_kf =
                best_candidate_independent_support_kf;
            constraint.raw_scan_context_support_similarity =
                best_candidate_independent_support_similarity;

            pending_first_loop_batch_.push_back(
                constraint);

            std::cout
                << "First loop batch candidate added"
                << " | member_mode="
                << (pending_first_loop_batch_.size() == 1
                        ? "ANCHOR"
                        : "FOLLOWUP")
                << " | historical_kf="
                << constraint.historical_keyframe_id
                << " | current_kf="
                << constraint.current_keyframe_id
                << " | batch="
                << pending_first_loop_batch_.size()
                << "/"
                << online_loop_first_batch_min_edges_
                << " | support="
                << online_loop_track_.support
                << "/"
                << online_loop_first_edge_min_support_
                << " | overlap=" << constraint.overlap
                << " | rmse=" << constraint.rmse << " m"
                << " | current_gap=" << batch_current_gap
                << " | historical_gap=" << batch_historical_gap
                << " | historical_segment_continuous="
                << (historical_segment_continuous ? "true" : "false")
                << " | historical_segment_arc_gap="
                << batch_segment_continuity_v225.trajectory_arc_gap_m
                << " m"
                << " | historical_segment_shared_kfs="
                << batch_segment_continuity_v225.shared_keyframes
                << " | temporal_track_continuous="
                << (temporal_track_continuous ? "true" : "false")
                << " | historical_delta=" << batch_historical_delta
                << " | historical_direction=" << batch_historical_direction
                << " | direction_jitter=+/-"
                << online_loop_first_batch_direction_jitter_tolerance_
                << "kf"
                << " | unique_history="
                << (historical_anchor_unique ? "true" : "false")
                << " | monotonic_history="
                << (historical_sequence_monotonic ? "true" : "false")
                << " | independent_global_retrieval="
                << (constraint.raw_scan_context_supported ? "true" : "false")
                << " | retrieval_support_kf="
                << constraint.raw_scan_context_support_kf
                << " | retrieval_similarity="
                << constraint.raw_scan_context_support_similarity
                << " | correction_consistency_mode=PREVIOUS_TRACK_LOCAL"
                << " | correction_consistency_dt="
                << batch_translation_error << " m"
                << " | correction_consistency_dR="
                << batch_rotation_error_deg << " deg"
                << std::endl;
        }
        else
        {
            std::cout
                << "First loop batch observation not added"
                << " | current_kf=" << current_keyframe.id
                << " | historical_kf="
                << best_candidate->candidate_id
                << " | strong_geometry="
                << (strong_first_geometry ? "true" : "false")
                << " | spacing_ok="
                << (spacing_ok ? "true" : "false")
                << " | correction_consistent="
                << (correction_consistent ? "true" : "false")
                << " | unique_history="
                << (historical_anchor_unique ? "true" : "false")
                << " | monotonic_history="
                << (historical_sequence_monotonic ? "true" : "false")
                << " | large_drift_evidence_ok="
                << (large_drift_evidence_ok ? "true" : "false")
                << " | strict_injected_followup="
                << (strict_injected_large_drift_followup ? "true" : "false")
                << " | local_cluster_member="
                << (local_strong_cluster_member ? "true" : "false")
                << " | independent_global_retrieval="
                << (best_candidate_independent_retrieval_supported ? "true" : "false")
                << " | raw_sc_supported="
                << (best_candidate_raw_sc_supported ? "true" : "false")
                << " | hierarchical_3m_supported="
                << (best_candidate_hierarchical_sc_supported ? "true" : "false")
                << " | current_gap=" << batch_current_gap
                << " | historical_gap=" << batch_historical_gap
                << " | historical_segment_continuous="
                << (historical_segment_continuous ? "true" : "false")
                << " | historical_segment_arc_gap="
                << batch_segment_continuity_v225.trajectory_arc_gap_m
                << " m"
                << " | historical_segment_shared_kfs="
                << batch_segment_continuity_v225.shared_keyframes
                << " | temporal_track_continuous="
                << (temporal_track_continuous ? "true" : "false")
                << " | historical_delta=" << batch_historical_delta
                << " | historical_direction=" << batch_historical_direction
                << " | direction_jitter=+/-"
                << online_loop_first_batch_direction_jitter_tolerance_
                << "kf"
                << " | correction_consistency_mode=PREVIOUS_TRACK_LOCAL"
                << " | correction_consistency_dt="
                << batch_translation_error << " m"
                << " | max_dt="
                << online_loop_first_batch_max_translation_error_
                << " m"
                << " | correction_consistency_dR="
                << batch_rotation_error_deg << " deg"
                << " | max_dR="
                << online_loop_first_batch_max_rotation_error_deg_
                << " deg"
                << std::endl;
        }

        if (current_first_loop_large_drift_candidate &&
            !best_candidate_independent_retrieval_supported)
        {
            std::cout
                << "Large-drift injected followup V20"
                << " | current_kf=" << current_keyframe.id
                << " | historical_kf=" << best_candidate->candidate_id
                << " | overlap=" << best_verification.overlap_ratio
                << " | rmse=" << best_verification.rmse
                << " | consistency_dt=" << batch_translation_error
                << " | consistency_dR=" << batch_rotation_error_deg
                << " | corridor_supported="
                << (best_candidate_corridor_supported ? "true" : "false")
                << " | corridor_distance="
                << best_candidate_corridor_distance << " m"
                << " | strict_followup="
                << (strict_injected_large_drift_followup
                        ? "PASS"
                        : "REJECT")
                << std::endl;
        }

        // Candidate collection and batch commit are separate operations.
        // A valid anchor may already be stored before temporal support reaches
        // the commit threshold.  Only the following combined condition opens
        // the transaction that stages loop edges into PoseGraph.
        const bool first_batch_has_enough_edges =
            pending_first_loop_batch_.size() >=
            online_loop_first_batch_min_edges_;

        const bool first_batch_has_enough_support =
            online_loop_track_.support >=
            online_loop_first_edge_min_support_;

        // Mode B: large-drift closure must be supported by a genuine
        // multi-anchor historical sequence AND independent place-recognition
        // evidence.  TRACK/neighborhood injection is still never treated as an
        // independent vote.
        std::size_t first_batch_independent_retrieval_members = 0;

        for (const PendingLoopConstraint &constraint :
             pending_first_loop_batch_)
        {
            if (constraint.raw_scan_context_supported)
            {
                ++first_batch_independent_retrieval_members;
            }
        }

        const bool first_batch_has_independent_retrieval =
            first_batch_independent_retrieval_members >=
            online_loop_large_drift_min_raw_sc_members_;

        // First-loop safety:
        // 在农业重复场景里，仅仅两次 independent retrieval 还不够。
        // 第一条真正进入 PoseGraph 的 loop 至少必须包含一次 LOCAL_STRONG 证据。
        const bool first_batch_has_local_strong_evidence =
            online_loop_track_.local_strong_support >= 1;

        const bool large_drift_sequence_ready =
            first_batch_has_enough_edges &&
            first_batch_has_enough_support &&
            first_batch_has_independent_retrieval &&
            first_batch_has_local_strong_evidence;

        // Mode A: LOCAL_STRONG_3M_SEGMENT_CLUSTER.
        // The graph still receives ONE representative loop factor, but the
        // confirmation is now based on physical 3m-segment continuity instead
        // of a fixed +/-N Keyframe-ID radius.
        bool local_strong_cluster_ready = false;
        std::size_t local_ready_cluster_gap =
            std::numeric_limits<std::size_t>::max();

        if (pending_first_loop_batch_.size() == 1 &&
            first_loop_local_strong_candidate)
        {
            const std::size_t stored_historical_kf =
                pending_first_loop_batch_.front()
                    .historical_keyframe_id;

            local_ready_cluster_gap =
                best_candidate->candidate_id >= stored_historical_kf
                    ? best_candidate->candidate_id -
                          stored_historical_kf
                    : stored_historical_kf -
                          best_candidate->candidate_id;

            local_strong_cluster_ready =
                online_loop_track_.local_strong_support >=
                    online_loop_first_edge_min_support_ &&
                local_strong_cluster_member &&
                temporal_track_continuous &&
                correction_consistent;
        }

        const bool first_batch_ready =
            large_drift_sequence_ready ||
            local_strong_cluster_ready;

        if (!first_batch_ready)
        {
            std::cout
                << "Keyframe loop decision"
                << " | current_kf=" << current_keyframe.id
                << " | decision=PENDING"
                << " | reason=WAIT_FIRST_LOOP_CONFIRMATION"
                << " | support="
                << online_loop_track_.support
                << "/"
                << online_loop_first_edge_min_support_
                << " | local_strong_support="
                << online_loop_track_.local_strong_support
                << "/"
                << online_loop_first_edge_min_support_
                << " | batch="
                << pending_first_loop_batch_.size()
                << "/"
                << online_loop_first_batch_min_edges_
                << " | independent_retrieval_batch="
                << first_batch_independent_retrieval_members
                << "/"
                << online_loop_large_drift_min_raw_sc_members_
                << " | independent_retrieval="
                << (first_batch_has_independent_retrieval
                        ? "true"
                        : "false")
                << " | sequence_ready="
                << (large_drift_sequence_ready ? "true" : "false")
                << " | local_cluster_ready="
                << (local_strong_cluster_ready ? "true" : "false")
                << std::endl;
            std::cout
                << "FR_LOOP_TRACE"
                << " | stage=DECISION"
                << " | current_kf=" << current_keyframe.id
                << " | historical_kf=" << best_candidate->candidate_id
                << " | decision=PENDING"
                << " | reason=WAIT_FIRST_LOOP_CONFIRMATION"
                << " | support=" << online_loop_track_.support
                << "/" << online_loop_first_edge_min_support_
                << " | batch=" << pending_first_loop_batch_.size()
                << "/" << online_loop_first_batch_min_edges_
                << " | independent_retrieval_batch=" << first_batch_independent_retrieval_members
                << " | independent_retrieval="
                << (first_batch_has_independent_retrieval ? "true" : "false")
                << std::endl;
            return;
        }

        const char *first_loop_commit_mode =
            local_strong_cluster_ready
                ? "LOCAL_STRONG_3M_SEGMENT_CLUSTER"
                : "TWO_OBSERVATION_3M_SEGMENT_CONFIRMED_SINGLE_FACTOR";

        std::cout
            << "First loop batch ready"
            << " | mode=" << first_loop_commit_mode
            << " | edges="
            << pending_first_loop_batch_.size()
            << " | support="
            << online_loop_track_.support
            << " | local_strong_support="
            << online_loop_track_.local_strong_support
            << " | independent_retrieval_batch="
            << first_batch_independent_retrieval_members
            << "/"
            << online_loop_first_batch_min_edges_
            << " | action=STAGE_AND_OPTIMIZE"
            << std::endl;

        std::cout
            << "FR_LOOP_TRACE"
            << " | stage=BATCH_READY"
            << " | current_kf=" << current_keyframe.id
            << " | mode=" << first_loop_commit_mode
            << " | observations=" << pending_first_loop_batch_.size()
            << " | support=" << online_loop_track_.support
            << " | local_strong_support="
            << online_loop_track_.local_strong_support
            << " | independent_retrieval_members=" << first_batch_independent_retrieval_members
            << " | action=STAGE_SINGLE_FACTOR"
            << std::endl;

        // V20 SIMPLE BASELINE:
        // The second observation CONFIRMS the place, but we insert only ONE
        // loop factor into PoseGraph.  This avoids the old failure mode where
        // 2-3 highly correlated constraints from the same short revisit pulled
        // the whole graph together as if they were independent evidence.
        const PendingLoopConstraint *first_loop_graph_constraint =
            &pending_first_loop_batch_.front();

        // Prefer an independently retrieved representative.  The large-drift anchor is
        // intentionally the most semantically independent observation.
        for (const PendingLoopConstraint &candidate_constraint :
             pending_first_loop_batch_)
        {
            if (candidate_constraint.raw_scan_context_supported &&
                !first_loop_graph_constraint->raw_scan_context_supported)
            {
                first_loop_graph_constraint =
                    &candidate_constraint;
            }
            else if (candidate_constraint.raw_scan_context_supported ==
                     first_loop_graph_constraint->raw_scan_context_supported)
            {
                const bool better_overlap =
                    candidate_constraint.overlap >
                    first_loop_graph_constraint->overlap + 1e-6;

                const bool same_overlap_better_rmse =
                    std::abs(candidate_constraint.overlap -
                             first_loop_graph_constraint->overlap) <= 1e-6 &&
                    candidate_constraint.rmse <
                        first_loop_graph_constraint->rmse;

                if (better_overlap || same_overlap_better_rmse)
                {
                    first_loop_graph_constraint =
                        &candidate_constraint;
                }
            }
        }

        std::vector<std::pair<std::size_t, std::size_t>>
            staged_loop_edges;

        staged_loop_edges.reserve(1);

        bool batch_stage_ok = true;

        do
        {
            const PendingLoopConstraint &constraint =
                *first_loop_graph_constraint;
            if (!pose_graph_.HasNode(
                    constraint.historical_keyframe_id) ||
                !pose_graph_.HasNode(
                    constraint.current_keyframe_id))
            {
                batch_stage_ok = false;
                break;
            }

            Eigen::Matrix<double, 6, 6> loop_information =
                Eigen::Matrix<double, 6, 6>::Identity();

            std::size_t loop_shadow_correspondences = 0;
            double loop_median_range =
                std::numeric_limits<double>::quiet_NaN();
            double loop_min_relative =
                std::numeric_limits<double>::quiet_NaN();

            bool dynamic_loop_information = false;

            const Keyframe *loop_current_keyframe =
                FindBackendKeyframeById(
                    constraint.current_keyframe_id);

            const Keyframe *loop_historical_keyframe =
                FindBackendKeyframeById(
                    constraint.historical_keyframe_id);

            pcl::PointCloud<LIDAR_POINT>::Ptr
                loop_historical_target_K;

            if (loop_current_keyframe != nullptr &&
                loop_historical_keyframe != nullptr &&
                loop_current_keyframe->cloud &&
                loop_historical_keyframe->T_WL.matrix().allFinite() &&
                BuildCandidateCenteredHistoricalTarget(
                    constraint.historical_keyframe_id,
                    loop_historical_target_K,
                    nullptr))
            {
                dynamic_loop_information =
                    BuildLoopShadowInformationFull6x6(
                        loop_current_keyframe->cloud,
                        loop_historical_target_K,
                        constraint.T_historical_current,
                        loop_verifier_.GetConfig(),
                        loop_information,
                        loop_shadow_correspondences,
                        loop_median_range,
                        loop_min_relative);
            }

            if (!pose_graph_.AddLoopEdge(
                    constraint.historical_keyframe_id,
                    constraint.current_keyframe_id,
                    constraint.T_historical_current,
                    loop_information))
            {
                batch_stage_ok = false;
                break;
            }

            double loop_max_offdiag = 0.0;
            double loop_max_tr_coupling = 0.0;

            ComputeLoopInformationStats(
                loop_information,
                loop_max_offdiag,
                loop_max_tr_coupling);

            staged_loop_edges.emplace_back(
                constraint.historical_keyframe_id,
                constraint.current_keyframe_id);

            std::cout
                << "FR_LOOP_TRACE"
                << " | stage=EDGE_STAGE"
                << " | mode=FIRST_LOOP"
                << " | from_kf=" << constraint.historical_keyframe_id
                << " | to_kf=" << constraint.current_keyframe_id
                << " | evidence_observations=" << pending_first_loop_batch_.size()
                << " | measurement_translation_norm="
                << constraint.T_historical_current.translation().norm()
                << " m"
                << std::endl;

            std::cout
                << "ONLINE Keyframe PoseGraph loop edge staged"
                << " | from_kf="
                << constraint.historical_keyframe_id
                << " | to_kf="
                << constraint.current_keyframe_id
                << " | edge_mode=" << first_loop_commit_mode
                << " | evidence_observations="
                << pending_first_loop_batch_.size()
                << " | staged_factor_index=1/1"
                << " | information_mode="
                << (dynamic_loop_information
                        ? "SHADOW_FULL_6X6"
                        : "IDENTITY_FALLBACK")
                << " | base_diag=["
                << loop_information(0, 0) << " "
                << loop_information(1, 1) << " "
                << loop_information(2, 2) << " "
                << loop_information(3, 3) << " "
                << loop_information(4, 4) << " "
                << loop_information(5, 5)
                << "]"
                << " | max_offdiag="
                << loop_max_offdiag
                << " | max_tr_coupling="
                << loop_max_tr_coupling
                << " | shadow_corr="
                << loop_shadow_correspondences
                << " | median_range="
                << loop_median_range
                << " | min_relative="
                << loop_min_relative
                << " | measurement_translation_norm="
                << constraint.T_historical_current.translation().norm()
                << " m"
                << std::endl;
        } while (false);

        if (!batch_stage_ok)
        {
            for (auto iterator =
                     staged_loop_edges.rbegin();
                 iterator != staged_loop_edges.rend();
                 ++iterator)
            {
                pose_graph_.RemoveLoopEdge(
                    iterator->first,
                    iterator->second);
            }

            pending_first_loop_batch_.clear();

            std::cerr
                << "First loop batch stage failed"
                << " | staged_edges="
                << staged_loop_edges.size()
                << " | action=ROLLBACK_ALL_AND_CLEAR_BATCH"
                << std::endl;
            return;
        }

        // V16 transactional pose snapshot.  Optimize() writes accepted g2o
        // estimates back into pose_graph_, so a post-optimization update guard
        // needs an explicit copy in order to restore the graph on rollback.
        const std::vector<PoseGraphNode>
            first_batch_pose_snapshot =
                pose_graph_.GetNodes();

        PoseGraphOptimizationResult optimization_result;

        const std::chrono::steady_clock::time_point
            first_batch_pgo_start =
                std::chrono::steady_clock::now();

        const bool first_batch_pgo_ok =
            pose_graph_optimizer_.Optimize(
                pose_graph_,
                optimization_result);

        loop_timing.pose_graph_optimize_ms +=
            ElapsedMilliseconds(
                first_batch_pgo_start,
                std::chrono::steady_clock::now());

        ++loop_timing.pose_graph_optimize_calls;

        if (!first_batch_pgo_ok)
        {
            std::size_t restored_nodes = 0;

            for (const PoseGraphNode &node :
                 first_batch_pose_snapshot)
            {
                if (pose_graph_.SetNodePose(
                        node.id,
                        node.T_WK))
                {
                    ++restored_nodes;
                }
            }

            std::size_t rollback_count = 0;

            for (auto iterator =
                     staged_loop_edges.rbegin();
                 iterator != staged_loop_edges.rend();
                 ++iterator)
            {
                if (pose_graph_.RemoveLoopEdge(
                        iterator->first,
                        iterator->second))
                {
                    ++rollback_count;
                }
            }

            pending_first_loop_batch_.clear();
            online_loop_track_ = OnlineLoopTrack();

            std::cout
                << "FR_LOOP_TRACE"
                << " | stage=PGO"
                << " | mode=FIRST_LOOP"
                << " | decision=REJECT_ROLLBACK"
                << " | reason=OPTIMIZER_REJECTED"
                << " | staged_edges=" << staged_loop_edges.size()
                << std::endl;

            std::cerr
                << "First loop batch optimization rejected"
                << " | staged_edges="
                << staged_loop_edges.size()
                << " | rolled_back=" << rollback_count
                << " | restored_nodes=" << restored_nodes
                << "/" << first_batch_pose_snapshot.size()
                << " | gravity_guard_passed="
                << (optimization_result.gravity_guard_passed
                        ? "true"
                        : "false")
                << " | shape_guard_passed="
                << (optimization_result.trajectory_shape_guard_passed
                        ? "true"
                        : "false")
                << " | xy_pca_before="
                << optimization_result.xy_pca_ratio_before
                << " | xy_pca_after="
                << optimization_result.xy_pca_ratio_after
                << " | path_length_ratio="
                << optimization_result.path_length_ratio
                << " | action=RESTORE_POSES_REMOVE_LOOPS_RESET_TRACK"
                << std::endl;
            return;
        }

        // ----------------------------------------------------------------
        // V18 consensus-aware first-PGO update guard.
        //
        // A genuine large-drift loop is allowed to make a large correction,
        // but only after TWO locally consistent historical observations have
        // confirmed the same revisit neighborhood.
        //
        // LOCAL_STRONG keeps the conservative 5 m / 20 deg limits.
        // LARGE_DRIFT scales the numerical update ceiling from the consensus
        // itself, and additionally requires the optimized latest anchor to land
        // near the world pose predicted by that consensus.
        // ----------------------------------------------------------------
        const Eigen::Isometry3d first_batch_expected_correction =
            first_loop_graph_constraint->T_loop_correction;

        const double first_batch_expected_translation =
            first_batch_expected_correction.translation().norm();

        const double first_batch_expected_rotation =
            RelativeRotationDeg(
                Eigen::Isometry3d::Identity(),
                first_batch_expected_correction);

        double first_batch_translation_limit =
            online_loop_pgo_max_translation_update_;

        double first_batch_rotation_limit =
            online_loop_pgo_max_rotation_update_deg_;

        if (large_drift_sequence_ready)
        {
            first_batch_translation_limit =
                std::min(
                    online_loop_large_drift_pgo_translation_cap_,
                    std::max(
                        online_loop_pgo_max_translation_update_,
                        online_loop_large_drift_pgo_translation_scale_ *
                                first_batch_expected_translation +
                            online_loop_large_drift_pgo_translation_margin_));

            // V19.1:
            // Do NOT scale the large-drift decision from the absolute maximum
            // node rotation.  A long graph can legitimately distribute a large
            // world-yaw correction over many vertices even when every adjacent
            // odometry edge bends only a few degrees.
            //
            // Keep only the 110 deg catastrophic cap here.  Real safety comes
            // from the local odometry-deformation and staged-loop residual tests
            // computed below.
            first_batch_rotation_limit =
                online_loop_large_drift_pgo_rotation_cap_deg_;
        }

        bool first_batch_anchor_target_ok = true;
        double first_batch_anchor_target_translation_error =
            std::numeric_limits<double>::quiet_NaN();
        double first_batch_anchor_target_rotation_error =
            std::numeric_limits<double>::quiet_NaN();

        if (large_drift_sequence_ready)
        {
            const PendingLoopConstraint &guard_anchor =
                *first_loop_graph_constraint;

            const PoseGraphNode *optimized_anchor =
                pose_graph_.GetNode(
                    guard_anchor.current_keyframe_id);

            const PoseGraphNode *snapshot_anchor = nullptr;

            for (const PoseGraphNode &snapshot_node :
                 first_batch_pose_snapshot)
            {
                if (snapshot_node.id ==
                    guard_anchor.current_keyframe_id)
                {
                    snapshot_anchor =
                        &snapshot_node;
                    break;
                }
            }

            if (optimized_anchor == nullptr ||
                snapshot_anchor == nullptr ||
                !optimized_anchor->T_WK.matrix().allFinite() ||
                !snapshot_anchor->T_WK.matrix().allFinite())
            {
                first_batch_anchor_target_ok = false;
            }
            else
            {
                const Eigen::Isometry3d expected_anchor_pose =
                    guard_anchor.T_loop_correction *
                    snapshot_anchor->T_WK;

                if (!expected_anchor_pose.matrix().allFinite())
                {
                    first_batch_anchor_target_ok = false;
                }
                else
                {
                    first_batch_anchor_target_translation_error =
                        (optimized_anchor->T_WK.translation() -
                         expected_anchor_pose.translation())
                            .norm();

                    first_batch_anchor_target_rotation_error =
                        RelativeRotationDeg(
                            expected_anchor_pose,
                            optimized_anchor->T_WK);

                    first_batch_anchor_target_ok =
                        std::isfinite(
                            first_batch_anchor_target_translation_error) &&
                        std::isfinite(
                            first_batch_anchor_target_rotation_error) &&
                        first_batch_anchor_target_translation_error <=
                            online_loop_large_drift_pgo_anchor_target_translation_error_ &&
                        first_batch_anchor_target_rotation_error <=
                            online_loop_large_drift_pgo_anchor_target_rotation_error_deg_;
                }
            }
        }

        // ----------------------------------------------------------------
        // V19.1 LOCAL graph-deformation diagnostics.
        //
        // "max_rotation_update" is an ABSOLUTE world-pose change of one node.
        // For a long trajectory this can be tens of degrees even when the
        // correction field is smooth.  What detects a graph being "pulled
        // apart" is the RELATIVE deformation between adjacent odometry nodes.
        // ----------------------------------------------------------------
        double max_local_odom_translation_deformation = 0.0;
        double max_local_odom_rotation_deformation_deg = 0.0;
        std::size_t worst_local_odom_from =
            std::numeric_limits<std::size_t>::max();
        std::size_t worst_local_odom_to =
            std::numeric_limits<std::size_t>::max();

        bool local_odom_deformation_finite = true;

        for (std::size_t snapshot_index = 1;
             snapshot_index < first_batch_pose_snapshot.size();
             ++snapshot_index)
        {
            const PoseGraphNode &before_previous =
                first_batch_pose_snapshot[snapshot_index - 1];

            const PoseGraphNode &before_current =
                first_batch_pose_snapshot[snapshot_index];

            // PoseGraph keyframes are sequential in the current backend.
            // Skip any non-consecutive pair rather than inventing an odom edge.
            if (before_current.id !=
                before_previous.id + 1)
            {
                continue;
            }

            const PoseGraphNode *after_previous =
                pose_graph_.GetNode(
                    before_previous.id);

            const PoseGraphNode *after_current =
                pose_graph_.GetNode(
                    before_current.id);

            if (after_previous == nullptr ||
                after_current == nullptr ||
                !before_previous.T_WK.matrix().allFinite() ||
                !before_current.T_WK.matrix().allFinite() ||
                !after_previous->T_WK.matrix().allFinite() ||
                !after_current->T_WK.matrix().allFinite())
            {
                local_odom_deformation_finite = false;
                break;
            }

            const Eigen::Isometry3d before_relative =
                before_previous.T_WK.inverse() *
                before_current.T_WK;

            const Eigen::Isometry3d after_relative =
                after_previous->T_WK.inverse() *
                after_current->T_WK;

            const Eigen::Isometry3d deformation =
                before_relative.inverse() *
                after_relative;

            if (!deformation.matrix().allFinite())
            {
                local_odom_deformation_finite = false;
                break;
            }

            const double deformation_translation =
                deformation.translation().norm();

            const double deformation_rotation_deg =
                RelativeRotationDeg(
                    Eigen::Isometry3d::Identity(),
                    deformation);

            if (!std::isfinite(deformation_translation) ||
                !std::isfinite(deformation_rotation_deg))
            {
                local_odom_deformation_finite = false;
                break;
            }

            if (deformation_translation >
                max_local_odom_translation_deformation)
            {
                max_local_odom_translation_deformation =
                    deformation_translation;
            }

            if (deformation_rotation_deg >
                max_local_odom_rotation_deformation_deg)
            {
                max_local_odom_rotation_deformation_deg =
                    deformation_rotation_deg;
                worst_local_odom_from =
                    before_previous.id;
                worst_local_odom_to =
                    before_current.id;
            }
        }

        const bool local_odom_deformation_ok =
            !large_drift_sequence_ready ||
            (local_odom_deformation_finite &&
             max_local_odom_translation_deformation <=
                 online_loop_large_drift_max_local_odom_translation_deformation_ &&
             max_local_odom_rotation_deformation_deg <=
                 online_loop_large_drift_max_local_odom_rotation_deformation_deg_);

        // ----------------------------------------------------------------
        // V19.1 staged-loop residual check after optimization.
        //
        // Every independent loop factor used for the first transaction must
        // still be reasonably satisfied by the optimized graph.  This prevents
        // accepting a numerically smooth graph that simply ignored one of the
        // the staged loop factor.
        // ----------------------------------------------------------------
        double max_staged_loop_residual_translation = 0.0;
        double max_staged_loop_residual_rotation_deg = 0.0;
        bool staged_loop_residuals_finite = true;

        do
        {
            const PendingLoopConstraint &constraint =
                *first_loop_graph_constraint;

            const PoseGraphNode *optimized_historical =
                pose_graph_.GetNode(
                    constraint.historical_keyframe_id);

            const PoseGraphNode *optimized_current =
                pose_graph_.GetNode(
                    constraint.current_keyframe_id);

            if (optimized_historical == nullptr ||
                optimized_current == nullptr ||
                !optimized_historical->T_WK.matrix().allFinite() ||
                !optimized_current->T_WK.matrix().allFinite() ||
                !constraint.T_historical_current.matrix().allFinite())
            {
                staged_loop_residuals_finite = false;
                break;
            }

            const Eigen::Isometry3d optimized_measurement =
                optimized_historical->T_WK.inverse() *
                optimized_current->T_WK;

            const Eigen::Isometry3d loop_residual =
                constraint.T_historical_current.inverse() *
                optimized_measurement;

            if (!loop_residual.matrix().allFinite())
            {
                staged_loop_residuals_finite = false;
                break;
            }

            const double residual_translation =
                loop_residual.translation().norm();

            const double residual_rotation_deg =
                RelativeRotationDeg(
                    Eigen::Isometry3d::Identity(),
                    loop_residual);

            if (!std::isfinite(residual_translation) ||
                !std::isfinite(residual_rotation_deg))
            {
                staged_loop_residuals_finite = false;
                break;
            }

            max_staged_loop_residual_translation =
                std::max(
                    max_staged_loop_residual_translation,
                    residual_translation);

            max_staged_loop_residual_rotation_deg =
                std::max(
                    max_staged_loop_residual_rotation_deg,
                    residual_rotation_deg);
        } while (false);

        const bool staged_loop_residuals_ok =
            !large_drift_sequence_ready ||
            (staged_loop_residuals_finite &&
             max_staged_loop_residual_translation <=
                 online_loop_large_drift_max_loop_residual_translation_ &&
             max_staged_loop_residual_rotation_deg <=
                 online_loop_large_drift_max_loop_residual_rotation_deg_);

        const bool chi2_improved =
            std::isfinite(optimization_result.chi2_before) &&
            std::isfinite(optimization_result.chi2_after) &&
            optimization_result.chi2_after <
                optimization_result.chi2_before;

        const bool first_batch_path_length_ratio_ok =
            std::isfinite(
                optimization_result.path_length_ratio) &&
            optimization_result.path_length_ratio >=
                online_loop_first_pgo_min_path_length_ratio_ &&
            optimization_result.path_length_ratio <=
                online_loop_first_pgo_max_path_length_ratio_;

        const bool first_batch_update_guard_passed =
            std::isfinite(
                optimization_result.max_translation_update) &&
            std::isfinite(
                optimization_result.max_rotation_update_deg) &&
            optimization_result.max_translation_update <=
                first_batch_translation_limit &&
            optimization_result.max_rotation_update_deg <=
                first_batch_rotation_limit &&
            first_batch_anchor_target_ok &&
            local_odom_deformation_ok &&
            staged_loop_residuals_ok &&
            first_batch_path_length_ratio_ok &&
            (!large_drift_sequence_ready ||
             chi2_improved);

        std::cout
            << "First loop baseline PGO guard V20"
            << " | path_length_ratio="
            << optimization_result.path_length_ratio
            << " | path_length_ratio_limit=["
            << online_loop_first_pgo_min_path_length_ratio_
            << ","
            << online_loop_first_pgo_max_path_length_ratio_
            << "]"
            << " | path_length_ratio_ok="
            << (first_batch_path_length_ratio_ok
                    ? "true"
                    : "false")
            << " | mode=" << first_loop_commit_mode
            << " | expected_translation="
            << first_batch_expected_translation << " m"
            << " | expected_rotation="
            << first_batch_expected_rotation << " deg"
            << " | actual_max_translation="
            << optimization_result.max_translation_update << " m"
            << " | actual_max_rotation="
            << optimization_result.max_rotation_update_deg << " deg"
            << " | limit_translation="
            << first_batch_translation_limit << " m"
            << " | limit_rotation="
            << first_batch_rotation_limit << " deg"
            << " | anchor_target_dt="
            << first_batch_anchor_target_translation_error << " m"
            << " | anchor_target_dR="
            << first_batch_anchor_target_rotation_error << " deg"
            << " | anchor_target_ok="
            << (first_batch_anchor_target_ok ? "true" : "false")
            << " | max_local_odom_def_dt="
            << max_local_odom_translation_deformation << " m"
            << " | max_local_odom_def_dR="
            << max_local_odom_rotation_deformation_deg << " deg"
            << " | worst_local_odom_edge="
            << worst_local_odom_from << "->" << worst_local_odom_to
            << " | local_odom_ok="
            << (local_odom_deformation_ok ? "true" : "false")
            << " | max_loop_residual_dt="
            << max_staged_loop_residual_translation << " m"
            << " | max_loop_residual_dR="
            << max_staged_loop_residual_rotation_deg << " deg"
            << " | loop_residuals_ok="
            << (staged_loop_residuals_ok ? "true" : "false")
            << " | chi2_improved="
            << (chi2_improved ? "true" : "false")
            << " | decision="
            << (first_batch_update_guard_passed ? "PASS" : "ROLLBACK")
            << std::endl;

        if (!first_batch_update_guard_passed)
        {
            std::size_t restored_nodes = 0;

            for (const PoseGraphNode &node :
                 first_batch_pose_snapshot)
            {
                if (pose_graph_.SetNodePose(
                        node.id,
                        node.T_WK))
                {
                    ++restored_nodes;
                }
            }

            std::size_t rollback_count = 0;

            for (auto iterator =
                     staged_loop_edges.rbegin();
                 iterator != staged_loop_edges.rend();
                 ++iterator)
            {
                if (pose_graph_.RemoveLoopEdge(
                        iterator->first,
                        iterator->second))
                {
                    ++rollback_count;
                }
            }

            pending_first_loop_batch_.clear();
            online_loop_track_ = OnlineLoopTrack();

            std::cout
                << "FR_LOOP_TRACE"
                << " | stage=PGO"
                << " | mode=FIRST_LOOP"
                << " | decision=REJECT_ROLLBACK"
                << " | reason=UPDATE_GUARD"
                << " | max_update_dt=" << optimization_result.max_translation_update << " m"
                << " | max_update_dR=" << optimization_result.max_rotation_update_deg << " deg"
                << " | chi2_before=" << optimization_result.chi2_before
                << " | chi2_after=" << optimization_result.chi2_after
                << std::endl;

            std::cerr
                << "First loop batch PGO update guard rejected"
                << " | max_translation_update="
                << optimization_result.max_translation_update
                << " m"
                << " | limit_translation="
                << first_batch_translation_limit
                << " m"
                << " | max_rotation_update="
                << optimization_result.max_rotation_update_deg
                << " deg"
                << " | limit_rotation="
                << first_batch_rotation_limit
                << " deg"
                << " | anchor_target_dt="
                << first_batch_anchor_target_translation_error
                << " m"
                << " | anchor_target_dR="
                << first_batch_anchor_target_rotation_error
                << " deg"
                << " | max_local_odom_def_dt="
                << max_local_odom_translation_deformation
                << " m"
                << " | max_local_odom_def_dR="
                << max_local_odom_rotation_deformation_deg
                << " deg"
                << " | local_odom_ok="
                << (local_odom_deformation_ok ? "true" : "false")
                << " | max_loop_residual_dt="
                << max_staged_loop_residual_translation
                << " m"
                << " | max_loop_residual_dR="
                << max_staged_loop_residual_rotation_deg
                << " deg"
                << " | loop_residuals_ok="
                << (staged_loop_residuals_ok ? "true" : "false")
                << " | chi2_improved="
                << (chi2_improved ? "true" : "false")
                << " | restored_nodes=" << restored_nodes
                << "/" << first_batch_pose_snapshot.size()
                << " | rolled_back_edges=" << rollback_count
                << "/" << staged_loop_edges.size()
                << " | action=RESTORE_POSES_REMOVE_LOOPS_RESET_TRACK"
                << std::endl;
            return;
        }

        loop_timing.optimization_accepted = true;
        loop_timing.loop_edge_accepted = true;

        const PendingLoopConstraint &last_constraint =
            *first_loop_graph_constraint;

        has_last_online_loop_edge_ = true;
        last_online_loop_current_keyframe_id_ =
            last_constraint.current_keyframe_id;
        last_online_loop_historical_keyframe_id_ =
            last_constraint.historical_keyframe_id;
        last_online_loop_measurement_ =
            last_constraint.T_historical_current;

        // V2.4: the first committed loop becomes the initial cluster anchor.
        g_post_loop_new_cluster_track =
            PostLoopClusterTrackState();

        const std::size_t accepted_batch_size =
            pending_first_loop_batch_.size();
        const std::size_t accepted_graph_factor_count =
            staged_loop_edges.size();

        pending_first_loop_batch_.clear();

        std::cout
            << "ONLINE Keyframe PoseGraph first loop batch accepted"
            << " | mode=" << first_loop_commit_mode
            << " | evidence_observations=" << accepted_batch_size
            << " | graph_factors=" << accepted_graph_factor_count
            << " | last_from_kf="
            << last_online_loop_historical_keyframe_id_
            << " | last_to_kf="
            << last_online_loop_current_keyframe_id_
            << " | total_loop_edges="
            << pose_graph_.LoopEdgeCount()
            << std::endl;

        std::cout
            << "FR_LOOP_TRACE"
            << " | stage=PGO"
            << " | mode=FIRST_LOOP"
            << " | decision=ACCEPT"
            << " | from_kf=" << last_online_loop_historical_keyframe_id_
            << " | to_kf=" << last_online_loop_current_keyframe_id_
            << " | observations=" << accepted_batch_size
            << " | graph_factors=" << accepted_graph_factor_count
            << " | chi2_before=" << optimization_result.chi2_before
            << " | chi2_after=" << optimization_result.chi2_after
            << " | max_update_dt=" << optimization_result.max_translation_update << " m"
            << " | max_update_dR=" << optimization_result.max_rotation_update_deg << " deg"
            << " | gravity_guard="
            << (optimization_result.gravity_guard_passed ? "PASS" : "FAIL")
            << std::endl;

        std::cout
            << "G2O Keyframe PoseGraph optimized"
            << " | iterations=" << optimization_result.iterations
            << " | chi2_before=" << optimization_result.chi2_before
            << " | chi2_after=" << optimization_result.chi2_after
            << " | optimized_nodes=" << optimization_result.optimized_nodes
            << " | odom_edges=" << optimization_result.odometry_edges
            << " | loop_edges=" << optimization_result.loop_edges
            << " | gravity_edges=" << optimization_result.gravity_edges
            << " | max_translation_update="
            << optimization_result.max_translation_update << " m"
            << " | max_rotation_update="
            << optimization_result.max_rotation_update_deg << " deg"
            << " | max_droll="
            << optimization_result.max_roll_update_deg << " deg"
            << " | max_dpitch="
            << optimization_result.max_pitch_update_deg << " deg"
            << " | max_dyaw="
            << optimization_result.max_yaw_update_deg << " deg"
            << " | max_gravity_tilt="
            << optimization_result.max_gravity_tilt_error_deg << " deg"
            << " | xy_pca_before="
            << optimization_result.xy_pca_ratio_before
            << " | xy_pca_after="
            << optimization_result.xy_pca_ratio_after
            << " | path_length_ratio="
            << optimization_result.path_length_ratio
            << " | guards=PASS"
            << std::endl;

        const std::chrono::steady_clock::time_point
            first_batch_map_odom_start =
                std::chrono::steady_clock::now();

        const bool first_batch_map_odom_ok =
            UpdateMapOdomCorrection(
                last_online_loop_current_keyframe_id_);

        loop_timing.map_odom_ms +=
            ElapsedMilliseconds(
                first_batch_map_odom_start,
                std::chrono::steady_clock::now());

        if (!first_batch_map_odom_ok)
        {
            std::cerr
                << "Map->odom correction update failed"
                << " | anchor_kf="
                << last_online_loop_current_keyframe_id_
                << std::endl;
        }

        const std::chrono::steady_clock::time_point
            first_batch_global_map_start =
                std::chrono::steady_clock::now();

        const bool global_map_rebuilt =
            RebuildGlobalMapSnapshots();

        loop_timing.global_map_rebuild_ms +=
            ElapsedMilliseconds(
                first_batch_global_map_start,
                std::chrono::steady_clock::now());

        if (!global_map_rebuilt)
        {
            std::cerr
                << "Global map snapshot rebuild failed"
                << " | keyframes=" << backend_keyframes_.size()
                << " | graph_nodes=" << pose_graph_.NodeCount()
                << std::endl;
        }
        else
        {
            const std::chrono::steady_clock::time_point
                first_batch_refinement_start =
                    std::chrono::steady_clock::now();

            const bool refinement_ok =
                RebuildPostPgoRefinedMap();

            loop_timing.refinement_ms +=
                ElapsedMilliseconds(
                    first_batch_refinement_start,
                    std::chrono::steady_clock::now());

            if (!refinement_ok)
            {
                std::cerr
                    << "Post-PGO refined map rebuild skipped/failed"
                    << " | global_revision=" << global_map_revision_
                    << " | keyframes=" << backend_keyframes_.size()
                    << std::endl;
            }
        }

        return;
    }

    if (online_loop_track_.support <
        online_loop_min_support_)
    {
        std::cout
            << "Keyframe loop decision"
            << " | current_kf=" << current_keyframe.id
            << " | decision=PENDING"
            << " | reason=TEMPORAL_SUPPORT"
            << " | support=" << online_loop_track_.support
            << "/" << online_loop_min_support_
            << std::endl;
        std::cout
            << "FR_LOOP_TRACE"
            << " | stage=DECISION"
            << " | current_kf=" << current_keyframe.id
            << " | historical_kf=" << best_candidate->candidate_id
            << " | decision=PENDING"
            << " | reason=TEMPORAL_SUPPORT"
            << " | support=" << online_loop_track_.support
            << "/" << online_loop_min_support_
            << std::endl;
        return;
    }

    // Subsequent sequence loop: one verified candidate may be considered after
    // the initial multi-edge batch has already established the revisit geometry.
    std::size_t edge_historical_keyframe_id =
        representative_historical_keyframe_id;
    std::size_t edge_current_keyframe_id =
        current_keyframe.id;

    const BackendSubmapSnapshot *representative_historical_submap =
        FindBestFinishedSubmapForKeyframe(
            representative_historical_keyframe_id);

    std::size_t edge_historical_submap_id =
        representative_historical_submap != nullptr
            ? representative_historical_submap->id
            : best_historical_submap->id;
    std::size_t edge_current_submap_id =
        current_submap_id;
    Eigen::Isometry3d edge_measurement =
        T_representative_L;

    std::cout
        << "LOOP_EDGE_REANCHOR_V22_7"
        << " | current_kf=" << edge_current_keyframe_id
        << " | segment_anchor_kf="
        << best_candidate->candidate_id
        << " | representative_kf="
        << edge_historical_keyframe_id
        << " | representative_distance="
        << representative_distance_m << " m"
        << " | action=USE_REPRESENTATIVE_ENDPOINT"
        << std::endl;

    // V2.4 diagnostic label only; assigned after cluster classification.
    const char *sequence_edge_mode =
        "SEQUENCE_CYCLE_OK";

    // ------------------------------------------------------------------------
    // V9: Multi-loop edge sparsification.
    //
    // Temporal loop tracking above is allowed to run for every Keyframe, but
    // PoseGraph should not receive a nearly identical loop factor at every
    // frame.  Later loop edges are inserted only after BOTH the current and
    // historical trajectories have progressed enough since the last inserted
    // loop factor.
    // ------------------------------------------------------------------------
    std::size_t loop_edge_current_gap = 0;
    std::size_t loop_edge_historical_gap = 0;

    if (!first_online_loop_edge)
    {
        if (edge_current_keyframe_id <
            last_online_loop_current_keyframe_id_)
        {
            std::cout
                << "Keyframe loop decision"
                << " | current_kf=" << edge_current_keyframe_id
                << " | historical_kf="
                << edge_historical_keyframe_id
                << " | decision=REJECT"
                << " | reason=NON_MONOTONIC_CURRENT_KEYFRAME_ID"
                << " | last_loop_current_kf="
                << last_online_loop_current_keyframe_id_
                << std::endl;
            return;
        }

        loop_edge_current_gap =
            edge_current_keyframe_id -
            last_online_loop_current_keyframe_id_;

        loop_edge_historical_gap =
            edge_historical_keyframe_id >=
                    last_online_loop_historical_keyframe_id_
                ? edge_historical_keyframe_id -
                      last_online_loop_historical_keyframe_id_
                : last_online_loop_historical_keyframe_id_ -
                      edge_historical_keyframe_id;

        const bool current_spacing_ok =
            loop_edge_current_gap >=
            min_online_loop_edge_current_keyframe_spacing_;

        const bool historical_spacing_ok =
            loop_edge_historical_gap >=
            min_online_loop_edge_historical_keyframe_spacing_;

        const bool loop_edge_spacing_ok =
            current_spacing_ok && historical_spacing_ok;

        // V22.7 IMPORTANT: do NOT return on spacing yet.  SAME_CLUSTER cycle
        // consistency (or NEW_CLUSTER support) must run first.  Only evidence
        // that survives those checks may be visualized as yellow spacing-only
        // evidence.

        // --------------------------------------------------------------------
        // V2.4: post-loop continuity is cluster-aware.
        //
        // SAME_CLUSTER:
        //     Keep the existing strict V10 loop-to-loop cycle test.
        //
        // NEW_CLUSTER:
        //     The previous accepted loop is too remote to be a meaningful
        //     local cycle anchor.  Build independent temporal evidence from
        //     several locally consistent verified observations instead.
        // --------------------------------------------------------------------
        const bool same_loop_cluster =
            loop_edge_current_gap <=
                kPostLoopSameClusterMaxCurrentGap &&
            loop_edge_historical_gap <=
                kPostLoopSameClusterMaxHistoricalGap;

        sequence_edge_mode =
            same_loop_cluster
                ? "SEQUENCE_CYCLE_OK"
                : "NEW_CLUSTER_CONFIRMED";

        if (same_loop_cluster)
        {
            if (g_post_loop_new_cluster_track.valid)
            {
                std::cout
                    << "Post-loop new cluster track V2.4"
                    << " | action=RESET_RETURN_TO_SAME_CLUSTER"
                    << " | pending_support="
                    << g_post_loop_new_cluster_track.support
                    << std::endl;

                g_post_loop_new_cluster_track =
                    PostLoopClusterTrackState();
            }

            // --------------------------------------------------------------
            // Existing V10 same-cluster SE(3) cycle consistency.
            //
            //     Z1 * B  ~=  A * Z2
            //     E = (A * Z2)^-1 * (Z1 * B)
            // --------------------------------------------------------------
            const Keyframe *last_historical_keyframe =
                FindBackendKeyframeById(
                    last_online_loop_historical_keyframe_id_);
            const Keyframe *last_current_loop_keyframe =
                FindBackendKeyframeById(
                    last_online_loop_current_keyframe_id_);
            const Keyframe *new_historical_keyframe =
                FindBackendKeyframeById(
                    edge_historical_keyframe_id);
            const Keyframe *new_current_loop_keyframe =
                FindBackendKeyframeById(
                    edge_current_keyframe_id);

            if (last_historical_keyframe == nullptr ||
                last_current_loop_keyframe == nullptr ||
                new_historical_keyframe == nullptr ||
                new_current_loop_keyframe == nullptr)
            {
                std::cout
                    << "Keyframe loop decision"
                    << " | current_kf=" << edge_current_keyframe_id
                    << " | historical_kf="
                    << edge_historical_keyframe_id
                    << " | decision=REJECT"
                    << " | reason=LOOP_CYCLE_MISSING_KEYFRAME"
                    << std::endl;
                return;
            }

            const Eigen::Isometry3d A_historical =
                last_historical_keyframe->T_WL.inverse() *
                new_historical_keyframe->T_WL;

            const Eigen::Isometry3d B_current =
                last_current_loop_keyframe->T_WL.inverse() *
                new_current_loop_keyframe->T_WL;

            const Eigen::Isometry3d path_via_previous_loop =
                last_online_loop_measurement_ *
                B_current;

            const Eigen::Isometry3d path_via_new_loop =
                A_historical *
                edge_measurement;

            const Eigen::Isometry3d cycle_error =
                path_via_new_loop.inverse() *
                path_via_previous_loop;

            const double cycle_translation_error =
                cycle_error.translation().norm();

            const double cycle_rotation_error_deg =
                RelativeRotationDeg(
                    Eigen::Isometry3d::Identity(),
                    cycle_error);

            const bool cycle_finite =
                A_historical.matrix().allFinite() &&
                B_current.matrix().allFinite() &&
                path_via_previous_loop.matrix().allFinite() &&
                path_via_new_loop.matrix().allFinite() &&
                cycle_error.matrix().allFinite() &&
                std::isfinite(cycle_translation_error) &&
                std::isfinite(cycle_rotation_error_deg);

            const bool cycle_consistent =
                cycle_finite &&
                cycle_translation_error <=
                    online_loop_cycle_max_translation_error_ &&
                cycle_rotation_error_deg <=
                    online_loop_cycle_max_rotation_error_deg_;

            // V22.8 confidence split:
            //   strict cycle -> eligible to continue toward a PGO factor;
            //   relaxed cycle -> verified TRACK_ONLY visualization/evidence.
            // The relaxed branch NEVER reaches AddLoopEdge().
            const bool relaxed_track_cycle_consistent =
                cycle_finite &&
                cycle_translation_error <=
                    kLoopTrackRelaxedCycleMaxTranslationM &&
                cycle_rotation_error_deg <=
                    kLoopTrackRelaxedCycleMaxRotationDeg;

            std::cout
                << "Keyframe loop cluster V2.4"
                << " | classification=SAME_CLUSTER"
                << " | previous="
                << last_online_loop_historical_keyframe_id_
                << "->" << last_online_loop_current_keyframe_id_
                << " | new=" << edge_historical_keyframe_id
                << "->" << edge_current_keyframe_id
                << " | current_gap=" << loop_edge_current_gap
                << "/" << kPostLoopSameClusterMaxCurrentGap
                << " | historical_gap=" << loop_edge_historical_gap
                << "/" << kPostLoopSameClusterMaxHistoricalGap
                << std::endl;

            std::cout
                << "Keyframe loop cycle consistency"
                << " | previous="
                << last_online_loop_historical_keyframe_id_
                << "->" << last_online_loop_current_keyframe_id_
                << " | new=" << edge_historical_keyframe_id
                << "->" << edge_current_keyframe_id
                << " | translation_error="
                << cycle_translation_error << " m"
                << " | max_translation_error="
                << online_loop_cycle_max_translation_error_ << " m"
                << " | rotation_error="
                << cycle_rotation_error_deg << " deg"
                << " | max_rotation_error="
                << online_loop_cycle_max_rotation_error_deg_ << " deg"
                << " | consistent="
                << (cycle_consistent ? "true" : "false")
                << std::endl;

            std::cout
                << "FR_LOOP_TRACE"
                << " | stage=CYCLE"
                << " | cluster=SAME_CLUSTER"
                << " | previous="
                << last_online_loop_historical_keyframe_id_
                << "->" << last_online_loop_current_keyframe_id_
                << " | current=" << edge_historical_keyframe_id
                << "->" << edge_current_keyframe_id
                << " | dt=" << cycle_translation_error << " m"
                << " | dt_limit="
                << online_loop_cycle_max_translation_error_ << " m"
                << " | dR=" << cycle_rotation_error_deg << " deg"
                << " | dR_limit="
                << online_loop_cycle_max_rotation_error_deg_ << " deg"
                << " | decision="
                << (cycle_consistent ? "PASS" : "REJECT")
                << std::endl;

            if (!cycle_consistent)
            {
                if (relaxed_track_cycle_consistent)
                {
                    std::cout
                        << "Keyframe loop decision"
                        << " | current_kf=" << edge_current_keyframe_id
                        << " | historical_kf="
                        << edge_historical_keyframe_id
                        << " | decision=TRACK_ONLY"
                        << " | reason=LOOP_TRACK_CYCLE_RELAXED"
                        << " | cluster=SAME_CLUSTER"
                        << " | cycle_translation_error="
                        << cycle_translation_error << " m"
                        << " | strict_translation_limit="
                        << online_loop_cycle_max_translation_error_ << " m"
                        << " | relaxed_translation_limit="
                        << kLoopTrackRelaxedCycleMaxTranslationM << " m"
                        << " | cycle_rotation_error="
                        << cycle_rotation_error_deg << " deg"
                        << " | strict_rotation_limit="
                        << online_loop_cycle_max_rotation_error_deg_ << " deg"
                        << " | relaxed_rotation_limit="
                        << kLoopTrackRelaxedCycleMaxRotationDeg << " deg"
                        << " | action=TRACK_ONLY_NEVER_PGO"
                        << std::endl;

                    std::cout
                        << "FR_LOOP_TRACE"
                        << " | stage=CYCLE"
                        << " | cluster=SAME_CLUSTER"
                        << " | current=" << edge_historical_keyframe_id
                        << "->" << edge_current_keyframe_id
                        << " | decision=TRACK_ONLY_RELAXED"
                        << " | dt=" << cycle_translation_error << " m"
                        << " | dR=" << cycle_rotation_error_deg << " deg"
                        << std::endl;

                    fr_slam_debug::RecordLoopDecisionDebugEdge(
                        edge_historical_keyframe_id,
                        edge_current_keyframe_id,
                        fr_slam_debug::LoopDecisionDebugKind::TRACK_ONLY_CYCLE_RELAXED);
                    return;
                }

                std::cout
                    << "Keyframe loop decision"
                    << " | current_kf=" << edge_current_keyframe_id
                    << " | historical_kf="
                    << edge_historical_keyframe_id
                    << " | decision=TRACK_ONLY"
                    << " | reason=LOOP_CYCLE_INCONSISTENT"
                    << " | cluster=SAME_CLUSTER"
                    << " | cycle_translation_error="
                    << cycle_translation_error << " m"
                    << " | cycle_rotation_error="
                    << cycle_rotation_error_deg << " deg"
                    << std::endl;

                // Hard cycle disagreement remains orange.  It passed local
                // geometry/graph/temporal checks but is too inconsistent even
                // for the relaxed track-only gate.
                fr_slam_debug::RecordLoopDecisionDebugEdge(
                    edge_historical_keyframe_id,
                    edge_current_keyframe_id,
                    fr_slam_debug::LoopDecisionDebugKind::CYCLE_INCONSISTENT);
                return;
            }
        }
        else
        {
            // --------------------------------------------------------------
            // V2.4 new-cluster confirmation.
            //
            // Reaching here already means the candidate passed the existing
            // geometry, graph and online temporal-track gates.  We only add a
            // new condition: several consecutive candidates from this remote
            // revisit region must agree locally with one another.
            // --------------------------------------------------------------
            bool new_cluster_extends = false;
            std::size_t new_cluster_current_gap = 0;
            std::size_t new_cluster_historical_gap = 0;

            double new_cluster_translation_error =
                std::numeric_limits<double>::infinity();
            double new_cluster_rotation_error_deg =
                std::numeric_limits<double>::infinity();

            if (g_post_loop_new_cluster_track.valid)
            {
                new_cluster_current_gap =
                    edge_current_keyframe_id >=
                            g_post_loop_new_cluster_track
                                .last_current_keyframe_id
                        ? edge_current_keyframe_id -
                              g_post_loop_new_cluster_track
                                  .last_current_keyframe_id
                        : g_post_loop_new_cluster_track
                                  .last_current_keyframe_id -
                              edge_current_keyframe_id;

                new_cluster_historical_gap =
                    edge_historical_keyframe_id >=
                            g_post_loop_new_cluster_track
                                .last_historical_keyframe_id
                        ? edge_historical_keyframe_id -
                              g_post_loop_new_cluster_track
                                  .last_historical_keyframe_id
                        : g_post_loop_new_cluster_track
                                  .last_historical_keyframe_id -
                              edge_historical_keyframe_id;

                const Eigen::Isometry3d T_W_L_cluster_prediction =
                    g_post_loop_new_cluster_track.T_loop_correction *
                    current_keyframe.T_WL;

                const Eigen::Isometry3d new_cluster_local_error =
                    T_W_L_cluster_prediction.inverse() *
                    T_W_L_loop;

                if (T_W_L_cluster_prediction.matrix().allFinite() &&
                    new_cluster_local_error.matrix().allFinite())
                {
                    new_cluster_translation_error =
                        new_cluster_local_error.translation().norm();

                    new_cluster_rotation_error_deg =
                        RelativeRotationDeg(
                            Eigen::Isometry3d::Identity(),
                            new_cluster_local_error);
                }

                new_cluster_extends =
                    new_cluster_current_gap <=
                        kPostLoopNewClusterMaxCurrentGap &&
                    new_cluster_historical_gap <=
                        kPostLoopNewClusterMaxHistoricalGap &&
                    std::isfinite(new_cluster_translation_error) &&
                    std::isfinite(new_cluster_rotation_error_deg) &&
                    new_cluster_translation_error <=
                        kPostLoopNewClusterMaxTranslationError &&
                    new_cluster_rotation_error_deg <=
                        kPostLoopNewClusterMaxRotationErrorDeg;
            }

            if (!g_post_loop_new_cluster_track.valid ||
                !new_cluster_extends)
            {
                const std::size_t previous_pending_support =
                    g_post_loop_new_cluster_track.valid
                        ? g_post_loop_new_cluster_track.support
                        : 0;

                g_post_loop_new_cluster_track =
                    PostLoopClusterTrackState();
                g_post_loop_new_cluster_track.valid = true;
                g_post_loop_new_cluster_track.last_current_keyframe_id =
                    edge_current_keyframe_id;
                g_post_loop_new_cluster_track.last_historical_keyframe_id =
                    edge_historical_keyframe_id;
                g_post_loop_new_cluster_track.support = 1;
                g_post_loop_new_cluster_track.T_loop_correction =
                    T_loop_correction;

                std::cout
                    << "Post-loop new cluster track V2.4"
                    << " | action=START"
                    << " | current_kf=" << edge_current_keyframe_id
                    << " | historical_kf="
                    << edge_historical_keyframe_id
                    << " | previous_pending_support="
                    << previous_pending_support
                    << " | current_gap_from_last_edge="
                    << loop_edge_current_gap
                    << " | historical_gap_from_last_edge="
                    << loop_edge_historical_gap
                    << " | support=1/"
                    << kPostLoopNewClusterMinSupport
                    << std::endl;
            }
            else
            {
                ++g_post_loop_new_cluster_track.support;
                g_post_loop_new_cluster_track.last_current_keyframe_id =
                    edge_current_keyframe_id;
                g_post_loop_new_cluster_track.last_historical_keyframe_id =
                    edge_historical_keyframe_id;
                g_post_loop_new_cluster_track.T_loop_correction =
                    T_loop_correction;

                std::cout
                    << "Post-loop new cluster track V2.4"
                    << " | action=EXTEND"
                    << " | current_kf=" << edge_current_keyframe_id
                    << " | historical_kf="
                    << edge_historical_keyframe_id
                    << " | current_gap="
                    << new_cluster_current_gap
                    << "/" << kPostLoopNewClusterMaxCurrentGap
                    << " | historical_gap="
                    << new_cluster_historical_gap
                    << "/" << kPostLoopNewClusterMaxHistoricalGap
                    << " | local_dt="
                    << new_cluster_translation_error << " m"
                    << "/" << kPostLoopNewClusterMaxTranslationError
                    << " m"
                    << " | local_dR="
                    << new_cluster_rotation_error_deg << " deg"
                    << "/" << kPostLoopNewClusterMaxRotationErrorDeg
                    << " deg"
                    << " | support="
                    << g_post_loop_new_cluster_track.support
                    << "/" << kPostLoopNewClusterMinSupport
                    << std::endl;
            }

            std::cout
                << "Keyframe loop cluster V2.4"
                << " | classification=NEW_CLUSTER"
                << " | previous="
                << last_online_loop_historical_keyframe_id_
                << "->" << last_online_loop_current_keyframe_id_
                << " | new=" << edge_historical_keyframe_id
                << "->" << edge_current_keyframe_id
                << " | current_gap=" << loop_edge_current_gap
                << " | same_cluster_current_limit="
                << kPostLoopSameClusterMaxCurrentGap
                << " | historical_gap=" << loop_edge_historical_gap
                << " | same_cluster_historical_limit="
                << kPostLoopSameClusterMaxHistoricalGap
                << " | new_cluster_support="
                << g_post_loop_new_cluster_track.support
                << "/" << kPostLoopNewClusterMinSupport
                << std::endl;

            if (g_post_loop_new_cluster_track.support <
                kPostLoopNewClusterMinSupport)
            {
                std::cout
                    << "Keyframe loop decision"
                    << " | current_kf=" << edge_current_keyframe_id
                    << " | historical_kf="
                    << edge_historical_keyframe_id
                    << " | decision=TRACK_ONLY"
                    << " | reason=NEW_LOOP_CLUSTER_SUPPORT"
                    << " | support="
                    << g_post_loop_new_cluster_track.support
                    << "/" << kPostLoopNewClusterMinSupport
                    << std::endl;

                std::cout
                    << "FR_LOOP_TRACE"
                    << " | stage=CLUSTER"
                    << " | classification=NEW_CLUSTER"
                    << " | current=" << edge_historical_keyframe_id
                    << "->" << edge_current_keyframe_id
                    << " | decision=PENDING"
                    << " | reason=NEW_LOOP_CLUSTER_SUPPORT"
                    << " | support="
                    << g_post_loop_new_cluster_track.support
                    << "/" << kPostLoopNewClusterMinSupport
                    << std::endl;

                // V22.1: orange pending evidence.  This is deliberately kept
                // separate from yellow LOOP_EDGE_SPACING evidence because the
                // new cluster has not yet accumulated enough support.
                fr_slam_debug::RecordLoopDecisionDebugEdge(
                    edge_historical_keyframe_id,
                    edge_current_keyframe_id,
                    fr_slam_debug::LoopDecisionDebugKind::NEW_CLUSTER_PENDING);
                return;
            }

            std::cout
                << "Post-loop new cluster confirmed V2.4"
                << " | historical_kf="
                << edge_historical_keyframe_id
                << " | current_kf=" << edge_current_keyframe_id
                << " | support="
                << g_post_loop_new_cluster_track.support
                << "/" << kPostLoopNewClusterMinSupport
                << " | action=ALLOW_NEW_CLUSTER_EDGE"
                << std::endl;

            std::cout
                << "FR_LOOP_TRACE"
                << " | stage=CLUSTER"
                << " | classification=NEW_CLUSTER"
                << " | current=" << edge_historical_keyframe_id
                << "->" << edge_current_keyframe_id
                << " | decision=PASS"
                << " | support="
                << g_post_loop_new_cluster_track.support
                << "/" << kPostLoopNewClusterMinSupport
                << std::endl;
        }

        // --------------------------------------------------------------------
        // V22.7 postponed spacing gate.
        //
        // Reaching here means:
        //   * geometry PASS
        //   * graph PASS
        //   * temporal PASS
        //   * SAME_CLUSTER cycle PASS, or NEW_CLUSTER support confirmed
        //
        // Only now may spacing turn this observation into a yellow debug edge.
        // --------------------------------------------------------------------
        if (!loop_edge_spacing_ok)
        {
            std::cout
                << "Keyframe loop decision"
                << " | current_kf=" << edge_current_keyframe_id
                << " | historical_kf="
                << edge_historical_keyframe_id
                << " | segment_anchor_kf="
                << best_candidate->candidate_id
                << " | decision=TRACK_ONLY"
                << " | reason=LOOP_EDGE_SPACING_CYCLE_VERIFIED"
                << " | current_gap=" << loop_edge_current_gap
                << " | min_current_gap="
                << min_online_loop_edge_current_keyframe_spacing_
                << " | historical_gap="
                << loop_edge_historical_gap
                << " | min_historical_gap="
                << min_online_loop_edge_historical_keyframe_spacing_
                << " | support=" << online_loop_track_.support
                << std::endl;

            std::cout
                << "FR_LOOP_TRACE"
                << " | stage=DECISION"
                << " | current_kf=" << edge_current_keyframe_id
                << " | historical_kf="
                << edge_historical_keyframe_id
                << " | segment_anchor_kf="
                << best_candidate->candidate_id
                << " | decision=TRACK_ONLY"
                << " | reason=LOOP_EDGE_SPACING_CYCLE_VERIFIED"
                << " | current_gap=" << loop_edge_current_gap
                << " | historical_gap=" << loop_edge_historical_gap
                << " | support=" << online_loop_track_.support
                << std::endl;

            fr_slam_debug::RecordLoopDecisionDebugEdge(
                edge_historical_keyframe_id,
                edge_current_keyframe_id,
                fr_slam_debug::LoopDecisionDebugKind::TRACK_ONLY_SPACING);
            return;
        }
    }

    // Both Keyframes entered PoseGraph immediately when they were created.
    if (!pose_graph_.HasNode(edge_historical_keyframe_id) ||
        !pose_graph_.HasNode(edge_current_keyframe_id))
    {
        std::cerr
            << "ONLINE Keyframe PoseGraph loop edge failed"
            << " | reason=MISSING_KEYFRAME_VERTEX"
            << " | historical_kf=" << edge_historical_keyframe_id
            << " | current_kf=" << edge_current_keyframe_id
            << std::endl;
        return;
    }

    Eigen::Matrix<double, 6, 6> loop_information =
        Eigen::Matrix<double, 6, 6>::Identity();

    std::size_t loop_shadow_correspondences = 0;
    double loop_median_range =
        std::numeric_limits<double>::quiet_NaN();
    double loop_min_relative =
        std::numeric_limits<double>::quiet_NaN();

    pcl::PointCloud<LIDAR_POINT>::Ptr
        edge_historical_target_K;

    const bool edge_target_ok =
        BuildCandidateCenteredHistoricalTarget(
            edge_historical_keyframe_id,
            edge_historical_target_K,
            nullptr);

    const bool dynamic_loop_information =
        edge_target_ok &&
        BuildLoopShadowInformationFull6x6(
            current_keyframe.cloud,
            edge_historical_target_K,
            edge_measurement,
            loop_verifier_.GetConfig(),
            loop_information,
            loop_shadow_correspondences,
            loop_median_range,
            loop_min_relative);

    if (!pose_graph_.AddLoopEdge(
            edge_historical_keyframe_id,
            edge_current_keyframe_id,
            edge_measurement,
            loop_information))
    {
        std::cerr
            << "ONLINE Keyframe PoseGraph loop edge failed"
            << " | from_kf=" << edge_historical_keyframe_id
            << " | to_kf=" << edge_current_keyframe_id
            << std::endl;
        return;
    }

    double loop_max_offdiag = 0.0;
    double loop_max_tr_coupling = 0.0;

    ComputeLoopInformationStats(
        loop_information,
        loop_max_offdiag,
        loop_max_tr_coupling);

    // Gravity Guard V1 makes loop insertion transactional.  The loop edge is
    // temporarily present while g2o evaluates it, but the online loop-sequence
    // anchors are NOT committed yet.  If optimization violates the gravity
    // hard guard, this exact loop edge is removed again.
    std::cout
        << "FR_LOOP_TRACE"
        << " | stage=EDGE_STAGE"
        << " | mode=SEQUENCE"
        << " | from_kf=" << edge_historical_keyframe_id
        << " | to_kf=" << edge_current_keyframe_id
        << " | support=" << online_loop_track_.support
        << " | edge_mode=" << sequence_edge_mode
        << " | current_gap=" << loop_edge_current_gap
        << " | historical_gap=" << loop_edge_historical_gap
        << " | measurement_translation_norm="
        << edge_measurement.translation().norm() << " m"
        << std::endl;

    std::cout
        << "ONLINE Keyframe PoseGraph loop edge staged"
        << " | from_kf=" << edge_historical_keyframe_id
        << " | to_kf=" << edge_current_keyframe_id
        << " | geometry_target_submap=" << edge_historical_submap_id
        << " | current_submap=" << edge_current_submap_id
        << " | support=" << online_loop_track_.support
        << " | edge_mode="
        << sequence_edge_mode
        << " | information_mode="
        << (dynamic_loop_information
                ? "SHADOW_FULL_6X6"
                : "IDENTITY_FALLBACK")
        << " | base_diag=["
        << loop_information(0, 0) << " "
        << loop_information(1, 1) << " "
        << loop_information(2, 2) << " "
        << loop_information(3, 3) << " "
        << loop_information(4, 4) << " "
        << loop_information(5, 5)
        << "]"
        << " | max_offdiag="
        << loop_max_offdiag
        << " | max_tr_coupling="
        << loop_max_tr_coupling
        << " | shadow_corr="
        << loop_shadow_correspondences
        << " | median_range="
        << loop_median_range
        << " | min_relative="
        << loop_min_relative
        << " | current_edge_gap=" << loop_edge_current_gap
        << " | historical_edge_gap=" << loop_edge_historical_gap
        << " | measurement_translation_norm="
        << edge_measurement.translation().norm() << " m"
        << " | loop_edges_staged=" << pose_graph_.LoopEdgeCount()
        << std::endl;

    // ========================================================================
    // V8: the first real backend optimization.
    //
    // PoseGraph contains:
    //     VertexSE3 = Keyframe pose
    //     EdgeSE3   = KF odometry / KF loop measurement
    //
    // The optimizer writes corrected poses back ONLY to pose_graph_.
    // T_WL_ and the frontend KeyframeManager poses remain unchanged.
    // Therefore the green frontend path remains continuous while the PoseGraph
    // markers show the corrected backend trajectory.
    // ========================================================================
    const std::vector<PoseGraphNode>
        sequence_pose_snapshot =
            pose_graph_.GetNodes();

    PoseGraphOptimizationResult optimization_result;

    const std::chrono::steady_clock::time_point
        sequence_pgo_start =
            std::chrono::steady_clock::now();

    const bool sequence_pgo_ok =
        pose_graph_optimizer_.Optimize(
            pose_graph_,
            optimization_result);

    loop_timing.pose_graph_optimize_ms +=
        ElapsedMilliseconds(
            sequence_pgo_start,
            std::chrono::steady_clock::now());

    ++loop_timing.pose_graph_optimize_calls;

    if (!sequence_pgo_ok)
    {
        std::size_t restored_nodes = 0;

        for (const PoseGraphNode &node :
             sequence_pose_snapshot)
        {
            if (pose_graph_.SetNodePose(
                    node.id,
                    node.T_WK))
            {
                ++restored_nodes;
            }
        }

        const bool loop_edge_rolled_back =
            pose_graph_.RemoveLoopEdge(
                edge_historical_keyframe_id,
                edge_current_keyframe_id);

        online_loop_track_ = OnlineLoopTrack();
        g_post_loop_new_cluster_track =
            PostLoopClusterTrackState();

        std::cout
            << "FR_LOOP_TRACE"
            << " | stage=PGO"
            << " | mode=SEQUENCE"
            << " | decision=REJECT_ROLLBACK"
            << " | reason=OPTIMIZER_REJECTED"
            << " | from_kf=" << edge_historical_keyframe_id
            << " | to_kf=" << edge_current_keyframe_id
            << std::endl;

        std::cerr
            << "G2O Keyframe PoseGraph optimization rejected"
            << " | nodes=" << pose_graph_.NodeCount()
            << " | edges_after_rollback=" << pose_graph_.EdgeCount()
            << " | loop_rollback="
            << (loop_edge_rolled_back ? "true" : "false")
            << " | restored_nodes=" << restored_nodes
            << "/" << sequence_pose_snapshot.size()
            << " | gravity_guard_passed="
            << (optimization_result.gravity_guard_passed ? "true" : "false")
            << " | max_gravity_tilt="
            << optimization_result.max_gravity_tilt_error_deg << " deg"
            << " | worst_gravity_kf="
            << optimization_result.worst_gravity_keyframe_id
            << " | shape_guard_passed="
            << (optimization_result.trajectory_shape_guard_passed
                    ? "true"
                    : "false")
            << " | xy_pca_before="
            << optimization_result.xy_pca_ratio_before
            << " | xy_pca_after="
            << optimization_result.xy_pca_ratio_after
            << " | path_length_ratio="
            << optimization_result.path_length_ratio
            << " | action=RESTORE_POSES_REMOVE_LOOP_RESET_TRACK"
            << std::endl;
        return;
    }

    const bool sequence_update_guard_passed =
        std::isfinite(
            optimization_result.max_translation_update) &&
        std::isfinite(
            optimization_result.max_rotation_update_deg) &&
        optimization_result.max_translation_update <=
            online_loop_pgo_max_translation_update_ &&
        optimization_result.max_rotation_update_deg <=
            online_loop_pgo_max_rotation_update_deg_;

    if (!sequence_update_guard_passed)
    {
        std::size_t restored_nodes = 0;

        for (const PoseGraphNode &node :
             sequence_pose_snapshot)
        {
            if (pose_graph_.SetNodePose(
                    node.id,
                    node.T_WK))
            {
                ++restored_nodes;
            }
        }

        const bool loop_edge_rolled_back =
            pose_graph_.RemoveLoopEdge(
                edge_historical_keyframe_id,
                edge_current_keyframe_id);

        online_loop_track_ = OnlineLoopTrack();
        g_post_loop_new_cluster_track =
            PostLoopClusterTrackState();

        std::cout
            << "FR_LOOP_TRACE"
            << " | stage=PGO"
            << " | mode=SEQUENCE"
            << " | decision=REJECT_ROLLBACK"
            << " | reason=UPDATE_GUARD"
            << " | from_kf=" << edge_historical_keyframe_id
            << " | to_kf=" << edge_current_keyframe_id
            << " | max_update_dt=" << optimization_result.max_translation_update << " m"
            << " | max_update_dR=" << optimization_result.max_rotation_update_deg << " deg"
            << " | chi2_before=" << optimization_result.chi2_before
            << " | chi2_after=" << optimization_result.chi2_after
            << std::endl;

        std::cerr
            << "G2O Keyframe PoseGraph PGO update guard rejected"
            << " | max_translation_update="
            << optimization_result.max_translation_update
            << " m"
            << " | limit_translation="
            << online_loop_pgo_max_translation_update_
            << " m"
            << " | max_rotation_update="
            << optimization_result.max_rotation_update_deg
            << " deg"
            << " | limit_rotation="
            << online_loop_pgo_max_rotation_update_deg_
            << " deg"
            << " | restored_nodes=" << restored_nodes
            << "/" << sequence_pose_snapshot.size()
            << " | loop_rollback="
            << (loop_edge_rolled_back ? "true" : "false")
            << " | action=RESTORE_POSES_REMOVE_LOOP_RESET_TRACK"
            << std::endl;
        return;
    }

    // Optimization + gravity validation + V16 update guard succeeded.
    // Only now is this loop considered a real backend constraint for future
    // sequence/cycle logic.
    loop_timing.optimization_accepted = true;
    loop_timing.loop_edge_accepted = true;

    has_last_online_loop_edge_ = true;

    // V22.1: if the same pair was previously visible only as debug evidence,
    // remove that debug record now that it is a real red PoseGraph factor.
    fr_slam_debug::RemoveLoopDecisionDebugEdge(
        edge_historical_keyframe_id,
        edge_current_keyframe_id);

    last_online_loop_current_keyframe_id_ =
        edge_current_keyframe_id;
    last_online_loop_historical_keyframe_id_ =
        edge_historical_keyframe_id;
    last_online_loop_measurement_ =
        edge_measurement;

    // V2.4: a newly accepted edge is now the local cluster anchor.
    g_post_loop_new_cluster_track =
        PostLoopClusterTrackState();

    std::cout
        << "ONLINE Keyframe PoseGraph loop edge accepted"
        << " | from_kf=" << edge_historical_keyframe_id
        << " | to_kf=" << edge_current_keyframe_id
        << " | edge_mode="
        << sequence_edge_mode
        << " | loop_edges=" << pose_graph_.LoopEdgeCount()
        << std::endl;

    std::cout
        << "FR_LOOP_TRACE"
        << " | stage=PGO"
        << " | mode=SEQUENCE"
        << " | decision=ACCEPT"
        << " | from_kf=" << edge_historical_keyframe_id
        << " | to_kf=" << edge_current_keyframe_id
        << " | edge_mode=" << sequence_edge_mode
        << " | chi2_before=" << optimization_result.chi2_before
        << " | chi2_after=" << optimization_result.chi2_after
        << " | max_update_dt=" << optimization_result.max_translation_update << " m"
        << " | max_update_dR=" << optimization_result.max_rotation_update_deg << " deg"
        << " | gravity_guard="
        << (optimization_result.gravity_guard_passed ? "PASS" : "FAIL")
        << " | loop_edges=" << pose_graph_.LoopEdgeCount()
        << std::endl;

    std::cout
        << "G2O Keyframe PoseGraph optimized"
        << " | iterations=" << optimization_result.iterations
        << " | chi2_before=" << optimization_result.chi2_before
        << " | chi2_after=" << optimization_result.chi2_after
        << " | optimized_nodes=" << optimization_result.optimized_nodes
        << " | odom_edges=" << optimization_result.odometry_edges
        << " | loop_edges=" << optimization_result.loop_edges
        << " | gravity_edges=" << optimization_result.gravity_edges
        << " | max_translation_update="
        << optimization_result.max_translation_update << " m"
        << " | max_rotation_update="
        << optimization_result.max_rotation_update_deg << " deg"
        << " | max_droll="
        << optimization_result.max_roll_update_deg << " deg"
        << " | max_dpitch="
        << optimization_result.max_pitch_update_deg << " deg"
        << " | max_dyaw="
        << optimization_result.max_yaw_update_deg << " deg"
        << " | mean_gravity_tilt="
        << optimization_result.mean_gravity_tilt_error_deg << " deg"
        << " | max_gravity_tilt="
        << optimization_result.max_gravity_tilt_error_deg << " deg"
        << " | xy_pca_before="
        << optimization_result.xy_pca_ratio_before
        << " | xy_pca_after="
        << optimization_result.xy_pca_ratio_after
        << " | path_length_ratio="
        << optimization_result.path_length_ratio
        << " | gravity_guard=PASS"
        << " | shape_guard=PASS"
        << std::endl;

    // ------------------------------------------------------------------------
    // Connect the corrected backend/map frame to the still-continuous frontend
    // odometry frame.  Use the current loop Keyframe as an exact common anchor:
    //
    //     T_map_odom = T_WK(current, optimized) * T_WL(current, raw)^-1
    //
    // IMPORTANT: this does NOT overwrite Keyframe::T_WL or the live frontend
    // pose.  It only creates a bridge for corrected real-time output.
    // ------------------------------------------------------------------------
    const std::chrono::steady_clock::time_point
        sequence_map_odom_start =
            std::chrono::steady_clock::now();

    const bool sequence_map_odom_ok =
        UpdateMapOdomCorrection(
            current_keyframe.id);

    loop_timing.map_odom_ms +=
        ElapsedMilliseconds(
            sequence_map_odom_start,
            std::chrono::steady_clock::now());

    if (!sequence_map_odom_ok)
    {
        std::cerr
            << "Map->odom correction update failed"
            << " | anchor_kf=" << current_keyframe.id
            << std::endl;
    }

    // ------------------------------------------------------------------------
    // The PoseGraph optimization changed only Keyframe poses.  Rebuild the
    // global point-cloud snapshots now so RViz can compare the SAME Keyframe
    // clouds before and after backend correction.
    //
    // This is intentionally executed only after successful G2O optimization,
    // not on every LiDAR frame.
    // ------------------------------------------------------------------------
    const std::chrono::steady_clock::time_point
        sequence_global_map_start =
            std::chrono::steady_clock::now();

    const bool global_map_rebuilt =
        RebuildGlobalMapSnapshots();

    loop_timing.global_map_rebuild_ms +=
        ElapsedMilliseconds(
            sequence_global_map_start,
            std::chrono::steady_clock::now());

    if (!global_map_rebuilt)
    {
        std::cerr
            << "Global map snapshot rebuild failed"
            << " | keyframes=" << backend_keyframes_.size()
            << " | graph_nodes=" << pose_graph_.NodeCount()
            << std::endl;
    }
    else
    {
        const std::chrono::steady_clock::time_point
            sequence_refinement_start =
                std::chrono::steady_clock::now();

        const bool refinement_ok =
            RebuildPostPgoRefinedMap();

        loop_timing.refinement_ms +=
            ElapsedMilliseconds(
                sequence_refinement_start,
                std::chrono::steady_clock::now());

        if (!refinement_ok)
        {
            // Refinement is an OPTIONAL backend product.  A failure here must not
            // invalidate the already-successful PoseGraph optimization or the raw
            // / optimized global-map snapshots.
            std::cerr
                << "Post-PGO refined map rebuild skipped/failed"
                << " | global_revision=" << global_map_revision_
                << " | keyframes=" << backend_keyframes_.size()
                << std::endl;
        }
    }
}

// ============================================================================
// UpdateIncrementalGlobalMaps()
//
// Keyframe clouds + poses remain the authoritative backend data.  The global
// point clouds are derived caches split into small BACKEND-ONLY Keyframe blocks.
//
// This is intentionally independent from frontend SubmapManager lifecycle:
//
//     frontend Submap -> Scan-to-LocalMap / loop geometry
//     backend block   -> visualization/export cache only
//
// New Keyframe:
//     normally dirties one raw block + one optimized block.
//
// PoseGraph optimization:
//     compares cached T_WK against the new graph solution and rebuilds only
//     blocks containing Keyframes whose pose changed beyond the dirty gate.
//
// Per-block VoxelGrid is applied during block rebuild.  The published global
// cloud is assembled from already-filtered blocks, so there is no million-point
// global VoxelGrid pass on every update.
// ============================================================================
