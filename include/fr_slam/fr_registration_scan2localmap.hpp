#pragma once

#include <cstddef>
#include <limits>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include <pcl/point_cloud.h>

#include "fr_slam/fr_keyframe.hpp"
#include "fr_slam/fr_incremental_global_map.hpp"
#include "fr_slam/fr_keyframe_detector.hpp"
#include "fr_slam/fr_keyframe_manager.hpp"
#include "fr_slam/fr_lidar_registration.hpp"
#include "fr_slam/fr_lidar_registration_config.hpp"
#include "fr_slam/fr_loop_detector.hpp"
#include "fr_slam/fr_loop_verifier.hpp"
#include "fr_slam/fr_point_types.hpp"
#include "fr_slam/fr_pose_graph.hpp"
#include "fr_slam/fr_pose_graph_optimizer.hpp"
#include "fr_slam/fr_submap_manager.hpp"

// ============================================================================
// Scan-to-Active-Submap frontend + KEYFRAME PoseGraph backend.
//
// Responsibilities are deliberately separated:
//
//     Frontend tracking target:
//         Active / Previous+Active Submap
//
//     Loop place recognition:
//         Current Keyframe -> historical Keyframe Scan Context DB
//
//     Loop geometry verification:
//         Current Keyframe -> candidate-centered historical FINISHED Submap
//
//     Backend optimization state:
//         one PoseGraph vertex per Keyframe
//
// Submaps are NOT PoseGraph vertices in V6. They are auxiliary geometry
// containers for frontend registration and loop ICP verification.
// ============================================================================
class RegistrationScan2LocalMap
{
public:
    RegistrationScan2LocalMap(
        const LidarRegistrationConfig &registration_config,
        const LocalMapConfig &local_map_config);

    bool AddFrame(
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &cloud_lidar,
        double timestamp,
        Eigen::Isometry3d &T_WL,
        LidarRegistrationResult &registration_result,
        const Eigen::Quaterniond *imu_relative_rotation = nullptr);

    Eigen::Isometry3d GetPose() const;

    pcl::PointCloud<LIDAR_POINT>::ConstPtr
    GetLocalMap() const;

    // ------------------------------------------------------------------------
    // Backend global-map snapshots.
    //
    // /raw_keyframe_map:
    //     same Keyframe clouds transformed by immutable frontend T_WL poses.
    //
    // /optimized_map:
    //     same Keyframe clouds transformed by current PoseGraph T_WK poses.
    //
    // Both are maintained incrementally from the same Keyframe source data.
    // New Keyframes normally rebuild one small backend block; PoseGraph updates
    // rebuild only dirty blocks whose cached optimized pose changed.
    // ------------------------------------------------------------------------
    pcl::PointCloud<LIDAR_POINT>::ConstPtr
    GetRawKeyframeMap() const;

    pcl::PointCloud<LIDAR_POINT>::ConstPtr
    GetOptimizedMap() const;

    // Post-PGO local geometric refinement result.  This map is built from the
    // SAME Keyframe clouds, but selected revisit Keyframes may use a small
    // refined pose obtained by point-to-plane registration against frozen
    // historical geometry.  PoseGraphNode::T_WK is never overwritten.
    pcl::PointCloud<LIDAR_POINT>::ConstPtr
    GetRefinedMap() const;

    // Debug snapshots for the latest ACCEPTED Post-PGO refinement window.
    // All three clouds live in the backend/map frame and refer to the same
    // local optimization group:
    //
    //   historical_target : frozen historical LocalMap used as ICP target
    //   current_before    : current-window Keyframes at pure G2O poses
    //   current_after     : same current-window Keyframes at refined poses
    //
    // They are visualization-only and are never fed back into frontend or PGO.
    pcl::PointCloud<LIDAR_POINT>::ConstPtr
    GetRefinementHistoricalTarget() const;

    pcl::PointCloud<LIDAR_POINT>::ConstPtr
    GetRefinementCurrentBefore() const;

    pcl::PointCloud<LIDAR_POINT>::ConstPtr
    GetRefinementCurrentAfter() const;

    // Equals the GlobalMapRevision() for which the three debug snapshots were
    // created. Zero means no accepted refinement debug group is available.
    std::size_t RefinementDebugRevision() const;

    // Equals the GlobalMapRevision() from which /refined_map was generated.
    // Zero means no refined snapshot is currently available.
    std::size_t RefinedMapRevision() const;

    // Incremented whenever the backend incremental-map cache changes.  The ROS
    // wrapper therefore publishes on new Keyframes / backend corrections, but
    // still skips ordinary non-Keyframe LiDAR scans.
    std::size_t GlobalMapRevision() const;

    // ------------------------------------------------------------------------
    // Backend map-frame <-> frontend odom-frame bridge.
    //
    // The live frontend keeps producing the continuous raw pose T_WL in its
    // odometry/world frame.  After G2O, PoseGraphNode::T_WK lives in the
    // corrected backend/map frame.  For one anchor Keyframe k:
    //
    //     T_map_odom = T_WK(k) * T_WL(k)^-1
    //
    // and any later raw frontend pose can be represented in the corrected
    // frame without overwriting the frontend state:
    //
    //     T_map_L = T_map_odom * T_WL(raw)
    //
    // Before the first backend correction T_map_odom is Identity.
    // ------------------------------------------------------------------------
    bool HasMapOdomCorrection() const;

    Eigen::Isometry3d GetMapOdomCorrection() const;

    Eigen::Isometry3d GetCorrectedPose(
        const Eigen::Isometry3d &T_odom_lidar) const;

    std::size_t MapOdomRevision() const;

    std::size_t LocalMapFrameCount() const;
    std::size_t LocalMapPointCount() const;

    std::size_t KeyframeCount() const;
    const std::vector<Keyframe> &GetKeyframes() const;

    std::size_t SubmapCount() const;
    const Submap *GetActiveSubmap() const;
    const Submap *GetPreviousSubmap() const;

    // ------------------------------------------------------------------------
    // Backend PoseGraph diagnostics / access.
    //
    // V6 graph semantics:
    //     Node id == Keyframe id
    //     Node pose == T_WK
    // ------------------------------------------------------------------------
    const PoseGraph &GetPoseGraph() const;
    std::size_t PoseGraphNodeCount() const;
    std::size_t PoseGraphEdgeCount() const;
    std::size_t PoseGraphOdometryEdgeCount() const;
    std::size_t PoseGraphLoopEdgeCount() const;

    void Reset();

private:
    // Every committed Keyframe immediately enters the backend graph.
    //
    //     KF_i -> Vertex i
    //
    // and for i > 0:
    //
    //     Z_(i-1,i) = T_WK_(i-1)^-1 * T_WK_i
    //
    // This is independent from Submap finishing.
    bool AddKeyframeToPoseGraph(
        const Keyframe &keyframe);

    const Submap *FindSubmapById(
        std::size_t submap_id) const;

    const Keyframe *FindKeyframeById(
        std::size_t keyframe_id) const;

    // Candidate Keyframes can belong to two Submaps because of overlap. Choose
    // a FINISHED Submap that contains the candidate and places it closest to
    // the center of that Submap's keyframe window.
    const Submap *FindBestFinishedSubmapForKeyframe(
        std::size_t keyframe_id) const;

    // Called immediately after a new Keyframe is committed.
    //
    // Candidate retrieval is Keyframe-based. Historical Submap is used only as
    // a geometry target. The accepted loop is finally converted to:
    //
    //     T_Khistorical_Kcurrent
    //
    // and added between Keyframe PoseGraph vertices.
    void DetectAndVerifyLoopFromKeyframe(
        const Keyframe &current_keyframe,
        std::size_t current_submap_id);

    // Incrementally update backend global-map caches from Keyframe source data.
    //
    // Keyframes are grouped into backend-only fixed-size blocks.  This mapping
    // cache is intentionally independent from frontend Active/Finished Submap
    // lifecycle.  New Keyframes dirty only one block; PoseGraph correction
    // dirties only blocks whose cached T_WK changed beyond a small threshold.
    bool UpdateIncrementalGlobalMaps(
        const char *reason,
        bool clear_refined_overrides);

    // Compatibility wrapper used after successful main PoseGraph optimization.
    // It now performs dirty-block updates rather than a full-map rebuild.
    bool RebuildGlobalMapSnapshots();

    // Post-PGO refinement V3: local-window multi-pose optimization.
    //
    // V1 was too free: each revisit Keyframe was refined independently.
    // V2 was too rigid: the whole revisit window shared one Delta_T_map.
    //
    // V3 keeps one SE(3) state per current-window Keyframe.  For each Keyframe
    // we first run point-to-plane registration against ONE frozen historical
    // LocalMap and convert the successful result into a geometry anchor edge.
    // Consecutive current Keyframes are simultaneously tied by their frozen
    // G2O relative poses.  A small temporary PoseGraph therefore optimizes:
    //
    //     local odometry consistency + historical-map geometry anchors
    //
    // The main PoseGraphNode::T_WK is never overwritten.  Only /refined_map
    // uses the temporary local-window solution.
    bool RebuildPostPgoRefinedMap();

    // Update the constant left-multiplicative bridge between the continuous
    // frontend odometry frame and the latest corrected backend map frame.
    // This is called only after a successful PoseGraph optimization.
    bool UpdateMapOdomCorrection(
        std::size_t anchor_keyframe_id);

    struct OnlineLoopTrack
    {
        bool valid = false;

        std::size_t last_current_submap_id = 0;
        std::size_t last_current_keyframe_id = 0;
        std::size_t last_historical_keyframe_id = 0;
        std::size_t last_historical_submap_id = 0;

        std::size_t support = 0;

        // Historical keyframe progression direction:
        //
        //   0  -> direction not locked yet
        //  +1  -> historical keyframe ids should mainly increase
        //  -1  -> historical keyframe ids should mainly decrease
        //
        // Small +/- jitter is tolerated by
        // online_loop_historical_backtrack_tolerance_.
        int historical_direction = 0;

        // Left-multiplicative world correction implied by the loop:
        //
        //     T_WK_loop ~= T_loop_correction * T_WK_frontend
        //
        // It is independent from current Submap transitions.
        Eigen::Isometry3d T_loop_correction =
            Eigen::Isometry3d::Identity();
    };

private:
    LidarRegistration registration_;

    // Frontend / geometry organization only.
    SubmapManager submap_manager_;

    // Backend state: ONE KEYFRAME -> ONE PoseGraphNode.
    PoseGraph pose_graph_;

    // V8: g2o optimizer. It updates only PoseGraph node estimates.
    // The live Scan-to-LocalMap frontend T_WL_ remains untouched.
    PoseGraphOptimizer pose_graph_optimizer_;

    LoopDetector loop_detector_;
    LoopVerifier loop_verifier_;

    std::size_t max_loop_candidates_to_verify_ = 3;

    // Local-neighborhood exclusion after a candidate KF is associated with a
    // historical Submap. Submaps are used only to decide whether the candidate
    // is merely a nearby trajectory neighbor.
    std::size_t min_loop_submap_separation_ = 5;

    OnlineLoopTrack online_loop_track_;
    std::size_t online_loop_min_support_ = 3;
    std::size_t online_loop_max_current_keyframe_gap_ = 3;
    std::size_t online_loop_max_current_submap_gap_ = 1;
    // Broad neighborhood gate kept from V6.
    std::size_t online_loop_max_historical_keyframe_gap_ = 15;
    std::size_t online_loop_max_historical_submap_gap_ = 1;

    // V7: explicit Historical-KF sequence consistency.
    //
    // If current keyframes are consecutive, the matched historical keyframe
    // should move only a small number of keyframes along the historical
    // trajectory.  The allowed step scales with current_gap so one missed
    // loop candidate does not break a good track.
    std::size_t online_loop_max_historical_progression_step_ = 6;

    // Once a historical direction is locked, allow a tiny amount of
    // Scan-Context / ICP anchor jitter in the opposite direction.
    std::size_t online_loop_historical_backtrack_tolerance_ = 1;

    double online_loop_track_translation_error_ = 2.0;
    double online_loop_track_rotation_error_deg_ = 10.0;

    // Candidate-anchored ICP should not need a huge correction.
    double max_loop_icp_correction_translation_ = 5.0;
    double max_loop_icp_correction_rotation_deg_ = 45.0;

    // Keyframe graph sanity gate. Compare the world pose of the current
    // Keyframe implied by the loop against its frontend T_WL estimate.
    double max_loop_graph_correction_translation_ = 20.0;
    double max_loop_graph_correction_rotation_deg_ = 45.0;

    // ------------------------------------------------------------------------
    // Multi-loop edge insertion state.
    //
    // Loop detection / ICP verification / temporal tracking keeps running for
    // every new Keyframe in a revisited area.  We only sparsify the factors
    // that are actually inserted into PoseGraph.
    //
    // Example:
    //     KF501 -> Hist4     ADD
    //     KF502 -> Hist4     TRACK_ONLY
    //     KF503 -> Hist5     TRACK_ONLY
    //     KF505 -> Hist6     ADD
    //
    // This replaces the old hard rule "one loop edge per current Submap".
    // ------------------------------------------------------------------------
    bool has_last_online_loop_edge_ = false;

    std::size_t last_online_loop_current_keyframe_id_ =
        std::numeric_limits<std::size_t>::max();

    std::size_t last_online_loop_historical_keyframe_id_ =
        std::numeric_limits<std::size_t>::max();

    // Keep loop factors informative but not overly correlated.  With the
    // current ~0.5 m Keyframe translation threshold, 4 current KFs is roughly
    // a couple of metres in translation-dominated regions.
    std::size_t min_online_loop_edge_current_keyframe_spacing_ = 4;

    // Require the historical anchor to make visible progress as well, so many
    // current KFs are not repeatedly constrained to the exact same old KF.
    std::size_t min_online_loop_edge_historical_keyframe_spacing_ = 2;

    // ------------------------------------------------------------------------
    // V11: first-loop BATCH confirmation.
    //
    // A single loop edge can over-constrain a long odometry chain and let g2o
    // distribute a large XY/yaw correction in an unrealistic way.  Therefore
    // the FIRST backend optimization is delayed until several independent,
    // spatially separated loop constraints all support the same world-frame
    // correction.
    // ------------------------------------------------------------------------
    // The first anchor may be collected before this support is reached.
    // This threshold is checked only when deciding whether the whole first
    // loop batch is mature enough to be staged into PoseGraph / g2o.
    std::size_t online_loop_first_edge_min_support_ = 4;

    // Strict geometry gate for the FIRST anchor only.
    double online_loop_first_edge_min_overlap_ = 0.95;
    double online_loop_first_edge_max_rmse_ = 0.35;
    double online_loop_first_edge_max_icp_translation_ = 0.75;
    double online_loop_first_edge_max_icp_rotation_deg_ = 3.0;

    // Follow-up members are mainly judged by whether they imply the same
    // world correction as the anchor.  Keep only a wider ICP safety gate here.
    double online_loop_first_batch_followup_max_icp_translation_ = 1.0;
    double online_loop_first_batch_followup_max_icp_rotation_deg_ = 15.0;

    struct PendingLoopConstraint
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        std::size_t historical_keyframe_id =
            std::numeric_limits<std::size_t>::max();
        std::size_t current_keyframe_id =
            std::numeric_limits<std::size_t>::max();
        std::size_t historical_submap_id =
            std::numeric_limits<std::size_t>::max();
        std::size_t current_submap_id =
            std::numeric_limits<std::size_t>::max();

        // Loop measurement inserted into PoseGraph:
        //     Z_hc = T_Kh_Kc
        Eigen::Isometry3d T_historical_current =
            Eigen::Isometry3d::Identity();

        // Left-multiplicative world correction implied by this loop:
        //     C = T_WL_loop * T_WL_frontend^-1
        // Multiple first-loop constraints must agree on C before the batch is
        // allowed to enter g2o.
        Eigen::Isometry3d T_loop_correction =
            Eigen::Isometry3d::Identity();

        double overlap = 0.0;
        double rmse = std::numeric_limits<double>::infinity();
        double correction_translation =
            std::numeric_limits<double>::infinity();
        double correction_rotation_deg =
            std::numeric_limits<double>::infinity();
    };

    std::vector<
        PendingLoopConstraint,
        Eigen::aligned_allocator<PendingLoopConstraint>>
        pending_first_loop_batch_;

    // First optimization requires several independent anchors.  These gates
    // are intentionally looser than later sparse-loop spacing because they are
    // used only to collect a short, locally consistent loop sequence.
    // Two independent loop anchors are sufficient for the first optimization.
    // Candidate collection and batch commit are intentionally separate:
    //   - an anchor can be stored at support=3,
    //   - the batch is committed only when support>=4 and edges>=2.
    std::size_t online_loop_first_batch_min_edges_ = 2;
    std::size_t online_loop_first_batch_current_spacing_ = 1;
    std::size_t online_loop_first_batch_historical_spacing_ = 1;

    // With only two edges in the first batch, require tighter agreement of the
    // left-multiplicative world correction implied by both loop constraints.
    double online_loop_first_batch_max_translation_error_ = 0.6;
    double online_loop_first_batch_max_rotation_error_deg_ = 2.5;

    // Measurement of the last loop factor that was ACTUALLY inserted into
    // PoseGraph.  For the next sparse loop factor we compare two paths:
    //
    //     Z_prev * B_current
    //
    // against
    //
    //     A_historical * Z_new
    //
    // and require the resulting cycle error to remain small.
    Eigen::Isometry3d last_online_loop_measurement_ =
        Eigen::Isometry3d::Identity();

    double online_loop_cycle_max_translation_error_ = 0.50;
    double online_loop_cycle_max_rotation_error_deg_ = 3.0;

    PreparedLidarTarget prepared_tracking_target_;

    KeyframeDetector keyframe_detector_;
    KeyframeManager keyframe_manager_;

    // ------------------------------------------------------------------------
    // Incremental backend map cache.
    //
    // Keyframe cloud + pose remain the source of truth.  This class only caches
    // voxelized backend blocks for visualization/export and is never fed back
    // into frontend Scan-to-LocalMap registration.
    //
    // The block grouping is BACKEND-ONLY and deliberately independent from the
    // frontend SubmapManager.  This avoids binding global-map maintenance to
    // Active/Finished Submap lifecycle.
    // ------------------------------------------------------------------------
    IncrementalGlobalMap incremental_global_map_{
        10,     // Keyframes per backend map block.
        0.30f,  // Per-block VoxelGrid leaf size.
        0.01,   // Dirty translation threshold: 1 cm.
        0.05};  // Dirty rotation threshold: 0.05 deg.

    // Latest accepted local-refinement debug group.  These clouds deliberately
    // remain separate so RViz can reveal whether a visible discontinuity comes
    // from the historical target, the current revisit window, or from geometry
    // outside the selected refinement window.
    pcl::PointCloud<LIDAR_POINT>::Ptr refinement_historical_target_debug_ =
        pcl::make_shared<pcl::PointCloud<LIDAR_POINT>>();

    pcl::PointCloud<LIDAR_POINT>::Ptr refinement_current_before_debug_ =
        pcl::make_shared<pcl::PointCloud<LIDAR_POINT>>();

    pcl::PointCloud<LIDAR_POINT>::Ptr refinement_current_after_debug_ =
        pcl::make_shared<pcl::PointCloud<LIDAR_POINT>>();

    std::size_t refinement_debug_revision_ = 0;

    // Pose array from the latest refinement pass, in the same order as the
    // KeyframeManager snapshot used for that pass.  Unrefined entries simply
    // contain the original graph pose.
    std::vector<
        Eigen::Isometry3d,
        Eigen::aligned_allocator<Eigen::Isometry3d>>
        refined_keyframe_poses_;

    std::vector<bool> refined_keyframe_pose_was_adjusted_;

    std::size_t global_map_revision_ = 0;

    // Source backend-map revision used to create the current refined layer.
    std::size_t refined_map_revision_ = 0;

    // Post-PGO local-window refinement V3.
    //
    // Current side: one variable SE(3) pose per Keyframe in a short revisit
    // window ending at the accepted loop endpoint.
    // Historical side: one frozen LocalMap around the historical loop endpoint.
    std::size_t refinement_local_window_ = 15;
    std::size_t refinement_historical_keyframe_window_ = 20;
    double refinement_historical_radius_ = 4.0;
    std::size_t refinement_min_historical_keyframe_gap_ = 30;

    std::size_t refinement_min_current_keyframes_ = 3;
    std::size_t refinement_min_historical_keyframes_ = 3;
    std::size_t refinement_max_historical_keyframes_ = 16;

    // Frozen historical LocalMap preparation.
    float refinement_target_voxel_leaf_size_ = 0.25f;
    std::size_t refinement_min_target_points_ = 1000;

    // Per-Keyframe point-to-plane geometry-anchor gate.  These ICP results are
    // NOT written directly into the final map; they become soft absolute-pose
    // constraints in the temporary local PoseGraph.
    std::size_t refinement_geometry_min_correspondences_ = 500;
    double refinement_geometry_max_rmse_ = 0.25;
    double refinement_geometry_max_translation_correction_ = 0.40;
    double refinement_geometry_max_rotation_correction_deg_ = 3.0;
    std::size_t refinement_min_geometry_anchors_ = 3;

    // Temporary local PoseGraph weights.  Odometry is deliberately stronger
    // than each individual geometry anchor so the revisit trajectory can bend
    // gradually but cannot make frame-to-frame jumps.  Geometry edges are Loop
    // type edges and therefore inherit the existing Huber kernel.
    double refinement_local_odom_information_scale_ = 10.0;
    double refinement_geometry_information_scale_ = 1.0;

    // Final local-window safety gate relative to the frozen G2O solution.
    // If any optimized Keyframe exceeds this small-correction envelope, the
    // whole window is rejected and keeps the pure G2O poses.
    double refinement_window_max_translation_update_ = 0.30;
    double refinement_window_max_rotation_update_deg_ = 2.0;

    // Latest backend correction that maps a raw frontend pose into the
    // corrected PoseGraph/map frame:
    //
    //     T_map_L = T_map_odom_ * T_odom_L
    //
    // It stays constant between backend optimizations, so the frontend remains
    // continuous and every new scan can be represented consistently in the
    // corrected map frame.
    Eigen::Isometry3d T_map_odom_ =
        Eigen::Isometry3d::Identity();

    bool has_map_odom_correction_ = false;
    std::size_t map_odom_revision_ = 0;
    std::size_t map_odom_anchor_keyframe_id_ =
        std::numeric_limits<std::size_t>::max();

    Eigen::Isometry3d T_WL_ =
        Eigen::Isometry3d::Identity();

    Eigen::Isometry3d last_relative_transform_ =
        Eigen::Isometry3d::Identity();

    std::size_t consecutive_rejected_frames_ = 0;
    std::size_t max_recovery_prediction_steps_ = 5;

    bool initialized_ = false;

    double max_accepted_rmse_ = 0.15;
    std::size_t min_accepted_correspondences_ = 1000;
};
