#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include <pcl/point_cloud.h>

#include "fr_slam/mapping/fr_keyframe.hpp"
#include "fr_slam/mapping/fr_incremental_global_map.hpp"
#include "fr_slam/mapping/fr_keyframe_detector.hpp"
#include "fr_slam/mapping/fr_keyframe_manager.hpp"
#include "fr_slam/lidar/fr_lidar_registration.hpp"
#include "fr_slam/lidar/fr_lidar_registration_config.hpp"
#include "fr_slam/loop/fr_loop_detector.hpp"
#include "fr_slam/loop/fr_loop_verifier.hpp"
#include "fr_slam/common/fr_point_types.hpp"
#include "fr_slam/backend/fr_pose_graph.hpp"
#include "fr_slam/backend/fr_pose_graph_optimizer.hpp"
#include "fr_slam/mapping/fr_submap_manager.hpp"

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
        const LocalMapConfig &local_map_config,
        const LoopDetectorConfig &loop_detector_config =
            LoopDetectorConfig());

    ~RegistrationScan2LocalMap();

    bool AddFrame(
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &cloud_lidar,
        double timestamp,
        Eigen::Isometry3d &T_WL,
        LidarRegistrationResult &registration_result,
        const Eigen::Quaterniond *imu_relative_rotation = nullptr);

    Eigen::Isometry3d GetPose() const;

    pcl::PointCloud<LIDAR_POINT>::ConstPtr
    GetLocalMap() const;

    struct BackendMapSnapshot
    {
        pcl::PointCloud<LIDAR_POINT>::ConstPtr raw_map;
        pcl::PointCloud<LIDAR_POINT>::ConstPtr optimized_map;
        pcl::PointCloud<LIDAR_POINT>::ConstPtr refined_map;

        pcl::PointCloud<LIDAR_POINT>::ConstPtr refinement_historical_target;
        pcl::PointCloud<LIDAR_POINT>::ConstPtr refinement_current_before;
        pcl::PointCloud<LIDAR_POINT>::ConstPtr refinement_current_after;

        std::size_t global_revision = 0;
        std::size_t refined_revision = 0;
        std::size_t refinement_debug_revision = 0;
    };

    BackendMapSnapshot GetBackendMapSnapshot() const;

    // ------------------------------------------------------------------------
    // Latest loop-ICP RViz diagnostic.
    //
    // All three clouds are already expressed in the same PoseGraph/world
    // frame so the ROS wrapper can publish them directly:
    //
    //   historical_target_world : ICP target geometry
    //   initial_aligned_world    : current KF cloud transformed by the
    //                              supplied ICP initial guess
    //   final_aligned_world      : same current KF cloud transformed by the
    //                              final LoopVerifier ICP result
    //
    // This snapshot is visualization-only. It is updated for reverse-loop
    // hypotheses even when the hypothesis is later rejected by ICP/graph
    // gates, which is exactly what is needed to diagnose longitudinal sliding.
    // ------------------------------------------------------------------------
    struct LoopIcpDebugSnapshot
    {
        pcl::PointCloud<LIDAR_POINT>::ConstPtr historical_target_world;
        pcl::PointCloud<LIDAR_POINT>::ConstPtr initial_aligned_world;
        pcl::PointCloud<LIDAR_POINT>::ConstPtr final_aligned_world;

        std::size_t revision = 0;
        std::size_t current_keyframe_id = 0;
        std::size_t historical_keyframe_id = 0;

        std::string initial_guess_name;

        double correction_translation =
            std::numeric_limits<double>::quiet_NaN();

        double correction_rotation_deg =
            std::numeric_limits<double>::quiet_NaN();
    };

    LoopIcpDebugSnapshot GetLoopIcpDebugSnapshot() const;

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
    PoseGraph GetPoseGraphSnapshot() const;

    // Backward-compatible API.  This returns a thread-local copy of the
    // latest immutable backend snapshot.
    const PoseGraph &GetPoseGraph() const;

    std::size_t PoseGraphNodeCount() const;
    std::size_t PoseGraphEdgeCount() const;
    std::size_t PoseGraphOdometryEdgeCount() const;
    std::size_t PoseGraphLoopEdgeCount() const;

    void Reset();

private:
    // ========================================================================
    // Asynchronous backend snapshots / jobs.
    // ========================================================================
    struct BackendSubmapSnapshot
    {
        std::size_t id =
            std::numeric_limits<std::size_t>::max();

        Eigen::Isometry3d T_WS =
            Eigen::Isometry3d::Identity();

        std::vector<std::size_t> keyframe_ids;

        pcl::PointCloud<LIDAR_POINT>::ConstPtr cloud_S;
    };

    struct BackendKeyframeJob
    {
        Keyframe keyframe;

        std::size_t current_submap_id =
            std::numeric_limits<std::size_t>::max();

        bool has_finished_submap = false;
        BackendSubmapSnapshot finished_submap;
    };

    void StartBackendWorker();
    void StopBackendWorker();
    void BackendLoop();

    void ProcessBackendJob(
        const BackendKeyframeJob &job);

    bool EnqueueBackendKeyframe(
        const Keyframe &keyframe,
        std::size_t current_submap_id);

    bool BuildFinishedSubmapSnapshot(
        std::size_t submap_id,
        BackendSubmapSnapshot &snapshot) const;

    void StoreBackendFinishedSubmap(
        const BackendSubmapSnapshot &snapshot);

    void RefreshBackendOutputSnapshot();

    const BackendSubmapSnapshot *
    FindBackendSubmapById(
        std::size_t submap_id) const;

    const Keyframe *FindBackendKeyframeById(
        std::size_t keyframe_id) const;

    const BackendSubmapSnapshot *
    FindBestFinishedSubmapForKeyframe(
        std::size_t keyframe_id) const;

    // V19: build a candidate-KEYFRAME-centered historical LocalMap.
    //
    // Target coordinates are the candidate Keyframe frame K:
    //
    //     p_K = T_K_N * p_N
    //     T_K_N = T_WK^-1 * T_WN
    //
    // This makes candidate K1/K2/K3 genuinely different geometry anchors even
    // when all of them belong to the same finished Submap.
    bool BuildCandidateCenteredHistoricalTarget(
        std::size_t historical_keyframe_id,
        pcl::PointCloud<LIDAR_POINT>::Ptr &target_K,
        std::vector<std::size_t> *included_keyframe_ids = nullptr) const;

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

    // Called immediately after a new Keyframe is committed.
    //
    // Candidate retrieval AND geometry verification are Keyframe-based.
    // A small candidate-centered historical window is built around each
    // historical KF; finished Submaps are used only for history/separation
    // bookkeeping. The accepted loop is directly:
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

        // V17.1:
        // Number of consecutive LOCAL_STRONG confirmations for the SAME
        // historical anchor.  This is deliberately separate from generic
        // temporal support so a large-drift false match cannot become mature
        // merely by repeating the same historical KF several times.
        std::size_t local_strong_support = 0;

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
    struct ReverseLoopSeedTrack
    {
        bool valid = false;

        std::size_t last_current_keyframe_id = 0;
        std::size_t last_historical_keyframe_id = 0;

        std::size_t support = 0;

        // 真正观察到 historical KF 向反方向推进的次数。
        // 只看到同一个 historical KF 重复出现，不算方向证据。
        std::size_t reverse_progress_events = 0;
    };

    ReverseLoopSeedTrack reverse_loop_seed_track_;

    std::size_t reverse_loop_seed_min_support_ = 3;
    std::size_t reverse_loop_seed_min_progress_events_ = 2;

    std::size_t reverse_loop_seed_max_current_gap_ = 2;
    std::size_t reverse_loop_seed_max_historical_step_ = 4;

    std::int64_t reverse_loop_seed_forward_jitter_ = 1;

    // 这个模式只处理“前端认为位置已经比较接近”的反向重访。
    // 不拿它处理几十米的大漂移。
    double reverse_loop_seed_max_frontend_distance_ = 2.0;

    // 比现在单帧 reverse hypothesis 更严格。
    double reverse_loop_seed_min_yaw_deg_ = 150.0;

private:
    // Frontend registration instance.
    LidarRegistration registration_;

    // Separate backend registration instance for Post-PGO refinement.
    LidarRegistration backend_refinement_registration_;

    // Frontend / geometry organization only.
    SubmapManager submap_manager_;

    // Backend worker queue.  Every Keyframe is preserved; if the backend falls
    // behind we warn instead of dropping graph states.
    std::deque<BackendKeyframeJob> backend_queue_;
    mutable std::mutex backend_queue_mutex_;
    std::condition_variable backend_condition_;
    std::thread backend_thread_;
    std::atomic<bool> backend_running_{false};
    std::size_t backend_backlog_warning_threshold_ = 6;

    // Backend-owned immutable history copied from the frontend.
    std::vector<Keyframe> backend_keyframes_;
    std::vector<BackendSubmapSnapshot> backend_finished_submaps_;

    // Backend state: ONE KEYFRAME -> ONE PoseGraphNode.
    PoseGraph pose_graph_;

    // V8: g2o optimizer. It updates only PoseGraph node estimates.
    // The live Scan-to-LocalMap frontend T_WL_ remains untouched.
    PoseGraphOptimizer pose_graph_optimizer_;

    LoopDetector loop_detector_;
    LoopVerifier loop_verifier_;

    // V20 Simple Loop Baseline: 2-frame confirmation + 1 graph factor:
    // verify up to five UNIQUE historical Submaps.  Scan Context yaw is tested
    // together with its explicit 180-degree complementary mode.  A verified
    // temporal loop track propagates its loop-implied world correction as an
    // independent TRACK_PREDICTED continuation hypothesis; support==1 gets a
    // short tentative grace window but still bypasses no safety gate.
    // The old value of 3 was applied before Submap de-duplication, so three
    // adjacent Scan Context Keyframes from one wrong historical region could
    // consume the whole ICP budget while a true revisit at rank 4/5 was never
    // verified.
    std::size_t max_loop_candidates_to_verify_ = 5;

    // Local-neighborhood exclusion is intentionally Keyframe-based and reuses
    // LoopDetectorConfig::min_keyframe_id_separation.  Submaps remain geometry
    // containers only; Submap IDs do not define loop temporal separation.

    OnlineLoopTrack online_loop_track_;
    std::size_t online_loop_min_support_ = 3;

    // Mature active track: keep the original strict continuity window.
    std::size_t online_loop_max_current_keyframe_gap_ = 3;

    // V15 tentative track: after one graph-consistent observation, allow a
    // short Scan Context dropout before declaring the track stale.  This only
    // controls candidate continuation; geometry/graph/temporal gates remain
    // mandatory and no loop edge is submitted from this setting alone.
    std::size_t online_loop_tentative_max_current_keyframe_gap_ = 5;

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
    double max_loop_graph_correction_translation_ = 5.0;

    // V20.2 strict graph correction gate.
    //
    // Unknown first-loop seeds are only allowed to disagree with the
    // frontend pose by a few metres.
    //
    // A trusted temporal track may receive a slightly larger continuation
    // cap, but must never reopen the old 20-25 m large-error corridor.
    //
    // As soon as ONE strict first-loop anchor has actually entered the pending
    // batch AND that anchor has raw Scan Context provenance, the immediately
    // following geometrically consistent observation may use the adaptive
    // continuation cap.  This fixes the V19.5 dead-zone:
    //
    //     KF146->94  = 19.059 m  (strict raw-SC seed, PASS)
    //     KF147->93  = 20.0366 m (only +3.66 cm over 20 m, but V19.5
    //                              still used SEED_FIXED because support==1)
    //
    // Adaptive mode still requires corrected-space corridor support and local
    // agreement with the previous loop-track prediction.  Therefore this is
    // NOT a global 20 -> 25 m relaxation.
    std::size_t online_loop_graph_followup_min_track_support_ = 1;
    double online_loop_graph_followup_translation_cap_ = 5.0;

    // The old 45 deg gate rejected geometrically verified revisits before
    // they could accumulate temporal support.  Large corrections are still
    // protected by sequence consistency and the first-loop batch checks.
    double max_loop_graph_correction_rotation_deg_ = 100.0;

    // V16: backend transaction hard guard.
    //
    // Even if g2o's gravity/shape guards pass, an accepted loop must not be
    // allowed to move the already-built trajectory by an implausibly large
    // amount in one optimization.  The graph node poses and newly staged loop
    // edges are rolled back if either limit is exceeded.
    //
    // The known-good terminal loop from the current bag was about 1.08 m /
    // 1.19 deg, while the false KF149..151 -> KF96 batch produced
    // 18.89 m / 69.14 deg.  These conservative limits separate those cases
    // while keeping room for meaningful drift correction.
    double online_loop_pgo_max_translation_update_ = 5.0;
    double online_loop_pgo_max_rotation_update_deg_ = 20.0;

    // V17.1 LOCAL_STRONG repeated-anchor confirmation.
    //
    // Repeated observations of the exact same historical KF are allowed to
    // increase first-loop temporal support ONLY inside this tight envelope.
    // This separates:
    //
    //   true terminal revisit: ~1.5--2.6 m / ~1--4 deg
    //   false middle revisit : ~19 m (despite good overlap/RMSE)
    //
    // Repeated observations still never create duplicate graph factors.
    double online_loop_first_local_min_overlap_ = 0.90;
    double online_loop_first_local_max_rmse_ = 0.45;
    double online_loop_first_local_max_graph_translation_ = 3.0;
    double online_loop_first_local_max_graph_rotation_deg_ = 10.0;

    // V19.3 LOCAL_STRONG_CLUSTER.
    //
    // Neighboring historical representatives (e.g. KF20 and KF21) are treated
    // as the same physical local loop cluster for temporal confirmation.
    // They do NOT create duplicate graph factors.
    std::size_t online_loop_first_local_cluster_radius_ = 2;

    // V18 LARGE_DRIFT_CONSENSUS.
    //
    // A large graph correction is NOT automatically a false loop.  It may be
    // the very drift that loop closure must remove.  Once the first verified
    // large-drift anchor exists, actively probe a small historical-KF
    // neighborhood instead of waiting for Scan Context to independently return
    // a different anchor on the next frame.
    //
    // Example:
    //     first anchor: current 149 -> history 96
    //     next frames actively test history 95/97, then 94/98, ...
    //
    // All injected candidates still pass the SAME ICP / graph / temporal gates.
    std::size_t online_loop_large_drift_neighborhood_radius_ = 4;

    // V19.3 anti-self-confirmation policy.
    //
    // The first LARGE_DRIFT anchor MUST be independently seeded by raw Scan
    // Context.  Follow-up anchors may come from TRACK/neighborhood injection
    // only when their geometry is substantially stronger than the ordinary
    // first-loop gate and their correction agrees tightly with the seed.
    //
    // This fixes both observed failure modes:
    //   false V19.1 : 97->148,98->149,99->150
    //                 follow-up overlap ~= 0.898 / 0.897 -> blocked
    //   true V19.2  : 95->148,96->149,97->150
    //                 follow-up overlap ~= 0.952 / 0.914 -> allowed
    std::size_t online_loop_large_drift_raw_sc_support_radius_ = 2;
    std::size_t online_loop_large_drift_min_raw_sc_members_ = 1;

    double online_loop_large_drift_injected_min_overlap_ = 0.90;
    double online_loop_large_drift_injected_max_rmse_ = 0.50;
    double online_loop_large_drift_injected_max_consistency_translation_ = 1.20;
    double online_loop_large_drift_injected_max_consistency_rotation_deg_ = 8.0;

    // V19.4 drift-aware revisit corridor search.
    //
    // IMPORTANT:
    // Do NOT search historical KFs around the RAW frontend world position.
    // That would reject exactly the drift that loop closure is supposed to
    // correct.
    //
    // Once one trusted large-drift seed exists, use its correction:
    //
    //     T_WL_pred = C_seed * T_WL_frontend
    //
    // and search historical KFs around this CORRECTED predicted pose.
    // This creates a spatial candidate corridor on the old trajectory even
    // when the raw current trajectory has drifted metres away in RViz.
    bool online_loop_drift_aware_corridor_enabled_ = true;
    double online_loop_drift_aware_corridor_radius_ = 4.0;
    std::size_t online_loop_drift_aware_corridor_max_candidates_ = 8;

    // An injected large-drift follow-up must either be supported by raw Scan
    // Context OR lie inside the corrected-space revisit corridor.  Geometry
    // and correction-consistency gates still apply afterwards.
    double online_loop_drift_aware_corridor_accept_radius_ = 4.0;

    // V19 candidate-centered geometry target.
    //
    // Each historical candidate KF owns a small local window [K-r, K+r].
    // The clouds are transformed into the candidate KF frame before ICP.
    // With ~0.5 m Keyframe spacing, radius=2 gives roughly a 2 m local support
    // length while remaining specific enough to distinguish neighboring KFs.
    std::size_t online_loop_candidate_target_half_window_ = 2;
    std::size_t online_loop_candidate_target_min_keyframes_ = 2;
    double online_loop_candidate_target_voxel_leaf_size_ = 0.25;

    // During large-drift consensus only, allow several candidate KFs from the
    // SAME historical Submap to be verified.  This is necessary because
    // neighboring KFs such as 96/97/98 normally belong to one frozen Submap.
    // Normal Scan Context discovery still verifies only one candidate per
    // historical Submap.
    std::size_t online_loop_large_drift_same_submap_verify_budget_ = 3;

    // Additional full-ICP trials allowed only while building the first
    // large-drift consensus.  The normal global candidate budget remains 5.
    std::size_t online_loop_large_drift_extra_verify_budget_ = 6;

    // A pending first-loop correction outside LOCAL_STRONG enters the
    // large-drift consensus path.
    double online_loop_large_drift_trigger_translation_ = 3.0;
    double online_loop_large_drift_trigger_rotation_deg_ = 10.0;
    // V20.2 large-drift physical plausibility guard.
    //
    // Compare the loop-implied translation correction against the raw frontend
    // odometry arc length between historical KF and current KF.
    //
    // A genuine long-term loop may correct several metres after travelling a
    // long distance.  A repetitive-scene false loop often asks to remove a very
    // large fraction of the travelled path itself.
    double online_loop_large_drift_min_arc_length_for_ratio_ = 10.0;
    double online_loop_large_drift_max_correction_path_ratio_ = 0.35;

    // V20.2 first-loop PGO trajectory-length guard.
    //
    // A first loop must not globally compress or stretch the raw trajectory by an
    // implausibly large amount merely to satisfy one loop factor.
    double online_loop_first_pgo_min_path_length_ratio_ = 0.93;
    double online_loop_first_pgo_max_path_length_ratio_ = 1.07;

    // V18 adaptive first-PGO transaction guard.
    //
    // LOCAL_STRONG still uses the conservative fixed 5 m / 20 deg limits.
    // LARGE_DRIFT_MONOTONIC_SEQUENCE may legitimately require a larger update,
    // but the allowed update must scale with the independently agreed loop
    // correction and remains capped.
    double online_loop_large_drift_pgo_translation_scale_ = 1.35;
    double online_loop_large_drift_pgo_translation_margin_ = 1.0;
    double online_loop_large_drift_pgo_translation_cap_ = 25.0;

    double online_loop_large_drift_pgo_rotation_scale_ = 1.35;
    double online_loop_large_drift_pgo_rotation_margin_deg_ = 5.0;
    double online_loop_large_drift_pgo_rotation_cap_deg_ = 110.0;

    // After optimization, the latest current KF must end close to the world pose
    // implied by the consensus correction.  This checks the RESULT against the
    // loop consensus rather than merely accepting a large numerical update.
    double online_loop_large_drift_pgo_anchor_target_translation_error_ = 3.0;
    double online_loop_large_drift_pgo_anchor_target_rotation_error_deg_ = 15.0;

    // V19.1: large-drift PGO must be judged by LOCAL graph deformation, not
    // by the maximum absolute yaw change of an arbitrary node.
    //
    // The 2026-09-03 V19 run produced:
    //   max absolute node yaw update : 46.81 deg   (old guard rejected)
    //   max adjacent odom deformation: 5.54 deg
    //   max adjacent translation def.: 0.735 m
    //   max staged-loop residual      : 0.652 m / 10.23 deg
    //   anchor target error           : 0.444 m / 7.72 deg
    //
    // That is a smooth distributed correction, not a single catastrophic
    // vertex jump.  These local limits are therefore the V19.1 safety test.
    double online_loop_large_drift_max_local_odom_translation_deformation_ = 1.0;
    double online_loop_large_drift_max_local_odom_rotation_deformation_deg_ = 7.0;

    double online_loop_large_drift_max_loop_residual_translation_ = 1.0;
    double online_loop_large_drift_max_loop_residual_rotation_deg_ = 12.0;

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
    // V16.2: repeated observations of one historical KF may confirm temporal
    // persistence, but they never create repeated graph factors.  Three
    // graph-consistent observations are required before the first transaction.
    std::size_t online_loop_first_edge_min_support_ = 2;

    // Strict geometry gate for the FIRST anchor only.
    // Calibrated against the current online log: verified revisits are around
    // 0.70--0.75 overlap and 0.50 m RMSE.  Safety comes from requiring a
    // consistent multi-keyframe sequence, not from an unreachable 0.88 gate.
    double online_loop_first_edge_min_overlap_ = 0.70;
    double online_loop_first_edge_max_rmse_ = 0.58;
    double online_loop_first_edge_max_icp_translation_ = 3.0;
    double online_loop_first_edge_max_icp_rotation_deg_ = 35.0;

    // Follow-up members are mainly judged by whether they imply the same
    // world correction as the anchor.  Keep only a wider ICP safety gate here.
    double online_loop_first_batch_followup_max_icp_translation_ = 3.5;
    double online_loop_first_batch_followup_max_icp_rotation_deg_ = 35.0;

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

        // V19.2 provenance:
        // true only when THIS current frame independently retrieved the same
        // historical neighborhood through raw Scan Context.  Track-predicted
        // and large-drift-neighborhood injected candidates are hypotheses, not
        // independent evidence, and are not allowed to grow the first
        // large-drift PoseGraph batch by themselves.
        bool raw_scan_context_supported = false;
        std::size_t raw_scan_context_support_kf =
            std::numeric_limits<std::size_t>::max();
        double raw_scan_context_support_similarity = 0.0;
    };

    std::vector<
        PendingLoopConstraint,
        Eigen::aligned_allocator<PendingLoopConstraint>>
        pending_first_loop_batch_;

    // V17.1 dual-mode first-loop transaction.
    //
    // Mode A: LOCAL_STRONG_CLUSTER
    //   - one UNIQUE graph edge is enough,
    //   - historical representatives may jitter within +/-2 KF,
    //   - 3 LOCAL_STRONG observations inside that cluster confirm the loop,
    //   - the batch keeps ONE representative factor only.
    //
    // Mode B: LARGE_DRIFT_MONOTONIC_SEQUENCE
    //   - requires 3 DISTINCT historical KFs,
    //   - historical ids must progress monotonically,
    //   - all three constraints must agree on the same loop correction,
    //   - after the first large-drift anchor, V18 actively verifies nearby
    //     historical KFs so Scan Context does not have to rediscover each one,
    //   - first-PGO update limits scale with the agreed correction and the
    //     optimized anchor must land near the consensus-implied target pose.
    //
    // Later loop edges keep the original fixed 5 m / 20 deg rollback guard.
    std::size_t online_loop_first_batch_min_edges_ = 2;
    std::size_t online_loop_first_batch_current_spacing_ = 1;
    std::size_t online_loop_first_batch_historical_spacing_ = 1;

    // V20 baseline: two consecutive current KFs must confirm the same local
    // historical neighborhood.  This is intentionally simpler than the old
    // multi-stage direction state machine.
    std::size_t online_loop_first_batch_max_historical_gap_ = 4;

    // V19.6: the Keyframe-centered historical target uses +/-2 KFs.  The
    // winning center id can therefore jitter by one KF without representing
    // real trajectory reversal.  Do not establish/reverse batch direction from
    // a single +/-1 center-id step.
    std::size_t online_loop_first_batch_direction_jitter_tolerance_ = 1;

    // V19.6: batch consistency is evaluated locally against the PREVIOUS
    // verified loop-track prediction, not forever against the first anchor.
    // Keep the numerical envelope unchanged.
    double online_loop_first_batch_max_translation_error_ = 1.5;
    double online_loop_first_batch_max_rotation_error_deg_ = 10.0;

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
        10,    // Keyframes per backend map block.
        0.30f, // Per-block VoxelGrid leaf size.
        0.01,  // Dirty translation threshold: 1 cm.
        0.05}; // Dirty rotation threshold: 0.05 deg.

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
    double refinement_geometry_max_translation_correction_ = 1.10;
    double refinement_geometry_max_rotation_correction_deg_ = 7.5;
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
    double refinement_window_max_translation_update_ = 1.20;
    double refinement_window_max_rotation_update_deg_ = 8.0;

    // Final local-window shape guard.  Absolute rigid motion is allowed by the
    // broad window gate above, but neighboring Keyframe odometry must remain
    // almost unchanged after refinement.
    double refinement_window_max_relative_odom_translation_change_ = 0.10;
    double refinement_window_max_relative_odom_rotation_change_deg_ = 1.0;

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

    // Thread-safe published backend snapshots.  Heavy backend work never holds
    // this mutex; the worker only swaps/copies outputs when one job completes.
    mutable std::mutex backend_output_mutex_;

    PoseGraph backend_pose_graph_snapshot_;

    pcl::PointCloud<LIDAR_POINT>::ConstPtr backend_raw_map_snapshot_;
    pcl::PointCloud<LIDAR_POINT>::ConstPtr backend_optimized_map_snapshot_;
    pcl::PointCloud<LIDAR_POINT>::ConstPtr backend_refined_map_snapshot_;

    pcl::PointCloud<LIDAR_POINT>::ConstPtr
        backend_refinement_historical_target_snapshot_;
    pcl::PointCloud<LIDAR_POINT>::ConstPtr
        backend_refinement_current_before_snapshot_;
    pcl::PointCloud<LIDAR_POINT>::ConstPtr
        backend_refinement_current_after_snapshot_;

    // Latest reverse-loop ICP visualization diagnostic.  These are kept
    // separate from GlobalMapRevision because rejected loop hypotheses do not
    // modify the map or PoseGraph, but still need to be visible in RViz.
    pcl::PointCloud<LIDAR_POINT>::ConstPtr
        backend_loop_icp_historical_target_snapshot_;

    pcl::PointCloud<LIDAR_POINT>::ConstPtr
        backend_loop_icp_initial_aligned_snapshot_;

    pcl::PointCloud<LIDAR_POINT>::ConstPtr
        backend_loop_icp_final_aligned_snapshot_;

    std::size_t backend_loop_icp_debug_revision_snapshot_ = 0;
    std::size_t backend_loop_icp_debug_current_kf_snapshot_ = 0;
    std::size_t backend_loop_icp_debug_historical_kf_snapshot_ = 0;

    std::string backend_loop_icp_debug_guess_name_snapshot_;

    double backend_loop_icp_debug_correction_translation_snapshot_ =
        -std::numeric_limits<double>::infinity();

    double backend_loop_icp_debug_correction_rotation_snapshot_ =
        std::numeric_limits<double>::quiet_NaN();

    // Within one current KF prefer the confirmed reverse-sequence diagnostic
    // over the generic reverse-frontend diagnostic. For equal priority keep
    // the larger translation excursion because it exposes ICP sliding most
    // clearly.
    int backend_loop_icp_debug_priority_snapshot_ = 0;

    std::size_t backend_global_map_revision_snapshot_ = 0;
    std::size_t backend_refined_map_revision_snapshot_ = 0;
    std::size_t backend_refinement_debug_revision_snapshot_ = 0;

    Eigen::Isometry3d backend_T_map_odom_snapshot_ =
        Eigen::Isometry3d::Identity();

    bool backend_has_map_odom_correction_snapshot_ = false;
    std::size_t backend_map_odom_revision_snapshot_ = 0;

    Eigen::Isometry3d T_WL_ =
        Eigen::Isometry3d::Identity();

    Eigen::Isometry3d last_relative_transform_ =
        Eigen::Isometry3d::Identity();

    std::size_t consecutive_rejected_frames_ = 0;
    std::size_t max_recovery_prediction_steps_ = 5;

    bool initialized_ = false;

    double max_accepted_rmse_ = 0.15;
    std::size_t min_accepted_correspondences_ = 100;
};
