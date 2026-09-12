#include "fr_slam/frontend/lo_frontend.hpp"
#include "fr_slam/frontend/ground_segmenter.hpp"
#include "fr_slam/frontend/ground_input_bridge.hpp"
#include "fr_slam/frontend/multi_plane_extractor.hpp"

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
#include "lo_frontend_ground_internal.hpp"

namespace
{
    const rclcpp::Logger kRecoveryLogger =
        rclcpp::get_logger("scan2local_map.recovery");

    // ========================================================================
    // Frontend Robust ICP V1 experiment isolation.
    //
    // The realtime Scan-to-LocalMap registration uses the caller-provided
    // robust configuration. Post-PGO refinement is deliberately kept on the
    // old unweighted ICP behavior for this experiment, so any trajectory
    // change can be attributed to the FRONTEND robust kernel rather than to
    // a simultaneous change in backend refinement.
    LidarRegistrationConfig MakeBackendRefinementRegistrationConfig(
        const LidarRegistrationConfig &frontend_config)
    {
        LidarRegistrationConfig backend_config =
            frontend_config;

        backend_config.enable_huber_loss =
            false;

        // Keep post-PGO refinement on the legacy perturbation during the
        // controlled Degeneracy V2A/V2B experiment.  Only the realtime frontend
        // uses the sensor-centered + median-range-normalized Hessian/retraction in this run.
        backend_config.enable_sensor_centered_perturbation =
            false;

        backend_config.enable_hessian_scale_normalization =
            false;

        return backend_config;
    }

    double ElapsedMilliseconds(
        const std::chrono::steady_clock::time_point &start,
        const std::chrono::steady_clock::time_point &end)
    {
        return std::chrono::duration<double, std::milli>(
                   end - start)
            .count();
    }

    Eigen::Isometry3d ProjectPoseToPlanar(
        const Eigen::Isometry3d &input_pose)
    {
        Eigen::Isometry3d output_pose =
            input_pose;

        const Eigen::Matrix3d rotation =
            input_pose.rotation();

        const double yaw =
            std::atan2(
                rotation(1, 0),
                rotation(0, 0));

        output_pose.linear() =
            Eigen::AngleAxisd(
                yaw,
                Eigen::Vector3d::UnitZ())
                .toRotationMatrix();

        output_pose.translation().z() =
            0.0;

        return output_pose;
    }

    struct FrameTimingDiagnostics
    {
        std::chrono::steady_clock::time_point start =
            std::chrono::steady_clock::now();

        double ground_segment_ms = 0.0;
        double primary_align_ms = 0.0;
        double ground_refine_ms = 0.0;
        double recovery_coarse_ms = 0.0;
        double recovery_refine_ms = 0.0;
        double keyframe_decision_ms = 0.0;
        double keyframe_store_ms = 0.0;
        double pose_graph_insert_ms = 0.0;
        double global_map_incremental_ms = 0.0;
        double submap_insert_ms = 0.0;
        double loop_backend_ms = 0.0;
        double backend_enqueue_ms = 0.0;
        double prepare_target_ms = 0.0;

        std::size_t keyframes_before = 0;
        std::size_t keyframes_after = 0;

        bool first_frame = false;
        bool accepted = false;
        bool keyframe = false;
        bool recovery_triggered = false;
        bool coarse_recovery_accepted = false;
    };

    class FrameTimingReporter
    {
    public:
        explicit FrameTimingReporter(
            FrameTimingDiagnostics &diagnostics)
            : diagnostics_(diagnostics)
        {
        }

        ~FrameTimingReporter() = default;

    private:
        FrameTimingDiagnostics &diagnostics_;
    };

    // ============================================================================
    // ============================================================================
    // BuildOdometryInformationV2()
    //
    // Full 6x6 relative information shape.
    //
    // Design goal:
    //     Keep EXACTLY the V1.1 diagonal confidence values while restoring the
    //     off-diagonal coupling carried by the Hessian-derived covariance.
    //
    // Steps:
    //     1. World [r,t] covariance -> current LiDAR [r,t].
    //     2. Reorder to g2o [t,r].
    //     3. Compute the same normalized diagonal confidence as V1.1.
    //     4. Recover precision shape Q = C^-1.
    //     5. Standardize Q to unit diagonal.
    //     6. Restore the V1.1 diagonal through congruence scaling.
    // ============================================================================
    bool BuildOdometryInformationV2(
        const LidarRegistrationResult &registration_result,
        const Eigen::Isometry3d &T_WL,
        Eigen::Matrix<double, 6, 6> &information)
    {
        information =
            Eigen::Matrix<double, 6, 6>::Identity();

        if (!registration_result
                 .hessian_relative_covariance_valid ||
            !registration_result
                 .hessian_relative_covariance
                 .allFinite() ||
            !T_WL.matrix().allFinite())
        {
            return false;
        }

        // --------------------------------------------------------------------
        // 1. World [r,t] -> current LiDAR [r,t].
        // --------------------------------------------------------------------
        const Eigen::Matrix3d R_LW =
            T_WL.rotation().transpose();

        Eigen::Matrix<double, 6, 6> world_to_lidar =
            Eigen::Matrix<double, 6, 6>::Zero();

        world_to_lidar.block<3, 3>(0, 0) =
            R_LW;

        world_to_lidar.block<3, 3>(3, 3) =
            R_LW;

        const Eigen::Matrix<double, 6, 6>
            covariance_lidar_rt =
                world_to_lidar *
                registration_result.hessian_relative_covariance *
                world_to_lidar.transpose();

        if (!covariance_lidar_rt.allFinite())
        {
            return false;
        }

        // --------------------------------------------------------------------
        // 2. [rx ry rz tx ty tz] -> [tx ty tz rx ry rz].
        // --------------------------------------------------------------------
        Eigen::Matrix<double, 6, 6> rt_to_tr =
            Eigen::Matrix<double, 6, 6>::Zero();

        rt_to_tr.block<3, 3>(0, 3) =
            Eigen::Matrix3d::Identity();

        rt_to_tr.block<3, 3>(3, 0) =
            Eigen::Matrix3d::Identity();

        Eigen::Matrix<double, 6, 6> covariance_tr =
            rt_to_tr *
            covariance_lidar_rt *
            rt_to_tr.transpose();

        covariance_tr =
            0.5 *
            (covariance_tr +
             covariance_tr.transpose());

        if (!covariance_tr.allFinite())
        {
            return false;
        }

        // --------------------------------------------------------------------
        // 3. Keep EXACTLY the V1.1 diagonal confidence definition.
        // --------------------------------------------------------------------
        constexpr double minimum_directional_confidence =
            0.01;

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
                    minimum_directional_confidence,
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
                    minimum_directional_confidence,
                    1.0);
        }

        // --------------------------------------------------------------------
        // 4. Recover full precision shape Q = C^-1 by eigendecomposition.
        // --------------------------------------------------------------------
        Eigen::SelfAdjointEigenSolver<
            Eigen::Matrix<double, 6, 6>>
            covariance_solver(
                covariance_tr);

        if (covariance_solver.info() !=
            Eigen::Success)
        {
            return false;
        }

        const Eigen::Matrix<double, 6, 1>
            covariance_eigenvalues =
                covariance_solver.eigenvalues();

        if (!covariance_eigenvalues.allFinite() ||
            covariance_eigenvalues.minCoeff() <=
                1.0e-12)
        {
            return false;
        }

        Eigen::Matrix<double, 6, 6>
            inverse_eigenvalues =
                Eigen::Matrix<double, 6, 6>::Zero();

        for (int i = 0;
             i < 6;
             ++i)
        {
            inverse_eigenvalues(i, i) =
                1.0 /
                covariance_eigenvalues(i);
        }

        Eigen::Matrix<double, 6, 6>
            precision_shape =
                covariance_solver.eigenvectors() *
                inverse_eigenvalues *
                covariance_solver.eigenvectors().transpose();

        precision_shape =
            0.5 *
            (precision_shape +
             precision_shape.transpose());

        if (!precision_shape.allFinite())
        {
            return false;
        }

        // --------------------------------------------------------------------
        // 5. Standardize precision shape to unit diagonal.
        //
        //     J_ij = Q_ij / sqrt(Q_ii * Q_jj)
        //
        // J remains SPD because this is a diagonal congruence transform.
        // --------------------------------------------------------------------
        Eigen::Matrix<double, 6, 6>
            standardized_precision =
                Eigen::Matrix<double, 6, 6>::Zero();

        for (int i = 0;
             i < 6;
             ++i)
        {
            if (!std::isfinite(
                    precision_shape(i, i)) ||
                precision_shape(i, i) <=
                    0.0)
            {
                return false;
            }

            for (int j = 0;
                 j < 6;
                 ++j)
            {
                if (!std::isfinite(
                        precision_shape(j, j)) ||
                    precision_shape(j, j) <=
                        0.0)
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

        // --------------------------------------------------------------------
        // 6. Restore V1.1 diagonal confidence:
        //
        //     Omega_V2 = W * J * W
        //     W = diag(sqrt(confidence_tr))
        //
        // Therefore Omega_V2(i,i) == confidence_tr(i).
        // --------------------------------------------------------------------
        Eigen::Matrix<double, 6, 6>
            confidence_scale =
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
            return false;
        }

        // --------------------------------------------------------------------
        // 7. Strict SPD validation before storing the edge information.
        // --------------------------------------------------------------------
        Eigen::SelfAdjointEigenSolver<
            Eigen::Matrix<double, 6, 6>>
            information_solver(
                information,
                Eigen::EigenvaluesOnly);

        if (information_solver.info() !=
            Eigen::Success)
        {
            return false;
        }

        const double minimum_information_eigenvalue =
            information_solver
                .eigenvalues()
                .minCoeff();

        if (!std::isfinite(
                minimum_information_eigenvalue) ||
            minimum_information_eigenvalue <=
                1.0e-9)
        {
            return false;
        }

        return true;
    }
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

    // ============================================================================
    // RunCoarsePointToPointRecovery()
    //
    // Coarse tracking-recovery registration.
    //
    // source:
    //     current LiDAR scan in CURRENT LiDAR coordinates.
    //
    // target:
    //     Active Submap in WORLD coordinates.
    //
    // Therefore the final ICP transform has the same semantic meaning as the
    // normal Scan-to-LocalMap result:
    //
    //     T_target_source = T_WL_current
    //
    // The normal LiDAR/IMU prediction is still used as the initial guess.
    //
    // This stage is NOT the final pose estimator. Its job is only:
    //
    //     "find a better basin of attraction"
    //
    // for the existing point-to-plane registration.
    //
    // The final pose still has to pass:
    //
    //     coarse point-to-point ICP
    //                 |
    //                 v
    //     point-to-plane Scan-to-LocalMap refine
    //                 |
    //                 v
    //             Quality Gate
    //
    // ============================================================================
    bool RunCoarsePointToPointRecovery(
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &source_lidar,
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &target_world,
        const Eigen::Isometry3d &initial_guess,
        Eigen::Isometry3d &T_WL_coarse,
        double &fitness_score)
    {
        T_WL_coarse = initial_guess;
        fitness_score = std::numeric_limits<double>::infinity();

        if (!source_lidar ||
            !target_world ||
            source_lidar->empty() ||
            target_world->empty())
        {
            return false;
        }

        const pcl::PointCloud<pcl::PointXYZ>::Ptr source_xyz =
            ConvertToXYZ(source_lidar);

        const pcl::PointCloud<pcl::PointXYZ>::Ptr target_xyz =
            ConvertToXYZ(target_world);

        // A very small cloud is not meaningful for coarse recovery.
        if (source_xyz->size() < 100 ||
            target_xyz->size() < 100)
        {
            return false;
        }

        pcl::IterativeClosestPoint<
            pcl::PointXYZ,
            pcl::PointXYZ>
            icp;

        icp.setInputSource(source_xyz);
        icp.setInputTarget(target_xyz);

        // Recovery deliberately uses a wider capture range than normal
        // point-to-plane tracking.
        //
        // These are V2 experiment parameters, not final tuned values.
        icp.setMaxCorrespondenceDistance(2.0);
        icp.setMaximumIterations(30);
        icp.setTransformationEpsilon(1.0e-5);
        icp.setEuclideanFitnessEpsilon(1.0e-5);

        pcl::PointCloud<pcl::PointXYZ> aligned_cloud;

        const Eigen::Matrix4f initial_guess_float =
            initial_guess.matrix().cast<float>();

        icp.align(
            aligned_cloud,
            initial_guess_float);

        if (!icp.hasConverged())
        {
            return false;
        }

        const Eigen::Matrix4f final_transform =
            icp.getFinalTransformation();

        if (!final_transform.allFinite())
        {
            return false;
        }

        fitness_score =
            icp.getFitnessScore(2.0);

        if (!std::isfinite(fitness_score))
        {
            return false;
        }

        T_WL_coarse.matrix() =
            final_transform.cast<double>();

        return T_WL_coarse.matrix().allFinite();
    }

} // namespace


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
        Eigen::Vector3d normal_W = Eigen::Vector3d::UnitX();
        double d_W = 0.0;
        Eigen::Vector3d center_W = Eigen::Vector3d::Zero();
        double radius_m = 0.0;
        double quality = 0.0;
        std::size_t point_count = 0;
    };

    struct WallAssociationV2PhysicalWall
    {
        std::size_t cluster_id = 0;
        std::vector<std::size_t> member_plane_indices;
        std::size_t representative_plane_index = 0;

        Eigen::Vector3d normal_W = Eigen::Vector3d::UnitX();
        double d_W = 0.0;
        Eigen::Vector3d center_W = Eigen::Vector3d::Zero();
        double radius_m = 0.0;
        double quality = 0.0;
        std::size_t total_points = 0;
    };

    struct WallAssociationV2Track
    {
        std::size_t id = 0;

        // Last observation: used only for short-horizon identity matching.
        Eigen::Vector3d last_normal_W = Eigen::Vector3d::UnitX();
        double last_d_W = 0.0;
        Eigen::Vector3d last_center_W = Eigen::Vector3d::Zero();
        double last_radius_m = 0.0;

        // Frozen reference: created only after consecutive confirmation.
        bool stable = false;
        Eigen::Vector3d reference_normal_W = Eigen::Vector3d::UnitX();
        double reference_d_W = 0.0;

        // Confirmation accumulator.  Reset when temporal continuity breaks.
        Eigen::Vector3d confirmation_normal_sum_W = Eigen::Vector3d::Zero();
        double confirmation_d_sum_W = 0.0;
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
        Eigen::Vector3d reference_normal_W = Eigen::Vector3d::UnitX();
        double reference_d_W = 0.0;

        // Last observed support is used only for re-identification.
        Eigen::Vector3d last_center_W = Eigen::Vector3d::Zero();
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
        Eigen::Vector3d reid_pending_last_normal_W =
            Eigen::Vector3d::UnitX();
        double reid_pending_last_d_W = 0.0;
        Eigen::Vector3d reid_pending_last_center_W =
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
        bool persistent_reacquired = false;  // TRUE long-gap reacquisition.
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

    WallAssociationV2Runtime &GetWallAssociationV2Runtime()
    {
        static WallAssociationV2Runtime runtime;
        return runtime;
    }

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

    double WallAssociationV2TangentialDistance(
        const Eigen::Vector3d &center_a_W,
        const Eigen::Vector3d &center_b_W,
        const Eigen::Vector3d &normal_W)
    {
        if (!center_a_W.allFinite() ||
            !center_b_W.allFinite() ||
            !normal_W.allFinite() ||
            normal_W.norm() < 1.0e-9)
        {
            return std::numeric_limits<double>::quiet_NaN();
        }

        const Eigen::Vector3d unit_normal_W =
            normal_W.normalized();

        const Eigen::Vector3d delta_W =
            center_a_W - center_b_W;

        const Eigen::Vector3d tangential_delta_W =
            delta_W -
            unit_normal_W *
                unit_normal_W.dot(delta_W);

        return tangential_delta_W.norm();
    }

    double WallAssociationV2SupportGap(
        const Eigen::Vector3d &center_a_W,
        const double radius_a_m,
        const Eigen::Vector3d &center_b_W,
        const double radius_b_m,
        const Eigen::Vector3d &normal_W)
    {
        const double tangential_distance_m =
            WallAssociationV2TangentialDistance(
                center_a_W,
                center_b_W,
                normal_W);

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
        const Eigen::Isometry3d &T_WL,
        std::size_t &raw_wall_candidates)
    {
        constexpr double kClusterMaximumNormalDifferenceDeg = 5.0;
        constexpr double kClusterMaximumPlaneDistanceDifferenceM = 0.20;
        constexpr double kClusterMaximumSupportGapM = 1.50;

        std::vector<WallAssociationV2Observation> observations;

        const Eigen::Matrix3d R_WL =
            T_WL.rotation();

        const Eigen::Vector3d t_WL =
            T_WL.translation();

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
            observation.normal_W = R_WL * plane.normal_L;
            observation.d_W =
                plane.d -
                observation.normal_W.dot(t_WL);
            observation.center_W =
                R_WL * plane.center_L + t_WL;
            observation.radius_m =
                std::max(0.0, plane.radius_m);
            observation.quality =
                plane.constraint_quality_score;
            observation.point_count =
                plane.point_count;

            if (!observation.normal_W.allFinite() ||
                observation.normal_W.norm() < 1.0e-9 ||
                !std::isfinite(observation.d_W) ||
                !observation.center_W.allFinite())
            {
                continue;
            }

            WallAssociationV2CanonicalizePlane(
                observation.normal_W,
                observation.d_W);

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
                        observations[i].normal_W,
                        observations[j].normal_W);

                if (!std::isfinite(normal_difference_deg) ||
                    normal_difference_deg >
                        kClusterMaximumNormalDifferenceDeg)
                {
                    continue;
                }

                Eigen::Vector3d normal_j_W =
                    observations[j].normal_W;

                double d_j_W =
                    observations[j].d_W;

                if (normal_j_W.dot(
                        observations[i].normal_W) < 0.0)
                {
                    normal_j_W = -normal_j_W;
                    d_j_W = -d_j_W;
                }

                const double plane_distance_difference_m =
                    std::abs(
                        observations[i].d_W -
                        d_j_W);

                if (plane_distance_difference_m >
                    kClusterMaximumPlaneDistanceDifferenceM)
                {
                    continue;
                }

                Eigen::Vector3d mean_normal_W =
                    observations[i].normal_W +
                    normal_j_W;

                if (!mean_normal_W.allFinite() ||
                    mean_normal_W.norm() < 1.0e-9)
                {
                    continue;
                }

                mean_normal_W.normalize();

                const double support_gap_m =
                    WallAssociationV2SupportGap(
                        observations[i].center_W,
                        observations[i].radius_m,
                        observations[j].center_W,
                        observations[j].radius_m,
                        mean_normal_W);

                if (!std::isfinite(support_gap_m) ||
                    support_gap_m >
                        kClusterMaximumSupportGapM)
                {
                    continue;
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

            Eigen::Vector3d normal_sum_W =
                Eigen::Vector3d::Zero();

            Eigen::Vector3d center_sum_W =
                Eigen::Vector3d::Zero();

            double d_sum_W = 0.0;
            double weight_sum = 0.0;
            std::size_t total_points = 0;

            for (const std::size_t observation_index : members)
            {
                const WallAssociationV2Observation &observation =
                    observations[observation_index];

                Eigen::Vector3d aligned_normal_W =
                    observation.normal_W;

                double aligned_d_W =
                    observation.d_W;

                if (aligned_normal_W.dot(
                        representative.normal_W) < 0.0)
                {
                    aligned_normal_W =
                        -aligned_normal_W;

                    aligned_d_W =
                        -aligned_d_W;
                }

                const double weight =
                    std::max(
                        1.0,
                        static_cast<double>(
                            observation.point_count)) *
                    std::max(
                        0.10,
                        observation.quality);

                normal_sum_W +=
                    weight * aligned_normal_W;

                d_sum_W +=
                    weight * aligned_d_W;

                center_sum_W +=
                    weight * observation.center_W;

                weight_sum += weight;
                total_points += observation.point_count;
            }

            if (!normal_sum_W.allFinite() ||
                normal_sum_W.norm() < 1.0e-9 ||
                weight_sum <= 0.0)
            {
                continue;
            }

            WallAssociationV2PhysicalWall physical_wall;
            physical_wall.cluster_id = cluster_index;
            physical_wall.representative_plane_index =
                representative.plane_index;
            physical_wall.normal_W =
                normal_sum_W.normalized();
            physical_wall.d_W =
                d_sum_W / weight_sum;
            physical_wall.center_W =
                center_sum_W / weight_sum;
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
                        observation.center_W,
                        physical_wall.center_W,
                        physical_wall.normal_W);

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
                physical_wall.normal_W,
                physical_wall.d_W);

            physical_walls.push_back(
                physical_wall);
        }

        return physical_walls;
    }

    WallAssociationV2FrameDebug RunWallAssociationV3(
        const ::fr_slam::MultiPlaneExtractionResult &result,
        const Eigen::Isometry3d &T_WL,
        const std::size_t frame_index)
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
                T_WL,
                debug.raw_wall_candidates);

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

        WallAssociationV2Runtime &runtime =
            GetWallAssociationV2Runtime();

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

                    return
                        (frame_index - track.last_seen_frame) >
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

            persistent->last_center_W = candidate.center_W;
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
                        candidate.normal_W,
                        persistent.reference_normal_W);

                if (!std::isfinite(normal_difference_deg) ||
                    normal_difference_deg >
                        kDuplicateMaximumNormalDifferenceDeg)
                {
                    continue;
                }

                Eigen::Vector3d aligned_normal_W =
                    candidate.normal_W;
                double aligned_d_W =
                    candidate.d_W;

                if (aligned_normal_W.dot(
                        persistent.reference_normal_W) < 0.0)
                {
                    aligned_normal_W = -aligned_normal_W;
                    aligned_d_W = -aligned_d_W;
                }

                const double plane_distance_difference_m =
                    std::abs(
                        aligned_d_W -
                        persistent.reference_d_W);

                if (plane_distance_difference_m >
                    kDuplicateMaximumPlaneDistanceDifferenceM)
                {
                    continue;
                }

                const double support_gap_m =
                    WallAssociationV2SupportGap(
                        candidate.center_W,
                        candidate.radius_m,
                        persistent.last_center_W,
                        persistent.last_radius_m,
                        persistent.reference_normal_W);

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

                track.reference_normal_W =
                    best_duplicate->reference_normal_W;

                track.reference_d_W =
                    best_duplicate->reference_d_W;

                candidate_debug.persistent_rebound = true;
                candidate_debug.persistent_wall_id =
                    best_duplicate->id;

                candidate_debug.reference_normal_difference_deg =
                    WallAssociationV2NormalAngleDeg(
                        candidate.normal_W,
                        best_duplicate->reference_normal_W);

                Eigen::Vector3d aligned_normal_W =
                    candidate.normal_W;

                double aligned_d_W =
                    candidate.d_W;

                if (aligned_normal_W.dot(
                        best_duplicate->reference_normal_W) < 0.0)
                {
                    aligned_normal_W =
                        -aligned_normal_W;

                    aligned_d_W =
                        -aligned_d_W;
                }

                candidate_debug.reference_plane_distance_difference_m =
                    std::abs(
                        aligned_d_W -
                        best_duplicate->reference_d_W);

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

                best_duplicate->last_center_W =
                    candidate.center_W;

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
            persistent.reference_normal_W =
                track.reference_normal_W;
            persistent.reference_d_W =
                track.reference_d_W;
            persistent.last_center_W =
                candidate.center_W;
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
                        candidate.normal_W,
                        track.last_normal_W);

                if (!std::isfinite(normal_difference_deg) ||
                    normal_difference_deg >
                        kMaximumNormalDifferenceDeg)
                {
                    continue;
                }

                Eigen::Vector3d aligned_normal_W =
                    candidate.normal_W;

                double aligned_d_W =
                    candidate.d_W;

                if (aligned_normal_W.dot(
                        track.last_normal_W) < 0.0)
                {
                    aligned_normal_W =
                        -aligned_normal_W;

                    aligned_d_W =
                        -aligned_d_W;
                }

                const double plane_distance_difference_m =
                    std::abs(
                        aligned_d_W -
                        track.last_d_W);

                if (plane_distance_difference_m >
                    kMaximumPlaneDistanceDifferenceM)
                {
                    continue;
                }

                const double support_gap_m =
                    WallAssociationV2SupportGap(
                        candidate.center_W,
                        candidate.radius_m,
                        track.last_center_W,
                        track.last_radius_m,
                        track.last_normal_W);

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

            Eigen::Vector3d aligned_normal_W =
                candidate.normal_W;

            double aligned_d_W =
                candidate.d_W;

            if (aligned_normal_W.dot(
                    track.last_normal_W) < 0.0)
            {
                aligned_normal_W =
                    -aligned_normal_W;

                aligned_d_W =
                    -aligned_d_W;
            }

            if (!track.stable)
            {
                if (!temporally_contiguous ||
                    track.confirmation_samples == 0U)
                {
                    track.confirmation_normal_sum_W =
                        aligned_normal_W;

                    track.confirmation_d_sum_W =
                        aligned_d_W;

                    track.confirmation_samples = 1U;
                    track.consecutive_hits = 1U;
                }
                else
                {
                    Eigen::Vector3d confirmation_normal_W =
                        aligned_normal_W;

                    if (confirmation_normal_W.dot(
                            track.confirmation_normal_sum_W) < 0.0)
                    {
                        confirmation_normal_W =
                            -confirmation_normal_W;

                        aligned_d_W =
                            -aligned_d_W;
                    }

                    track.confirmation_normal_sum_W +=
                        confirmation_normal_W;

                    track.confirmation_d_sum_W +=
                        aligned_d_W;

                    ++track.confirmation_samples;
                    ++track.consecutive_hits;
                }

                if (track.consecutive_hits >=
                        kStableConsecutiveHits &&
                    track.confirmation_samples > 0U &&
                    track.confirmation_normal_sum_W.allFinite() &&
                    track.confirmation_normal_sum_W.norm() > 1.0e-9)
                {
                    track.reference_normal_W =
                        track.confirmation_normal_sum_W.normalized();

                    track.reference_d_W =
                        track.confirmation_d_sum_W /
                        static_cast<double>(
                            track.confirmation_samples);

                    WallAssociationV2CanonicalizePlane(
                        track.reference_normal_W,
                        track.reference_d_W);

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

            track.last_normal_W =
                aligned_normal_W.normalized();
            track.last_d_W = aligned_d_W;
            track.last_center_W = candidate.center_W;
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
                        candidate.normal_W,
                        track.reference_normal_W);

                Eigen::Vector3d reference_aligned_normal_W =
                    candidate.normal_W;

                double reference_aligned_d_W =
                    candidate.d_W;

                if (reference_aligned_normal_W.dot(
                        track.reference_normal_W) < 0.0)
                {
                    reference_aligned_normal_W =
                        -reference_aligned_normal_W;

                    reference_aligned_d_W =
                        -reference_aligned_d_W;
                }

                candidate_debug.reference_plane_distance_difference_m =
                    std::abs(
                        reference_aligned_d_W -
                        track.reference_d_W);

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
                        candidate.normal_W,
                        persistent.reference_normal_W);

                if (!std::isfinite(normal_difference_deg) ||
                    normal_difference_deg >
                        kReidMaximumNormalDifferenceDeg)
                {
                    continue;
                }

                Eigen::Vector3d aligned_normal_W =
                    candidate.normal_W;

                double aligned_d_W =
                    candidate.d_W;

                if (aligned_normal_W.dot(
                        persistent.reference_normal_W) < 0.0)
                {
                    aligned_normal_W =
                        -aligned_normal_W;

                    aligned_d_W =
                        -aligned_d_W;
                }

                const double plane_distance_difference_m =
                    std::abs(
                        aligned_d_W -
                        persistent.reference_d_W);

                if (plane_distance_difference_m >
                    kReidMaximumPlaneDistanceDifferenceM)
                {
                    continue;
                }

                const double support_gap_m =
                    WallAssociationV2SupportGap(
                        candidate.center_W,
                        candidate.radius_m,
                        persistent.last_center_W,
                        persistent.last_radius_m,
                        persistent.reference_normal_W);

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

            Eigen::Vector3d aligned_normal_W =
                candidate.normal_W;

            double aligned_d_W =
                candidate.d_W;

            if (aligned_normal_W.dot(
                    persistent.reference_normal_W) < 0.0)
            {
                aligned_normal_W =
                    -aligned_normal_W;

                aligned_d_W =
                    -aligned_d_W;
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

                linked_live_track->last_normal_W =
                    aligned_normal_W.normalized();

                linked_live_track->last_d_W =
                    aligned_d_W;

                linked_live_track->last_center_W =
                    candidate.center_W;

                linked_live_track->last_radius_m =
                    candidate.radius_m;

                linked_live_track->stable =
                    true;

                linked_live_track->static_confirmed =
                    true;

                linked_live_track->reference_normal_W =
                    persistent.reference_normal_W;

                linked_live_track->reference_d_W =
                    persistent.reference_d_W;

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
                    linked_live_track->
                        static_residual_samples;

                candidate_debug.static_normal_rms_deg =
                    StaticNormalRmsDeg(
                        *linked_live_track);

                candidate_debug.static_plane_d_rms_m =
                    StaticPlaneDRmsM(
                        *linked_live_track);

                persistent.last_center_W =
                    candidate.center_W;

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
                        aligned_normal_W,
                        persistent.reid_pending_last_normal_W);

                const double pending_plane_distance_difference_m =
                    std::abs(
                        aligned_d_W -
                        persistent.reid_pending_last_d_W);

                const double pending_support_gap_m =
                    WallAssociationV2SupportGap(
                        candidate.center_W,
                        candidate.radius_m,
                        persistent.reid_pending_last_center_W,
                        persistent.reid_pending_last_radius_m,
                        persistent.reference_normal_W);

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

            persistent.reid_pending_last_normal_W =
                aligned_normal_W.normalized();

            persistent.reid_pending_last_d_W =
                aligned_d_W;

            persistent.reid_pending_last_center_W =
                candidate.center_W;

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

            track.last_normal_W =
                aligned_normal_W.normalized();

            track.last_d_W =
                aligned_d_W;

            track.last_center_W =
                candidate.center_W;

            track.last_radius_m =
                candidate.radius_m;

            track.stable =
                true;

            track.static_confirmed =
                true;

            track.reference_normal_W =
                persistent.reference_normal_W;

            track.reference_d_W =
                persistent.reference_d_W;

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

            persistent.last_center_W =
                candidate.center_W;

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
            track.last_normal_W = candidate.normal_W;
            track.last_d_W = candidate.d_W;
            track.last_center_W = candidate.center_W;
            track.last_radius_m = candidate.radius_m;
            track.confirmation_normal_sum_W = candidate.normal_W;
            track.confirmation_d_sum_W = candidate.d_W;
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
                    track.confirmation_normal_sum_W =
                        Eigen::Vector3d::Zero();
                    track.confirmation_d_sum_W = 0.0;
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


RegistrationScan2LocalMap::RegistrationScan2LocalMap(
    const LidarRegistrationConfig &registration_config,
    const LocalMapConfig &local_map_config,
    const LoopDetectorConfig &loop_detector_config,
    const LoopRuntimeConfig &loop_runtime_config,
    const GroundConstraintConfig &ground_constraint_config,
    bool planar_motion_mode)
    : planar_motion_mode_(planar_motion_mode),
      registration_(registration_config),
      backend_refinement_registration_(
          MakeBackendRefinementRegistrationConfig(
              registration_config)),

      // Submap V1 defaults:
      //     15 keyframes / Submap
      //     5-keyframe overlap
      //
      // local_map_config is reused by the LocalMap builder inside each Submap.
      submap_manager_(
          SubmapManagerConfig(),
          local_map_config),

      loop_detector_(loop_detector_config),
      loop_consistency_checker_(loop_runtime_config.consistency),

      // Create a keyframe when either:
      //
      //     translation >= 0.5 m
      //
      // OR
      //
      //     rotation >= 5 deg
      //
      // The comparison is made against the LAST KEYFRAME pose,
      // not against the previous ordinary LiDAR frame.
      keyframe_detector_(0.5, 5.0)
{
    loop_runtime_config_ =
        loop_runtime_config;

    min_online_loop_edge_current_keyframe_spacing_ =
        std::max<std::size_t>(
            1,
            loop_runtime_config
                .min_online_loop_edge_current_keyframe_spacing);

    min_online_loop_edge_historical_keyframe_spacing_ =
        std::max<std::size_t>(
            1,
            loop_runtime_config
                .min_online_loop_edge_historical_keyframe_spacing);

    GroundIcpRuntime *ground_icp_runtime =
        RegisterGroundIcpRuntime(
            this,
            registration_config,
            ground_constraint_config);




    const LoopDetectorConfig &active_loop_config =
        loop_detector_.GetConfig();



    const LoopConsistencyConfig &active_consistency =
        loop_consistency_checker_.GetConfig();


    RefreshBackendOutputSnapshot();
    StartBackendWorker();
}

RegistrationScan2LocalMap::~RegistrationScan2LocalMap()
{
    StopBackendWorker();
    RemoveGroundIcpRuntime(this);
}

// ============================================================================
// Backend worker lifecycle.
// ============================================================================

bool RegistrationScan2LocalMap::AddFrame(
    const pcl::PointCloud<LIDAR_POINT>::ConstPtr &cloud_lidar,
    double timestamp,
    Eigen::Isometry3d &T_WL,
    LidarRegistrationResult &registration_result,
    const Eigen::Quaterniond *imu_relative_rotation)
{
    FrameTimingDiagnostics frame_timing;
    frame_timing.keyframes_before =
        keyframe_manager_.Size();

    frame_timing.keyframes_after =
        frame_timing.keyframes_before;

    FrameTimingReporter frame_timing_reporter(
        frame_timing);

    // =========================================================================
    // 0. Validate input.
    // =========================================================================
    if (!cloud_lidar || cloud_lidar->empty())
    {
        std::cerr
            << "RegistrationScan2LocalMap::AddFrame(): input cloud is empty."
            << std::endl;
        return false;
    }

    if (!std::isfinite(timestamp))
    {
        std::cerr
            << "RegistrationScan2LocalMap::AddFrame(): timestamp is invalid."
            << std::endl;
        return false;
    }

    // =========================================================================
    // Ground ICP V1.1 dual-branch input stage.
    //
    // Ordinary Scan-to-LocalMap keeps using cloud_lidar, i.e. the final sparse
    // registration cloud after the normal Voxel/SOR/ROR chain.
    //
    // Ground V4.0 instead consumes the one-shot Basic/ROI/CropBox cloud captured
    // by PreProcessor::preprocess() BEFORE the coarse registration voxel and
    // outlier filters.  Inside SegmentFrontendGround it is brought back to the
    // validated Ground-V4 operating point with a dedicated 0.15 m voxel.
    //
    // If the dense bridge is unavailable for any reason, we fall back to the
    // original registration cloud.  This preserves the V1 fail-safe behavior.
    // =========================================================================
    pcl::PointCloud<LIDAR_POINT>::ConstPtr ground_input_cloud =
        fr_slam::ConsumeGroundIcpDenseInput();

    const char *ground_input_source =
        "BASIC_BRIDGE";

    if (!ground_input_cloud ||
        ground_input_cloud->empty())
    {
        ground_input_cloud =
            cloud_lidar;

        ground_input_source =
            "REGISTRATION_FALLBACK";
    }

    const std::chrono::steady_clock::time_point
        ground_segment_start =
            std::chrono::steady_clock::now();

    const fr_slam::GroundSegmentationResult frontend_ground_result =
        SegmentFrontendGround(
            this,
            ground_input_cloud,
            ground_input_source);

    frame_timing.ground_segment_ms +=
        ElapsedMilliseconds(
            ground_segment_start,
            std::chrono::steady_clock::now());

    // =========================================================================
    // Multi-Plane Frontend V1.5.1
    //
    // V1.0 : BTC-style 3D voxel plane extraction + coplanar merge.
    // V1.1 : latest PlaneSet exposed to ROS node for RViz.
    // V1.2 : conservative Ground / Horizontal / Vertical / Oblique
    //        classification relative to current frontend up direction.
    //
    // Diagnostics only: NO pose/ICP/LocalMap/Ground/Loop/PGO feedback.
    // =========================================================================
    const fr_slam::MultiPlaneExtractor multi_plane_extractor;

    Eigen::Vector3d multi_plane_up_direction_L =
        T_WL_.rotation().transpose() *
        Eigen::Vector3d::UnitZ();

    if (!multi_plane_up_direction_L.allFinite() ||
        multi_plane_up_direction_L.norm() < 1.0e-9)
    {
        multi_plane_up_direction_L =
            Eigen::Vector3d::UnitZ();
    }

    fr_slam::MultiPlaneExtractionResult multi_plane_result =
        multi_plane_extractor.Extract(
            ground_input_cloud,
            multi_plane_up_direction_L);

    // V1.3 refinement is diagnostics-only. No plane residual is applied yet.
    fr_slam::RefinePlaneConstraintEligibility(
        multi_plane_result,
        frontend_ground_result,
        multi_plane_extractor.GetConfig(),
        T_WL_.rotation(),
        T_WL_.translation());

    fr_slam::StoreLatestMultiPlaneResult(
        multi_plane_result);

    static std::size_t multi_plane_debug_frame = 0;
    ++multi_plane_debug_frame;

    const WallAssociationV2FrameDebug wall_association_v3 =
        RunWallAssociationV3(
            multi_plane_result,
            T_WL_,
            multi_plane_debug_frame);

    for (std::size_t wall_index = 0;
         wall_index < wall_association_v3.candidate_debug.size();
         ++wall_index)
    {
        const WallAssociationV2CandidateDebug &association =
            wall_association_v3.candidate_debug[wall_index];

        if (!association.persistent_created &&
            !association.persistent_fallback &&
            !association.persistent_reacquired &&
            !association.persistent_rebound &&
            !association.reid_pending)
        {
            continue;
        }

        const char *event_name =
            association.persistent_reacquired
                ? "TRUE_REACQUIRED"
                : (association.persistent_fallback
                       ? "PERSISTENT_FALLBACK"
                       : (association.persistent_rebound
                              ? "REBOUND_TO_PERSISTENT"
                              : (association.reid_pending
                                     ? "REID_PENDING"
                                     : "STATIC_NEW")));

        std::cout
            << "WALL_PERSIST_V31_EVENT"
            << " | frame=" << multi_plane_debug_frame
            << " | physical_id=" << wall_index
            << " | live_track_id=" << association.track_id
            << " | persistent_wall_id="
            << association.persistent_wall_id
            << " | event=" << event_name
            << " | reid_pending_count="
            << association.reid_pending_count
            << " | observation_span="
            << association.observation_span_frames
            << " | total_hits="
            << association.total_hits
            << " | normal_rms_deg="
            << association.static_normal_rms_deg
            << " | plane_d_rms_m="
            << association.static_plane_d_rms_m
            << std::endl;
    }

    if ((multi_plane_debug_frame % 20U) == 0U)
    {
        std::cout
            << "MULTIPLANE_V151"
            << " | frame=" << multi_plane_debug_frame
            << " | input_points=" << multi_plane_result.input_points
            << " | voxels=" << multi_plane_result.occupied_voxels
            << " | raw_planes=" << multi_plane_result.raw_plane_patches
            << " | merged_before_filter="
            << multi_plane_result.merged_plane_count_before_filter
            << " | output_planes=" << multi_plane_result.planes.size()
            << " | ground=" << multi_plane_result.ground_candidate_count
            << " | horizontal=" << multi_plane_result.horizontal_plane_count
            << " | vertical=" << multi_plane_result.vertical_plane_count
            << " | oblique=" << multi_plane_result.oblique_plane_count
            << " | eligible_ground="
            << multi_plane_result.eligible_ground_plane_count
            << " | eligible_walls="
            << multi_plane_result.eligible_wall_plane_count
            << " | support_candidates="
            << multi_plane_result.support_candidate_count
            << " | active_support="
            << multi_plane_result.active_support_plane_count
            << " | active_mode="
            << fr_slam::SupportSurfaceModeName(
                   multi_plane_result.active_support_mode)
            << " | pending_count="
            << multi_plane_result.support_pending_confirmation_count
            << " | local_ground_candidates="
            << multi_plane_result.local_ground_candidate_count
            << " | active_ground="
            << multi_plane_result.active_ground_plane_count
            << " | ground_mode="
            << fr_slam::SupportSurfaceModeName(
                   multi_plane_result.active_support_mode)
            << " | wall_candidates="
            << multi_plane_result.wall_constraint_candidate_count
            << std::endl;

        std::cout
            << "WALL_ASSOC_V2_SUMMARY"
            << " | frame=" << multi_plane_debug_frame
            << " | raw_wall_candidates="
            << wall_association_v3.raw_wall_candidates
            << " | physical_wall_candidates="
            << wall_association_v3.physical_wall_candidates
            << " | fragments_merged="
            << wall_association_v3.fragments_merged
            << " | matched_tracks="
            << wall_association_v3.matched_tracks
            << " | new_tracks="
            << wall_association_v3.new_tracks
            << " | stable_walls="
            << wall_association_v3.stable_walls
            << " | active_walls="
            << wall_association_v3.active_walls
            << " | stable_tracks_alive="
            << wall_association_v3.stable_tracks_alive
            << " | tracks_alive="
            << wall_association_v3.tracks_alive
            << std::endl;

        std::cout
            << "WALL_PERSIST_V31_SUMMARY"
            << " | frame=" << multi_plane_debug_frame
            << " | live_tracks="
            << wall_association_v3.tracks_alive
            << " | live_stable="
            << wall_association_v3.stable_tracks_alive
            << " | static_confirmed_current="
            << wall_association_v3.static_confirmed_current
            << " | transient_current="
            << wall_association_v3.transient_current
            << " | persistent_walls="
            << wall_association_v3.persistent_walls
            << " | active_static_walls="
            << wall_association_v3.active_static_walls
            << " | dormant_walls="
            << wall_association_v3.dormant_walls
            << " | persistent_fallback="
            << wall_association_v3.fallback_this_frame
            << " | true_reacquired="
            << wall_association_v3.reacquired_this_frame
            << " | rebound="
            << wall_association_v3.rebound_this_frame
            << " | reid_pending="
            << wall_association_v3.reid_pending_current
            << std::endl;

        for (std::size_t wall_index = 0;
             wall_index < wall_association_v3.physical_walls.size();
             ++wall_index)
        {
            const WallAssociationV2PhysicalWall &wall =
                wall_association_v3.physical_walls[wall_index];

            const WallAssociationV2CandidateDebug &association =
                wall_association_v3.candidate_debug[wall_index];

            std::cout
                << "WALL_ASSOC_V2_CLUSTER"
                << " | frame=" << multi_plane_debug_frame
                << " | physical_id=" << wall_index
                << " | member_count="
                << wall.member_plane_indices.size()
                << " | member_ids=[";

            for (std::size_t member_index = 0;
                 member_index < wall.member_plane_indices.size();
                 ++member_index)
            {
                if (member_index > 0U)
                {
                    std::cout << ",";
                }

                std::cout
                    << wall.member_plane_indices[member_index];
            }

            std::cout
                << "]"
                << " | representative="
                << wall.representative_plane_index
                << " | quality=" << wall.quality
                << " | points=" << wall.total_points
                << " | radius=" << wall.radius_m
                << " | normal_W=["
                << wall.normal_W.x() << ","
                << wall.normal_W.y() << ","
                << wall.normal_W.z() << "]"
                << " | d_W=" << wall.d_W
                << std::endl;

            const char *association_status =
                "UNMATCHED";

            if (association.persistent_reacquired)
            {
                association_status =
                    "TRUE_REACQUIRED";
            }
            else if (association.persistent_fallback)
            {
                association_status =
                    "PERSISTENT_FALLBACK";
            }
            else if (association.persistent_rebound)
            {
                association_status =
                    "REBOUND_TO_PERSISTENT";
            }
            else if (association.reid_pending)
            {
                association_status =
                    "REID_PENDING";
            }
            else if (association.matched_existing_track)
            {
                association_status =
                    "MATCH";
            }
            else if (association.new_track)
            {
                association_status =
                    "NEW";
            }
            else if (association.low_quality_untracked)
            {
                association_status =
                    "LOW_QUALITY";
            }

            std::cout
                << "WALL_ASSOC_V2_TRACK"
                << " | frame=" << multi_plane_debug_frame
                << " | physical_id=" << wall_index
                << " | representative="
                << wall.representative_plane_index
                << " | track_id=" << association.track_id
                << " | status="
                << association_status
                << " | stable="
                << (association.stable ? 1 : 0)
                << " | active_wall="
                << (association.constraint_ready ? 1 : 0)
                << " | consecutive_hits="
                << association.consecutive_hits
                << " | total_hits="
                << association.total_hits
                << " | misses="
                << association.missed_frames
                << " | normal_diff_deg="
                << association.normal_difference_deg
                << " | plane_dist_diff_m="
                << association.plane_distance_difference_m
                << " | support_gap_m="
                << association.tangential_support_gap_m
                << " | assoc_score="
                << association.association_score
                << " | ref_normal_diff_deg="
                << association.reference_normal_difference_deg
                << " | ref_plane_dist_diff_m="
                << association.reference_plane_distance_difference_m
                << " | quality=" << wall.quality
                << " | members="
                << wall.member_plane_indices.size()
                << std::endl;

            const bool has_persistent_wall =
                association.persistent_wall_id !=
                std::numeric_limits<std::size_t>::max();

            const char *persistent_state =
                "TRANSIENT";

            if (association.low_quality_untracked)
            {
                persistent_state =
                    "UNTRACKED";
            }
            else if (association.persistent_reacquired)
            {
                persistent_state =
                    "TRUE_REACQUIRED";
            }
            else if (association.persistent_fallback)
            {
                persistent_state =
                    "PERSISTENT_FALLBACK";
            }
            else if (association.persistent_rebound)
            {
                persistent_state =
                    "REBOUND_TO_PERSISTENT";
            }
            else if (association.reid_pending)
            {
                persistent_state =
                    "REID_PENDING";
            }
            else if (association.persistent_created)
            {
                persistent_state =
                    "STATIC_NEW";
            }
            else if (has_persistent_wall)
            {
                persistent_state =
                    association.constraint_ready
                        ? "ACTIVE_STATIC"
                        : "PERSISTENT_OBSERVED";
            }
            else if (association.static_confirmed)
            {
                persistent_state =
                    "STATIC_CONFIRMED";
            }

            std::cout
                << "WALL_PERSIST_V31"
                << " | frame=" << multi_plane_debug_frame
                << " | physical_id=" << wall_index
                << " | live_track_id=" << association.track_id
                << " | persistent_valid="
                << (has_persistent_wall ? 1 : 0)
                << " | persistent_wall_id="
                << association.persistent_wall_id
                << " | state=" << persistent_state
                << " | static_confirmed="
                << (association.static_confirmed ? 1 : 0)
                << " | persistent_created="
                << (association.persistent_created ? 1 : 0)
                << " | fallback="
                << (association.persistent_fallback ? 1 : 0)
                << " | true_reacquired="
                << (association.persistent_reacquired ? 1 : 0)
                << " | rebound="
                << (association.persistent_rebound ? 1 : 0)
                << " | reid_pending="
                << (association.reid_pending ? 1 : 0)
                << " | reid_pending_count="
                << association.reid_pending_count
                << " | active_static="
                << ((has_persistent_wall &&
                     association.constraint_ready &&
                     !association.reid_pending)
                        ? 1
                        : 0)
                << " | observation_span="
                << association.observation_span_frames
                << " | total_hits="
                << association.total_hits
                << " | static_samples="
                << association.static_residual_samples
                << " | normal_rms_deg="
                << association.static_normal_rms_deg
                << " | plane_d_rms_m="
                << association.static_plane_d_rms_m
                << " | persistent_observations="
                << association.persistent_total_observations
                << " | quality=" << wall.quality
                << std::endl;
        }

        const std::size_t print_count =
            std::min<std::size_t>(
                8,
                multi_plane_result.planes.size());

        for (std::size_t plane_index = 0;
             plane_index < print_count;
             ++plane_index)
        {
            const fr_slam::PlaneObservation &plane =
                multi_plane_result.planes[plane_index];

            std::cout
                << "MULTIPLANE_V151_PLANE"
                << " | frame=" << multi_plane_debug_frame
                << " | id=" << plane_index
                << " | type="
                << fr_slam::PlaneSemanticTypeName(
                       plane.semantic_type)
                << " | angle_to_up_deg=" << plane.angle_to_up_deg
                << " | points=" << plane.point_count
                << " | patches=" << plane.merged_patch_count
                << " | center=["
                << plane.center_L.x() << ","
                << plane.center_L.y() << ","
                << plane.center_L.z() << "]"
                << " | normal=["
                << plane.normal_L.x() << ","
                << plane.normal_L.y() << ","
                << plane.normal_L.z() << "]"
                << " | d=" << plane.d
                << " | radius=" << plane.radius_m
                << " | min_eigen=" << plane.minimum_eigenvalue
                << " | ground_score=" << plane.ground_candidate_score
                << " | eligible=" << (plane.constraint_eligible ? 1 : 0)
                << " | quality=" << plane.constraint_quality_score
                << " | reject_mask=" << plane.constraint_rejection_mask
                << " | v4_normal_diff_deg="
                << plane.ground_v4_normal_difference_deg
                << " | v4_distance_diff_m="
                << plane.ground_v4_plane_distance_difference_m
                << " | support_candidate="
                << (plane.support_candidate ? 1 : 0)
                << " | active_support="
                << (plane.active_support ? 1 : 0)
                << " | support_mode="
                << fr_slam::SupportSurfaceModeName(
                       plane.support_mode)
                << " | support_score="
                << plane.support_selection_score
                << " | support_v4_confirmed="
                << (plane.support_v4_confirmed ? 1 : 0)
                << " | temporal_normal_diff_deg="
                << plane.support_temporal_normal_difference_deg
                << " | temporal_distance_diff_m="
                << plane.support_temporal_plane_distance_difference_m
                << " | lateral_center_m="
                << plane.support_lateral_center_m
                << " | lateral_sigma_m="
                << plane.support_lateral_sigma_m
                << " | corridor_gap_m="
                << plane.support_corridor_gap_m
                << " | local_ground_candidate="
                << (plane.local_ground_candidate ? 1 : 0)
                << " | active_ground="
                << (plane.active_ground ? 1 : 0)
                << " | wall_candidate="
                << (plane.wall_constraint_candidate ? 1 : 0)
                << " | local_long_m="
                << plane.local_ground_longitudinal_center_m
                << " | local_lat_m="
                << plane.local_ground_lateral_center_m
                << " | sigma_long_m="
                << plane.local_ground_longitudinal_sigma_m
                << " | sigma_lat_m="
                << plane.local_ground_lateral_sigma_m
                << " | roi_gap_long_m="
                << plane.local_ground_longitudinal_roi_gap_m
                << " | roi_gap_lat_m="
                << plane.local_ground_lateral_roi_gap_m
                << " | origin_gap_m="
                << plane.local_ground_origin_footprint_gap_m
                << " | local_ground_score="
                << plane.local_ground_selection_score
                << std::endl;
        }
    }

    // =========================================================================
    // 1. First frame initialization.
    //
    // There is no LocalMap yet, so the first frame does not run ICP.
    // The first LiDAR pose defines the SLAM World origin:
    //
    //     T_WL0 = Identity
    //
    // The first frame becomes KF0 and creates Active Submap 0.
    // =========================================================================
    if (!initialized_)
    {
        frame_timing.first_frame = true;
        frame_timing.keyframe = true;

        // The very first accepted LiDAR scan defines the World origin.
        //
        // Therefore:
        //
        //     LiDAR_0 == World
        //
        // and:
        //
        //     T_WL0 = I
        T_WL_ = Eigen::Isometry3d::Identity();

        // At startup there is no previous LiDAR motion yet.
        //
        // Identity means:
        //
        //     "predict no relative motion"
        //
        // until registration estimates the first real relative transform.
        last_relative_transform_ = Eigen::Isometry3d::Identity();

        // No frame has been rejected before the first frame.
        consecutive_rejected_frames_ = 0;

        // ---------------------------------------------------------------------
        // 1.1 Store KF0 in complete historical KeyframeManager.
        // ---------------------------------------------------------------------
        const std::chrono::steady_clock::time_point
            first_keyframe_store_start =
                std::chrono::steady_clock::now();

        const bool first_keyframe_stored =
            keyframe_manager_.AddKeyframe(
                timestamp,
                T_WL_,
                cloud_lidar,
                nullptr,
                ground_input_cloud);

        frame_timing.keyframe_store_ms +=
            ElapsedMilliseconds(
                first_keyframe_store_start,
                std::chrono::steady_clock::now());

        frame_timing.keyframes_after =
            keyframe_manager_.Size();

        if (!first_keyframe_stored)
        {
            std::cerr
                << "RegistrationScan2LocalMap::AddFrame(): "
                << "failed to store first keyframe."
                << std::endl;
            return false;
        }

        const Keyframe *first_keyframe =
            keyframe_manager_.Latest();

        if (first_keyframe == nullptr)
        {
            std::cerr
                << "RegistrationScan2LocalMap::AddFrame(): "
                << "first keyframe pointer is null."
                << std::endl;
            return false;
        }

        // ---------------------------------------------------------------------
        // 1.2 Backend work is asynchronous.
        //
        // PoseGraph / global map / BTC retrieval / loop verification are no
        // longer executed on the LiDAR processing thread.
        // ---------------------------------------------------------------------

        // ---------------------------------------------------------------------
        // 1.3 Create Active Submap 0 and insert KF0.
        //
        // Submap internally reuses LocalMap for:
        //
        //     transform -> merge -> voxel
        // ---------------------------------------------------------------------
        const std::chrono::steady_clock::time_point
            first_submap_start =
                std::chrono::steady_clock::now();

        const bool first_submap_ok =
            submap_manager_.AddKeyframe(
                *first_keyframe);

        frame_timing.submap_insert_ms +=
            ElapsedMilliseconds(
                first_submap_start,
                std::chrono::steady_clock::now());

        if (!first_submap_ok)
        {
            std::cerr
                << "RegistrationScan2LocalMap::AddFrame(): "
                << "failed to initialize Active Submap."
                << std::endl;
            return false;
        }

        // ---------------------------------------------------------------------
        // 1.5 Prepare point-to-plane target from the current Submap tracking map.
        // ---------------------------------------------------------------------
        const std::chrono::steady_clock::time_point
            first_prepare_target_start =
                std::chrono::steady_clock::now();

        const bool first_prepare_target_ok =
            registration_.PrepareTarget(
                submap_manager_.GetTrackingMap(),
                prepared_tracking_target_);

        frame_timing.prepare_target_ms +=
            ElapsedMilliseconds(
                first_prepare_target_start,
                std::chrono::steady_clock::now());

        if (!first_prepare_target_ok)
        {
            std::cerr
                << "RegistrationScan2LocalMap::AddFrame(): "
                << "failed to prepare first Submap tracking target."
                << std::endl;
            return false;
        }

        const std::chrono::steady_clock::time_point
            first_backend_enqueue_start =
                std::chrono::steady_clock::now();

        const bool first_backend_enqueued =
            EnqueueBackendKeyframe(
                *first_keyframe,
                submap_manager_.ActiveSubmapId());

        frame_timing.backend_enqueue_ms +=
            ElapsedMilliseconds(
                first_backend_enqueue_start,
                std::chrono::steady_clock::now());

        if (!first_backend_enqueued)
        {
            std::cerr
                << "Async backend enqueue failed"
                << " | keyframe=" << first_keyframe->id
                << std::endl;
        }

        // The first keyframe becomes the detector reference pose.
        keyframe_detector_.SetLastKeyframePose(T_WL_);

        initialized_ = true;

        // ---------------------------------------------------------------------
        // 1.6 First frame has no ICP, so create a successful result manually.
        // ---------------------------------------------------------------------
        registration_result = LidarRegistrationResult();
        registration_result.success = true;
        registration_result.converged = true;
        registration_result.iterations = 0;
        registration_result.correspondences = 0;
        registration_result.rmse = 0.0;
        registration_result.T_target_source = T_WL_;

        T_WL = T_WL_;


        frame_timing.accepted = true;
        frame_timing.keyframes_after =
            keyframe_manager_.Size();

        return true;
    }

    // =========================================================================
    // 2. Save previous ACCEPTED pose.
    //
    // Do not modify T_WL_ until the candidate registration passes the quality
    // gate. This follows:
    //
    //     Calculate -> Validate -> Commit
    // =========================================================================
    // T_WL_ always represents the LAST ACCEPTED LiDAR pose.
    //
    // If the immediately previous processed scan was rejected, T_WL_ still
    // points to an older accepted scan. This behavior is intentional.
    const Eigen::Isometry3d T_WL_previous = T_WL_;

    // =========================================================================
    // 3. Recovery-aware LiDAR constant-motion prediction.
    //
    // Normal tracking:
    //
    //     consecutive_rejected_frames_ = 0
    //     prediction_steps             = 1
    //
    // so the behavior is exactly the same as before:
    //
    //     T_guess = T_last_accepted * DeltaT
    //
    //
    // Short tracking interruption:
    //
    //     F100 accepted
    //     F101 rejected
    //     F102 current
    //
    // then:
    //
    //     consecutive_rejected_frames_ = 1
    //     prediction_steps             = 2
    //
    // and:
    //
    //     T_guess = T_F100 * DeltaT * DeltaT
    //
    // This lets the LiDAR translation prediction span rejected frames.
    //
    // IMPORTANT:
    //
    // The number of extrapolation steps is capped. Constant-motion
    // extrapolation is useful for SHORT recovery only. It must not grow
    // without bound after tracking has been lost for a long time.
    // =========================================================================
    // Number of one-frame motion increments that should be extrapolated.
    //
    // Examples:
    //
    // rejected = 0
    //     current is the next frame after last accepted
    //     -> use 1 motion step
    //
    // rejected = 1
    //     one frame was missed between last accepted and current
    //     -> use 2 motion steps
    //
    // rejected = 4
    //     -> use 5 motion steps
    //
    // rejected >= max limit
    //     -> keep the prediction capped
    //
    // The cap prevents a stale motion model from being extrapolated to
    // absurd distances after long-term tracking loss.
    const std::size_t prediction_steps =
        std::min(
            consecutive_rejected_frames_ + 1,
            max_recovery_prediction_steps_);

    // Accumulated relative prediction from the last accepted LiDAR frame
    // toward the current LiDAR frame.
    //
    // Start from Identity:
    //
    //     DeltaT_pred = I
    Eigen::Isometry3d predicted_relative_transform =
        Eigen::Isometry3d::Identity();

    // Repeated composition:
    //
    // prediction_steps = 1:
    //
    //     DeltaT_pred = DeltaT
    //
    // prediction_steps = 2:
    //
    //     DeltaT_pred = DeltaT * DeltaT
    //
    // prediction_steps = 3:
    //
    //     DeltaT_pred = DeltaT * DeltaT * DeltaT
    //
    // Because transforms are SE(3) transforms, we compose them by matrix /
    // Isometry multiplication rather than multiplying translation values alone.
    for (std::size_t i = 0;
         i < prediction_steps;
         ++i)
    {
        predicted_relative_transform =
            predicted_relative_transform *
            last_relative_transform_;
    }

    // Convert the relative motion prediction into a global pose prediction:
    //
    //     T_WL_guess
    //         =
    //     T_WL_last_accepted * T_last_accepted_current_prediction
    Eigen::Isometry3d initial_guess =
        T_WL_previous *
        predicted_relative_transform;

    // =========================================================================
    // 4. Optional IMU rotation prediction.
    //
    // Keep translation from LiDAR constant motion and replace only rotation:
    //
    //     R_WL_guess = R_WL_previous * R_relative(IMU)
    // =========================================================================
    if (imu_relative_rotation != nullptr &&
        imu_relative_rotation->coeffs().allFinite() &&
        imu_relative_rotation->norm() > 1.0e-12)
    {
        // Copy instead of modifying the caller's quaternion.
        Eigen::Quaterniond delta_q = *imu_relative_rotation;

        // Numerical integrations can introduce a very small norm error.
        // Normalize before converting to a rotation matrix.
        delta_q.normalize();

        // IMPORTANT:
        //
        // We overwrite ONLY the rotation.
        //
        // Translation remains the LiDAR constant-motion / recovery prediction.
        //
        //     R_WL_guess
        //         =
        //     R_WL_last_accepted * R_relative_IMU
        //
        // This is useful because IMU is very good at short-term rotational
        // prediction, while raw IMU translation integration is currently not
        // trusted in this frontend.
        initial_guess.linear() =
            T_WL_previous.rotation() * delta_q.toRotationMatrix();
    }

    // =========================================================================
    // 4.1 Tracking / recovery prediction diagnostics.
    //
    // This makes it obvious in the log whether we are:
    //
    //     TRACKING:
    //         no rejected frame is pending.
    //
    //     RECOVERY:
    //         one or more frames have been rejected and the translation
    //         prediction is being extrapolated across the missing interval.
    // =========================================================================

    // =========================================================================
    // 5. Primary Scan-to-LocalMap registration.
    //
    // source:
    //     current scan in CURRENT LiDAR coordinates.
    //
    // target:
    //     LocalMap in World coordinates.
    //
    // Therefore:
    //
    //     result.T_target_source = T_WL_current
    //
    // directly. Do NOT multiply by the previous global pose again.
    // =========================================================================
    LidarRegistrationResult result;

    const std::chrono::steady_clock::time_point
        primary_align_start =
            std::chrono::steady_clock::now();

    bool registration_success =
        registration_.Align(
            cloud_lidar,
            prepared_tracking_target_,
            initial_guess,
            result);

    LidarRegistrationResult ground_refined_result;

    GroundJointIcpStatus primary_ground_status =
        GroundJointIcpStatus::NotEligible;

    const bool primary_general_candidate_is_trusted =
        registration_success &&
        result.success &&
        result.T_target_source.matrix().allFinite() &&
        std::isfinite(result.rmse) &&
        result.rmse <= max_accepted_rmse_ &&
        result.correspondences >=
            min_accepted_correspondences_;

    if (primary_general_candidate_is_trusted)
    {
        primary_ground_status =
            RunTrustedGroundPlaneRefinementV14(
                this,
                cloud_lidar,
                prepared_tracking_target_,
                frontend_ground_result,
                result,
                max_accepted_rmse_,
                min_accepted_correspondences_,
                timestamp,
                "PRIMARY",
                ground_refined_result);
    }

    if (primary_ground_status ==
            GroundJointIcpStatus::Success &&
        ground_refined_result.success &&
        ground_refined_result
            .T_target_source
            .matrix()
            .allFinite() &&
        std::isfinite(
            ground_refined_result.rmse) &&
        ground_refined_result.rmse <=
            max_accepted_rmse_ &&
        ground_refined_result.correspondences >=
            min_accepted_correspondences_)
    {
        result = ground_refined_result;
        registration_success = result.success;
    }

    frame_timing.primary_align_ms +=
        ElapsedMilliseconds(
            primary_align_start,
            std::chrono::steady_clock::now());

    // =========================================================================
    // 5.1 Candidate quality check helper.
    //
    // IMPORTANT:
    //
    // This lambda does NOT commit anything.
    //
    // It only answers:
    //
    //     "Is this candidate good enough to continue?"
    //
    // The same rule is used for:
    //
    //     primary point-to-plane result
    //
    // and
    //
    //     point-to-plane result after coarse recovery.
    // =========================================================================
    const auto CandidatePassesQualityGate =
        [this](
            bool align_success,
            const LidarRegistrationResult &candidate)
        -> bool
    {
        if (!align_success ||
            !candidate.success)
        {
            return false;
        }

        if (!candidate.T_target_source
                 .matrix()
                 .allFinite())
        {
            return false;
        }

        if (!std::isfinite(candidate.rmse) ||
            candidate.rmse >
                max_accepted_rmse_)
        {
            return false;
        }

        if (candidate.correspondences <
            min_accepted_correspondences_)
        {
            return false;
        }

        return true;
    };

    bool candidate_passed =
        CandidatePassesQualityGate(
            registration_success,
            result);

    // =========================================================================
    // 5.2 Tracking Recovery V2
    //
    // Trigger:
    //
    //     normal point-to-plane Scan-to-LocalMap candidate is not good enough.
    //
    // Recovery strategy:
    //
    //     current LiDAR scan
    //              |
    //              v
    //     coarse Point-to-Point ICP
    //     against the current LocalMap
    //              |
    //              v
    //       T_WL_coarse
    //              |
    //              v
    //     existing Point-to-Plane
    //     Scan-to-LocalMap refinement
    //              |
    //              v
    //         same Quality Gate
    //
    // Why this is useful:
    //
    // The failure log shows point-to-plane Hessian degeneracy dominated by
    // translation directions. The old Recovery V1 only extrapolates the
    // constant-motion model. It does not create a new translation observation.
    //
    // Point-to-point ICP is used here only as a coarse reacquisition mechanism
    // to provide a fresh translation estimate from LiDAR geometry.
    //
    // IMU behavior is unchanged:
    //
    //     IMU rotation still helped build `initial_guess`.
    //
    // No LiDAR-IMU state fusion is added here.
    // =========================================================================
    bool used_coarse_recovery = false;
    double coarse_fitness =
        std::numeric_limits<double>::infinity();

    if (!candidate_passed)
    {
        frame_timing.recovery_triggered = true;


        // -----------------------------------------------------------------
        // ROS2 debug block:
        //
        // Print exactly WHY point-to-plane failed and how far the primary
        // candidate moved away from the prediction.
        //
        // Grep:
        //
        //     RECOVERY_DEBUG
        // -----------------------------------------------------------------
        const char *primary_reject_reason =
            "UNKNOWN";

        if (!registration_success ||
            !result.success)
        {
            primary_reject_reason =
                "ALIGN_FAILED";
        }
        else if (!result.T_target_source
                      .matrix()
                      .allFinite())
        {
            primary_reject_reason =
                "NONFINITE_TRANSFORM";
        }
        else if (!std::isfinite(result.rmse) ||
                 result.rmse >
                     max_accepted_rmse_)
        {
            primary_reject_reason =
                "RMSE";
        }
        else if (result.correspondences <
                 min_accepted_correspondences_)
        {
            primary_reject_reason =
                "CORRESPONDENCE";
        }

        Eigen::Vector3d primary_correction =
            Eigen::Vector3d::Constant(
                std::numeric_limits<double>::quiet_NaN());

        if (result.T_target_source.matrix().allFinite())
        {
            primary_correction =
                result.T_target_source.translation() -
                initial_guess.translation();
        }

        const double primary_rotation_correction_deg =
            result.T_target_source.matrix().allFinite()
                ? RelativeRotationDeg(
                      initial_guess,
                      result.T_target_source)
                : std::numeric_limits<double>::quiet_NaN();



        Eigen::Isometry3d T_WL_coarse =
            initial_guess;

        const std::chrono::steady_clock::time_point
            recovery_coarse_start =
                std::chrono::steady_clock::now();

        const bool coarse_success =
            RunCoarsePointToPointRecovery(
                cloud_lidar,
                submap_manager_.GetTrackingMap(),
                initial_guess,
                T_WL_coarse,
                coarse_fitness);

        frame_timing.recovery_coarse_ms +=
            ElapsedMilliseconds(
                recovery_coarse_start,
                std::chrono::steady_clock::now());


        const Eigen::Vector3d coarse_correction =
            T_WL_coarse.translation() -
            initial_guess.translation();

        const double coarse_rotation_correction_deg =
            RelativeRotationDeg(
                initial_guess,
                T_WL_coarse);


        if (coarse_success)
        {
            // -------------------------------------------------------------
            // Coarse ICP is NOT accepted as final odometry.
            //
            // It becomes only a new initial guess for the original
            // point-to-plane Scan-to-LocalMap optimizer.
            // -------------------------------------------------------------
            LidarRegistrationResult refined_result;

            const std::chrono::steady_clock::time_point
                recovery_refine_start =
                    std::chrono::steady_clock::now();

            bool refined_success =
                registration_.Align(
                    cloud_lidar,
                    prepared_tracking_target_,
                    T_WL_coarse,
                    refined_result);

            LidarRegistrationResult
                recovery_ground_refined_result;

            GroundJointIcpStatus
                recovery_ground_status =
                    GroundJointIcpStatus::NotEligible;

            if (CandidatePassesQualityGate(
                    refined_success,
                    refined_result))
            {
                recovery_ground_status =
                    RunTrustedGroundPlaneRefinementV14(
                        this,
                        cloud_lidar,
                        prepared_tracking_target_,
                        frontend_ground_result,
                        refined_result,
                        max_accepted_rmse_,
                        min_accepted_correspondences_,
                        timestamp,
                        "RECOVERY",
                        recovery_ground_refined_result);
            }

            if (recovery_ground_status ==
                    GroundJointIcpStatus::Success &&
                CandidatePassesQualityGate(
                    true,
                    recovery_ground_refined_result))
            {
                refined_result =
                    recovery_ground_refined_result;
                refined_success =
                    refined_result.success;
            }

            frame_timing.recovery_refine_ms +=
                ElapsedMilliseconds(
                    recovery_refine_start,
                    std::chrono::steady_clock::now());

            const bool refined_passed =
                CandidatePassesQualityGate(
                    refined_success,
                    refined_result);


            const Eigen::Vector3d refine_correction =
                refined_result.T_target_source.translation() -
                T_WL_coarse.translation();

            const double refine_rotation_correction_deg =
                RelativeRotationDeg(
                    T_WL_coarse,
                    refined_result.T_target_source);


            // Keep the refined diagnostics as the final candidate diagnostics.
            //
            // Even if refinement still fails, the caller can see why.
            result =
                refined_result;

            registration_success =
                refined_success;

            candidate_passed =
                refined_passed;

            if (candidate_passed)
            {
                used_coarse_recovery = true;
                frame_timing.coarse_recovery_accepted = true;
            }
        }
    }

    // =========================================================================
    // 5.3 Ground ICP V1.4 Frozen World-Plane Anchor.
    //
    // The candidate above always starts with ordinary all-scene ICP.  A trusted
    // Ground frame may then make a bounded roll/pitch/z-only correction against
    // the frozen reference plane.  Any safety-gate failure keeps the ordinary
    // ICP candidate unchanged.
    // =========================================================================

    // Always expose the final candidate diagnostics to the caller.
    registration_result = result;

    // Reaching max iterations is only a warning. The Quality Gate below is
    // still the final acceptance rule.
    if (registration_success &&
        result.success &&
        !result.converged)
    {
        std::cerr
            << "RegistrationScan2LocalMap::AddFrame(): "
            << "registration reached max iterations."
            << " corr="
            << result.correspondences
            << " rmse="
            << result.rmse
            << std::endl;
    }

    // =========================================================================
    // 6. Final Quality Gate.
    //
    // Only ONE rejection counter update happens here.
    //
    // That is important because one LiDAR frame may have attempted:
    //
    //     primary registration
    //     + coarse recovery
    //     + refined registration
    //
    // but it is still only ONE LiDAR frame.
    //
    // A rejected result must NOT:
    //
    //     update T_WL_,
    //     update last_relative_transform_,
    //     create a keyframe,
    //     update LocalMap,
    //     replace prepared_tracking_target_.
    // =========================================================================
    if (!candidate_passed)
    {
        ++consecutive_rejected_frames_;

        if (!registration_success ||
            !result.success)
        {
            std::cerr
                << "RegistrationScan2LocalMap::AddFrame(): "
                << "registration/recovery failed."
                << " | consecutive_rejected="
                << consecutive_rejected_frames_
                << std::endl;
        }
        else if (!result.T_target_source
                      .matrix()
                      .allFinite())
        {
            std::cerr
                << "RegistrationScan2LocalMap::AddFrame(): "
                << "registration rejected because transform contains NaN/Inf."
                << " | consecutive_rejected="
                << consecutive_rejected_frames_
                << std::endl;
        }
        else if (!std::isfinite(result.rmse) ||
                 result.rmse >
                     max_accepted_rmse_)
        {
        }
        else
        {
            std::cerr
                << "RegistrationScan2LocalMap::AddFrame(): "
                << "registration rejected by correspondence count."
                << " corr="
                << result.correspondences
                << " min="
                << min_accepted_correspondences_
                << " | consecutive_rejected="
                << consecutive_rejected_frames_
                << std::endl;
        }

        frame_timing.keyframes_after =
            keyframe_manager_.Size();

        return false;
    }

    if (planar_motion_mode_)
    {
        const Eigen::Isometry3d unprojected_pose =
            result.T_target_source;

        result.T_target_source =
            ProjectPoseToPlanar(
                unprojected_pose);

        registration_result =
            result;

    }

    if (used_coarse_recovery)
    {

        const Eigen::Vector3d final_jump =
            result.T_target_source.translation() -
            T_WL_previous.translation();

        const double final_rotation_jump_deg =
            RelativeRotationDeg(
                T_WL_previous,
                result.T_target_source);

    }

    // Candidate global pose estimated directly by Scan-to-LocalMap.
    //
    // Because:
    //
    //     target = World-frame LocalMap
    //     source = current LiDAR scan
    //
    // registration returns:
    //
    //     T_target_source
    //         =
    //     T_WL_current
    //
    // There is NO additional pose accumulation here.
    const Eigen::Isometry3d T_WL_current = result.T_target_source;

    // =========================================================================
    // 7. Relative LiDAR motion for next-frame constant-motion prediction.
    //
    //     T_previous_current = T_WL_previous^-1 * T_WL_current
    // =========================================================================
    // Relative transform from the LAST ACCEPTED pose to the current accepted
    // candidate:
    //
    //     T_previous_current
    //         =
    //     T_WL_previous^-1 * T_WL_current
    //
    // Normal tracking:
    //     this spans one LiDAR frame interval.
    //
    // Recovery:
    //     this may span several LiDAR frame intervals.
    //
    // This distinction is critical later when updating the one-frame motion
    // model.
    const Eigen::Isometry3d T_previous_current =
        T_WL_previous.inverse() * T_WL_current;

    if (!T_previous_current.matrix().allFinite())
    {
        ++consecutive_rejected_frames_;

        std::cerr
            << "RegistrationScan2LocalMap::AddFrame(): "
            << "relative transform contains NaN/Inf."
            << " | consecutive_rejected="
            << consecutive_rejected_frames_
            << std::endl;

        return false;
    }

    // =========================================================================
    // 8. Keyframe decision.
    //
    // IMPORTANT:
    //     Compare with LAST KEYFRAME, not previous ordinary scan.
    // =========================================================================
    // These are distances relative to the LAST KEYFRAME pose.
    //
    // They are diagnostic outputs from KeyframeDetector and are not the
    // frame-to-frame odometry increments.
    double keyframe_translation = 0.0;
    double keyframe_rotation_deg = 0.0;

    const std::chrono::steady_clock::time_point
        keyframe_decision_start =
            std::chrono::steady_clock::now();

    const bool is_keyframe =
        keyframe_detector_.ShouldCreateKeyframe(
            T_WL_current,
            keyframe_translation,
            keyframe_rotation_deg);

    frame_timing.keyframe_decision_ms +=
        ElapsedMilliseconds(
            keyframe_decision_start,
            std::chrono::steady_clock::now());

    frame_timing.keyframe =
        is_keyframe;

    // =========================================================================
    // 9. Keyframe / Submap update.
    //
    // ONLY keyframes:
    //
    //     KeyframeManager -> complete historical record
    //     SubmapManager   -> Active/Previous lifecycle
    //     LocalMap        -> internal Submap cloud builder
    // =========================================================================
    if (is_keyframe)
    {
        // ---------------------------------------------------------------------
        // 9.1 Store historical Keyframe.
        // ---------------------------------------------------------------------
        const std::chrono::steady_clock::time_point
            keyframe_store_start =
                std::chrono::steady_clock::now();

        Eigen::Matrix<double, 6, 6> odom_information =
            Eigen::Matrix<double, 6, 6>::Identity();

        const bool dynamic_information_valid =
            BuildOdometryInformationV2(
                result,
                T_WL_current,
                odom_information);

        double maximum_absolute_off_diagonal =
            0.0;

        double maximum_translation_rotation_coupling =
            0.0;

        if (dynamic_information_valid)
        {
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

                    maximum_absolute_off_diagonal =
                        std::max(
                            maximum_absolute_off_diagonal,
                            std::abs(
                                odom_information(i, j)));

                    const bool translation_rotation_pair =
                        (i < 3 && j >= 3) ||
                        (i >= 3 && j < 3);

                    if (translation_rotation_pair)
                    {
                        maximum_translation_rotation_coupling =
                            std::max(
                                maximum_translation_rotation_coupling,
                                std::abs(
                                    odom_information(i, j)));
                    }
                }
            }
        }

        const bool keyframe_stored =
            keyframe_manager_.AddKeyframe(
                timestamp,
                T_WL_current,
                cloud_lidar,
                dynamic_information_valid
                    ? &odom_information
                    : nullptr,
                ground_input_cloud);

        frame_timing.keyframe_store_ms +=
            ElapsedMilliseconds(
                keyframe_store_start,
                std::chrono::steady_clock::now());

        frame_timing.keyframes_after =
            keyframe_manager_.Size();

        if (!keyframe_stored)
        {
            std::cerr
                << "RegistrationScan2LocalMap::AddFrame(): "
                << "failed to store historical keyframe."
                << std::endl;
            return false;
        }

        const Keyframe *new_keyframe =
            keyframe_manager_.Latest();

        if (new_keyframe == nullptr)
        {
            std::cerr
                << "RegistrationScan2LocalMap::AddFrame(): "
                << "latest keyframe pointer is null."
                << std::endl;
            return false;
        }

        // ---------------------------------------------------------------------
        // 9.2 Backend work is asynchronous.
        //
        // The frontend only stores the Keyframe and updates its tracking
        // Submap. PoseGraph / map / BTC retrieval / LoopVerifier run later in
        // backend_thread_.
        // ---------------------------------------------------------------------

        // ---------------------------------------------------------------------
        // 9.3 Capture the frontend Submap that OWNS this Keyframe BEFORE insertion.
        //
        // AddKeyframe() may fill it and immediately create the next overlapping
        // Active Submap. We keep this id only for local-neighborhood rejection,
        // temporal continuity, and redundant-loop suppression. It is NOT a
        // PoseGraph vertex id.
        // ---------------------------------------------------------------------
        const Submap *current_owner_before_add =
            submap_manager_.ActiveSubmap();

        if (current_owner_before_add == nullptr ||
            !current_owner_before_add->has_origin_pose ||
            !current_owner_before_add->T_WS.matrix().allFinite())
        {
            std::cerr
                << "RegistrationScan2LocalMap::AddFrame(): "
                << "current Active Submap is invalid before keyframe insertion."
                << std::endl;
            return false;
        }

        const std::size_t current_owner_submap_id =
            current_owner_before_add->id;

        // Add Keyframe to Active Submap. If it becomes full, SubmapManager may
        // finish it and create the next Active Submap here.
        const std::chrono::steady_clock::time_point
            submap_insert_start =
                std::chrono::steady_clock::now();

        const bool submap_insert_ok =
            submap_manager_.AddKeyframe(
                *new_keyframe);

        frame_timing.submap_insert_ms +=
            ElapsedMilliseconds(
                submap_insert_start,
                std::chrono::steady_clock::now());

        if (!submap_insert_ok)
        {
            std::cerr
                << "RegistrationScan2LocalMap::AddFrame(): "
                << "failed to update SubmapManager."
                << std::endl;
            return false;
        }

        if (submap_manager_.LastAddStartedNewSubmap())
        {
            // V6: Submap transition changes only frontend/local-map geometry.
            // PoseGraph vertices were already added per Keyframe above.
        }

        // ---------------------------------------------------------------------
        // 9.5 Submap tracking map changed -> rebuild registration target.
        // ---------------------------------------------------------------------
        PreparedLidarTarget new_prepared_active_submap;

        const std::chrono::steady_clock::time_point
            prepare_target_start =
                std::chrono::steady_clock::now();

        const bool prepare_target_ok =
            registration_.PrepareTarget(
                submap_manager_.GetTrackingMap(),
                new_prepared_active_submap);

        frame_timing.prepare_target_ms +=
            ElapsedMilliseconds(
                prepare_target_start,
                std::chrono::steady_clock::now());

        if (!prepare_target_ok)
        {
            std::cerr
                << "RegistrationScan2LocalMap::AddFrame(): "
                << "failed to prepare updated Submap tracking target."
                << std::endl;
            return false;
        }

        prepared_tracking_target_ =
            std::move(
                new_prepared_active_submap);

        const std::chrono::steady_clock::time_point
            backend_enqueue_start =
                std::chrono::steady_clock::now();

        const bool backend_enqueued =
            EnqueueBackendKeyframe(
                *new_keyframe,
                current_owner_submap_id);

        frame_timing.backend_enqueue_ms +=
            ElapsedMilliseconds(
                backend_enqueue_start,
                std::chrono::steady_clock::now());

        if (!backend_enqueued)
        {
            std::cerr
                << "Async backend enqueue failed"
                << " | keyframe=" << new_keyframe->id
                << std::endl;
        }

        keyframe_detector_.SetLastKeyframePose(
            T_WL_current);
    }

    // =========================================================================
    // Tracking recovery state.
    //
    // IMPORTANT:
    //
    // If the previous frames were rejected, then:
    //
    //     T_previous_current
    //
    // is NOT a one-frame LiDAR motion.
    //
    // Example:
    //
    //     F100 accepted
    //     F101 rejected
    //     F102 rejected
    //     F103 rejected
    //     F104 accepted
    //
    // Then:
    //
    //     T_previous_current
    //         =
    //     T_F100^-1 * T_F104
    //
    // This is a FOUR-frame accumulated motion.
    //
    // Therefore, after a recovery we must NOT directly store it into:
    //
    //     last_relative_transform_
    //
    // because last_relative_transform_ is supposed to represent approximately
    // ONE LiDAR-frame motion for the next constant-motion prediction.
    // =========================================================================
    // If this value is true, the current candidate is the FIRST accepted
    // frame after one or more consecutive rejected scans.
    //
    // Example:
    //
    //     F100 accepted
    //     F101 rejected
    //     F102 rejected
    //     F103 accepted  <-- recovered_from_tracking_loss == true
    const bool recovered_from_tracking_loss =
        consecutive_rejected_frames_ > 0;

    // =========================================================================
    // Motion Model Gate V1
    //
    // PURPOSE:
    //
    // A pose can pass the ordinary Quality Gate:
    //
    //     transform is finite
    //     RMSE is acceptable
    //     correspondence count is acceptable
    //
    // but its frame-to-frame motion can still be suspicious.
    //
    // This is especially possible in geometrically weak / degenerate scenes:
    // point-to-plane residual can remain small even though one weakly
    // constrained translation direction jumps too much.
    //
    // IMPORTANT DESIGN DECISION:
    //
    //     Pose acceptance
    //
    // and
    //
    //     Motion-model update
    //
    // are TWO DIFFERENT decisions.
    //
    // If the current pose passed the normal Quality Gate, we still accept:
    //
    //     T_WL_current
    //
    // However, if its relative translation suddenly becomes much larger than
    // the previous reliable one-frame motion, we do NOT allow that suspicious
    // relative transform to overwrite:
    //
    //     last_relative_transform_
    //
    // The old reliable motion model is kept for the next initial guess.
    //
    // Example from the failure log:
    //
    //     previous reliable translation ~= 0.125 m
    //     current relative translation  ~= 0.291 m
    //
    //     ratio ~= 2.32
    //
    // The pose may still be usable, but 0.291 m should not immediately become
    // the next constant-motion predictor.
    //
    // V1 checks translation only. Later we can also incorporate Hessian
    // degeneracy / weak-direction diagnostics into this decision.
    // =========================================================================
    const double previous_motion_translation =
        last_relative_transform_.translation().norm();

    const double current_motion_translation =
        T_previous_current.translation().norm();

    // Do not compute a meaningful ratio when the previous motion is almost
    // zero, otherwise tiny numerical motion could make the ratio explode.
    constexpr double min_reference_translation = 0.05;

    // A new one-frame translation larger than 2x the previous reliable
    // one-frame translation is considered suspicious in V1.
    constexpr double max_translation_ratio = 2.0;

    // Additional absolute floor:
    //
    // We only block a "large jump" if the current translation itself is at
    // least 0.20 m. This avoids treating harmless changes such as:
    //
    //     0.02 m -> 0.05 m
    //
    // as a serious motion-model fault just because the ratio is large.
    constexpr double min_suspicious_translation = 0.20;

    double motion_translation_ratio = 1.0;

    if (previous_motion_translation >
        min_reference_translation)
    {
        motion_translation_ratio =
            current_motion_translation /
            previous_motion_translation;
    }

    bool motion_model_update_allowed =
        !used_coarse_recovery;

    // If coarse recovery was required, the final pose may be valid but this
    // frame is deliberately NOT used to refresh the one-frame constant-motion
    // predictor. The next normal accepted frame can rebuild that predictor.
    if (used_coarse_recovery)
    {
    }

    // Recovery frames are already handled separately:
    //
    //     T_previous_current
    //
    // spans multiple LiDAR intervals after rejected frames, so it must never
    // be used directly as a one-frame motion model.
    //
    // Therefore Motion Model Gate V1 is only evaluated during normal tracking.
    if (!recovered_from_tracking_loss &&
        previous_motion_translation >
            min_reference_translation &&
        current_motion_translation >
            min_suspicious_translation &&
        motion_translation_ratio >
            max_translation_ratio)
    {
        motion_model_update_allowed = false;
    }

    if (recovered_from_tracking_loss)
    {
    }

    // =========================================================================
    // Commit accepted global pose.
    //
    // Every accepted frame updates the global LiDAR pose, including a frame
    // that was successfully recovered.
    // =========================================================================
    // Commit the accepted global pose to internal state.
    //
    // This is always safe here because:
    //
    //     registration succeeded
    //     Quality Gate passed
    //     keyframe update (if required) succeeded
    //
    // World-Z decomposition is intentionally evaluated HERE: Ground ICP has
    // already produced the final candidate and every later frontend operation
    // that can reject the frame has succeeded. Therefore this diagnostic never
    // accumulates an uncommitted/rejected pose.

    T_WL_ =
        T_WL_current;

    // Also write the same accepted pose to the caller's output parameter.
    T_WL =
        T_WL_current;

    // =========================================================================
    // Update constant-motion model.
    //
    // Normal TRACKING:
    //
    //     F100 accepted
    //     F101 accepted
    //
    // T_previous_current is truly one-frame motion, so we update:
    //
    //     last_relative_transform_ = T_F100^-1 * T_F101
    //
    //
    // RECOVERY:
    //
    //     F100 accepted
    //     F101 rejected
    //     F102 rejected
    //     F103 accepted
    //
    // T_previous_current is accumulated motion across THREE scan intervals.
    //
    // DO NOT use that accumulated transform as the next one-frame motion model.
    //
    // Instead, temporarily keep the last reliable single-frame motion.
    //
    // If the very next frame F104 is accepted normally, then:
    //
    //     T_F103^-1 * T_F104
    //
    // becomes a true one-frame motion again and the model will automatically
    // update on that frame.
    // =========================================================================
    if (!recovered_from_tracking_loss &&
        motion_model_update_allowed)
    {
        // NORMAL TRACKING + MOTION MODEL GATE PASSED:
        //
        // previous accepted frame and current frame are adjacent accepted
        // LiDAR scans, and the new relative translation is not suspicious.
        //
        // Therefore T_previous_current becomes the new one-frame predictor.
        last_relative_transform_ =
            T_previous_current;
    }
    else if (recovered_from_tracking_loss)
    {
        // RECOVERY:
        //
        // Keep the old reliable one-frame model because T_previous_current
        // spans multiple LiDAR intervals.
    }
    else
    {
        // NORMAL TRACKING, BUT MOTION MODEL GATE REJECTED THE UPDATE:
        //
        // IMPORTANT:
        //     The POSE is still accepted.
        //
        // Only the constant-motion predictor is protected from a suspicious
        // frame-to-frame jump.
    }

    // Recovery has now been successfully completed.
    //
    // The next scan starts again in normal TRACKING mode:
    //
    //     rejected_frames = 0
    //     prediction_steps = 1
    //
    // If that next scan is accepted, it will provide a fresh one-frame
    // relative transform and automatically refresh last_relative_transform_.
    consecutive_rejected_frames_ = 0;

    // =========================================================================
    // 12. Diagnostics.
    //
    // Expected after enough motion:
    //
    //     keyframes=25 | map_frames=10
    //
    // KeyframeManager = full history
    // LocalMap        = recent keyframe window
    // =========================================================================

    frame_timing.accepted = true;
    frame_timing.keyframes_after =
        keyframe_manager_.Size();

    return true;
}

// ============================================================================
// BuildCandidateCenteredHistoricalTarget() -- V19
//
// Build one small historical LocalMap in the candidate Keyframe frame K.
//
// IMPORTANT:
//   Loop retrieval unit      = Keyframe
//   Geometry verification   = Keyframe-centered LocalMap
//   PoseGraph loop endpoint = Keyframe
//
// This removes the old inconsistency where KF93/KF94/KF95 all verified against
// exactly the same finished Submap cloud and therefore were not truly
// independent geometric anchors.
// ============================================================================

Eigen::Isometry3d RegistrationScan2LocalMap::GetPose() const
{
    return T_WL_;
}

// Return the ACTUAL current registration target.
pcl::PointCloud<LIDAR_POINT>::ConstPtr
RegistrationScan2LocalMap::GetLocalMap() const
{
    return submap_manager_.GetTrackingMap();
}

RegistrationScan2LocalMap::BackendMapSnapshot
RegistrationScan2LocalMap::GetBackendMapSnapshot() const
{
    std::lock_guard<std::mutex> lock(
        backend_output_mutex_);

    BackendMapSnapshot snapshot;

    snapshot.raw_map =
        backend_raw_map_snapshot_;

    snapshot.optimized_map =
        backend_optimized_map_snapshot_;

    snapshot.refined_map =
        backend_refined_map_snapshot_;

    snapshot.refinement_historical_target =
        backend_refinement_historical_target_snapshot_;

    snapshot.refinement_current_before =
        backend_refinement_current_before_snapshot_;

    snapshot.refinement_current_after =
        backend_refinement_current_after_snapshot_;

    snapshot.global_revision =
        backend_global_map_revision_snapshot_;

    snapshot.refined_revision =
        backend_refined_map_revision_snapshot_;

    snapshot.refinement_debug_revision =
        backend_refinement_debug_revision_snapshot_;

    return snapshot;
}

pcl::PointCloud<LIDAR_POINT>::ConstPtr
RegistrationScan2LocalMap::GetRawKeyframeMap() const
{
    std::lock_guard<std::mutex> lock(
        backend_output_mutex_);

    return backend_raw_map_snapshot_;
}

pcl::PointCloud<LIDAR_POINT>::ConstPtr
RegistrationScan2LocalMap::GetOptimizedMap() const
{
    std::lock_guard<std::mutex> lock(
        backend_output_mutex_);

    return backend_optimized_map_snapshot_;
}

pcl::PointCloud<LIDAR_POINT>::ConstPtr
RegistrationScan2LocalMap::GetRefinedMap() const
{
    std::lock_guard<std::mutex> lock(
        backend_output_mutex_);

    return backend_refined_map_snapshot_;
}

pcl::PointCloud<LIDAR_POINT>::ConstPtr
RegistrationScan2LocalMap::GetRefinementHistoricalTarget() const
{
    std::lock_guard<std::mutex> lock(
        backend_output_mutex_);

    return backend_refinement_historical_target_snapshot_;
}

pcl::PointCloud<LIDAR_POINT>::ConstPtr
RegistrationScan2LocalMap::GetRefinementCurrentBefore() const
{
    std::lock_guard<std::mutex> lock(
        backend_output_mutex_);

    return backend_refinement_current_before_snapshot_;
}

pcl::PointCloud<LIDAR_POINT>::ConstPtr
RegistrationScan2LocalMap::GetRefinementCurrentAfter() const
{
    std::lock_guard<std::mutex> lock(
        backend_output_mutex_);

    return backend_refinement_current_after_snapshot_;
}

std::size_t RegistrationScan2LocalMap::RefinementDebugRevision() const
{
    std::lock_guard<std::mutex> lock(
        backend_output_mutex_);

    return backend_refinement_debug_revision_snapshot_;
}

std::size_t RegistrationScan2LocalMap::RefinedMapRevision() const
{
    std::lock_guard<std::mutex> lock(
        backend_output_mutex_);

    return backend_refined_map_revision_snapshot_;
}

std::size_t RegistrationScan2LocalMap::GlobalMapRevision() const
{
    std::lock_guard<std::mutex> lock(
        backend_output_mutex_);

    return backend_global_map_revision_snapshot_;
}

bool RegistrationScan2LocalMap::HasMapOdomCorrection() const
{
    std::lock_guard<std::mutex> lock(
        backend_output_mutex_);

    return backend_has_map_odom_correction_snapshot_;
}

Eigen::Isometry3d RegistrationScan2LocalMap::GetMapOdomCorrection() const
{
    std::lock_guard<std::mutex> lock(
        backend_output_mutex_);

    return backend_T_map_odom_snapshot_;
}

Eigen::Isometry3d RegistrationScan2LocalMap::GetCorrectedPose(
    const Eigen::Isometry3d &T_odom_lidar) const
{
    if (!T_odom_lidar.matrix().allFinite())
    {
        return Eigen::Isometry3d::Identity();
    }

    Eigen::Isometry3d T_map_odom =
        Eigen::Isometry3d::Identity();

    {
        std::lock_guard<std::mutex> lock(
            backend_output_mutex_);

        T_map_odom =
            backend_T_map_odom_snapshot_;
    }

    return T_map_odom *
           T_odom_lidar;
}

std::size_t RegistrationScan2LocalMap::MapOdomRevision() const
{
    std::lock_guard<std::mutex> lock(
        backend_output_mutex_);

    return backend_map_odom_revision_snapshot_;
}

// Backward-compatible diagnostic:
// now means number of keyframes in Active Submap.
std::size_t RegistrationScan2LocalMap::LocalMapFrameCount() const
{
    return submap_manager_.ActiveKeyframeCount();
}

std::size_t RegistrationScan2LocalMap::LocalMapPointCount() const
{
    return submap_manager_.TrackingPointCount();
}

std::size_t RegistrationScan2LocalMap::KeyframeCount() const
{
    return keyframe_manager_.Size();
}

const std::vector<Keyframe> &
RegistrationScan2LocalMap::GetKeyframes() const
{
    return keyframe_manager_.GetAllKeyframes();
}

std::size_t RegistrationScan2LocalMap::SubmapCount() const
{
    return submap_manager_.SubmapCount();
}

const Submap *
RegistrationScan2LocalMap::GetActiveSubmap() const
{
    return submap_manager_.ActiveSubmap();
}

const Submap *
RegistrationScan2LocalMap::GetPreviousSubmap() const
{
    return submap_manager_.PreviousSubmap();
}

PoseGraph
RegistrationScan2LocalMap::GetPoseGraphSnapshot() const
{
    std::lock_guard<std::mutex> lock(
        backend_output_mutex_);

    return backend_pose_graph_snapshot_;
}

const PoseGraph &
RegistrationScan2LocalMap::GetPoseGraph() const
{
    thread_local PoseGraph snapshot;

    snapshot =
        GetPoseGraphSnapshot();

    return snapshot;
}

std::size_t
RegistrationScan2LocalMap::PoseGraphNodeCount() const
{
    return GetPoseGraphSnapshot().NodeCount();
}

std::size_t
RegistrationScan2LocalMap::PoseGraphEdgeCount() const
{
    return GetPoseGraphSnapshot().EdgeCount();
}

std::size_t
RegistrationScan2LocalMap::PoseGraphOdometryEdgeCount() const
{
    return GetPoseGraphSnapshot().OdometryEdgeCount();
}

std::size_t
RegistrationScan2LocalMap::PoseGraphLoopEdgeCount() const
{
    return GetPoseGraphSnapshot().LoopEdgeCount();
}

// ============================================================================
// Reset()
//
// Return the complete Scan-to-LocalMap frontend to an uninitialized state.
//
// After Reset():
//
//     next LiDAR scan becomes KF0
//     T_WL = Identity
//     LocalMap is empty
//     historical KeyframeManager is empty
//     tracking recovery counter = 0
//     constant-motion model = Identity
//
// ============================================================================

void RegistrationScan2LocalMap::Reset()
{
    // Stop the backend first so no backend-owned state is being read/written
    // while the frontend and backend histories are cleared.
    StopBackendWorker();

    // Reset the pure-LiDAR Ground V4 temporal/anchor state together with the
    // rest of the frontend so a new SLAM session bootstraps its own clearance.
    ResetGroundIcpRuntime(this);

    T_WL_ =
        Eigen::Isometry3d::Identity();

    last_relative_transform_ =
        Eigen::Isometry3d::Identity();

    consecutive_rejected_frames_ = 0;

    // Frontend-only state.
    submap_manager_.Clear();
    prepared_tracking_target_ = PreparedLidarTarget();
    keyframe_manager_.Clear();
    keyframe_detector_.Reset();

    // Backend-only history/state.
    backend_keyframes_.clear();
    backend_finished_submaps_.clear();

    pose_graph_.Clear();
    loop_detector_.Clear();
    loop_verifier_.ClearCache();
    loop_consistency_checker_.Reset();
    incremental_global_map_.Clear();

    online_loop_track_ = OnlineLoopTrack();

    has_last_online_loop_edge_ = false;

    last_online_loop_current_keyframe_id_ =
        std::numeric_limits<std::size_t>::max();

    last_online_loop_historical_keyframe_id_ =
        std::numeric_limits<std::size_t>::max();

    last_online_loop_measurement_ =
        Eigen::Isometry3d::Identity();

    pending_first_loop_batch_.clear();

    // Create fresh debug clouds instead of mutating a cloud that may still be
    // referenced by a ROS-side snapshot.
    refinement_historical_target_debug_ =
        pcl::make_shared<pcl::PointCloud<LIDAR_POINT>>();

    refinement_current_before_debug_ =
        pcl::make_shared<pcl::PointCloud<LIDAR_POINT>>();

    refinement_current_after_debug_ =
        pcl::make_shared<pcl::PointCloud<LIDAR_POINT>>();

    refined_keyframe_poses_.clear();
    refined_keyframe_pose_was_adjusted_.clear();

    global_map_revision_ = 0;
    refined_map_revision_ = 0;
    refinement_debug_revision_ = 0;

    T_map_odom_ =
        Eigen::Isometry3d::Identity();

    has_map_odom_correction_ =
        false;

    map_odom_revision_ = 0;

    map_odom_anchor_keyframe_id_ =
        std::numeric_limits<std::size_t>::max();

    initialized_ = false;

    RefreshBackendOutputSnapshot();
    StartBackendWorker();
}
