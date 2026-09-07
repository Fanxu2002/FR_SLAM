#include "fr_slam/frontend/fr_lidar_frontend.hpp"
#include "fr_slam/frontend/fr_ground_segmenter.hpp"
#include "fr_slam/frontend/fr_ground_input_bridge.hpp"

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
#include "fr_lidar_frontend_ground_internal.hpp"

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
        double scan_context_insert_ms = 0.0;
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

RegistrationScan2LocalMap::RegistrationScan2LocalMap(
    const LidarRegistrationConfig &registration_config,
    const LocalMapConfig &local_map_config,
    const LoopDetectorConfig &loop_detector_config)
    : registration_(registration_config),
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
    GroundIcpRuntime *ground_icp_runtime =
        RegisterGroundIcpRuntime(
            this,
            registration_config);

    std::cout
        << "Ground ICP V1.3 Quality-Weighted Observable-Subspace Joint-Hessian"
        << " | enabled="
        << (ground_icp_runtime != nullptr &&
                    ground_icp_runtime->enabled
                ? "ON"
                : "OFF")
        << " | base_weight="
        << (ground_icp_runtime != nullptr
                ? ground_icp_runtime->base_weight
                : 0.0)
        << " | support_cap="
        << (ground_icp_runtime != nullptr
                ? ground_icp_runtime->maximum_support_points
                : 0UL)
        << " | min_ground_corr="
        << (ground_icp_runtime != nullptr
                ? ground_icp_runtime->minimum_ground_correspondences
                : 0UL)
        << " | ground_voxel="
        << (ground_icp_runtime != nullptr
                ? ground_icp_runtime->analysis_voxel_leaf_m
                : 0.0)
        << " | ground_input=BASIC_PRE_VOXEL_SOR_ROR"
        << " | main_solver=JOINT_HESSIAN"
        << " | ground_reference=LOCALMAP_GROUND_PLANES"
        << " | ground_quality=V4_CONFIDENCE_X_CLEARANCE"
        << " | ground_dofs=ROLL_PITCH_Z"
        << " | ground_xy_yaw=OFF"
        << " | post_refinement=OFF"
        << " | local_ground_map=PREPARED_LOCALMAP_PLANES"
        << " | anchor_is_residual=NO"
        << " | backend_unchanged=YES"
        << std::endl;

    std::cout
        << "Frontend Robust ICP V1 + Degeneracy V2B"
        << " | huber="
        << (registration_config.enable_huber_loss
                ? "ON"
                : "OFF")
        << " | delta="
        << registration_config.huber_delta
        << " m"
        << " | hard_pt2plane_gate="
        << registration_config.max_point_to_plane_distance
        << " m"
        << " | sensor_centered="
        << (registration_config.enable_sensor_centered_perturbation
                ? "ON"
                : "OFF")
        << " | hessian_scale="
        << (registration_config.enable_hessian_scale_normalization
                ? "MEDIAN_RANGE_V2B"
                : "OFF")
        << " | scale_min="
        << registration_config.hessian_scale_min_range
        << " m"
        << " | scale_max="
        << registration_config.hessian_scale_max_range
        << " m"
        << " | backend_refinement_huber=OFF"
        << " | backend_refinement_sensor_centered=OFF"
        << " | backend_refinement_hessian_scale=OFF"
        << std::endl;

    const LoopDetectorConfig &active_loop_config =
        loop_detector_.GetConfig();

    std::cout
        << "Scan Context V3 IoU Sparse-Coverage Guard"
        << " | rings=" << active_loop_config.scan_context.num_rings
        << " | sectors=" << active_loop_config.scan_context.num_sectors
        << " | radius=[" << active_loop_config.scan_context.min_radius
        << "," << active_loop_config.scan_context.max_radius << "] m"
        << " | min_points=" << active_loop_config.scan_context.min_valid_points
        << " | min_occupied_sectors="
        << active_loop_config.scan_context.min_occupied_sectors
        << " | min_occupied_cells="
        << active_loop_config.scan_context.min_occupied_cells
        << " | min_sector_coverage="
        << active_loop_config.scan_context.min_sector_coverage_ratio
        << " | coverage_penalty="
        << active_loop_config.scan_context.coverage_penalty_weight
        << " | max_distance="
        << active_loop_config.max_scan_context_distance
        << std::endl;

    std::cout
        << "Online Loop Gate V20.1 SIMPLE-BASELINE 2-Frame-Confirm + Single-Factor"
        << " | graph_seed_max_translation="
        << max_loop_graph_correction_translation_ << " m"
        << " | graph_followup_cap="
        << online_loop_graph_followup_translation_cap_ << " m"
        << " | graph_followup_min_support="
        << online_loop_graph_followup_min_track_support_
        << " | first_confirm_frames="
        << online_loop_first_edge_min_support_
        << " | first_evidence_edges="
        << online_loop_first_batch_min_edges_
        << " | first_graph_factors=1"
        << " | historical_neighbor_max_gap="
        << online_loop_first_batch_max_historical_gap_
        << "kf"
        << " | batch_consistency=PREVIOUS_TRACK_LOCAL"
        << " | graph_max_rotation="
        << max_loop_graph_correction_rotation_deg_ << " deg"
        << " | first_overlap="
        << online_loop_first_edge_min_overlap_
        << " | first_rmse="
        << online_loop_first_edge_max_rmse_ << " m"
        << " | first_icp_translation="
        << online_loop_first_edge_max_icp_translation_ << " m"
        << " | first_icp_rotation="
        << online_loop_first_edge_max_icp_rotation_deg_ << " deg"
        << " | batch_edges="
        << online_loop_first_batch_min_edges_
        << " | batch_support="
        << online_loop_first_edge_min_support_
        << " | batch_consistency_translation="
        << online_loop_first_batch_max_translation_error_ << " m"
        << " | batch_consistency_rotation="
        << online_loop_first_batch_max_rotation_error_deg_ << " deg"
        << " | verify_unique_submaps="
        << max_loop_candidates_to_verify_
        << " | hypothesis_policy=POSE_PLUS_SC_PLUS_180_COMPLEMENT_PLUS_TENTATIVE_TRACK_THEN_GRAPH_GATE"
        << " | pgo_max_update_translation="
        << online_loop_pgo_max_translation_update_ << " m"
        << " | pgo_max_update_rotation="
        << online_loop_pgo_max_rotation_update_deg_ << " deg"
        << " | local_strong=[overlap>="
        << online_loop_first_local_min_overlap_
        << ",rmse<="
        << online_loop_first_local_max_rmse_
        << ",graph_dt<="
        << online_loop_first_local_max_graph_translation_
        << "m,graph_dR<="
        << online_loop_first_local_max_graph_rotation_deg_
        << "deg,support>="
        << online_loop_first_edge_min_support_
        << ",unique_edge=1]"
        << " | large_drift_neighborhood=+/-"
        << online_loop_large_drift_neighborhood_radius_
        << " kf"
        << " | large_drift_same_submap_budget="
        << online_loop_large_drift_same_submap_verify_budget_
        << " | large_drift_extra_verify="
        << online_loop_large_drift_extra_verify_budget_
        << " | large_drift_pgo_caps=["
        << online_loop_large_drift_pgo_translation_cap_
        << "m,"
        << online_loop_large_drift_pgo_rotation_cap_deg_
        << "deg]"
        << " | geometry_target=KEYFRAME_CENTERED"
        << " | target_half_window="
        << online_loop_candidate_target_half_window_
        << " | target_voxel="
        << online_loop_candidate_target_voxel_leaf_size_
        << " m"
        << " | temporal_consistency=LOCAL_POSE_PREDICTION"
        << " | large_drift_guard=LOCAL_DEFORMATION"
        << " | max_local_odom_def=["
        << online_loop_large_drift_max_local_odom_translation_deformation_
        << "m,"
        << online_loop_large_drift_max_local_odom_rotation_deformation_deg_
        << "deg]"
        << " | max_loop_residual=["
        << online_loop_large_drift_max_loop_residual_translation_
        << "m,"
        << online_loop_large_drift_max_loop_residual_rotation_deg_
        << "deg]"
        << " | local_cluster_radius=+/-"
        << online_loop_first_local_cluster_radius_
        << "kf"
        << " | large_drift_seed=RAW_SC"
        << " | raw_sc_support_radius=+/-"
        << online_loop_large_drift_raw_sc_support_radius_
        << "kf"
        << " | injected_followup=[overlap>="
        << online_loop_large_drift_injected_min_overlap_
        << ",rmse<="
        << online_loop_large_drift_injected_max_rmse_
        << ",consistency_dt<="
        << online_loop_large_drift_injected_max_consistency_translation_
        << "m,consistency_dR<="
        << online_loop_large_drift_injected_max_consistency_rotation_deg_
        << "deg]"
        << " | drift_aware_corridor="
        << (online_loop_drift_aware_corridor_enabled_ ? "ON" : "OFF")
        << " | corridor_radius="
        << online_loop_drift_aware_corridor_radius_
        << "m"
        << " | corridor_max_candidates="
        << online_loop_drift_aware_corridor_max_candidates_
        << std::endl;

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
                cloud_lidar);

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
        // PoseGraph / global map / Scan Context / loop verification are no
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

        std::cout
            << "RegistrationScan2LocalMap"
            << " | first frame"
            << " | keyframe=true"
            << " | keyframe_id=0"
            << " | keyframes=" << keyframe_manager_.Size()
            << " | submaps=" << submap_manager_.SubmapCount()
            << " | active_submap=" << submap_manager_.ActiveSubmapId()
            << " | active_keyframes=" << submap_manager_.ActiveKeyframeCount()
            << " | target_mode="
            << (submap_manager_.IsTransitionActive()
                    ? "TRANSITION"
                    : "ACTIVE")
            << " | target_points="
            << submap_manager_.TrackingPointCount()
            << std::endl;

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

    const GroundJointIcpStatus primary_joint_status =
        RunTrustedGroundJointIcpV12(
            this,
            cloud_lidar,
            prepared_tracking_target_,
            frontend_ground_result,
            initial_guess,
            result);

    bool registration_success =
        false;

    if (primary_joint_status ==
        GroundJointIcpStatus::Success)
    {
        registration_success =
            result.success;
    }
    else
    {
        if (primary_joint_status ==
            GroundJointIcpStatus::Failed)
        {
            std::cout
                << "GROUND_ICP_V13"
                << " | stage=PRIMARY"
                << " | action=FALLBACK_GENERAL"
                << " | reason=JOINT_FAILED"
                << std::endl;
        }

        registration_success =
            registration_.Align(
                cloud_lidar,
                prepared_tracking_target_,
                initial_guess,
                result);
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

        std::cout
            << "Tracking Recovery V2 triggered"
            << " | primary_success="
            << (registration_success &&
                        result.success
                    ? "true"
                    : "false")
            << " | primary_corr="
            << result.correspondences
            << " | primary_rmse="
            << result.rmse
            << " | rejected_frames="
            << consecutive_rejected_frames_
            << std::endl;

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

        RCLCPP_WARN(
            kRecoveryLogger,
            "RECOVERY_DEBUG PRIMARY_FAIL"
            " | reason=%s"
            " | target_mode=%s"
            " | source_points=%zu"
            " | target_points=%zu"
            " | corr=%zu"
            " | rmse=%.6f"
            " | rejected_frames=%zu",
            primary_reject_reason,
            submap_manager_.IsTransitionActive()
                ? "TRANSITION"
                : "ACTIVE",
            cloud_lidar ? cloud_lidar->size() : 0UL,
            submap_manager_.TrackingPointCount(),
            result.correspondences,
            result.rmse,
            consecutive_rejected_frames_);

        RCLCPP_INFO(
            kRecoveryLogger,
            "RECOVERY_DEBUG POSE_BEFORE_COARSE"
            " | prev=[%.4f %.4f %.4f]"
            " | guess=[%.4f %.4f %.4f]"
            " | primary=[%.4f %.4f %.4f]"
            " | primary_minus_guess=[%.4f %.4f %.4f]"
            " | correction_norm=%.4f"
            " | correction_rot=%.3f deg",
            T_WL_previous.translation().x(),
            T_WL_previous.translation().y(),
            T_WL_previous.translation().z(),
            initial_guess.translation().x(),
            initial_guess.translation().y(),
            initial_guess.translation().z(),
            result.T_target_source.translation().x(),
            result.T_target_source.translation().y(),
            result.T_target_source.translation().z(),
            primary_correction.x(),
            primary_correction.y(),
            primary_correction.z(),
            primary_correction.norm(),
            primary_rotation_correction_deg);

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

        std::cout
            << "Recovery coarse Point-to-Point ICP"
            << " | success="
            << (coarse_success
                    ? "true"
                    : "false")
            << " | fitness="
            << coarse_fitness
            << " | correction_translation="
            << (T_WL_coarse.translation() -
                initial_guess.translation())
                   .norm()
            << " m"
            << std::endl;

        const Eigen::Vector3d coarse_correction =
            T_WL_coarse.translation() -
            initial_guess.translation();

        const double coarse_rotation_correction_deg =
            RelativeRotationDeg(
                initial_guess,
                T_WL_coarse);

        RCLCPP_INFO(
            kRecoveryLogger,
            "RECOVERY_DEBUG COARSE_POINT_TO_POINT"
            " | success=%s"
            " | fitness=%.6f"
            " | guess=[%.4f %.4f %.4f]"
            " | coarse=[%.4f %.4f %.4f]"
            " | delta=[%.4f %.4f %.4f]"
            " | delta_norm=%.4f"
            " | delta_rot=%.3f deg",
            coarse_success ? "true" : "false",
            coarse_fitness,
            initial_guess.translation().x(),
            initial_guess.translation().y(),
            initial_guess.translation().z(),
            T_WL_coarse.translation().x(),
            T_WL_coarse.translation().y(),
            T_WL_coarse.translation().z(),
            coarse_correction.x(),
            coarse_correction.y(),
            coarse_correction.z(),
            coarse_correction.norm(),
            coarse_rotation_correction_deg);

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

            const GroundJointIcpStatus recovery_joint_status =
                RunTrustedGroundJointIcpV12(
                    this,
                    cloud_lidar,
                    prepared_tracking_target_,
                    frontend_ground_result,
                    T_WL_coarse,
                    refined_result);

            bool recovery_used_ground_joint =
                recovery_joint_status ==
                GroundJointIcpStatus::Success;

            bool refined_success =
                false;

            if (recovery_used_ground_joint)
            {
                refined_success =
                    refined_result.success;
            }
            else
            {
                refined_success =
                    registration_.Align(
                        cloud_lidar,
                        prepared_tracking_target_,
                        T_WL_coarse,
                        refined_result);
            }

            frame_timing.recovery_refine_ms +=
                ElapsedMilliseconds(
                    recovery_refine_start,
                    std::chrono::steady_clock::now());

            const bool refined_passed =
                CandidatePassesQualityGate(
                    refined_success,
                    refined_result);

            std::cout
                << "Recovery Point-to-Plane refine"
                << " | success="
                << (refined_success &&
                            refined_result.success
                        ? "true"
                        : "false")
                << " | corr="
                << refined_result.correspondences
                << " | rmse="
                << refined_result.rmse
                << " | passed="
                << (refined_passed
                        ? "true"
                        : "false")
                << std::endl;

            const Eigen::Vector3d refine_correction =
                refined_result.T_target_source.translation() -
                T_WL_coarse.translation();

            const double refine_rotation_correction_deg =
                RelativeRotationDeg(
                    T_WL_coarse,
                    refined_result.T_target_source);

            RCLCPP_INFO(
                kRecoveryLogger,
                "RECOVERY_DEBUG REFINE_POINT_TO_PLANE"
                " | success=%s"
                " | passed=%s"
                " | corr=%zu"
                " | rmse=%.6f"
                " | coarse=[%.4f %.4f %.4f]"
                " | refined=[%.4f %.4f %.4f]"
                " | refine_delta=[%.4f %.4f %.4f]"
                " | refine_delta_norm=%.4f"
                " | refine_delta_rot=%.3f deg",
                (refined_success &&
                 refined_result.success)
                    ? "true"
                    : "false",
                refined_passed
                    ? "true"
                    : "false",
                refined_result.correspondences,
                refined_result.rmse,
                T_WL_coarse.translation().x(),
                T_WL_coarse.translation().y(),
                T_WL_coarse.translation().z(),
                refined_result.T_target_source.translation().x(),
                refined_result.T_target_source.translation().y(),
                refined_result.T_target_source.translation().z(),
                refine_correction.x(),
                refine_correction.y(),
                refine_correction.z(),
                refine_correction.norm(),
                refine_rotation_correction_deg);

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
    // 5.3 Ground ICP V1.3 Quality-Weighted Observable-Subspace Joint-Hessian.
    //
    // No post-refinement is performed here.  When Ground V4 is trusted, it
    // has already participated in EVERY main GN iteration through:
    //
    //     H_total = H_general + lambda_g * H_ground
    //     b_total = b_general + lambda_g * b_ground
    //
    // When Ground is not trusted, the candidate came from the unchanged
    // ordinary registration_.Align() fallback.
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
            std::cerr
                << "RegistrationScan2LocalMap::AddFrame(): "
                << "registration rejected by RMSE."
                << " rmse="
                << result.rmse
                << " max="
                << max_accepted_rmse_
                << " | consecutive_rejected="
                << consecutive_rejected_frames_
                << std::endl;
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

    if (used_coarse_recovery)
    {
        std::cout
            << "Tracking Recovery V2 accepted"
            << " | coarse_fitness="
            << coarse_fitness
            << " | refined_corr="
            << result.correspondences
            << " | refined_rmse="
            << result.rmse
            << std::endl;

        const Eigen::Vector3d final_jump =
            result.T_target_source.translation() -
            T_WL_previous.translation();

        const double final_rotation_jump_deg =
            RelativeRotationDeg(
                T_WL_previous,
                result.T_target_source);

        RCLCPP_WARN(
            kRecoveryLogger,
            "RECOVERY_DEBUG FINAL_ACCEPTED"
            " | prev=[%.4f %.4f %.4f]"
            " | final=[%.4f %.4f %.4f]"
            " | frame_jump=[%.4f %.4f %.4f]"
            " | jump_norm=%.4f"
            " | jump_rot=%.3f deg"
            " | coarse_fitness=%.6f"
            " | refined_corr=%zu"
            " | refined_rmse=%.6f",
            T_WL_previous.translation().x(),
            T_WL_previous.translation().y(),
            T_WL_previous.translation().z(),
            result.T_target_source.translation().x(),
            result.T_target_source.translation().y(),
            result.T_target_source.translation().z(),
            final_jump.x(),
            final_jump.y(),
            final_jump.z(),
            final_jump.norm(),
            final_rotation_jump_deg,
            coarse_fitness,
            result.correspondences,
            result.rmse);
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
                    : nullptr);

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
        // Submap. PoseGraph / map / Scan Context / LoopVerifier run later in
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
            std::cout
                << "Submap transition"
                << " | finished_submap="
                << submap_manager_.LastFinishedSubmapId()
                << " | previous_submap="
                << submap_manager_.PreviousSubmapId()
                << " | new_active_submap="
                << submap_manager_.ActiveSubmapId()
                << " | overlap_keyframes="
                << submap_manager_.ActiveKeyframeCount()
                << " | tracking_target=PREVIOUS+ACTIVE"
                << " | target_points="
                << submap_manager_.TrackingPointCount()
                << std::endl;
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
