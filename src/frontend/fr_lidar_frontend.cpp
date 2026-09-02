#include "fr_slam/fr_registration_scan2localmap.hpp"
#include "fr_slam/fr_ground_segmenter.hpp"
#include "fr_slam/fr_ground_input_bridge.hpp"

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

namespace
{
    const rclcpp::Logger kRecoveryLogger =
        rclcpp::get_logger("scan2local_map.recovery");

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

    struct RefinementTimingDiagnostics
    {
        std::chrono::steady_clock::time_point start =
            std::chrono::steady_clock::now();

        std::size_t keyframes = 0;
        std::size_t loop_anchors = 0;
        std::size_t groups_considered = 0;
        std::size_t groups_prepared = 0;
        std::size_t groups_optimized = 0;
        std::size_t groups_accepted = 0;
        std::size_t geometry_candidates = 0;
        std::size_t geometry_primary_selected = 0;
        std::size_t geometry_calls = 0;
        std::size_t geometry_fallback_calls = 0;
        std::size_t geometry_anchors = 0;

        double current_select_ms = 0.0;
        double historical_select_ms = 0.0;
        double historical_build_ms = 0.0;
        double historical_voxel_ms = 0.0;
        double prepare_target_ms = 0.0;
        double local_graph_build_ms = 0.0;
        double geometry_align_ms = 0.0;
        double local_pgo_ms = 0.0;
        double debug_clouds_ms = 0.0;
        double refined_map_update_ms = 0.0;
    };

    class RefinementTimingReporter
    {
    public:
        explicit RefinementTimingReporter(
            RefinementTimingDiagnostics &diagnostics)
            : diagnostics_(diagnostics)
        {
        }

        ~RefinementTimingReporter()
        {
            const double total_ms =
                ElapsedMilliseconds(
                    diagnostics_.start,
                    std::chrono::steady_clock::now());

            RCLCPP_INFO(
                kTimingLogger,
                "FR_TIMING REFINEMENT"
                " | total=%.3f ms"
                " | keyframes=%zu"
                " | loop_anchors=%zu"
                " | groups=%zu/%zu/%zu/%zu"
                " | current_select=%.3f"
                " | historical_select=%.3f"
                " | historical_build=%.3f"
                " | historical_voxel=%.3f"
                " | prepare_target=%.3f"
                " | local_graph=%.3f"
                " | geometry_align=%.3f"
                " | geometry_candidates=%zu"
                " | geometry_primary_selected=%zu"
                " | geometry_calls=%zu"
                " | geometry_fallback_calls=%zu"
                " | geometry_anchors=%zu"
                " | local_pgo=%.3f"
                " | debug_clouds=%.3f"
                " | refined_map_update=%.3f",
                total_ms,
                diagnostics_.keyframes,
                diagnostics_.loop_anchors,
                diagnostics_.groups_considered,
                diagnostics_.groups_prepared,
                diagnostics_.groups_optimized,
                diagnostics_.groups_accepted,
                diagnostics_.current_select_ms,
                diagnostics_.historical_select_ms,
                diagnostics_.historical_build_ms,
                diagnostics_.historical_voxel_ms,
                diagnostics_.prepare_target_ms,
                diagnostics_.local_graph_build_ms,
                diagnostics_.geometry_align_ms,
                diagnostics_.geometry_candidates,
                diagnostics_.geometry_primary_selected,
                diagnostics_.geometry_calls,
                diagnostics_.geometry_fallback_calls,
                diagnostics_.geometry_anchors,
                diagnostics_.local_pgo_ms,
                diagnostics_.debug_clouds_ms,
                diagnostics_.refined_map_update_ms);
        }

    private:
        RefinementTimingDiagnostics &diagnostics_;
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

    // ========================================================================
    // Ground ICP V1: Trusted Support soft reweighting for realtime frontend.
    //
    // Design rules:
    //   1. Ground V4.0 remains the trust gate.  support_plane_valid alone is
    //      NEVER sufficient.
    //   2. The learned LiDAR-to-support clearance anchor is ONLY a confidence
    //      gate/weight.  It is NOT an optimization residual and therefore does
    //      not force a fixed sensor height or fixed World-Z.
    //   3. The actual residual is still LiDAR geometry:
    //          current trusted support point -> LocalMap target plane.
    //   4. Ground correspondences are added as a bounded soft boost to the
    //      existing all-scene point-to-plane objective.
    //   5. Backend refinement / loop closure / Gravity Guard are untouched.
    // ========================================================================

    struct GroundIcpRuntime
    {
        explicit GroundIcpRuntime(
            const LidarRegistrationConfig &config)
            : registration_config(config)
        {
            const char *enable_env =
                std::getenv("FR_SLAM_GROUND_ICP_ENABLE");

            if (enable_env != nullptr)
            {
                const std::string value(enable_env);

                enabled =
                    !(value == "0" ||
                      value == "false" ||
                      value == "FALSE" ||
                      value == "off" ||
                      value == "OFF");
            }
        }

        fr_slam::GroundSegmenter segmenter;
        LidarRegistrationConfig registration_config;

        bool enabled = true;

        // Ground V4.0 was validated on a dedicated 0.15 m analysis voxel.
        // Keep that operating point even though the dense branch itself is the
        // Basic/ROI cloud (~8k-12k points on the current MID-360 data).
        double analysis_voxel_leaf_m = 0.15;

        // Extra information multiplier applied to trusted support
        // correspondences.  The actual frame weight is:
        //
        //   base_weight * confidence^2 * anchor_factor
        //
        // so borderline valid frames remain deliberately weak.
        double base_weight = 4.0;

        std::size_t maximum_support_points = 160;
        std::size_t minimum_ground_correspondences = 30;

        double target_normal_compatibility_deg = 20.0;
        double ground_maximum_residual_m = 0.15;
        double ground_huber_delta_m = 0.05;

        int maximum_refinement_iterations = 2;

        // Safety envelope for the EXTRA correction relative to the original
        // accepted Scan-to-LocalMap result.
        double maximum_total_rotation_correction_deg = 0.50;
        double maximum_total_translation_correction_m = 0.030;

        // The ordinary all-scene ICP objective is the guardrail.  A Ground
        // step may not meaningfully degrade it just to improve support points.
        double maximum_general_rmse_ratio = 1.03;
        double maximum_general_rmse_absolute_increase_m = 0.002;
        double minimum_general_correspondence_ratio = 0.85;
    };

    struct GroundIcpLinearization
    {
        Eigen::Matrix<double, 6, 6> H =
            Eigen::Matrix<double, 6, 6>::Zero();

        Eigen::Matrix<double, 6, 1> b =
            Eigen::Matrix<double, 6, 1>::Zero();

        std::size_t general_correspondences = 0;
        std::size_t ground_correspondences = 0;

        std::size_t general_downweighted_correspondences = 0;
        double general_min_robust_weight = 1.0;

        double general_raw_squared_error_sum = 0.0;
        double general_weighted_squared_error_sum = 0.0;
        double general_weight_sum = 0.0;

        double ground_raw_squared_error_sum = 0.0;
        double ground_weighted_squared_error_sum = 0.0;
        double ground_weight_sum = 0.0;

        std::vector<double> correspondence_ranges;

        double GeneralRmse() const
        {
            if (general_correspondences == 0)
            {
                return std::numeric_limits<double>::infinity();
            }

            return std::sqrt(
                general_raw_squared_error_sum /
                static_cast<double>(general_correspondences));
        }

        double GroundRmse() const
        {
            if (ground_correspondences == 0)
            {
                return std::numeric_limits<double>::infinity();
            }

            return std::sqrt(
                ground_raw_squared_error_sum /
                static_cast<double>(ground_correspondences));
        }

        double CombinedWeightedMeanSquaredError() const
        {
            const double total_weight =
                general_weight_sum +
                ground_weight_sum;

            if (!std::isfinite(total_weight) ||
                total_weight <= 1.0e-12)
            {
                return std::numeric_limits<double>::infinity();
            }

            return (general_weighted_squared_error_sum +
                    ground_weighted_squared_error_sum) /
                   total_weight;
        }
    };

    std::mutex &GroundIcpRuntimeMutex()
    {
        static std::mutex mutex;
        return mutex;
    }

    std::unordered_map<
        const RegistrationScan2LocalMap *,
        std::unique_ptr<GroundIcpRuntime>> &
    GroundIcpRuntimeMap()
    {
        static std::unordered_map<
            const RegistrationScan2LocalMap *,
            std::unique_ptr<GroundIcpRuntime>>
            runtime_map;

        return runtime_map;
    }

    GroundIcpRuntime *RegisterGroundIcpRuntime(
        const RegistrationScan2LocalMap *owner,
        const LidarRegistrationConfig &registration_config)
    {
        if (owner == nullptr)
        {
            return nullptr;
        }

        std::lock_guard<std::mutex> lock(
            GroundIcpRuntimeMutex());

        std::unique_ptr<GroundIcpRuntime> runtime =
            std::make_unique<GroundIcpRuntime>(
                registration_config);

        GroundIcpRuntime *runtime_pointer =
            runtime.get();

        GroundIcpRuntimeMap()[owner] =
            std::move(runtime);

        return runtime_pointer;
    }

    GroundIcpRuntime *GetGroundIcpRuntime(
        const RegistrationScan2LocalMap *owner)
    {
        std::lock_guard<std::mutex> lock(
            GroundIcpRuntimeMutex());

        const auto iterator =
            GroundIcpRuntimeMap().find(owner);

        if (iterator == GroundIcpRuntimeMap().end())
        {
            return nullptr;
        }

        return iterator->second.get();
    }

    void ResetGroundIcpRuntime(
        const RegistrationScan2LocalMap *owner)
    {
        GroundIcpRuntime *runtime =
            GetGroundIcpRuntime(owner);

        if (runtime != nullptr)
        {
            runtime->segmenter.Reset();
        }
    }

    void RemoveGroundIcpRuntime(
        const RegistrationScan2LocalMap *owner)
    {
        std::lock_guard<std::mutex> lock(
            GroundIcpRuntimeMutex());

        GroundIcpRuntimeMap().erase(owner);
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr
    ConvertToGroundAnalysisCloud(
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &cloud_lidar)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(
            new pcl::PointCloud<pcl::PointXYZ>);

        if (!cloud_lidar)
        {
            return cloud_xyz;
        }

        cloud_xyz->reserve(
            cloud_lidar->size());

        for (const LIDAR_POINT &point :
             cloud_lidar->points)
        {
            if (!std::isfinite(point.x) ||
                !std::isfinite(point.y) ||
                !std::isfinite(point.z))
            {
                continue;
            }

            pcl::PointXYZ point_xyz;
            point_xyz.x = point.x;
            point_xyz.y = point.y;
            point_xyz.z = point.z;

            cloud_xyz->push_back(
                point_xyz);
        }

        cloud_xyz->width =
            static_cast<std::uint32_t>(
                cloud_xyz->size());

        cloud_xyz->height = 1;
        cloud_xyz->is_dense = true;

        return cloud_xyz;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr
    VoxelizeGroundAnalysisCloud(
        const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud,
        double leaf_size_m)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(
            new pcl::PointCloud<pcl::PointXYZ>);

        if (!cloud ||
            cloud->empty())
        {
            return filtered;
        }

        if (!std::isfinite(leaf_size_m) ||
            leaf_size_m <= 0.0)
        {
            *filtered = *cloud;
            return filtered;
        }

        pcl::VoxelGrid<pcl::PointXYZ> voxel;
        voxel.setInputCloud(cloud);

        const float leaf =
            static_cast<float>(leaf_size_m);

        voxel.setLeafSize(
            leaf,
            leaf,
            leaf);

        voxel.filter(*filtered);

        return filtered;
    }

    fr_slam::GroundSegmentationResult
    SegmentFrontendGround(
        const RegistrationScan2LocalMap *owner,
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &cloud_lidar,
        const char *input_source)
    {
        (void)input_source;

        fr_slam::GroundSegmentationResult result;

        GroundIcpRuntime *runtime =
            GetGroundIcpRuntime(owner);

        if (runtime == nullptr ||
            !runtime->enabled)
        {
            return result;
        }

        const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz =
            ConvertToGroundAnalysisCloud(
                cloud_lidar);

        const pcl::PointCloud<pcl::PointXYZ>::Ptr analysis_cloud =
            VoxelizeGroundAnalysisCloud(
                cloud_xyz,
                runtime->analysis_voxel_leaf_m);

        result =
            runtime->segmenter.Segment(
                analysis_cloud);

        return result;
    }

    double GroundIcpAnchorFactor(
        const fr_slam::GroundSegmentationResult &ground_result)
    {
        if (!ground_result.support_clearance_anchor_valid ||
            !std::isfinite(
                ground_result.support_clearance_error_m) ||
            !std::isfinite(
                ground_result.support_clearance_anchor_tolerance_m) ||
            ground_result.support_clearance_anchor_tolerance_m <= 0.0)
        {
            return 1.0;
        }

        // A trusted frame sitting near the outer hard anchor gate should not
        // suddenly obtain the same optimizer strength as a frame centered on
        // the learned clearance distribution.
        const double sigma_m =
            std::max(
                0.025,
                0.5 *
                    ground_result
                        .support_clearance_anchor_tolerance_m);

        const double normalized_error =
            ground_result.support_clearance_error_m /
            sigma_m;

        return std::exp(
            -0.5 *
            normalized_error *
            normalized_error);
    }

    double ComputeGroundIcpWeight(
        const GroundIcpRuntime &runtime,
        const fr_slam::GroundSegmentationResult &ground_result)
    {
        const double confidence =
            std::clamp(
                ground_result.support_constraint_confidence,
                0.0,
                1.0);

        const double anchor_factor =
            GroundIcpAnchorFactor(
                ground_result);

        return std::clamp(
            runtime.base_weight *
                confidence *
                confidence *
                anchor_factor,
            0.0,
            runtime.base_weight);
    }

    bool BuildGroundIcpLinearization(
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &source,
        const PreparedLidarTarget &target,
        const Eigen::Isometry3d &T_target_source,
        const fr_slam::GroundSegmentationResult &ground_result,
        const GroundIcpRuntime &runtime,
        double ground_information_weight,
        GroundIcpLinearization &linearization)
    {
        linearization =
            GroundIcpLinearization();

        if (!source ||
            source->empty() ||
            !target.ready ||
            !target.cloud ||
            target.cloud->empty() ||
            !target.kdtree ||
            target.planes.size() != target.cloud->size() ||
            !T_target_source.matrix().allFinite())
        {
            return false;
        }

        const LidarRegistrationConfig &config =
            runtime.registration_config;

        if (config.knn <= 0)
        {
            return false;
        }

        const double maximum_correspondence_distance_squared =
            config.max_correspondence_distance *
            config.max_correspondence_distance;

        std::vector<int> neighbor_indices(
            static_cast<std::size_t>(config.knn));

        std::vector<float> neighbor_squared_distances(
            static_cast<std::size_t>(config.knn));

        const Eigen::Vector3d sensor_origin_target =
            T_target_source.translation();

        const bool use_sensor_centered =
            config.enable_sensor_centered_perturbation;

        const bool use_hessian_scale_normalization =
            config.enable_sensor_centered_perturbation &&
            config.enable_hessian_scale_normalization;

        if (use_hessian_scale_normalization)
        {
            linearization.correspondence_ranges.reserve(
                source->size());
        }

        // --------------------------------------------------------------------
        // A. Existing all-scene point-to-plane objective.
        // --------------------------------------------------------------------
        for (const LIDAR_POINT &source_point :
             source->points)
        {
            const Eigen::Vector3d p_source(
                static_cast<double>(source_point.x),
                static_cast<double>(source_point.y),
                static_cast<double>(source_point.z));

            if (!p_source.allFinite())
            {
                continue;
            }

            const Eigen::Vector3d p_target =
                T_target_source *
                p_source;

            LIDAR_POINT query_point{};
            query_point.x =
                static_cast<float>(p_target.x());
            query_point.y =
                static_cast<float>(p_target.y());
            query_point.z =
                static_cast<float>(p_target.z());

            const int found =
                target.kdtree->nearestKSearch(
                    query_point,
                    config.knn,
                    neighbor_indices,
                    neighbor_squared_distances);

            if (found <= 0)
            {
                continue;
            }

            int plane_index = -1;

            for (int j = 0;
                 j < found;
                 ++j)
            {
                const double neighbor_distance_squared =
                    static_cast<double>(
                        neighbor_squared_distances[static_cast<std::size_t>(j)]);

                if (neighbor_distance_squared >
                    maximum_correspondence_distance_squared)
                {
                    break;
                }

                const int candidate_index =
                    neighbor_indices[static_cast<std::size_t>(j)];

                if (candidate_index < 0)
                {
                    continue;
                }

                const std::size_t candidate =
                    static_cast<std::size_t>(
                        candidate_index);

                if (candidate >= target.planes.size() ||
                    target.planes[candidate].state !=
                        TargetPlane::State::Valid)
                {
                    continue;
                }

                plane_index =
                    candidate_index;

                break;
            }

            if (plane_index < 0)
            {
                continue;
            }

            const TargetPlane &plane =
                target.planes[static_cast<std::size_t>(
                    plane_index)];

            const double residual =
                plane.normal.dot(
                    p_target -
                    plane.point);

            if (!std::isfinite(residual) ||
                std::abs(residual) >
                    config.max_point_to_plane_distance)
            {
                continue;
            }

            Eigen::Matrix<double, 1, 6> J;

            if (use_sensor_centered)
            {
                const Eigen::Vector3d lever_arm_target =
                    p_target -
                    sensor_origin_target;

                J.block<1, 3>(0, 0) =
                    lever_arm_target.cross(
                                        plane.normal)
                        .transpose();

                if (use_hessian_scale_normalization)
                {
                    const double lever_arm_range =
                        lever_arm_target.norm();

                    if (std::isfinite(lever_arm_range) &&
                        lever_arm_range > 1.0e-9)
                    {
                        linearization
                            .correspondence_ranges
                            .push_back(
                                lever_arm_range);
                    }
                }
            }
            else
            {
                J.block<1, 3>(0, 0) =
                    p_target.cross(
                                plane.normal)
                        .transpose();
            }

            J.block<1, 3>(0, 3) =
                plane.normal.transpose();

            double robust_weight =
                1.0;

            const double absolute_residual =
                std::abs(residual);

            if (config.enable_huber_loss &&
                absolute_residual >
                    config.huber_delta)
            {
                robust_weight =
                    config.huber_delta /
                    absolute_residual;

                ++linearization
                      .general_downweighted_correspondences;
            }

            linearization.general_min_robust_weight =
                std::min(
                    linearization.general_min_robust_weight,
                    robust_weight);

            linearization.H.noalias() +=
                robust_weight *
                J.transpose() *
                J;

            linearization.b.noalias() +=
                robust_weight *
                J.transpose() *
                residual;

            linearization.general_raw_squared_error_sum +=
                residual *
                residual;

            linearization.general_weighted_squared_error_sum +=
                robust_weight *
                residual *
                residual;

            linearization.general_weight_sum +=
                robust_weight;

            ++linearization.general_correspondences;
        }

        // --------------------------------------------------------------------
        // B. Trusted Ground V4.0 extra point-to-plane information.
        //
        // Important: this is NOT residual(distance_to_anchor).  Ground points
        // are matched against ACTUAL LocalMap planes.  The source support normal
        // is used only to prevent an accidental support-point -> wall match.
        // --------------------------------------------------------------------
        if (!ground_result.support_constraint_valid ||
            !ground_result.support_plane_valid ||
            !ground_result.support_ground_cloud ||
            ground_result.support_ground_cloud->empty() ||
            ground_information_weight <= 1.0e-9)
        {
            return linearization.general_correspondences > 0;
        }

        Eigen::Vector3d support_normal_source =
            ground_result.support_ground_normal_L;

        const double support_normal_norm =
            support_normal_source.norm();

        if (!support_normal_source.allFinite() ||
            !std::isfinite(support_normal_norm) ||
            support_normal_norm <= 1.0e-12)
        {
            return linearization.general_correspondences > 0;
        }

        support_normal_source /=
            support_normal_norm;

        Eigen::Vector3d support_normal_target =
            T_target_source.rotation() *
            support_normal_source;

        const double support_normal_target_norm =
            support_normal_target.norm();

        if (!support_normal_target.allFinite() ||
            !std::isfinite(support_normal_target_norm) ||
            support_normal_target_norm <= 1.0e-12)
        {
            return linearization.general_correspondences > 0;
        }

        support_normal_target /=
            support_normal_target_norm;

        constexpr double kPi =
            3.14159265358979323846;

        const double normal_cosine_threshold =
            std::cos(
                runtime.target_normal_compatibility_deg *
                kPi /
                180.0);

        const std::size_t support_size =
            ground_result.support_ground_cloud->size();

        const std::size_t stride =
            std::max<std::size_t>(
                1,
                (support_size +
                 runtime.maximum_support_points -
                 1) /
                    std::max<std::size_t>(
                        1,
                        runtime.maximum_support_points));

        for (std::size_t point_index = 0;
             point_index < support_size;
             point_index += stride)
        {
            const pcl::PointXYZ &source_point =
                ground_result
                    .support_ground_cloud
                    ->points[point_index];

            const Eigen::Vector3d p_source(
                static_cast<double>(source_point.x),
                static_cast<double>(source_point.y),
                static_cast<double>(source_point.z));

            if (!p_source.allFinite())
            {
                continue;
            }

            const Eigen::Vector3d p_target =
                T_target_source *
                p_source;

            LIDAR_POINT query_point{};
            query_point.x =
                static_cast<float>(p_target.x());
            query_point.y =
                static_cast<float>(p_target.y());
            query_point.z =
                static_cast<float>(p_target.z());

            const int found =
                target.kdtree->nearestKSearch(
                    query_point,
                    config.knn,
                    neighbor_indices,
                    neighbor_squared_distances);

            if (found <= 0)
            {
                continue;
            }

            int plane_index = -1;

            for (int j = 0;
                 j < found;
                 ++j)
            {
                const double neighbor_distance_squared =
                    static_cast<double>(
                        neighbor_squared_distances[static_cast<std::size_t>(j)]);

                if (neighbor_distance_squared >
                    maximum_correspondence_distance_squared)
                {
                    break;
                }

                const int candidate_index =
                    neighbor_indices[static_cast<std::size_t>(j)];

                if (candidate_index < 0)
                {
                    continue;
                }

                const std::size_t candidate =
                    static_cast<std::size_t>(
                        candidate_index);

                if (candidate >= target.planes.size() ||
                    target.planes[candidate].state !=
                        TargetPlane::State::Valid)
                {
                    continue;
                }

                const TargetPlane &candidate_plane =
                    target.planes[candidate];

                const double normal_alignment =
                    std::abs(
                        candidate_plane.normal.dot(
                            support_normal_target));

                if (!std::isfinite(normal_alignment) ||
                    normal_alignment <
                        normal_cosine_threshold)
                {
                    continue;
                }

                plane_index =
                    candidate_index;

                break;
            }

            if (plane_index < 0)
            {
                continue;
            }

            const TargetPlane &plane =
                target.planes[static_cast<std::size_t>(
                    plane_index)];

            const double residual =
                plane.normal.dot(
                    p_target -
                    plane.point);

            if (!std::isfinite(residual) ||
                std::abs(residual) >
                    runtime.ground_maximum_residual_m)
            {
                continue;
            }

            Eigen::Matrix<double, 1, 6> J;

            if (use_sensor_centered)
            {
                const Eigen::Vector3d lever_arm_target =
                    p_target -
                    sensor_origin_target;

                J.block<1, 3>(0, 0) =
                    lever_arm_target.cross(
                                        plane.normal)
                        .transpose();
            }
            else
            {
                J.block<1, 3>(0, 0) =
                    p_target.cross(
                                plane.normal)
                        .transpose();
            }

            J.block<1, 3>(0, 3) =
                plane.normal.transpose();

            // Ground observable-subspace projection.
            // Parameter order is [rx ry rz tx ty tz].
            // Ground is allowed to add direct information ONLY to
            // roll / pitch / z.  General ICP remains full 6-DoF.
            J(0, 2) = 0.0; // yaw
            J(0, 3) = 0.0; // x
            J(0, 4) = 0.0; // y

            double ground_robust_weight =
                1.0;

            const double absolute_residual =
                std::abs(residual);

            if (absolute_residual >
                runtime.ground_huber_delta_m)
            {
                ground_robust_weight =
                    runtime.ground_huber_delta_m /
                    absolute_residual;
            }

            const double final_weight =
                ground_information_weight *
                ground_robust_weight;

            linearization.H.noalias() +=
                final_weight *
                J.transpose() *
                J;

            linearization.b.noalias() +=
                final_weight *
                J.transpose() *
                residual;

            linearization.ground_raw_squared_error_sum +=
                residual *
                residual;

            linearization.ground_weighted_squared_error_sum +=
                final_weight *
                residual *
                residual;

            linearization.ground_weight_sum +=
                final_weight;

            ++linearization.ground_correspondences;
        }

        return linearization.general_correspondences > 0;
    }

    bool ComputeGroundIcpParameterUnscale(
        const GroundIcpLinearization &linearization,
        const LidarRegistrationConfig &config,
        Eigen::Matrix<double, 6, 6> &parameter_unscale,
        double &characteristic_length)
    {
        parameter_unscale =
            Eigen::Matrix<double, 6, 6>::Identity();

        characteristic_length =
            1.0;

        const bool use_hessian_scale_normalization =
            config.enable_sensor_centered_perturbation &&
            config.enable_hessian_scale_normalization;

        if (!use_hessian_scale_normalization)
        {
            return true;
        }

        if (linearization.correspondence_ranges.empty())
        {
            return false;
        }

        std::vector<double> ranges =
            linearization.correspondence_ranges;

        const std::size_t range_count =
            ranges.size();

        std::vector<double>::iterator middle =
            ranges.begin() +
            static_cast<std::ptrdiff_t>(
                range_count / 2);

        std::nth_element(
            ranges.begin(),
            middle,
            ranges.end());

        double median_range =
            *middle;

        if ((range_count % 2U) == 0U)
        {
            const std::vector<double>::iterator lower_middle =
                std::max_element(
                    ranges.begin(),
                    middle);

            if (lower_middle != middle)
            {
                median_range =
                    0.5 *
                    (median_range +
                     *lower_middle);
            }
        }

        if (!std::isfinite(median_range) ||
            median_range <= 0.0)
        {
            return false;
        }

        characteristic_length =
            std::clamp(
                median_range,
                config.hessian_scale_min_range,
                config.hessian_scale_max_range);

        const double inverse_length =
            1.0 /
            characteristic_length;

        parameter_unscale(0, 0) =
            inverse_length;

        parameter_unscale(1, 1) =
            inverse_length;

        parameter_unscale(2, 2) =
            inverse_length;

        return true;
    }

    bool SolveGroundIcpStep(
        const GroundIcpLinearization &linearization,
        const LidarRegistrationConfig &config,
        Eigen::Matrix<double, 6, 1> &dx,
        Eigen::Matrix<double, 6, 6> *final_H_analysis = nullptr,
        Eigen::Matrix<double, 6, 6> *final_parameter_unscale = nullptr)
    {
        dx =
            Eigen::Matrix<double, 6, 1>::Zero();

        Eigen::Matrix<double, 6, 6> parameter_unscale;
        double characteristic_length = 1.0;

        if (!ComputeGroundIcpParameterUnscale(
                linearization,
                config,
                parameter_unscale,
                characteristic_length))
        {
            return false;
        }

        (void)characteristic_length;

        const Eigen::Matrix<double, 6, 6> H_analysis =
            parameter_unscale.transpose() *
            linearization.H *
            parameter_unscale;

        const Eigen::Matrix<double, 6, 1> b_analysis =
            parameter_unscale.transpose() *
            linearization.b;

        if (!H_analysis.allFinite() ||
            !b_analysis.allFinite())
        {
            return false;
        }

        Eigen::SelfAdjointEigenSolver<
            Eigen::Matrix<double, 6, 6>>
            eigen_solver(
                H_analysis);

        if (eigen_solver.info() !=
            Eigen::Success)
        {
            return false;
        }

        const Eigen::Matrix<double, 6, 1> eigenvalues =
            eigen_solver.eigenvalues();

        const Eigen::Matrix<double, 6, 6> eigenvectors =
            eigen_solver.eigenvectors();

        if (!eigenvalues.allFinite() ||
            !eigenvectors.allFinite())
        {
            return false;
        }

        const double lambda_max =
            eigenvalues(5);

        if (!std::isfinite(lambda_max) ||
            lambda_max <=
                config.degeneracy_absolute_eigenvalue_threshold)
        {
            return false;
        }

        const bool use_hessian_scale_normalization =
            config.enable_sensor_centered_perturbation &&
            config.enable_hessian_scale_normalization;

        const double hard_relative_threshold =
            use_hessian_scale_normalization
                ? 0.01
                : config.degeneracy_relative_eigenvalue_threshold;

        const Eigen::Matrix<double, 6, 1> relative_eigenvalues =
            eigenvalues /
            lambda_max;

        const Eigen::Matrix<double, 6, 1> gradient_eigen =
            eigenvectors.transpose() *
            b_analysis;

        Eigen::Matrix<double, 6, 1> delta_eigen =
            Eigen::Matrix<double, 6, 1>::Zero();

        int usable_directions = 0;

        for (int i = 0;
             i < 6;
             ++i)
        {
            const double lambda =
                eigenvalues(i);

            const double relative_lambda =
                relative_eigenvalues(i);

            const bool strong_degenerate =
                !std::isfinite(lambda) ||
                lambda <=
                    config.degeneracy_absolute_eigenvalue_threshold ||
                !std::isfinite(relative_lambda) ||
                relative_lambda <
                    hard_relative_threshold;

            if (strong_degenerate)
            {
                continue;
            }

            delta_eigen(i) =
                -gradient_eigen(i) /
                lambda;

            ++usable_directions;
        }

        if (usable_directions <= 0)
        {
            return false;
        }

        const Eigen::Matrix<double, 6, 1> delta_analysis =
            eigenvectors *
            delta_eigen;

        dx =
            parameter_unscale *
            delta_analysis;

        if (!dx.allFinite())
        {
            return false;
        }

        if (final_H_analysis != nullptr)
        {
            *final_H_analysis =
                H_analysis;
        }

        if (final_parameter_unscale != nullptr)
        {
            *final_parameter_unscale =
                parameter_unscale;
        }

        return true;
    }

    Eigen::Isometry3d ApplyGroundIcpIncrement(
        const Eigen::Isometry3d &T_target_source,
        const Eigen::Matrix<double, 6, 1> &dx,
        const LidarRegistrationConfig &config)
    {
        Eigen::Isometry3d updated_pose =
            T_target_source;

        const Eigen::Vector3d delta_rotation =
            dx.head<3>();

        const Eigen::Vector3d delta_translation =
            dx.tail<3>();

        const Eigen::Matrix3d delta_R =
            Sophus::SO3d::exp(
                delta_rotation)
                .matrix();

        if (config.enable_sensor_centered_perturbation)
        {
            updated_pose.linear() =
                delta_R *
                updated_pose.rotation();

            updated_pose.translation() +=
                delta_translation;
        }
        else
        {
            Eigen::Isometry3d delta_T =
                Eigen::Isometry3d::Identity();

            delta_T.linear() =
                delta_R;

            delta_T.translation() =
                delta_translation;

            updated_pose =
                delta_T *
                updated_pose;
        }

        return updated_pose;
    }

    bool UpdateGroundIcpRelativeCovariance(
        const GroundIcpLinearization &linearization,
        const LidarRegistrationConfig &config,
        LidarRegistrationResult &result)
    {
        const bool use_hessian_scale_normalization =
            config.enable_sensor_centered_perturbation &&
            config.enable_hessian_scale_normalization;

        if (!use_hessian_scale_normalization)
        {
            return false;
        }

        Eigen::Matrix<double, 6, 6> parameter_unscale;
        double characteristic_length = 1.0;

        if (!ComputeGroundIcpParameterUnscale(
                linearization,
                config,
                parameter_unscale,
                characteristic_length))
        {
            return false;
        }

        const Eigen::Matrix<double, 6, 6> H_analysis =
            parameter_unscale.transpose() *
            linearization.H *
            parameter_unscale;

        Eigen::SelfAdjointEigenSolver<
            Eigen::Matrix<double, 6, 6>>
            eigen_solver(
                H_analysis);

        if (eigen_solver.info() !=
                Eigen::Success ||
            !eigen_solver.eigenvalues().allFinite() ||
            !eigen_solver.eigenvectors().allFinite())
        {
            return false;
        }

        const Eigen::Matrix<double, 6, 1> eigenvalues =
            eigen_solver.eigenvalues();

        const double lambda_max =
            eigenvalues(5);

        if (!std::isfinite(lambda_max) ||
            lambda_max <=
                config.degeneracy_absolute_eigenvalue_threshold)
        {
            return false;
        }

        const Eigen::Matrix<double, 6, 1> relative_eigenvalues =
            eigenvalues /
            lambda_max;

        Eigen::Matrix<double, 6, 6> relative_covariance =
            Eigen::Matrix<double, 6, 6>::Zero();

        for (int i = 0;
             i < 6;
             ++i)
        {
            if (!std::isfinite(
                    relative_eigenvalues(i)))
            {
                return false;
            }

            const double relative_information =
                std::clamp(
                    relative_eigenvalues(i),
                    0.01,
                    1.0);

            const Eigen::Matrix<double, 6, 1> direction =
                eigen_solver.eigenvectors().col(i);

            relative_covariance.noalias() +=
                (1.0 /
                 relative_information) *
                direction *
                direction.transpose();
        }

        relative_covariance =
            0.5 *
            (relative_covariance +
             relative_covariance.transpose());

        if (!relative_covariance.allFinite())
        {
            return false;
        }

        result.hessian_relative_covariance =
            relative_covariance;

        result.hessian_relative_covariance_valid =
            true;

        return true;
    }

    // ========================================================================
    // GROUND_ICP_V13_OBSERVABLE_SUBSPACE_JOINT
    //
    // Final frontend Ground design used in this branch:
    //
    //   E(T) = E_general_point_to_plane(T)
    //        + Q_g * E_ground_point_to_local_ground(T)
    //
    // Ground V4 supplies the trusted support points and frame quality Q_g.
    // The historical reference is NOT one frozen world plane.  Each trusted
    // support point is matched to a nearby, normal-compatible plane already
    // present in the current Prepared LocalMap target.
    //
    // Most importantly, the Ground Jacobian is explicitly projected onto the
    // physically observable ground subspace:
    //
    //       [rx, ry, rz, tx, ty, tz]
    //        ^   ^                ^
    //      roll pitch             z
    //
    // rz / tx / ty are set to zero for Ground residuals.  General ICP remains
    // full 6-DoF and continues to estimate x/y/yaw from all scene geometry.
    //
    // Therefore BOTH objectives enter the SAME Gauss-Newton system:
    //
    //   H_total = H_general + H_ground
    //   b_total = b_general + b_ground
    //   H_total * dx = -b_total
    //
    // This removes the V1.2B global frozen-height assumption while keeping the
    // desired quality-weighted roll/pitch/z Ground information.
    // ========================================================================

    enum class GroundJointIcpStatus
    {
        NotEligible,
        Success,
        Failed
    };

    GroundJointIcpStatus RunTrustedGroundJointIcpV12(
        const RegistrationScan2LocalMap *owner,
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &source,
        const PreparedLidarTarget &target,
        const fr_slam::GroundSegmentationResult &ground_result,
        const Eigen::Isometry3d &initial_guess,
        LidarRegistrationResult &result)
    {
        result =
            LidarRegistrationResult();

        result.T_target_source =
            initial_guess;

        GroundIcpRuntime *runtime =
            GetGroundIcpRuntime(owner);

        if (runtime == nullptr ||
            !runtime->enabled)
        {
            return GroundJointIcpStatus::NotEligible;
        }

        if (!ground_result.support_constraint_valid ||
            !ground_result.support_plane_valid ||
            !ground_result.support_ground_cloud ||
            ground_result.support_ground_cloud->empty())
        {

            return GroundJointIcpStatus::NotEligible;
        }

        const double ground_information_weight =
            ComputeGroundIcpWeight(
                *runtime,
                ground_result);

        if (!std::isfinite(ground_information_weight) ||
            ground_information_weight <= 1.0e-3)
        {

            return GroundJointIcpStatus::NotEligible;
        }

        const LidarRegistrationConfig &config =
            runtime->registration_config;

        result.robust_kernel_enabled =
            config.enable_huber_loss;

        result.robust_kernel_delta =
            config.huber_delta;

        Eigen::Isometry3d current_pose =
            initial_guess;

        bool completed_iteration =
            false;

        for (int iteration = 0;
             iteration < config.max_iterations;
             ++iteration)
        {
            GroundIcpLinearization linearization;

            // General Point-to-Plane + quality-weighted Ground residuals are
            // assembled together here.  The Ground block inside
            // BuildGroundIcpLinearization() has its Jacobian projected to
            // [roll, pitch, z] only.
            if (!BuildGroundIcpLinearization(
                    source,
                    target,
                    current_pose,
                    ground_result,
                    *runtime,
                    ground_information_weight,
                    linearization))
            {
                std::cout
                    << "GROUND_ICP_V13"
                    << " | stage=JOINT_ITER"
                    << " | iteration=" << iteration
                    << " | action=FALLBACK_GENERAL"
                    << " | reason=LINEARIZATION_FAILED"
                    << std::endl;

                return GroundJointIcpStatus::Failed;
            }

            if (linearization.general_correspondences <
                config.min_correspondences)
            {
                std::cout
                    << "GROUND_ICP_V13"
                    << " | stage=JOINT_ITER"
                    << " | iteration=" << iteration
                    << " | action=FALLBACK_GENERAL"
                    << " | reason=LOW_GENERAL_CORR"
                    << " | general_corr="
                    << linearization.general_correspondences
                    << " | min="
                    << config.min_correspondences
                    << std::endl;

                return GroundJointIcpStatus::Failed;
            }

            if (linearization.ground_correspondences <
                runtime->minimum_ground_correspondences)
            {
                std::cout
                    << "GROUND_ICP_V13"
                    << " | stage=JOINT_ITER"
                    << " | iteration=" << iteration
                    << " | action=FALLBACK_GENERAL"
                    << " | reason=LOW_GROUND_CORR"
                    << " | ground_corr="
                    << linearization.ground_correspondences
                    << " | min="
                    << runtime->minimum_ground_correspondences
                    << std::endl;

                return iteration == 0
                           ? GroundJointIcpStatus::NotEligible
                           : GroundJointIcpStatus::Failed;
            }

            Eigen::Matrix<double, 6, 1> dx =
                Eigen::Matrix<double, 6, 1>::Zero();

            if (!SolveGroundIcpStep(
                    linearization,
                    config,
                    dx))
            {
                std::cout
                    << "GROUND_ICP_V13"
                    << " | stage=JOINT_ITER"
                    << " | iteration=" << iteration
                    << " | action=FALLBACK_GENERAL"
                    << " | reason=SOLVE_FAILED"
                    << std::endl;

                return GroundJointIcpStatus::Failed;
            }

            const Eigen::Vector3d delta_rotation =
                dx.head<3>();

            const Eigen::Vector3d delta_translation =
                dx.tail<3>();

            const double dR =
                delta_rotation.norm();

            const double dT =
                delta_translation.norm();

            const double robust_rmse =
                linearization.general_weight_sum > 1.0e-12
                    ? std::sqrt(
                          linearization
                              .general_weighted_squared_error_sum /
                          linearization.general_weight_sum)
                    : std::numeric_limits<double>::infinity();

            const double downweighted_ratio =
                linearization.general_correspondences > 0
                    ? static_cast<double>(
                          linearization
                              .general_downweighted_correspondences) /
                          static_cast<double>(
                              linearization.general_correspondences)
                    : 0.0;

            const Eigen::Isometry3d trial_pose =
                ApplyGroundIcpIncrement(
                    current_pose,
                    dx,
                    config);

            if (!trial_pose.matrix().allFinite())
            {
                return GroundJointIcpStatus::Failed;
            }

            current_pose =
                trial_pose;

            completed_iteration =
                true;

            result.success =
                true;

            result.converged =
                false;

            result.iterations =
                iteration + 1;

            result.correspondences =
                linearization.general_correspondences;

            result.rmse =
                linearization.GeneralRmse();

            result.robust_downweighted_correspondences =
                linearization.general_downweighted_correspondences;

            result.robust_downweighted_ratio =
                downweighted_ratio;

            result.robust_effective_weight_sum =
                linearization.general_weight_sum;

            result.robust_min_weight =
                linearization.general_min_robust_weight;

            result.robust_rmse =
                robust_rmse;

            result.T_target_source =
                current_pose;

            if (dR <
                    config.rotation_convergence_threshold &&
                dT <
                    config.translation_convergence_threshold)
            {
                result.converged =
                    true;
                break;
            }
        }

        if (!completed_iteration ||
            !result.success)
        {
            return GroundJointIcpStatus::Failed;
        }

        GroundIcpLinearization final_linearization;

        if (!BuildGroundIcpLinearization(
                source,
                target,
                current_pose,
                ground_result,
                *runtime,
                ground_information_weight,
                final_linearization))
        {
            return GroundJointIcpStatus::Failed;
        }

        if (final_linearization.general_correspondences <
                config.min_correspondences ||
            final_linearization.ground_correspondences <
                runtime->minimum_ground_correspondences)
        {
            return GroundJointIcpStatus::Failed;
        }

        const double final_robust_rmse =
            final_linearization.general_weight_sum > 1.0e-12
                ? std::sqrt(
                      final_linearization
                          .general_weighted_squared_error_sum /
                      final_linearization.general_weight_sum)
                : std::numeric_limits<double>::infinity();

        const double final_downweighted_ratio =
            final_linearization.general_correspondences > 0
                ? static_cast<double>(
                      final_linearization
                          .general_downweighted_correspondences) /
                      static_cast<double>(
                          final_linearization.general_correspondences)
                : 0.0;

        result.correspondences =
            final_linearization.general_correspondences;

        result.rmse =
            final_linearization.GeneralRmse();

        result.robust_downweighted_correspondences =
            final_linearization.general_downweighted_correspondences;

        result.robust_downweighted_ratio =
            final_downweighted_ratio;

        result.robust_effective_weight_sum =
            final_linearization.general_weight_sum;

        result.robust_min_weight =
            final_linearization.general_min_robust_weight;

        result.robust_rmse =
            final_robust_rmse;

        result.T_target_source =
            current_pose;

        UpdateGroundIcpRelativeCovariance(
            final_linearization,
            config,
            result);

        return GroundJointIcpStatus::Success;
    }
} // namespace

RegistrationScan2LocalMap::RegistrationScan2LocalMap(
    const LidarRegistrationConfig &registration_config,
    const LocalMapConfig &local_map_config)
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
void RegistrationScan2LocalMap::StartBackendWorker()
{
    bool expected = false;

    if (!backend_running_.compare_exchange_strong(
            expected,
            true))
    {
        return;
    }

    backend_thread_ =
        std::thread(
            &RegistrationScan2LocalMap::BackendLoop,
            this);
}

void RegistrationScan2LocalMap::StopBackendWorker()
{
    if (!backend_running_.exchange(false))
    {
        if (backend_thread_.joinable())
        {
            backend_thread_.join();
        }

        return;
    }

    {
        std::lock_guard<std::mutex> lock(
            backend_queue_mutex_);

        backend_queue_.clear();
    }

    backend_condition_.notify_all();

    if (backend_thread_.joinable())
    {
        backend_thread_.join();
    }
}

bool RegistrationScan2LocalMap::BuildFinishedSubmapSnapshot(
    std::size_t submap_id,
    BackendSubmapSnapshot &snapshot) const
{
    for (const Submap &submap :
         submap_manager_.GetAllSubmaps())
    {
        if (submap.id != submap_id)
        {
            continue;
        }

        if (!submap.finished ||
            !submap.has_frozen_cloud ||
            !submap.has_origin_pose ||
            !submap.cloud_S ||
            submap.cloud_S->empty() ||
            !submap.T_WS.matrix().allFinite())
        {
            return false;
        }

        snapshot = BackendSubmapSnapshot();
        snapshot.id = submap.id;
        snapshot.T_WS = submap.T_WS;
        snapshot.keyframe_ids = submap.keyframe_ids;
        snapshot.cloud_S = submap.cloud_S;

        return true;
    }

    return false;
}

bool RegistrationScan2LocalMap::EnqueueBackendKeyframe(
    const Keyframe &keyframe,
    std::size_t current_submap_id)
{
    if (!backend_running_.load() ||
        !keyframe.cloud ||
        keyframe.cloud->empty() ||
        !keyframe.T_WL.matrix().allFinite())
    {
        return false;
    }

    BackendKeyframeJob job;
    job.keyframe = keyframe;
    job.current_submap_id = current_submap_id;

    if (submap_manager_.LastAddStartedNewSubmap())
    {
        BackendSubmapSnapshot finished_snapshot;

        if (BuildFinishedSubmapSnapshot(
                submap_manager_.LastFinishedSubmapId(),
                finished_snapshot))
        {
            job.has_finished_submap = true;
            job.finished_submap =
                std::move(finished_snapshot);
        }
    }

    std::size_t queue_size = 0;

    {
        std::lock_guard<std::mutex> lock(
            backend_queue_mutex_);

        if (!backend_running_.load())
        {
            return false;
        }

        backend_queue_.push_back(
            std::move(job));

        queue_size =
            backend_queue_.size();
    }

    backend_condition_.notify_one();

    if (queue_size >
        backend_backlog_warning_threshold_)
    {
        RCLCPP_WARN(
            kTimingLogger,
            "FR_BACKEND backlog growing"
            " | queued_keyframes=%zu"
            " | warning_threshold=%zu",
            queue_size,
            backend_backlog_warning_threshold_);
    }

    return true;
}

void RegistrationScan2LocalMap::StoreBackendFinishedSubmap(
    const BackendSubmapSnapshot &snapshot)
{
    if (snapshot.id ==
            std::numeric_limits<std::size_t>::max() ||
        !snapshot.cloud_S ||
        snapshot.cloud_S->empty() ||
        !snapshot.T_WS.matrix().allFinite())
    {
        return;
    }

    for (BackendSubmapSnapshot &stored :
         backend_finished_submaps_)
    {
        if (stored.id == snapshot.id)
        {
            stored = snapshot;
            return;
        }
    }

    backend_finished_submaps_.push_back(
        snapshot);
}

void RegistrationScan2LocalMap::RefreshBackendOutputSnapshot()
{
    std::lock_guard<std::mutex> lock(
        backend_output_mutex_);

    backend_pose_graph_snapshot_ =
        pose_graph_;

    backend_raw_map_snapshot_ =
        incremental_global_map_.GetRawMap();

    backend_optimized_map_snapshot_ =
        incremental_global_map_.GetOptimizedMap();

    backend_refined_map_snapshot_ =
        incremental_global_map_.GetRefinedMap();

    backend_refinement_historical_target_snapshot_ =
        refinement_historical_target_debug_;

    backend_refinement_current_before_snapshot_ =
        refinement_current_before_debug_;

    backend_refinement_current_after_snapshot_ =
        refinement_current_after_debug_;

    backend_global_map_revision_snapshot_ =
        global_map_revision_;

    backend_refined_map_revision_snapshot_ =
        refined_map_revision_;

    backend_refinement_debug_revision_snapshot_ =
        refinement_debug_revision_;

    backend_T_map_odom_snapshot_ =
        T_map_odom_;

    backend_has_map_odom_correction_snapshot_ =
        has_map_odom_correction_;

    backend_map_odom_revision_snapshot_ =
        map_odom_revision_;
}

void RegistrationScan2LocalMap::ProcessBackendJob(
    const BackendKeyframeJob &job)
{
    const std::chrono::steady_clock::time_point
        backend_start =
            std::chrono::steady_clock::now();

    double pose_graph_ms = 0.0;
    double global_map_ms = 0.0;
    double scan_context_ms = 0.0;
    double loop_ms = 0.0;

    if (job.has_finished_submap)
    {
        StoreBackendFinishedSubmap(
            job.finished_submap);
    }

    if (FindBackendKeyframeById(
            job.keyframe.id) == nullptr)
    {
        backend_keyframes_.push_back(
            job.keyframe);
    }

    const std::chrono::steady_clock::time_point
        pose_graph_start =
            std::chrono::steady_clock::now();

    const bool pose_graph_ok =
        AddKeyframeToPoseGraph(
            job.keyframe);

    pose_graph_ms =
        ElapsedMilliseconds(
            pose_graph_start,
            std::chrono::steady_clock::now());

    if (!pose_graph_ok)
    {
        std::cerr
            << "Async backend PoseGraph insert failed"
            << " | keyframe=" << job.keyframe.id
            << std::endl;

        RefreshBackendOutputSnapshot();
        return;
    }

    const std::chrono::steady_clock::time_point
        global_map_start =
            std::chrono::steady_clock::now();

    const bool global_map_ok =
        UpdateIncrementalGlobalMaps(
            "NEW_KEYFRAME",
            false);

    global_map_ms =
        ElapsedMilliseconds(
            global_map_start,
            std::chrono::steady_clock::now());

    if (!global_map_ok)
    {
        std::cerr
            << "Async backend global map update failed"
            << " | keyframe=" << job.keyframe.id
            << std::endl;
    }

    const std::chrono::steady_clock::time_point
        scan_context_start =
            std::chrono::steady_clock::now();

    const bool scan_context_ok =
        loop_detector_.AddKeyframe(
            job.keyframe);

    scan_context_ms =
        ElapsedMilliseconds(
            scan_context_start,
            std::chrono::steady_clock::now());

    if (scan_context_ok)
    {
        const std::chrono::steady_clock::time_point
            loop_start =
                std::chrono::steady_clock::now();

        DetectAndVerifyLoopFromKeyframe(
            job.keyframe,
            job.current_submap_id);

        loop_ms =
            ElapsedMilliseconds(
                loop_start,
                std::chrono::steady_clock::now());
    }
    else
    {
        std::cerr
            << "Async backend Scan Context registration failed"
            << " | keyframe=" << job.keyframe.id
            << std::endl;
    }

    RefreshBackendOutputSnapshot();

    std::size_t remaining_queue = 0;

    {
        std::lock_guard<std::mutex> lock(
            backend_queue_mutex_);

        remaining_queue =
            backend_queue_.size();
    }

    const double total_ms =
        ElapsedMilliseconds(
            backend_start,
            std::chrono::steady_clock::now());

    RCLCPP_INFO(
        kTimingLogger,
        "FR_TIMING BACKEND_JOB"
        " | keyframe=%zu"
        " | total=%.3f ms"
        " | pose_graph=%.3f"
        " | global_map=%.3f"
        " | scan_context_insert=%.3f"
        " | loop_backend=%.3f"
        " | remaining_queue=%zu"
        " | backend_keyframes=%zu"
        " | backend_submaps=%zu",
        job.keyframe.id,
        total_ms,
        pose_graph_ms,
        global_map_ms,
        scan_context_ms,
        loop_ms,
        remaining_queue,
        backend_keyframes_.size(),
        backend_finished_submaps_.size());
}

void RegistrationScan2LocalMap::BackendLoop()
{
    while (true)
    {
        BackendKeyframeJob job;

        {
            std::unique_lock<std::mutex> lock(
                backend_queue_mutex_);

            backend_condition_.wait(
                lock,
                [this]()
                {
                    return !backend_running_.load() ||
                           !backend_queue_.empty();
                });

            if (!backend_running_.load() &&
                backend_queue_.empty())
            {
                break;
            }

            if (backend_queue_.empty())
            {
                continue;
            }

            job =
                std::move(
                    backend_queue_.front());

            backend_queue_.pop_front();
        }

        try
        {
            ProcessBackendJob(
                job);
        }
        catch (const std::exception &exception)
        {
            std::cerr
                << "Async backend exception"
                << " | keyframe=" << job.keyframe.id
                << " | what=" << exception.what()
                << std::endl;
        }
        catch (...)
        {
            std::cerr
                << "Async backend unknown exception"
                << " | keyframe=" << job.keyframe.id
                << std::endl;
        }
    }
}

// ============================================================================
// AddFrame()
//
// Process ONE LiDAR scan.
//
// Parameters:
//
// cloud_lidar:
//     Current processed LiDAR cloud.
//
//     Coordinate frame:
//         current LiDAR frame.
//
//     It is expected to have already gone through the outer pipeline such as:
//
//         deskew
//         filtering
//         voxelization
//
// timestamp:
//     Current LiDAR scan start timestamp in seconds.
//
//     This is stored in KeyframeManager when the frame becomes a keyframe.
//
// T_WL:
//     Output parameter.
//
//     On success:
//         receives the current accepted LiDAR -> World pose.
//
//     On rejection:
//         this function returns false and does NOT commit the rejected pose.
//
// registration_result:
//     Output registration diagnostics:
//
//         success
//         converged
//         correspondences
//         rmse
//         T_target_source
//         ...
//
// imu_relative_rotation:
//     Optional relative LiDAR rotation prediction derived from IMU.
//
//     In the current frontend design, IMU contributes ONLY to the rotational
//     part of the registration initial guess.
//
// Returns:
//
//     true:
//         current frame is accepted by the frontend.
//
//     false:
//         current frame is rejected / cannot be processed.
//
// ============================================================================

bool RegistrationScan2LocalMap::AddKeyframeToPoseGraph(
    const Keyframe &keyframe)
{
    if (!keyframe.T_WL.matrix().allFinite())
    {
        std::cerr
            << "Keyframe PoseGraph: invalid Keyframe pose"
            << " | keyframe=" << keyframe.id
            << std::endl;
        return false;
    }

    if (pose_graph_.HasNode(keyframe.id))
    {
        // Idempotent protection. A Keyframe should normally enter exactly once.
        return true;
    }

    const bool fixed =
        pose_graph_.NodeCount() == 0;

    const Keyframe *previous_keyframe = nullptr;
    Eigen::Isometry3d Z_previous_current =
        Eigen::Isometry3d::Identity();

    const std::vector<Keyframe> &all_keyframes =
        backend_keyframes_;

    if (!fixed)
    {
        // The current Keyframe has already been appended to KeyframeManager,
        // therefore its sequential predecessor is the second-to-last entry.
        // This does not assume Keyframe IDs are perfectly contiguous.
        if (all_keyframes.size() < 2)
        {
            return false;
        }

        previous_keyframe =
            &all_keyframes[all_keyframes.size() - 2];

        if (previous_keyframe->id == keyframe.id ||
            !previous_keyframe->T_WL.matrix().allFinite() ||
            !pose_graph_.HasNode(previous_keyframe->id))
        {
            std::cerr
                << "Keyframe PoseGraph: previous Keyframe/node missing"
                << " | current_kf=" << keyframe.id
                << " | previous_kf=" << previous_keyframe->id
                << std::endl;
            return false;
        }

        Z_previous_current =
            previous_keyframe->T_WL.inverse() *
            keyframe.T_WL;

        if (!Z_previous_current.matrix().allFinite())
        {
            std::cerr
                << "Keyframe PoseGraph: non-finite odometry measurement"
                << " | from=" << previous_keyframe->id
                << " | to=" << keyframe.id
                << std::endl;
            return false;
        }
    }

    // Initial graph pose for the new Keyframe.
    //
    // Before the first backend optimization this is identical to keyframe.T_WL.
    // After g2o has corrected the previous graph vertex, initialize every new
    // Keyframe by chaining the immutable frontend odometry measurement from
    // the corrected previous graph estimate:
    //
    //     X_i_initial = X_(i-1)_optimized * Z_(i-1,i)
    //
    // This keeps future graph vertices in the corrected backend/map frame while
    // the live frontend continues in its own continuous odometry frame.
    Eigen::Isometry3d T_WK_graph_initial =
        keyframe.T_WL;

    if (!fixed)
    {
        const PoseGraphNode *previous_graph_node =
            pose_graph_.GetNode(previous_keyframe->id);

        if (previous_graph_node == nullptr ||
            !previous_graph_node->T_WK.matrix().allFinite())
        {
            return false;
        }

        T_WK_graph_initial =
            previous_graph_node->T_WK *
            Z_previous_current;

        if (!T_WK_graph_initial.matrix().allFinite())
        {
            return false;
        }
    }

    if (!pose_graph_.AddNode(
            keyframe.id,
            T_WK_graph_initial,
            fixed))
    {
        std::cerr
            << "Keyframe PoseGraph: AddNode failed"
            << " | keyframe=" << keyframe.id
            << std::endl;
        return false;
    }

    // ------------------------------------------------------------------------
    // Gravity Guard V1: store an IMMUTABLE frontend tilt reference.
    //
    // Do not use T_WK_graph_initial here.  After the first backend correction
    // that pose already lives in the corrected map frame.  The raw frontend
    // Keyframe pose T_WL remains the physical IMU/Scan-to-LocalMap reference.
    //
    //     gravity_L_ref = R_WL(raw)^T * UnitZ
    //
    // A pure world-yaw correction does not change this vector, so the backend
    // remains free to close heading drift while artificial roll/pitch tilt is
    // penalized by the PoseGraph optimizer.
    // ------------------------------------------------------------------------
    Eigen::Vector3d gravity_L_reference =
        keyframe.T_WL.rotation().transpose() *
        Eigen::Vector3d::UnitZ();

    if (!gravity_L_reference.allFinite() ||
        gravity_L_reference.norm() < 1.0e-9 ||
        !pose_graph_.SetNodeGravityReference(
            keyframe.id,
            gravity_L_reference))
    {
        std::cerr
            << "Keyframe PoseGraph: gravity reference failed"
            << " | keyframe=" << keyframe.id
            << std::endl;
        return false;
    }

    gravity_L_reference.normalize();

    // After the first backend optimization, the new graph-pose chaining and
    // the explicit map->odom bridge should predict the SAME corrected pose:
    //
    //     previous_T_WK * (previous_T_WL^-1 * current_T_WL)
    //       ==
    //     T_map_odom * current_T_WL
    //
    // This diagnostic catches frame-convention mistakes immediately.
    if (has_map_odom_correction_)
    {
        const Eigen::Isometry3d T_WK_from_bridge =
            T_map_odom_ *
            keyframe.T_WL;

        const Eigen::Isometry3d T_bridge_error =
            T_WK_graph_initial.inverse() *
            T_WK_from_bridge;

        const double bridge_translation_error =
            T_bridge_error.translation().norm();

        const Eigen::AngleAxisd bridge_rotation_error(
            T_bridge_error.rotation());

        const double bridge_rotation_error_deg =
            std::abs(
                bridge_rotation_error.angle()) *
            180.0 /
            3.14159265358979323846;

        std::cout
            << "Map->odom bridge Keyframe consistency"
            << " | keyframe=" << keyframe.id
            << " | translation_error="
            << bridge_translation_error << " m"
            << " | rotation_error="
            << bridge_rotation_error_deg << " deg"
            << " | correction_revision="
            << map_odom_revision_
            << std::endl;
    }

    if (fixed)
    {
        return true;
    }

    Eigen::Matrix<double, 6, 6> information =
        Eigen::Matrix<double, 6, 6>::Identity();

    const char *information_mode =
        "IDENTITY_FALLBACK";

    if (keyframe.has_odom_information &&
        keyframe.odom_information.allFinite())
    {
        const Eigen::Matrix<double, 6, 6>
            symmetric_information =
                0.5 *
                (keyframe.odom_information +
                 keyframe.odom_information.transpose());

        const double asymmetry =
            (keyframe.odom_information -
             keyframe.odom_information.transpose())
                .cwiseAbs()
                .maxCoeff();

        Eigen::SelfAdjointEigenSolver<
            Eigen::Matrix<double, 6, 6>>
            solver(
                symmetric_information,
                Eigen::EigenvaluesOnly);

        const bool information_valid =
            std::isfinite(asymmetry) &&
            asymmetry <= 1.0e-8 &&
            solver.info() == Eigen::Success &&
            solver.eigenvalues().allFinite() &&
            solver.eigenvalues().minCoeff() > 1.0e-9;

        if (information_valid)
        {
            information =
                symmetric_information;

            information_mode =
                "V2B_FULL_6X6_V2";
        }
    }

    double pose_graph_maximum_off_diagonal =
        0.0;

    double pose_graph_maximum_tr_coupling =
        0.0;

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

            pose_graph_maximum_off_diagonal =
                std::max(
                    pose_graph_maximum_off_diagonal,
                    std::abs(
                        information(i, j)));

            const bool translation_rotation_pair =
                (i < 3 && j >= 3) ||
                (i >= 3 && j < 3);

            if (translation_rotation_pair)
            {
                pose_graph_maximum_tr_coupling =
                    std::max(
                        pose_graph_maximum_tr_coupling,
                        std::abs(
                            information(i, j)));
            }
        }
    }

    if (!pose_graph_.AddOdometryEdge(
            previous_keyframe->id,
            keyframe.id,
            Z_previous_current,
            information))
    {
        std::cerr
            << "Keyframe PoseGraph: AddOdometryEdge failed"
            << " | from=" << previous_keyframe->id
            << " | to=" << keyframe.id
            << std::endl;
        return false;
    }

    return true;
}

// ============================================================================
// Backend-only snapshot lookup helpers.
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
    // Startup / local-neighborhood protection.
    //
    // With min_loop_submap_separation_ = 5, S0...S4 cannot yet have a
    // historical Submap far enough away in graph topology to be considered a
    // loop. This prevents the first straight/local trajectory segment from
    // generating red loop edges while preserving Keyframe-driven querying.
    // ---------------------------------------------------------------------
    if (current_submap_id < min_loop_submap_separation_)
    {
        std::cout
            << "Keyframe loop detection skipped"
            << " | current_kf=" << current_keyframe.id
            << " | current_submap=" << current_submap_id
            << " | min_submap_gap=" << min_loop_submap_separation_
            << " | reason=INSUFFICIENT_SUBMAP_HISTORY"
            << std::endl;
        return;
    }

    const std::chrono::steady_clock::time_point
        scan_context_detect_start =
            std::chrono::steady_clock::now();

    const std::vector<LoopCandidate> candidates =
        loop_detector_.Detect(
            current_keyframe.id);

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
        << std::endl;

    for (const LoopCandidate &candidate : candidates)
    {
        std::cout
            << "Keyframe loop candidate"
            << " | current_kf=" << candidate.current_id
            << " | historical_kf=" << candidate.candidate_id
            << " | sc_distance=" << candidate.scan_context_distance
            << " | sc_similarity=" << candidate.scan_context_similarity
            << " | sector_shift=" << candidate.sector_shift
            << " | yaw_shift=" << candidate.yaw_shift_deg
            << " deg"
            << " | time_separation=" << candidate.time_separation_sec
            << " s"
            << " | pose_distance=" << candidate.distance
            << " m"
            << std::endl;
    }

    if (candidates.empty())
    {
        return;
    }

    const std::size_t verify_count =
        std::min(
            max_loop_candidates_to_verify_,
            candidates.size());

    const LoopCandidate *best_candidate = nullptr;
    const BackendSubmapSnapshot *best_historical_submap = nullptr;
    LoopVerificationResult best_verification;

    // Do not run ICP repeatedly against the same historical Submap merely
    // because several nearby candidate Keyframes map to it.
    std::vector<std::size_t> verified_historical_submaps;

    for (std::size_t index = 0;
         index < verify_count;
         ++index)
    {
        const LoopCandidate &candidate =
            candidates[index];

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
        // Reject normal local-trajectory overlap before ICP.
        //
        // Keyframe Scan Context is allowed to retrieve any sufficiently old
        // Keyframe. However, once the candidate is mapped to its owning
        // historical Submap, nearby Submaps are NOT loop closures; they are
        // simply the same local trajectory neighborhood.
        //
        // This gate fixes the failure pattern observed in the current log:
        //     S0  -> S2
        //     S0  -> S3
        //     S23 -> S26
        //     S24 -> S27
        //     S40 -> S43
        // all had Submap gap 2 or 3 and should never have reached ICP /
        // temporal acceptance.
        // ----------------------------------------------------------------
        if (historical_submap->id >= current_submap_id)
        {
            std::cout
                << "Keyframe loop candidate rejected"
                << " | current_kf=" << current_keyframe.id
                << " | historical_kf=" << candidate.candidate_id
                << " | current_submap=" << current_submap_id
                << " | historical_submap=" << historical_submap->id
                << " | reason=NOT_HISTORICAL_SUBMAP"
                << std::endl;
            continue;
        }

        const std::size_t submap_gap =
            current_submap_id - historical_submap->id;

        if (submap_gap < min_loop_submap_separation_)
        {
            std::cout
                << "Keyframe loop candidate rejected"
                << " | current_kf=" << current_keyframe.id
                << " | historical_kf=" << candidate.candidate_id
                << " | current_submap=" << current_submap_id
                << " | historical_submap=" << historical_submap->id
                << " | submap_gap=" << submap_gap
                << " | min_submap_gap="
                << min_loop_submap_separation_
                << " | reason=LOCAL_NEIGHBORHOOD"
                << std::endl;
            continue;
        }

        if (std::find(
                verified_historical_submaps.begin(),
                verified_historical_submaps.end(),
                historical_submap->id) !=
            verified_historical_submaps.end())
        {
            continue;
        }

        verified_historical_submaps.push_back(
            historical_submap->id);

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

        const Eigen::Isometry3d T_H_K =
            historical_submap->T_WS.inverse() *
            historical_keyframe->T_WL;

        if (!T_H_K.matrix().allFinite())
        {
            continue;
        }

        std::vector<std::pair<const char *, Eigen::Isometry3d>>
            initial_guesses;

        initial_guesses.emplace_back(
            "CANDIDATE_POSE",
            T_H_K);

        if (std::isfinite(candidate.yaw_shift_deg))
        {
            const double yaw_rad =
                candidate.yaw_shift_deg *
                M_PI / 180.0;

            Eigen::Isometry3d positive = T_H_K;
            positive.linear() =
                T_H_K.rotation() *
                Eigen::AngleAxisd(
                    yaw_rad,
                    Eigen::Vector3d::UnitZ())
                    .toRotationMatrix();

            Eigen::Isometry3d negative = T_H_K;
            negative.linear() =
                T_H_K.rotation() *
                Eigen::AngleAxisd(
                    -yaw_rad,
                    Eigen::Vector3d::UnitZ())
                    .toRotationMatrix();

            initial_guesses.emplace_back(
                "CANDIDATE_SC_POSITIVE",
                positive);

            initial_guesses.emplace_back(
                "CANDIDATE_SC_NEGATIVE",
                negative);
        }

        // --------------------------------------------------------------------
        // LoopVerifier V3: cheap pre-score -> ranked full ICP -> early exit.
        //
        // Previous behavior ran one FULL ICP for every initial guess.  With
        // CANDIDATE_POSE / +SC / -SC this meant three 50-iteration ICP runs per
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
        //   * As soon as one ranked hypothesis passes the ORIGINAL full
        //     LoopVerifier geometry gate, we stop.
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

            const bool score_ok =
                loop_verifier_.ScoreInitialGuess(
                    current_keyframe.cloud,
                    historical_submap->cloud_S,
                    ranked_guess.transform,
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
        const char *best_initial_guess_name = "NONE";

        const LoopVerifierConfig &verifier_config =
            loop_verifier_.GetConfig();

        for (const RankedInitialGuess &ranked_guess : ranked_guesses)
        {
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

            const bool trial_success =
                loop_verifier_.Verify(
                    current_keyframe.cloud,
                    historical_submap->cloud_S,
                    ranked_guess.transform,
                    std::numeric_limits<double>::quiet_NaN(),
                    trial);

            loop_timing.verifier_ms +=
                ElapsedMilliseconds(
                    verifier_start,
                    std::chrono::steady_clock::now());

            ++loop_timing.verifier_calls;

            if (!trial_success)
            {
                continue;
            }

            bool trial_better =
                !verification_success;

            if (!trial_better &&
                trial.accepted != verification.accepted)
            {
                trial_better = trial.accepted;
            }
            else if (!trial_better &&
                     trial.accepted == verification.accepted &&
                     trial.overlap_ratio >
                         verification.overlap_ratio + 1.0e-12)
            {
                trial_better = true;
            }
            else if (!trial_better &&
                     trial.accepted == verification.accepted &&
                     std::abs(
                         trial.overlap_ratio -
                         verification.overlap_ratio) <= 1.0e-12 &&
                     trial.rmse < verification.rmse)
            {
                trial_better = true;
            }

            if (trial_better)
            {
                verification = trial;
                verification_success = true;
                best_initial_guess_name = ranked_guess.name;
            }

            // The initial guesses were already ranked by the cheap geometric
            // pre-score.  Once a hypothesis passes the unchanged full geometry
            // gate, spending another 1-2 complete ICP runs gives little benefit
            // compared with its backend cost.
            if (trial.accepted)
            {
                break;
            }
        }

        std::cout
            << "Keyframe loop verification"
            << " | current_kf=" << current_keyframe.id
            << " | current_submap=" << current_submap_id
            << " | historical_kf=" << candidate.candidate_id
            << " | historical_submap=" << historical_submap->id
            << " | success="
            << (verification_success ? "true" : "false")
            << " | converged="
            << (verification.converged ? "true" : "false")
            << " | accepted="
            << (verification.accepted ? "true" : "false")
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
            << std::endl;

        if (!verification_success ||
            !verification.accepted)
        {
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
    // Convert the ICP result from historical Submap coordinates to the
    // historical CANDIDATE KEYFRAME coordinates.
    //
    // H = historical geometry Submap
    // K = historical candidate Keyframe
    // L = current Keyframe
    //
    //     T_H_K = T_WH^-1 * T_WK
    //     T_H_L = ICP result
    //
    // therefore:
    //
    //     T_K_L = T_H_K^-1 * T_H_L
    //
    // T_K_L is the real loop-factor measurement inserted into the Keyframe
    // PoseGraph.
    // ------------------------------------------------------------------------
    const Keyframe *best_historical_keyframe =
        FindBackendKeyframeById(best_candidate->candidate_id);

    if (best_historical_keyframe == nullptr ||
        !best_historical_keyframe->T_WL.matrix().allFinite())
    {
        return;
    }

    const Eigen::Isometry3d T_H_K =
        best_historical_submap->T_WS.inverse() *
        best_historical_keyframe->T_WL;

    const Eigen::Isometry3d T_K_L =
        T_H_K.inverse() *
        best_verification.T_target_source;

    if (!T_H_K.matrix().allFinite() ||
        !T_K_L.matrix().allFinite())
    {
        return;
    }

    // What world pose would this loop imply for the CURRENT KEYFRAME?
    const Eigen::Isometry3d T_W_L_loop =
        best_historical_keyframe->T_WL *
        T_K_L;

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

    const bool frontend_loop_drift_graph_gate_pass =
        std::isfinite(graph_correction_translation) &&
        std::isfinite(graph_correction_rotation) &&
        graph_correction_translation <=
            max_loop_graph_correction_translation_ &&
        graph_correction_rotation <=
            max_loop_graph_correction_rotation_deg_;

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
        << " | overlap=" << best_verification.overlap_ratio
        << " | rmse=" << best_verification.rmse << " m"
        << " | graph_correction_translation="
        << graph_correction_translation << " m"
        << " | graph_correction_rotation="
        << graph_correction_rotation << " deg"
        << std::endl;

    // Strong independent protection against large false loop corrections.
    if (!std::isfinite(graph_correction_translation) ||
        !std::isfinite(graph_correction_rotation) ||
        graph_correction_translation >
            max_loop_graph_correction_translation_ ||
        graph_correction_rotation >
            max_loop_graph_correction_rotation_deg_)
    {
        std::cout
            << "Keyframe loop decision"
            << " | current_kf=" << current_keyframe.id
            << " | decision=REJECT"
            << " | reason=GRAPH_CORRECTION_TOO_LARGE"
            << std::endl;
        return;
    }

    // ------------------------------------------------------------------------
    // Keyframe-level temporal consistency.
    //
    // The loop track is expressed as a world correction on the current
    // Keyframe, so normal frontend Submap transitions do not reset it.
    //
    // Instead compare the LEFT-MULTIPLICATIVE WORLD CORRECTION implied by each
    // verified current Keyframe:
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
        // that track is considered stale.  In that case the current candidate
        // is allowed to start a NEW track instead of being rejected forever.
        track_stale =
            current_gap > online_loop_max_current_keyframe_gap_;

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
                current_gap <= online_loop_max_current_keyframe_gap_;

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

            track_translation_error =
                (T_loop_correction.translation() -
                 online_loop_track_.T_loop_correction.translation())
                    .norm();

            track_rotation_error =
                RelativeRotationDeg(
                    online_loop_track_.T_loop_correction,
                    T_loop_correction);

            correction_ok =
                std::isfinite(track_translation_error) &&
                std::isfinite(track_rotation_error) &&
                track_translation_error <=
                    online_loop_track_translation_error_ &&
                track_rotation_error <=
                    online_loop_track_rotation_error_deg_;

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
                    << " | action=PRESERVE_TRACK"
                    << std::endl;

                return;
            }
        }
    }

    if (!had_track || track_stale)
    {
        // No usable previous track: start from the current verified loop.
        online_loop_track_ = OnlineLoopTrack();
        online_loop_track_.valid = true;
        online_loop_track_.support = 1;
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
        ++online_loop_track_.support;

        // Lock the historical traversal direction once we observe a movement
        // larger than the permitted one-keyframe jitter.  Zero delta is fine:
        // two neighboring current KFs may legitimately choose the same
        // historical anchor KF.
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

    std::cout
        << "Keyframe loop temporal"
        << " | current_kf=" << current_keyframe.id
        << " | current_submap=" << current_submap_id
        << " | historical_kf=" << best_candidate->candidate_id
        << " | historical_submap=" << best_historical_submap->id
        << " | support=" << online_loop_track_.support
        << "/" << online_loop_min_support_
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

        const bool common_geometry_ok =
            geometry_finite &&
            best_verification.overlap_ratio >=
                online_loop_first_edge_min_overlap_ &&
            best_verification.rmse <=
                online_loop_first_edge_max_rmse_;

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

            spacing_ok =
                batch_current_gap >=
                    online_loop_first_batch_current_spacing_ &&
                batch_historical_gap >=
                    online_loop_first_batch_historical_spacing_;

            const Eigen::Isometry3d correction_error =
                pending_first_loop_batch_.front()
                    .T_loop_correction.inverse() *
                T_loop_correction;

            batch_translation_error =
                correction_error.translation().norm();

            batch_rotation_error_deg =
                RelativeRotationDeg(
                    Eigen::Isometry3d::Identity(),
                    correction_error);

            correction_consistent =
                correction_error.matrix().allFinite() &&
                std::isfinite(batch_translation_error) &&
                std::isfinite(batch_rotation_error_deg) &&
                batch_translation_error <=
                    online_loop_first_batch_max_translation_error_ &&
                batch_rotation_error_deg <=
                    online_loop_first_batch_max_rotation_error_deg_;
        }

        if (strong_first_geometry &&
            spacing_ok &&
            correction_consistent)
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
                << " | current_gap=" << batch_current_gap
                << " | historical_gap=" << batch_historical_gap
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

        const bool first_batch_ready =
            first_batch_has_enough_edges &&
            first_batch_has_enough_support;

        if (!first_batch_ready)
        {
            std::cout
                << "Keyframe loop decision"
                << " | current_kf=" << current_keyframe.id
                << " | decision=PENDING"
                << " | reason=WAIT_FIRST_LOOP_BATCH"
                << " | support="
                << online_loop_track_.support
                << "/"
                << online_loop_first_edge_min_support_
                << " | batch="
                << pending_first_loop_batch_.size()
                << "/"
                << online_loop_first_batch_min_edges_
                << " | enough_edges="
                << (first_batch_has_enough_edges ? "true" : "false")
                << " | enough_support="
                << (first_batch_has_enough_support ? "true" : "false")
                << std::endl;
            return;
        }

        std::cout
            << "First loop batch ready"
            << " | edges="
            << pending_first_loop_batch_.size()
            << " | support="
            << online_loop_track_.support
            << " | action=STAGE_AND_OPTIMIZE"
            << std::endl;

        std::vector<std::pair<std::size_t, std::size_t>>
            staged_loop_edges;

        staged_loop_edges.reserve(
            pending_first_loop_batch_.size());

        bool batch_stage_ok = true;

        for (const PendingLoopConstraint &constraint :
             pending_first_loop_batch_)
        {
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

            const BackendSubmapSnapshot *loop_historical_submap =
                FindBackendSubmapById(
                    constraint.historical_submap_id);

            if (loop_current_keyframe != nullptr &&
                loop_historical_keyframe != nullptr &&
                loop_historical_submap != nullptr &&
                loop_current_keyframe->cloud &&
                loop_historical_submap->cloud_S &&
                loop_historical_submap->T_WS.matrix().allFinite() &&
                loop_historical_keyframe->T_WL.matrix().allFinite())
            {
                const Eigen::Isometry3d T_H_K_for_information =
                    loop_historical_submap->T_WS.inverse() *
                    loop_historical_keyframe->T_WL;

                const Eigen::Isometry3d T_H_L_for_information =
                    T_H_K_for_information *
                    constraint.T_historical_current;

                if (T_H_K_for_information.matrix().allFinite() &&
                    T_H_L_for_information.matrix().allFinite())
                {
                    dynamic_loop_information =
                        BuildLoopShadowInformationFull6x6(
                            loop_current_keyframe->cloud,
                            loop_historical_submap->cloud_S,
                            T_H_L_for_information,
                            loop_verifier_.GetConfig(),
                            loop_information,
                            loop_shadow_correspondences,
                            loop_median_range,
                            loop_min_relative);
                }
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
                << "ONLINE Keyframe PoseGraph loop edge staged"
                << " | from_kf="
                << constraint.historical_keyframe_id
                << " | to_kf="
                << constraint.current_keyframe_id
                << " | edge_mode=FIRST_BATCH"
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
        }

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

            std::cerr
                << "First loop batch optimization rejected"
                << " | staged_edges="
                << staged_loop_edges.size()
                << " | rolled_back=" << rollback_count
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
                << " | action=ROLLBACK_ALL"
                << std::endl;
            return;
        }

        loop_timing.optimization_accepted = true;
        loop_timing.loop_edge_accepted = true;

        const PendingLoopConstraint &last_constraint =
            pending_first_loop_batch_.back();

        has_last_online_loop_edge_ = true;
        last_online_loop_current_keyframe_id_ =
            last_constraint.current_keyframe_id;
        last_online_loop_historical_keyframe_id_ =
            last_constraint.historical_keyframe_id;
        last_online_loop_measurement_ =
            last_constraint.T_historical_current;

        const std::size_t accepted_batch_size =
            pending_first_loop_batch_.size();

        pending_first_loop_batch_.clear();

        std::cout
            << "ONLINE Keyframe PoseGraph first loop batch accepted"
            << " | batch_edges=" << accepted_batch_size
            << " | last_from_kf="
            << last_online_loop_historical_keyframe_id_
            << " | last_to_kf="
            << last_online_loop_current_keyframe_id_
            << " | total_loop_edges="
            << pose_graph_.LoopEdgeCount()
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
        return;
    }

    // Subsequent sequence loop: one verified candidate may be considered after
    // the initial multi-edge batch has already established the revisit geometry.
    std::size_t edge_historical_keyframe_id =
        best_candidate->candidate_id;
    std::size_t edge_current_keyframe_id =
        current_keyframe.id;
    std::size_t edge_historical_submap_id =
        best_historical_submap->id;
    std::size_t edge_current_submap_id =
        current_submap_id;
    Eigen::Isometry3d edge_measurement =
        T_K_L;

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

        if (!current_spacing_ok ||
            !historical_spacing_ok)
        {
            std::cout
                << "Keyframe loop decision"
                << " | current_kf=" << edge_current_keyframe_id
                << " | historical_kf="
                << edge_historical_keyframe_id
                << " | decision=TRACK_ONLY"
                << " | reason=LOOP_EDGE_SPACING"
                << " | current_gap=" << loop_edge_current_gap
                << " | min_current_gap="
                << min_online_loop_edge_current_keyframe_spacing_
                << " | historical_gap="
                << loop_edge_historical_gap
                << " | min_historical_gap="
                << min_online_loop_edge_historical_keyframe_spacing_
                << " | support=" << online_loop_track_.support
                << std::endl;

            return;
        }

        // --------------------------------------------------------------------
        // V10: loop-to-loop SE(3) cycle consistency.
        //
        // Last inserted loop:
        //     Z1 = T_h1_c1
        // New candidate loop:
        //     Z2 = T_h2_c2
        // Historical odometry increment:
        //     A = T_h1_h2
        // Current odometry increment:
        //     B = T_c1_c2
        //
        // Both paths from h1 to c2 should agree:
        //
        //     Z1 * B  ~=  A * Z2
        //
        // Cycle error:
        //
        //     E = (A * Z2)^-1 * (Z1 * B)
        // --------------------------------------------------------------------
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

        if (!cycle_consistent)
        {
            std::cout
                << "Keyframe loop decision"
                << " | current_kf=" << edge_current_keyframe_id
                << " | historical_kf="
                << edge_historical_keyframe_id
                << " | decision=TRACK_ONLY"
                << " | reason=LOOP_CYCLE_INCONSISTENT"
                << " | cycle_translation_error="
                << cycle_translation_error << " m"
                << " | cycle_rotation_error="
                << cycle_rotation_error_deg << " deg"
                << std::endl;
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

    const bool dynamic_loop_information =
        BuildLoopShadowInformationFull6x6(
            current_keyframe.cloud,
            best_historical_submap->cloud_S,
            best_verification.T_target_source,
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
        << "ONLINE Keyframe PoseGraph loop edge staged"
        << " | from_kf=" << edge_historical_keyframe_id
        << " | to_kf=" << edge_current_keyframe_id
        << " | geometry_target_submap=" << edge_historical_submap_id
        << " | current_submap=" << edge_current_submap_id
        << " | support=" << online_loop_track_.support
        << " | edge_mode="
        << "SEQUENCE_CYCLE_OK"
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
        const bool loop_edge_rolled_back =
            pose_graph_.RemoveLoopEdge(
                edge_historical_keyframe_id,
                edge_current_keyframe_id);

        std::cerr
            << "G2O Keyframe PoseGraph optimization rejected"
            << " | nodes=" << pose_graph_.NodeCount()
            << " | edges_after_rollback=" << pose_graph_.EdgeCount()
            << " | loop_rollback="
            << (loop_edge_rolled_back ? "true" : "false")
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
            << std::endl;
        return;
    }

    // Optimization + gravity validation succeeded.  Only now is this loop
    // considered a real backend constraint for future sequence/cycle logic.
    loop_timing.optimization_accepted = true;
    loop_timing.loop_edge_accepted = true;

    has_last_online_loop_edge_ = true;
    last_online_loop_current_keyframe_id_ =
        edge_current_keyframe_id;
    last_online_loop_historical_keyframe_id_ =
        edge_historical_keyframe_id;
    last_online_loop_measurement_ =
        edge_measurement;

    std::cout
        << "ONLINE Keyframe PoseGraph loop edge accepted"
        << " | from_kf=" << edge_historical_keyframe_id
        << " | to_kf=" << edge_current_keyframe_id
        << " | edge_mode="
        << "SEQUENCE_CYCLE_OK"
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
bool RegistrationScan2LocalMap::UpdateIncrementalGlobalMaps(
    const char *reason,
    bool clear_refined_overrides)
{
    const std::chrono::steady_clock::time_point
        map_update_start =
            std::chrono::steady_clock::now();

    const std::vector<Keyframe> &keyframes =
        backend_keyframes_;

    if (keyframes.empty() ||
        pose_graph_.NodeCount() == 0)
    {
        return false;
    }

    std::vector<
        Eigen::Isometry3d,
        Eigen::aligned_allocator<Eigen::Isometry3d>>
        optimized_poses(
            keyframes.size(),
            Eigen::Isometry3d::Identity());

    std::vector<bool>
        optimized_pose_valid(
            keyframes.size(),
            false);

    std::size_t missing_graph_nodes = 0;

    const std::chrono::steady_clock::time_point
        pose_snapshot_start =
            std::chrono::steady_clock::now();

    for (std::size_t i = 0;
         i < keyframes.size();
         ++i)
    {
        const PoseGraphNode *node =
            pose_graph_.GetNode(
                keyframes[i].id);

        if (node == nullptr ||
            !node->T_WK.matrix().allFinite())
        {
            ++missing_graph_nodes;
            continue;
        }

        optimized_poses[i] =
            node->T_WK;

        optimized_pose_valid[i] =
            true;
    }

    const double pose_snapshot_ms =
        ElapsedMilliseconds(
            pose_snapshot_start,
            std::chrono::steady_clock::now());

    IncrementalGlobalMap::UpdateStats raw_stats;
    IncrementalGlobalMap::UpdateStats optimized_stats;

    const std::chrono::steady_clock::time_point
        raw_update_start =
            std::chrono::steady_clock::now();

    const bool raw_ok =
        incremental_global_map_.UpdateRaw(
            keyframes,
            raw_stats);

    const double raw_update_ms =
        ElapsedMilliseconds(
            raw_update_start,
            std::chrono::steady_clock::now());

    const std::chrono::steady_clock::time_point
        optimized_update_start =
            std::chrono::steady_clock::now();

    const bool optimized_ok =
        incremental_global_map_.UpdateOptimized(
            keyframes,
            optimized_poses,
            optimized_pose_valid,
            clear_refined_overrides,
            optimized_stats);

    const double optimized_update_ms =
        ElapsedMilliseconds(
            optimized_update_start,
            std::chrono::steady_clock::now());

    const pcl::PointCloud<LIDAR_POINT>::ConstPtr raw_map =
        incremental_global_map_.GetRawMap();

    const pcl::PointCloud<LIDAR_POINT>::ConstPtr optimized_map =
        incremental_global_map_.GetOptimizedMap();

    if (!raw_ok ||
        !optimized_ok ||
        !raw_map ||
        !optimized_map ||
        raw_map->empty() ||
        optimized_map->empty())
    {
        return false;
    }

    ++global_map_revision_;

    // During ordinary incremental growth an already-accepted local refinement
    // remains valid and IncrementalGlobalMap preserves its override blocks.
    // After a new main PoseGraph optimization, however, old overrides are
    // deliberately cleared and RebuildPostPgoRefinedMap() will create a new
    // refinement result for this graph revision.
    if (clear_refined_overrides)
    {
        refined_map_revision_ = 0;
    }
    else
    {
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr refined_map =
            incremental_global_map_.GetRefinedMap();

        if (refined_map &&
            !refined_map->empty())
        {
            refined_map_revision_ =
                global_map_revision_;
        }
    }

    std::cout
        << "Incremental global map update"
        << " | reason=" << (reason != nullptr ? reason : "UNKNOWN")
        << " | revision=" << global_map_revision_
        << " | keyframes=" << keyframes.size()
        << " | missing_graph_nodes=" << missing_graph_nodes
        << " | keyframes_per_block="
        << incremental_global_map_.KeyframesPerBlock()
        << " | total_blocks="
        << incremental_global_map_.BlockCount()
        << " | raw_dirty_keyframes=" << raw_stats.dirty_keyframes
        << " | raw_dirty_blocks=" << raw_stats.dirty_blocks
        << " | raw_rebuilt_blocks=" << raw_stats.rebuilt_blocks
        << " | raw_reused_blocks=" << raw_stats.reused_blocks
        << " | raw_points=" << raw_map->size()
        << " | opt_dirty_keyframes=" << optimized_stats.dirty_keyframes
        << " | opt_dirty_blocks=" << optimized_stats.dirty_blocks
        << " | opt_rebuilt_blocks=" << optimized_stats.rebuilt_blocks
        << " | opt_reused_blocks=" << optimized_stats.reused_blocks
        << " | optimized_points=" << optimized_map->size()
        << " | block_voxel_leaf="
        << incremental_global_map_.VoxelLeafSize() << " m"
        << std::endl;

    const double map_update_total_ms =
        ElapsedMilliseconds(
            map_update_start,
            std::chrono::steady_clock::now());

    RCLCPP_INFO(
        kTimingLogger,
        "FR_TIMING GLOBAL_MAP"
        " | reason=%s"
        " | total=%.3f ms"
        " | pose_snapshot=%.3f"
        " | raw_update=%.3f"
        " | optimized_update=%.3f"
        " | keyframes=%zu"
        " | raw_dirty_blocks=%zu"
        " | optimized_dirty_blocks=%zu"
        " | raw_points=%zu"
        " | optimized_points=%zu",
        reason != nullptr ? reason : "UNKNOWN",
        map_update_total_ms,
        pose_snapshot_ms,
        raw_update_ms,
        optimized_update_ms,
        keyframes.size(),
        raw_stats.dirty_blocks,
        optimized_stats.dirty_blocks,
        raw_map->size(),
        optimized_map->size());

    return true;
}

// ============================================================================
// RebuildGlobalMapSnapshots()
//
// Compatibility wrapper retained for the existing post-G2O call site.  The
// implementation is now incremental dirty-block maintenance, not a full map
// replay + global VoxelGrid.
// ============================================================================
bool RegistrationScan2LocalMap::RebuildGlobalMapSnapshots()
{
    return UpdateIncrementalGlobalMaps(
        "POSE_GRAPH",
        true);
}

// ============================================================================
// RebuildPostPgoRefinedMap()
//
// Post-PGO refinement V4: sparse-geometry local-window multi-pose optimization.
//
// Every current-window Keyframe still gets its own SE(3) variable and every
// consecutive odometry edge is retained.  The expensive point-to-plane
// registration is evaluated only on a uniformly distributed subset of
// Keyframes.  Those accepted geometry anchors constrain the whole local graph,
// and the odometry chain propagates the correction to unregistered Keyframes.
// The temporary graph solution is used only for
// /refined_map and never overwrites the main PoseGraph.
//
// Why V4 exists:
//     V1: each Keyframe independently registered to a LocalMap -> too free.
//     V2: one shared rigid correction for the whole window -> too rigid.
//     V3: local pose graph works well, but performs Full Align for every KF.
//     V4: keep the V3 graph, but sparsify only the expensive geometry anchors.
//
// V4 uses the same small temporary Keyframe PoseGraph:
//
//     fixed map anchor (Identity)
//          |       |       |       geometry edges from point-to-plane ICP
//          v       v       v
//        KF_i --- KF_i+1 --- KF_i+2 --- ...
//             odom       odom
//
// The geometry edges allow different Keyframes to receive different small
// corrections.  The stronger consecutive odometry edges prevent frame-to-frame
// jumps and preserve local trajectory continuity.
//
// Important: this is a practical Local-Pose-Graph refinement, not yet direct
// point-residual LiDAR bundle adjustment.  Point-to-plane ICP is first reduced
// to one soft SE(3) observation per accepted Keyframe.
//
// Safety rules:
//   1. Only accepted main-graph Loop edges define refinement regions.
//   2. Historical geometry is frozen from the G2O pose snapshot.
//   3. The main PoseGraph and frontend T_WL are never overwritten.
//   4. Each geometry observation must pass correspondence/RMSE/small-delta gates.
//   5. The whole local-window result must remain inside a final small-update gate.
//   6. Overlapping accepted refinement windows are skipped conservatively.
// ============================================================================

bool RegistrationScan2LocalMap::UpdateMapOdomCorrection(
    std::size_t anchor_keyframe_id)
{
    const Keyframe *anchor_keyframe =
        FindBackendKeyframeById(
            anchor_keyframe_id);

    const PoseGraphNode *anchor_node =
        pose_graph_.GetNode(
            anchor_keyframe_id);

    if (anchor_keyframe == nullptr ||
        anchor_node == nullptr ||
        !anchor_keyframe->T_WL.matrix().allFinite() ||
        !anchor_node->T_WK.matrix().allFinite())
    {
        return false;
    }

    const Eigen::Isometry3d T_map_odom_new =
        anchor_node->T_WK *
        anchor_keyframe->T_WL.inverse();

    if (!T_map_odom_new.matrix().allFinite())
    {
        return false;
    }

    const Eigen::AngleAxisd correction_rotation(
        T_map_odom_new.rotation());

    const double correction_rotation_deg =
        std::abs(
            correction_rotation.angle()) *
        180.0 /
        3.14159265358979323846;

    // Sanity check: applying the bridge to the raw anchor pose must recover the
    // optimized graph pose to numerical precision.
    const Eigen::Isometry3d T_map_K_check =
        T_map_odom_new *
        anchor_keyframe->T_WL;

    const Eigen::Isometry3d T_check_error =
        anchor_node->T_WK.inverse() *
        T_map_K_check;

    const double check_translation_error =
        T_check_error.translation().norm();

    const Eigen::AngleAxisd check_rotation(
        T_check_error.rotation());

    const double check_rotation_error_deg =
        std::abs(
            check_rotation.angle()) *
        180.0 /
        3.14159265358979323846;

    if (!std::isfinite(check_translation_error) ||
        !std::isfinite(check_rotation_error_deg) ||
        check_translation_error > 1.0e-6 ||
        check_rotation_error_deg > 1.0e-6)
    {
        std::cerr
            << "Map->odom correction anchor consistency check failed"
            << " | anchor_kf=" << anchor_keyframe_id
            << " | translation_error=" << check_translation_error << " m"
            << " | rotation_error=" << check_rotation_error_deg << " deg"
            << std::endl;
        return false;
    }

    T_map_odom_ =
        T_map_odom_new;

    has_map_odom_correction_ =
        true;

    map_odom_anchor_keyframe_id_ =
        anchor_keyframe_id;

    ++map_odom_revision_;

    std::cout
        << "Map->odom correction updated"
        << " | revision=" << map_odom_revision_
        << " | anchor_kf=" << map_odom_anchor_keyframe_id_
        << " | translation=["
        << T_map_odom_.translation().x() << " "
        << T_map_odom_.translation().y() << " "
        << T_map_odom_.translation().z() << "]"
        << " | translation_norm="
        << T_map_odom_.translation().norm() << " m"
        << " | rotation="
        << correction_rotation_deg << " deg"
        << " | anchor_check_translation="
        << check_translation_error << " m"
        << " | anchor_check_rotation="
        << check_rotation_error_deg << " deg"
        << std::endl;

    return true;
}

// Return the latest ACCEPTED LiDAR -> World pose.
//
// Rejected scans never modify this value.
bool RegistrationScan2LocalMap::RebuildPostPgoRefinedMap()
{
    RefinementTimingDiagnostics refinement_timing;
    RefinementTimingReporter refinement_timing_reporter(
        refinement_timing);

    const std::vector<Keyframe> &keyframes =
        backend_keyframes_;

    refinement_timing.keyframes =
        keyframes.size();

    if (keyframes.empty() ||
        pose_graph_.NodeCount() == 0 ||
        global_map_revision_ == 0)
    {
        return false;
    }

    refined_map_revision_ = 0;

    // Publish-side snapshots may still reference the previous debug clouds.
    // Replace the handles instead of clearing those shared buffers in place.
    refinement_historical_target_debug_ =
        pcl::make_shared<pcl::PointCloud<LIDAR_POINT>>();

    refinement_current_before_debug_ =
        pcl::make_shared<pcl::PointCloud<LIDAR_POINT>>();

    refinement_current_after_debug_ =
        pcl::make_shared<pcl::PointCloud<LIDAR_POINT>>();

    refinement_debug_revision_ = 0;

    refined_keyframe_poses_.clear();
    refined_keyframe_pose_was_adjusted_.clear();

    refined_keyframe_poses_.resize(
        keyframes.size(),
        Eigen::Isometry3d::Identity());

    refined_keyframe_pose_was_adjusted_.resize(
        keyframes.size(),
        false);

    // Immutable G2O snapshot.  V3 never feeds its own refined poses back into
    // neighbor selection, ICP targets, the live frontend, or the main graph.
    std::vector<
        Eigen::Isometry3d,
        Eigen::aligned_allocator<Eigen::Isometry3d>>
        frozen_graph_poses(
            keyframes.size(),
            Eigen::Isometry3d::Identity());

    std::vector<bool> graph_pose_valid(
        keyframes.size(),
        false);

    for (std::size_t i = 0;
         i < keyframes.size();
         ++i)
    {
        const PoseGraphNode *node =
            pose_graph_.GetNode(
                keyframes[i].id);

        if (node == nullptr ||
            !node->T_WK.matrix().allFinite())
        {
            continue;
        }

        frozen_graph_poses[i] =
            node->T_WK;

        refined_keyframe_poses_[i] =
            node->T_WK;

        graph_pose_valid[i] =
            true;
    }

    struct RefinementLoopAnchor
    {
        std::size_t historical_id = 0;
        std::size_t current_id = 0;
    };

    std::vector<RefinementLoopAnchor>
        loop_anchors;

    for (const PoseGraphEdge &edge :
         pose_graph_.GetEdges())
    {
        if (edge.type != PoseGraphEdgeType::Loop)
        {
            continue;
        }

        RefinementLoopAnchor anchor;

        anchor.historical_id =
            std::min(
                edge.from_id,
                edge.to_id);

        anchor.current_id =
            std::max(
                edge.from_id,
                edge.to_id);

        if (anchor.current_id <=
            anchor.historical_id)
        {
            continue;
        }

        if (anchor.current_id -
                anchor.historical_id <
            refinement_min_historical_keyframe_gap_)
        {
            continue;
        }

        loop_anchors.push_back(anchor);
    }

    if (loop_anchors.empty())
    {
        return false;
    }

    std::sort(
        loop_anchors.begin(),
        loop_anchors.end(),
        [](const RefinementLoopAnchor &lhs,
           const RefinementLoopAnchor &rhs)
        {
            return lhs.current_id <
                   rhs.current_id;
        });

    refinement_timing.loop_anchors =
        loop_anchors.size();

    std::size_t groups_considered = 0;
    std::size_t groups_prepared = 0;
    std::size_t groups_optimized = 0;
    std::size_t groups_accepted = 0;
    std::size_t groups_rejected_overlap = 0;
    std::size_t groups_rejected_quality = 0;
    std::size_t groups_rejected_large_update = 0;
    std::size_t geometry_candidates_total = 0;
    std::size_t geometry_primary_selected_total = 0;
    std::size_t geometry_attempts_total = 0;
    std::size_t geometry_fallback_calls_total = 0;
    std::size_t geometry_anchors_total = 0;
    std::size_t adjusted_keyframes = 0;

    // V4 keeps all local graph states and odometry edges, but limits the
    // expensive point-to-plane observations.  A 16-KF window therefore uses
    // at most 8 primary geometry registrations.  If too few primary anchors
    // pass the existing quality gates, up to 4 additional untested Keyframes
    // are tried before rejecting the refinement group.
    constexpr std::size_t kSparsePrimaryGeometryAnchors = 8;
    constexpr std::size_t kSparseMaxFallbackGeometryCalls = 4;

    std::cout
        << "Post-PGO local-window refinement V5 SPARSE+GATE_DIAG started"
        << " | global_revision=" << global_map_revision_
        << " | keyframes=" << keyframes.size()
        << " | loop_anchors=" << loop_anchors.size()
        << " | local_window=" << refinement_local_window_
        << " | historical_window=" << refinement_historical_keyframe_window_
        << " | historical_radius=" << refinement_historical_radius_ << " m"
        << " | min_geometry_anchors=" << refinement_min_geometry_anchors_
        << " | sparse_primary_anchors=" << kSparsePrimaryGeometryAnchors
        << " | sparse_fallback_max=" << kSparseMaxFallbackGeometryCalls
        << " | odom_info_scale=" << refinement_local_odom_information_scale_
        << " | geometry_info_scale=" << refinement_geometry_information_scale_
        << " | max_window_dt=" << refinement_window_max_translation_update_ << " m"
        << " | max_window_dR=" << refinement_window_max_rotation_update_deg_ << " deg"
        << std::endl;

    for (const RefinementLoopAnchor &anchor :
         loop_anchors)
    {
        std::size_t historical_anchor_index =
            std::numeric_limits<std::size_t>::max();

        std::size_t current_anchor_index =
            std::numeric_limits<std::size_t>::max();

        for (std::size_t i = 0;
             i < keyframes.size();
             ++i)
        {
            if (keyframes[i].id ==
                anchor.historical_id)
            {
                historical_anchor_index = i;
            }

            if (keyframes[i].id ==
                anchor.current_id)
            {
                current_anchor_index = i;
            }
        }

        if (historical_anchor_index ==
                std::numeric_limits<std::size_t>::max() ||
            current_anchor_index ==
                std::numeric_limits<std::size_t>::max() ||
            !graph_pose_valid[historical_anchor_index] ||
            !graph_pose_valid[current_anchor_index])
        {
            continue;
        }

        ++groups_considered;
        refinement_timing.groups_considered =
            groups_considered;

        // --------------------------------------------------------------------
        // Current revisit window: one independent SE(3) state per Keyframe.
        // The window ends at the current loop endpoint because those poses are
        // already available when the online loop is accepted.
        // --------------------------------------------------------------------
        const std::chrono::steady_clock::time_point
            current_select_start =
                std::chrono::steady_clock::now();

        std::vector<std::size_t>
            current_indices;

        current_indices.reserve(
            refinement_local_window_ + 1);

        for (std::size_t i = 0;
             i < keyframes.size();
             ++i)
        {
            if (!graph_pose_valid[i] ||
                !keyframes[i].cloud ||
                keyframes[i].cloud->empty())
            {
                continue;
            }

            if (keyframes[i].id >
                anchor.current_id)
            {
                continue;
            }

            const std::size_t current_gap =
                anchor.current_id -
                keyframes[i].id;

            if (current_gap >
                refinement_local_window_)
            {
                continue;
            }

            current_indices.push_back(i);
        }

        refinement_timing.current_select_ms +=
            ElapsedMilliseconds(
                current_select_start,
                std::chrono::steady_clock::now());

        if (current_indices.size() <
            refinement_min_current_keyframes_)
        {
            ++groups_rejected_quality;
            continue;
        }

        // Keep V3 conservative when multiple accepted loop windows overlap.
        // A future global local-BA pass can merge those windows explicitly.
        bool overlaps_previous_refinement = false;

        for (const std::size_t source_index :
             current_indices)
        {
            if (refined_keyframe_pose_was_adjusted_[source_index])
            {
                overlaps_previous_refinement = true;
                break;
            }
        }

        if (overlaps_previous_refinement)
        {
            ++groups_rejected_overlap;

            std::cout
                << "Post-PGO local-window refinement skipped"
                << " | historical_kf=" << anchor.historical_id
                << " | current_kf=" << anchor.current_id
                << " | reason=OVERLAPPING_REFINEMENT_WINDOW"
                << std::endl;

            continue;
        }

        // --------------------------------------------------------------------
        // Build one frozen HISTORICAL LocalMap tied to the historical endpoint
        // of the accepted loop edge.  Every current Keyframe uses exactly this
        // same target, so the geometry constraints are mutually comparable.
        // --------------------------------------------------------------------
        const std::chrono::steady_clock::time_point
            historical_select_start =
                std::chrono::steady_clock::now();

        std::vector<std::pair<double, std::size_t>>
            historical_candidates;

        historical_candidates.reserve(
            refinement_max_historical_keyframes_ * 2);

        const Eigen::Vector3d historical_anchor_position =
            frozen_graph_poses[historical_anchor_index]
                .translation();

        for (std::size_t i = 0;
             i < keyframes.size();
             ++i)
        {
            if (!graph_pose_valid[i] ||
                !keyframes[i].cloud ||
                keyframes[i].cloud->empty())
            {
                continue;
            }

            if (keyframes[i].id >=
                anchor.current_id)
            {
                continue;
            }

            const std::size_t current_to_historical_gap =
                anchor.current_id -
                keyframes[i].id;

            if (current_to_historical_gap <
                refinement_min_historical_keyframe_gap_)
            {
                continue;
            }

            const std::size_t historical_id_gap =
                keyframes[i].id >= anchor.historical_id
                    ? keyframes[i].id - anchor.historical_id
                    : anchor.historical_id - keyframes[i].id;

            if (historical_id_gap >
                refinement_historical_keyframe_window_)
            {
                continue;
            }

            const double distance =
                (frozen_graph_poses[i].translation() -
                 historical_anchor_position)
                    .norm();

            if (!std::isfinite(distance) ||
                distance >
                    refinement_historical_radius_)
            {
                continue;
            }

            historical_candidates.emplace_back(
                distance,
                i);
        }

        if (historical_candidates.size() <
            refinement_min_historical_keyframes_)
        {
            ++groups_rejected_quality;
            continue;
        }

        std::sort(
            historical_candidates.begin(),
            historical_candidates.end(),
            [](const auto &lhs,
               const auto &rhs)
            {
                return lhs.first < rhs.first;
            });

        if (historical_candidates.size() >
            refinement_max_historical_keyframes_)
        {
            historical_candidates.resize(
                refinement_max_historical_keyframes_);
        }

        refinement_timing.historical_select_ms +=
            ElapsedMilliseconds(
                historical_select_start,
                std::chrono::steady_clock::now());

        const std::chrono::steady_clock::time_point
            historical_build_start =
                std::chrono::steady_clock::now();

        pcl::PointCloud<LIDAR_POINT>::Ptr
            historical_target_accumulated =
                pcl::make_shared<
                    pcl::PointCloud<LIDAR_POINT>>();

        std::size_t estimated_historical_points = 0;

        for (const auto &candidate :
             historical_candidates)
        {
            estimated_historical_points +=
                keyframes[candidate.second].cloud->size();
        }

        historical_target_accumulated->reserve(
            estimated_historical_points);

        for (const auto &candidate :
             historical_candidates)
        {
            const std::size_t historical_index =
                candidate.second;

            pcl::PointCloud<LIDAR_POINT>
                historical_world_cloud;

            const Eigen::Matrix4f T_WK_float =
                frozen_graph_poses[historical_index]
                    .matrix()
                    .cast<float>();

            pcl::transformPointCloud(
                *keyframes[historical_index].cloud,
                historical_world_cloud,
                T_WK_float);

            *historical_target_accumulated +=
                historical_world_cloud;
        }

        refinement_timing.historical_build_ms +=
            ElapsedMilliseconds(
                historical_build_start,
                std::chrono::steady_clock::now());

        if (historical_target_accumulated->empty())
        {
            ++groups_rejected_quality;
            continue;
        }

        pcl::PointCloud<LIDAR_POINT>::Ptr
            historical_target_filtered =
                pcl::make_shared<
                    pcl::PointCloud<LIDAR_POINT>>();

        pcl::VoxelGrid<LIDAR_POINT>
            target_voxel;

        target_voxel.setLeafSize(
            refinement_target_voxel_leaf_size_,
            refinement_target_voxel_leaf_size_,
            refinement_target_voxel_leaf_size_);

        target_voxel.setInputCloud(
            historical_target_accumulated);

        const std::chrono::steady_clock::time_point
            historical_voxel_start =
                std::chrono::steady_clock::now();

        target_voxel.filter(
            *historical_target_filtered);

        refinement_timing.historical_voxel_ms +=
            ElapsedMilliseconds(
                historical_voxel_start,
                std::chrono::steady_clock::now());

        if (historical_target_filtered->size() <
            refinement_min_target_points_)
        {
            ++groups_rejected_quality;
            continue;
        }

        PreparedLidarTarget
            prepared_refinement_target;

        const std::chrono::steady_clock::time_point
            prepare_target_start =
                std::chrono::steady_clock::now();

        const bool prepare_target_ok =
            backend_refinement_registration_.PrepareTarget(
                historical_target_filtered,
                prepared_refinement_target);

        refinement_timing.prepare_target_ms +=
            ElapsedMilliseconds(
                prepare_target_start,
                std::chrono::steady_clock::now());

        if (!prepare_target_ok)
        {
            ++groups_rejected_quality;
            continue;
        }

        ++groups_prepared;
        refinement_timing.groups_prepared =
            groups_prepared;

        const std::chrono::steady_clock::time_point
            local_graph_start =
                std::chrono::steady_clock::now();

        // --------------------------------------------------------------------
        // Temporary LOCAL PoseGraph.
        //
        // local node 0:
        //     fixed map-frame anchor at Identity.
        //
        // local nodes 1..N:
        //     current-window Keyframe poses initialized from frozen G2O.
        //
        // Odometry edges preserve the current trajectory shape.
        // Geometry edges are soft absolute-pose observations produced by
        // Keyframe-to-HistoricalLocalMap point-to-plane registration.
        // --------------------------------------------------------------------
        PoseGraph local_pose_graph;

        if (!local_pose_graph.AddNode(
                0,
                Eigen::Isometry3d::Identity(),
                true))
        {
            ++groups_rejected_quality;
            continue;
        }

        std::vector<std::size_t>
            local_node_ids(
                current_indices.size(),
                0);

        bool local_graph_ok = true;

        for (std::size_t k = 0;
             k < current_indices.size();
             ++k)
        {
            const std::size_t local_node_id =
                k + 1;

            local_node_ids[k] =
                local_node_id;

            const std::size_t source_index =
                current_indices[k];

            if (!local_pose_graph.AddNode(
                    local_node_id,
                    frozen_graph_poses[source_index],
                    false))
            {
                local_graph_ok = false;
                break;
            }
        }

        if (!local_graph_ok)
        {
            ++groups_rejected_quality;
            continue;
        }

        Eigen::Matrix<double, 6, 6>
            odom_information =
                Eigen::Matrix<double, 6, 6>::Identity() *
                refinement_local_odom_information_scale_;

        for (std::size_t k = 1;
             k < current_indices.size();
             ++k)
        {
            const std::size_t previous_index =
                current_indices[k - 1];

            const std::size_t current_index =
                current_indices[k];

            const Eigen::Isometry3d Z_previous_current =
                frozen_graph_poses[previous_index].inverse() *
                frozen_graph_poses[current_index];

            if (!local_pose_graph.AddOdometryEdge(
                    local_node_ids[k - 1],
                    local_node_ids[k],
                    Z_previous_current,
                    odom_information))
            {
                local_graph_ok = false;
                break;
            }
        }

        if (!local_graph_ok)
        {
            ++groups_rejected_quality;
            continue;
        }

        refinement_timing.local_graph_build_ms +=
            ElapsedMilliseconds(
                local_graph_start,
                std::chrono::steady_clock::now());

        std::vector<bool>
            geometry_anchor_valid(
                current_indices.size(),
                false);

        std::vector<bool>
            geometry_was_tested(
                current_indices.size(),
                false);

        std::vector<std::size_t>
            geometry_correspondences(
                current_indices.size(),
                0);

        std::vector<double>
            geometry_rmse(
                current_indices.size(),
                std::numeric_limits<double>::infinity());

        std::vector<double>
            geometry_delta_t(
                current_indices.size(),
                std::numeric_limits<double>::infinity());

        std::vector<double>
            geometry_delta_R_deg(
                current_indices.size(),
                std::numeric_limits<double>::infinity());

        std::size_t geometry_anchor_count = 0;
        std::size_t geometry_primary_selected = 0;
        std::size_t geometry_fallback_calls = 0;

        // V5 diagnostic counters. These counters are observational only;
        // they do not change any geometry acceptance threshold or graph math.
        std::size_t geometry_attempts_group = 0;
        std::size_t geometry_reject_align_failed = 0;
        std::size_t geometry_reject_result_failed = 0;
        std::size_t geometry_reject_not_converged = 0;
        std::size_t geometry_reject_transform_nonfinite = 0;
        std::size_t geometry_reject_rmse_nonfinite = 0;
        std::size_t geometry_reject_corr_low = 0;
        std::size_t geometry_reject_rmse_high = 0;
        std::size_t geometry_reject_correction_nonfinite = 0;
        std::size_t geometry_reject_delta_t_high = 0;
        std::size_t geometry_reject_delta_R_high = 0;
        std::size_t geometry_reject_add_edge_failed = 0;

        geometry_candidates_total +=
            current_indices.size();

        refinement_timing.geometry_candidates +=
            current_indices.size();

        const Eigen::Matrix<double, 6, 6>
            geometry_information =
                Eigen::Matrix<double, 6, 6>::Identity() *
                refinement_geometry_information_scale_;

        // --------------------------------------------------------------------
        // V4 sparse geometry selection.
        //
        // Keep the first and last Keyframes and distribute the remaining
        // anchors approximately uniformly over the local window.  For the
        // current default 16-KF window this selects:
        //
        //     0, 2, 4, 6, 9, 11, 13, 15
        //
        // (exact integer rounding is intentional).  All 16 graph nodes and
        // all 15 odometry edges remain in the optimization.
        // --------------------------------------------------------------------
        const std::size_t primary_target_count =
            std::min(
                kSparsePrimaryGeometryAnchors,
                current_indices.size());

        std::vector<std::size_t>
            primary_geometry_indices;

        primary_geometry_indices.reserve(
            primary_target_count);

        if (primary_target_count == 1)
        {
            primary_geometry_indices.push_back(0);
        }
        else if (primary_target_count > 1)
        {
            const std::size_t last_index =
                current_indices.size() - 1;

            const std::size_t denominator =
                primary_target_count - 1;

            for (std::size_t slot = 0;
                 slot < primary_target_count;
                 ++slot)
            {
                const std::size_t numerator =
                    slot * last_index;

                // Rounded integer interpolation in [0, last_index].
                const std::size_t k =
                    (numerator + denominator / 2) /
                    denominator;

                if (primary_geometry_indices.empty() ||
                    primary_geometry_indices.back() != k)
                {
                    primary_geometry_indices.push_back(k);
                }
            }

            // Defensive guarantee: the current loop endpoint is always tested.
            if (primary_geometry_indices.empty() ||
                primary_geometry_indices.back() != last_index)
            {
                primary_geometry_indices.push_back(
                    last_index);
            }
        }

        geometry_primary_selected =
            primary_geometry_indices.size();

        geometry_primary_selected_total +=
            geometry_primary_selected;

        refinement_timing.geometry_primary_selected +=
            geometry_primary_selected;

        auto log_geometry_attempt =
            [&](const char *outcome,
                const char *reason,
                const std::size_t k,
                const std::size_t source_keyframe_id,
                const bool is_fallback,
                const std::size_t correspondences,
                const double rmse,
                const double correction_translation,
                const double correction_rotation_deg)
        {
            RCLCPP_INFO(
                kTimingLogger,
                "FR_REFINEMENT_GEOMETRY_ATTEMPT"
                " | historical_kf=%zu"
                " | current_kf=%zu"
                " | source_kf=%zu"
                " | local_index=%zu"
                " | sample=%s"
                " | outcome=%s"
                " | reason=%s"
                " | corr=%zu"
                " | rmse=%.6f"
                " | delta_t=%.6f"
                " | delta_R_deg=%.6f",
                anchor.historical_id,
                anchor.current_id,
                source_keyframe_id,
                k,
                is_fallback ? "fallback" : "primary",
                outcome,
                reason,
                correspondences,
                rmse,
                correction_translation,
                correction_rotation_deg);
        };

        auto try_geometry_anchor =
            [&](const std::size_t k,
                const bool is_fallback) -> bool
        {
            if (k >= current_indices.size() ||
                geometry_was_tested[k])
            {
                return true;
            }

            geometry_was_tested[k] = true;

            const std::size_t source_index =
                current_indices[k];

            const Keyframe &source_keyframe =
                keyframes[source_index];

            ++geometry_attempts_group;
            ++geometry_attempts_total;
            ++refinement_timing.geometry_calls;

            if (is_fallback)
            {
                ++geometry_fallback_calls;
                ++geometry_fallback_calls_total;
                ++refinement_timing.geometry_fallback_calls;
            }

            LidarRegistrationResult geometry_result;

            const std::chrono::steady_clock::time_point
                geometry_align_start =
                    std::chrono::steady_clock::now();

            const bool align_success =
                backend_refinement_registration_.Align(
                    source_keyframe.cloud,
                    prepared_refinement_target,
                    frozen_graph_poses[source_index],
                    geometry_result);

            refinement_timing.geometry_align_ms +=
                ElapsedMilliseconds(
                    geometry_align_start,
                    std::chrono::steady_clock::now());

            const double nan_value =
                std::numeric_limits<double>::quiet_NaN();

            if (!align_success)
            {
                ++geometry_reject_align_failed;
                log_geometry_attempt(
                    "REJECT",
                    "ALIGN_RETURN_FALSE",
                    k,
                    source_keyframe.id,
                    is_fallback,
                    0,
                    nan_value,
                    nan_value,
                    nan_value);
                return true;
            }

            if (!geometry_result.success)
            {
                ++geometry_reject_result_failed;
                log_geometry_attempt(
                    "REJECT",
                    "RESULT_SUCCESS_FALSE",
                    k,
                    source_keyframe.id,
                    is_fallback,
                    geometry_result.correspondences,
                    geometry_result.rmse,
                    nan_value,
                    nan_value);
                return true;
            }

            if (!geometry_result.converged)
            {
                ++geometry_reject_not_converged;
                log_geometry_attempt(
                    "REJECT",
                    "NOT_CONVERGED",
                    k,
                    source_keyframe.id,
                    is_fallback,
                    geometry_result.correspondences,
                    geometry_result.rmse,
                    nan_value,
                    nan_value);
                return true;
            }

            if (!geometry_result.T_target_source.matrix().allFinite())
            {
                ++geometry_reject_transform_nonfinite;
                log_geometry_attempt(
                    "REJECT",
                    "TRANSFORM_NONFINITE",
                    k,
                    source_keyframe.id,
                    is_fallback,
                    geometry_result.correspondences,
                    geometry_result.rmse,
                    nan_value,
                    nan_value);
                return true;
            }

            if (!std::isfinite(geometry_result.rmse))
            {
                ++geometry_reject_rmse_nonfinite;
                log_geometry_attempt(
                    "REJECT",
                    "RMSE_NONFINITE",
                    k,
                    source_keyframe.id,
                    is_fallback,
                    geometry_result.correspondences,
                    geometry_result.rmse,
                    nan_value,
                    nan_value);
                return true;
            }

            geometry_correspondences[k] =
                geometry_result.correspondences;

            geometry_rmse[k] =
                geometry_result.rmse;

            const Eigen::Isometry3d T_initial_geometry =
                frozen_graph_poses[source_index].inverse() *
                geometry_result.T_target_source;

            const double correction_translation =
                T_initial_geometry.translation().norm();

            const double correction_rotation_deg =
                RelativeRotationDeg(
                    frozen_graph_poses[source_index],
                    geometry_result.T_target_source);

            geometry_delta_t[k] =
                correction_translation;

            geometry_delta_R_deg[k] =
                correction_rotation_deg;

            // ------------------------------------------------------------
            // Refinement pose-delta diagnostics.
            //
            // T_local_delta answers:
            //     how much does this Keyframe need to move relative to its
            //     current post-PGO pose, expressed in the Keyframe-local
            //     perturbation convention used by the existing gate?
            //
            // T_world_delta answers:
            //     is there a coherent rigid correction in the map/world
            //     frame shared by several current-window Keyframes?
            //
            // If many good ICP results show nearly the same world-frame
            // yaw / translation, the current absolute 0.40 m / 3 deg gate is
            // rejecting a coherent correction rather than random outliers.
            // ------------------------------------------------------------
            const Eigen::Isometry3d T_local_delta =
                T_initial_geometry;

            const Eigen::Isometry3d T_world_delta =
                geometry_result.T_target_source *
                frozen_graph_poses[source_index].inverse();

            const Eigen::Vector3d local_rpy_zyx =
                T_local_delta.rotation().eulerAngles(2, 1, 0);

            const Eigen::Vector3d world_rpy_zyx =
                T_world_delta.rotation().eulerAngles(2, 1, 0);

            constexpr double kRadToDeg =
                180.0 / M_PI;

            const double local_yaw_deg =
                local_rpy_zyx[0] * kRadToDeg;
            const double local_pitch_deg =
                local_rpy_zyx[1] * kRadToDeg;
            const double local_roll_deg =
                local_rpy_zyx[2] * kRadToDeg;

            const double world_yaw_deg =
                world_rpy_zyx[0] * kRadToDeg;
            const double world_pitch_deg =
                world_rpy_zyx[1] * kRadToDeg;
            const double world_roll_deg =
                world_rpy_zyx[2] * kRadToDeg;

            RCLCPP_INFO(
                kTimingLogger,
                "FR_REFINEMENT_DELTA_COMPONENTS"
                " | historical_kf=%zu"
                " | current_kf=%zu"
                " | source_kf=%zu"
                " | local_index=%zu"
                " | sample=%s"
                " | corr=%zu"
                " | rmse=%.6f"
                " | local_dt=[%.6f %.6f %.6f]"
                " | local_rpy_deg=[%.6f %.6f %.6f]"
                " | world_dt=[%.6f %.6f %.6f]"
                " | world_rpy_deg=[%.6f %.6f %.6f]"
                " | delta_t_norm=%.6f"
                " | delta_R_deg=%.6f",
                anchor.historical_id,
                anchor.current_id,
                source_keyframe.id,
                k,
                is_fallback ? "fallback" : "primary",
                geometry_result.correspondences,
                geometry_result.rmse,
                T_local_delta.translation().x(),
                T_local_delta.translation().y(),
                T_local_delta.translation().z(),
                local_roll_deg,
                local_pitch_deg,
                local_yaw_deg,
                T_world_delta.translation().x(),
                T_world_delta.translation().y(),
                T_world_delta.translation().z(),
                world_roll_deg,
                world_pitch_deg,
                world_yaw_deg,
                correction_translation,
                correction_rotation_deg);

            if (geometry_result.correspondences <
                refinement_geometry_min_correspondences_)
            {
                ++geometry_reject_corr_low;
                log_geometry_attempt(
                    "REJECT",
                    "CORRESPONDENCES_LOW",
                    k,
                    source_keyframe.id,
                    is_fallback,
                    geometry_result.correspondences,
                    geometry_result.rmse,
                    correction_translation,
                    correction_rotation_deg);
                return true;
            }

            if (geometry_result.rmse >
                refinement_geometry_max_rmse_)
            {
                ++geometry_reject_rmse_high;
                log_geometry_attempt(
                    "REJECT",
                    "RMSE_HIGH",
                    k,
                    source_keyframe.id,
                    is_fallback,
                    geometry_result.correspondences,
                    geometry_result.rmse,
                    correction_translation,
                    correction_rotation_deg);
                return true;
            }

            if (!std::isfinite(correction_translation) ||
                !std::isfinite(correction_rotation_deg))
            {
                ++geometry_reject_correction_nonfinite;
                log_geometry_attempt(
                    "REJECT",
                    "CORRECTION_NONFINITE",
                    k,
                    source_keyframe.id,
                    is_fallback,
                    geometry_result.correspondences,
                    geometry_result.rmse,
                    correction_translation,
                    correction_rotation_deg);
                return true;
            }

            if (correction_translation >
                refinement_geometry_max_translation_correction_)
            {
                ++geometry_reject_delta_t_high;
                log_geometry_attempt(
                    "REJECT",
                    "DELTA_T_HIGH",
                    k,
                    source_keyframe.id,
                    is_fallback,
                    geometry_result.correspondences,
                    geometry_result.rmse,
                    correction_translation,
                    correction_rotation_deg);
                return true;
            }

            if (correction_rotation_deg >
                refinement_geometry_max_rotation_correction_deg_)
            {
                ++geometry_reject_delta_R_high;
                log_geometry_attempt(
                    "REJECT",
                    "DELTA_R_HIGH",
                    k,
                    source_keyframe.id,
                    is_fallback,
                    geometry_result.correspondences,
                    geometry_result.rmse,
                    correction_translation,
                    correction_rotation_deg);
                return true;
            }

            // Fixed node 0 is Identity in map coordinates. Therefore the
            // measurement 0 -> k is simply the absolute map pose returned by
            // the point-to-plane registration.
            if (!local_pose_graph.AddLoopEdge(
                    0,
                    local_node_ids[k],
                    geometry_result.T_target_source,
                    geometry_information))
            {
                ++geometry_reject_add_edge_failed;
                log_geometry_attempt(
                    "REJECT",
                    "ADD_LOOP_EDGE_FAILED",
                    k,
                    source_keyframe.id,
                    is_fallback,
                    geometry_result.correspondences,
                    geometry_result.rmse,
                    correction_translation,
                    correction_rotation_deg);
                return false;
            }

            geometry_anchor_valid[k] = true;

            ++geometry_anchor_count;
            ++geometry_anchors_total;

            refinement_timing.geometry_anchors =
                geometry_anchors_total;

            log_geometry_attempt(
                "ACCEPT",
                "OK",
                k,
                source_keyframe.id,
                is_fallback,
                geometry_result.correspondences,
                geometry_result.rmse,
                correction_translation,
                correction_rotation_deg);

            return true;
        };

        for (const std::size_t k :
             primary_geometry_indices)
        {
            if (!try_geometry_anchor(
                    k,
                    false))
            {
                local_graph_ok = false;
                break;
            }
        }

        // If the sparse primary set happens to fall on weak geometry, test a
        // few additional Keyframes.  The fallback is intentionally bounded so
        // V4 cannot silently return to V3's all-frame Full Align cost.
        if (local_graph_ok &&
            geometry_anchor_count <
                refinement_min_geometry_anchors_)
        {
            for (std::size_t k = 0;
                 k < current_indices.size() &&
                 geometry_anchor_count <
                     refinement_min_geometry_anchors_ &&
                 geometry_fallback_calls <
                     kSparseMaxFallbackGeometryCalls;
                 ++k)
            {
                if (geometry_was_tested[k])
                {
                    continue;
                }

                if (!try_geometry_anchor(
                        k,
                        true))
                {
                    local_graph_ok = false;
                    break;
                }
            }
        }

        RCLCPP_INFO(
            kTimingLogger,
            "FR_REFINEMENT_GEOMETRY_SUMMARY"
            " | historical_kf=%zu"
            " | current_kf=%zu"
            " | candidates=%zu"
            " | primary_selected=%zu"
            " | fallback_calls=%zu"
            " | attempts=%zu"
            " | accepted=%zu"
            " | reject_align_failed=%zu"
            " | reject_result_failed=%zu"
            " | reject_not_converged=%zu"
            " | reject_transform_nonfinite=%zu"
            " | reject_rmse_nonfinite=%zu"
            " | reject_corr_low=%zu"
            " | reject_rmse_high=%zu"
            " | reject_correction_nonfinite=%zu"
            " | reject_delta_t_high=%zu"
            " | reject_delta_R_high=%zu"
            " | reject_add_edge_failed=%zu"
            " | min_corr=%zu"
            " | max_rmse=%.6f"
            " | max_delta_t=%.6f"
            " | max_delta_R_deg=%.6f",
            anchor.historical_id,
            anchor.current_id,
            current_indices.size(),
            geometry_primary_selected,
            geometry_fallback_calls,
            geometry_attempts_group,
            geometry_anchor_count,
            geometry_reject_align_failed,
            geometry_reject_result_failed,
            geometry_reject_not_converged,
            geometry_reject_transform_nonfinite,
            geometry_reject_rmse_nonfinite,
            geometry_reject_corr_low,
            geometry_reject_rmse_high,
            geometry_reject_correction_nonfinite,
            geometry_reject_delta_t_high,
            geometry_reject_delta_R_high,
            geometry_reject_add_edge_failed,
            refinement_geometry_min_correspondences_,
            refinement_geometry_max_rmse_,
            refinement_geometry_max_translation_correction_,
            refinement_geometry_max_rotation_correction_deg_);

        std::cout
            << "Post-PGO sparse geometry sampling"
            << " | historical_kf=" << anchor.historical_id
            << " | current_kf=" << anchor.current_id
            << " | candidates=" << current_indices.size()
            << " | primary_selected=" << geometry_primary_selected
            << " | fallback_calls=" << geometry_fallback_calls
            << " | total_calls="
            << std::count(
                   geometry_was_tested.begin(),
                   geometry_was_tested.end(),
                   true)
            << " | accepted_anchors=" << geometry_anchor_count
            << std::endl;

        if (!local_graph_ok ||
            geometry_anchor_count <
                refinement_min_geometry_anchors_)
        {
            ++groups_rejected_quality;

            std::cout
                << "Post-PGO local-window refinement rejected"
                << " | historical_kf=" << anchor.historical_id
                << " | current_kf=" << anchor.current_id
                << " | current_kfs=" << current_indices.size()
                << " | geometry_anchors=" << geometry_anchor_count
                << " | required=" << refinement_min_geometry_anchors_
                << " | reason=INSUFFICIENT_GEOMETRY_ANCHORS"
                << std::endl;

            continue;
        }

        PoseGraphOptimizationResult
            local_optimization_result;

        const std::chrono::steady_clock::time_point
            local_pgo_start =
                std::chrono::steady_clock::now();

        const bool local_pgo_ok =
            pose_graph_optimizer_.Optimize(
                local_pose_graph,
                local_optimization_result);

        refinement_timing.local_pgo_ms +=
            ElapsedMilliseconds(
                local_pgo_start,
                std::chrono::steady_clock::now());

        if (!local_pgo_ok ||
            !local_optimization_result.success)
        {
            ++groups_rejected_quality;
            continue;
        }

        ++groups_optimized;
        refinement_timing.groups_optimized =
            groups_optimized;

        std::vector<
            Eigen::Isometry3d,
            Eigen::aligned_allocator<Eigen::Isometry3d>>
            optimized_local_poses(
                current_indices.size(),
                Eigen::Isometry3d::Identity());

        double max_pose_delta_t = 0.0;
        double max_pose_delta_R_deg = 0.0;
        double sum_pose_delta_t = 0.0;
        double sum_pose_delta_R_deg = 0.0;

        bool optimized_window_valid = true;

        for (std::size_t k = 0;
             k < current_indices.size();
             ++k)
        {
            const PoseGraphNode *local_node =
                local_pose_graph.GetNode(
                    local_node_ids[k]);

            if (local_node == nullptr ||
                !local_node->T_WK.matrix().allFinite())
            {
                optimized_window_valid = false;
                break;
            }

            const std::size_t source_index =
                current_indices[k];

            optimized_local_poses[k] =
                local_node->T_WK;

            const Eigen::Isometry3d T_initial_optimized =
                frozen_graph_poses[source_index].inverse() *
                optimized_local_poses[k];

            const double delta_t =
                T_initial_optimized.translation().norm();

            const double delta_R_deg =
                RelativeRotationDeg(
                    frozen_graph_poses[source_index],
                    optimized_local_poses[k]);

            if (!std::isfinite(delta_t) ||
                !std::isfinite(delta_R_deg))
            {
                optimized_window_valid = false;
                break;
            }

            max_pose_delta_t =
                std::max(
                    max_pose_delta_t,
                    delta_t);

            max_pose_delta_R_deg =
                std::max(
                    max_pose_delta_R_deg,
                    delta_R_deg);

            sum_pose_delta_t +=
                delta_t;

            sum_pose_delta_R_deg +=
                delta_R_deg;
        }

        double max_relative_odom_translation_change = 0.0;
        double max_relative_odom_rotation_change_deg = 0.0;

        if (optimized_window_valid)
        {
            for (std::size_t k = 1;
                 k < current_indices.size();
                 ++k)
            {
                const std::size_t previous_index =
                    current_indices[k - 1];

                const std::size_t current_index =
                    current_indices[k];

                const Eigen::Isometry3d Z_before =
                    frozen_graph_poses[previous_index].inverse() *
                    frozen_graph_poses[current_index];

                const Eigen::Isometry3d Z_after =
                    optimized_local_poses[k - 1].inverse() *
                    optimized_local_poses[k];

                const Eigen::Isometry3d Z_error =
                    Z_before.inverse() *
                    Z_after;

                max_relative_odom_translation_change =
                    std::max(
                        max_relative_odom_translation_change,
                        Z_error.translation().norm());

                max_relative_odom_rotation_change_deg =
                    std::max(
                        max_relative_odom_rotation_change_deg,
                        RelativeRotationDeg(
                            Z_before,
                            Z_after));
            }
        }

        const bool refinement_window_translation_ok =
            optimized_window_valid &&
            max_pose_delta_t <=
                refinement_window_max_translation_update_;

        const bool refinement_window_rotation_ok =
            optimized_window_valid &&
            max_pose_delta_R_deg <=
                refinement_window_max_rotation_update_deg_;

        const bool refinement_window_relative_translation_ok =
            optimized_window_valid &&
            max_relative_odom_translation_change <=
                refinement_window_max_relative_odom_translation_change_;

        const bool refinement_window_relative_rotation_ok =
            optimized_window_valid &&
            max_relative_odom_rotation_change_deg <=
                refinement_window_max_relative_odom_rotation_change_deg_;

        // Coherent refinement gate:
        //   1) allow a reasonably large absolute rigid correction after the
        //      global loop optimization;
        //   2) reject solutions that distort the local odometry shape.
        const bool refinement_window_gate_accepted =
            optimized_window_valid &&
            refinement_window_translation_ok &&
            refinement_window_rotation_ok &&
            refinement_window_relative_translation_ok &&
            refinement_window_relative_rotation_ok;

        RCLCPP_INFO(
            kTimingLogger,
            "FR_REFINEMENT_WINDOW_GATE | historical_kf=%zu | current_kf=%zu "
            "| geometry_anchors=%zu | chi2_before=%.9f | chi2_after=%.9f "
            "| max_delta_t=%.6f | max_delta_t_limit=%.6f "
            "| max_delta_R_deg=%.6f | max_delta_R_limit_deg=%.6f "
            "| max_rel_odom_dt=%.6f | max_rel_odom_dt_limit=%.6f "
            "| max_rel_odom_dR_deg=%.6f | max_rel_odom_dR_limit_deg=%.6f "
            "| optimized_window_valid=%s | translation_ok=%s | rotation_ok=%s "
            "| relative_translation_ok=%s | relative_rotation_ok=%s "
            "| decision=%s",
            anchor.historical_id,
            anchor.current_id,
            geometry_anchor_count,
            local_optimization_result.chi2_before,
            local_optimization_result.chi2_after,
            max_pose_delta_t,
            refinement_window_max_translation_update_,
            max_pose_delta_R_deg,
            refinement_window_max_rotation_update_deg_,
            max_relative_odom_translation_change,
            refinement_window_max_relative_odom_translation_change_,
            max_relative_odom_rotation_change_deg,
            refinement_window_max_relative_odom_rotation_change_deg_,
            optimized_window_valid ? "true" : "false",
            refinement_window_translation_ok ? "true" : "false",
            refinement_window_rotation_ok ? "true" : "false",
            refinement_window_relative_translation_ok ? "true" : "false",
            refinement_window_relative_rotation_ok ? "true" : "false",
            refinement_window_gate_accepted ? "ACCEPT" : "REJECT");

        if (!refinement_window_gate_accepted)
        {
            ++groups_rejected_large_update;

            const char *rejection_reason =
                (!refinement_window_relative_translation_ok ||
                 !refinement_window_relative_rotation_ok)
                    ? "RELATIVE_ODOM_SHAPE_CHANGE"
                    : "LARGE_LOCAL_WINDOW_UPDATE";

            std::cout
                << "Post-PGO local-window refinement rejected"
                << " | historical_kf=" << anchor.historical_id
                << " | current_kf=" << anchor.current_id
                << " | geometry_anchors=" << geometry_anchor_count
                << " | chi2_before=" << local_optimization_result.chi2_before
                << " | chi2_after=" << local_optimization_result.chi2_after
                << " | max_delta_t=" << max_pose_delta_t << " m"
                << " | max_delta_R=" << max_pose_delta_R_deg << " deg"
                << " | max_odom_rel_dt="
                << max_relative_odom_translation_change << " m"
                << " | max_odom_rel_dR="
                << max_relative_odom_rotation_change_deg << " deg"
                << " | reason=" << rejection_reason
                << std::endl;

            continue;
        }

        const double mean_pose_delta_t =
            sum_pose_delta_t /
            static_cast<double>(
                current_indices.size());

        const double mean_pose_delta_R_deg =
            sum_pose_delta_R_deg /
            static_cast<double>(
                current_indices.size());

        for (std::size_t k = 0;
             k < current_indices.size();
             ++k)
        {
            const std::size_t source_index =
                current_indices[k];

            refined_keyframe_poses_[source_index] =
                optimized_local_poses[k];

            refined_keyframe_pose_was_adjusted_[source_index] =
                true;

            ++adjusted_keyframes;

            const Eigen::Isometry3d T_initial_optimized =
                frozen_graph_poses[source_index].inverse() *
                optimized_local_poses[k];

            const double final_delta_t =
                T_initial_optimized.translation().norm();

            const double final_delta_R_deg =
                RelativeRotationDeg(
                    frozen_graph_poses[source_index],
                    optimized_local_poses[k]);

            std::cout
                << "Post-PGO local pose refined"
                << " | keyframe=" << keyframes[source_index].id
                << " | geometry_tested="
                << (geometry_was_tested[k] ? "true" : "false")
                << " | geometry_anchor="
                << (geometry_anchor_valid[k] ? "true" : "false")
                << " | geometry_corr=" << geometry_correspondences[k]
                << " | geometry_rmse=" << geometry_rmse[k]
                << " | geometry_delta_t=" << geometry_delta_t[k]
                << " | geometry_delta_R=" << geometry_delta_R_deg[k]
                << " | final_delta_t=" << final_delta_t
                << " | final_delta_R=" << final_delta_R_deg
                << std::endl;
        }

        // --------------------------------------------------------------------
        // Save one COHERENT debug triplet for this accepted refinement group.
        // If several non-overlapping loop groups are accepted in one pass, the
        // latest accepted group replaces the previous debug triplet.  This is
        // intentional: the three RViz topics must always describe exactly the
        // same historical/current window rather than a confusing mixture.
        // --------------------------------------------------------------------
        const std::chrono::steady_clock::time_point
            debug_clouds_start =
                std::chrono::steady_clock::now();

        pcl::PointCloud<LIDAR_POINT>::Ptr current_before_debug =
            pcl::make_shared<pcl::PointCloud<LIDAR_POINT>>();

        pcl::PointCloud<LIDAR_POINT>::Ptr current_after_debug =
            pcl::make_shared<pcl::PointCloud<LIDAR_POINT>>();

        std::size_t estimated_current_debug_points = 0;

        for (const std::size_t source_index :
             current_indices)
        {
            if (keyframes[source_index].cloud)
            {
                estimated_current_debug_points +=
                    keyframes[source_index].cloud->size();
            }
        }

        current_before_debug->reserve(
            estimated_current_debug_points);

        current_after_debug->reserve(
            estimated_current_debug_points);

        for (std::size_t k = 0;
             k < current_indices.size();
             ++k)
        {
            const std::size_t source_index =
                current_indices[k];

            if (!keyframes[source_index].cloud ||
                keyframes[source_index].cloud->empty())
            {
                continue;
            }

            pcl::PointCloud<LIDAR_POINT> before_world_cloud;
            pcl::PointCloud<LIDAR_POINT> after_world_cloud;

            pcl::transformPointCloud(
                *keyframes[source_index].cloud,
                before_world_cloud,
                frozen_graph_poses[source_index]
                    .matrix()
                    .cast<float>());

            pcl::transformPointCloud(
                *keyframes[source_index].cloud,
                after_world_cloud,
                optimized_local_poses[k]
                    .matrix()
                    .cast<float>());

            *current_before_debug +=
                before_world_cloud;

            *current_after_debug +=
                after_world_cloud;
        }

        refinement_historical_target_debug_ =
            pcl::make_shared<pcl::PointCloud<LIDAR_POINT>>(
                *historical_target_filtered);

        refinement_current_before_debug_ =
            current_before_debug;

        refinement_current_after_debug_ =
            current_after_debug;

        refinement_debug_revision_ =
            global_map_revision_;

        std::cout
            << "Post-PGO refinement debug snapshots updated"
            << " | revision=" << refinement_debug_revision_
            << " | historical_kf=" << anchor.historical_id
            << " | current_kf=" << anchor.current_id
            << " | historical_points="
            << refinement_historical_target_debug_->size()
            << " | current_before_points="
            << refinement_current_before_debug_->size()
            << " | current_after_points="
            << refinement_current_after_debug_->size()
            << std::endl;

        refinement_timing.debug_clouds_ms +=
            ElapsedMilliseconds(
                debug_clouds_start,
                std::chrono::steady_clock::now());

        ++groups_accepted;
        refinement_timing.groups_accepted =
            groups_accepted;

        std::cout
            << "Post-PGO local-window refined"
            << " | historical_kf=" << anchor.historical_id
            << " | current_kf=" << anchor.current_id
            << " | current_kfs=" << current_indices.size()
            << " | historical_kfs=" << historical_candidates.size()
            << " | target_points=" << historical_target_filtered->size()
            << " | geometry_candidates=" << current_indices.size()
            << " | geometry_primary_selected=" << geometry_primary_selected
            << " | geometry_fallback_calls=" << geometry_fallback_calls
            << " | geometry_anchors=" << geometry_anchor_count
            << " | odom_edges=" << (current_indices.size() - 1)
            << " | chi2_before=" << local_optimization_result.chi2_before
            << " | chi2_after=" << local_optimization_result.chi2_after
            << " | mean_delta_t=" << mean_pose_delta_t << " m"
            << " | max_delta_t=" << max_pose_delta_t << " m"
            << " | mean_delta_R=" << mean_pose_delta_R_deg << " deg"
            << " | max_delta_R=" << max_pose_delta_R_deg << " deg"
            << " | max_odom_rel_dt="
            << max_relative_odom_translation_change << " m"
            << " | max_odom_rel_dR="
            << max_relative_odom_rotation_change_deg << " deg"
            << std::endl;
    }

    // ------------------------------------------------------------------------
    // Incrementally update /refined_map.
    //
    // The optimized block map is the base.  Only backend blocks containing
    // accepted locally-refined Keyframes become refined override blocks.  If
    // no local window was accepted, the refined map simply mirrors optimized
    // blocks without replaying every Keyframe cloud.
    // ------------------------------------------------------------------------
    std::size_t used_keyframes = 0;

    for (std::size_t i = 0;
         i < keyframes.size();
         ++i)
    {
        if (graph_pose_valid[i] &&
            keyframes[i].cloud &&
            !keyframes[i].cloud->empty() &&
            refined_keyframe_poses_[i].matrix().allFinite())
        {
            ++used_keyframes;
        }
    }

    if (used_keyframes == 0)
    {
        return false;
    }

    IncrementalGlobalMap::UpdateStats
        refined_stats;

    const std::chrono::steady_clock::time_point
        refined_map_update_start =
            std::chrono::steady_clock::now();

    const bool refined_update_ok =
        incremental_global_map_.UpdateRefinedOverrides(
            keyframes,
            refined_keyframe_poses_,
            refined_keyframe_pose_was_adjusted_,
            refined_stats);

    const pcl::PointCloud<LIDAR_POINT>::ConstPtr refined_map =
        incremental_global_map_.GetRefinedMap();

    refinement_timing.refined_map_update_ms +=
        ElapsedMilliseconds(
            refined_map_update_start,
            std::chrono::steady_clock::now());

    if (!refined_update_ok)
    {
        return false;
    }

    if (!refined_map ||
        refined_map->empty())
    {
        return false;
    }

    refined_map_revision_ =
        global_map_revision_;

    std::cout
        << "Post-PGO local-window refined map updated incrementally"
        << " | revision=" << refined_map_revision_
        << " | groups_considered=" << groups_considered
        << " | groups_prepared=" << groups_prepared
        << " | groups_optimized=" << groups_optimized
        << " | groups_accepted=" << groups_accepted
        << " | groups_rejected_overlap=" << groups_rejected_overlap
        << " | groups_rejected_quality=" << groups_rejected_quality
        << " | groups_rejected_large_update="
        << groups_rejected_large_update
        << " | geometry_candidates=" << geometry_candidates_total
        << " | geometry_primary_selected=" << geometry_primary_selected_total
        << " | geometry_attempts=" << geometry_attempts_total
        << " | geometry_fallback_calls=" << geometry_fallback_calls_total
        << " | geometry_anchors=" << geometry_anchors_total
        << " | adjusted_keyframes=" << adjusted_keyframes
        << " | used_keyframes=" << used_keyframes
        << " | refined_dirty_keyframes="
        << refined_stats.dirty_keyframes
        << " | refined_dirty_blocks="
        << refined_stats.dirty_blocks
        << " | refined_rebuilt_blocks="
        << refined_stats.rebuilt_blocks
        << " | refined_reused_blocks="
        << refined_stats.reused_blocks
        << " | total_blocks="
        << refined_stats.total_blocks
        << " | refined_points="
        << refined_map->size()
        << std::endl;

    return true;
}

// ============================================================================
// UpdateMapOdomCorrection()
//
// The frontend raw pose and the backend optimized pose refer to the SAME
// physical Keyframe but live in two different coordinate frames after loop
// correction:
//
//     raw frontend:       T_odom_K  == Keyframe::T_WL
//     corrected backend:  T_map_K   == PoseGraphNode::T_WK
//
// Therefore:
//
//     T_map_K = T_map_odom * T_odom_K
//
// and:
//
//     T_map_odom = T_map_K * T_odom_K^-1
//
// Between backend optimizations this correction is held constant.  New raw
// frontend scans remain continuous, while their corrected pose is obtained by
// left-multiplying this bridge.
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
