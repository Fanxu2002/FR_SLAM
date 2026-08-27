#include "fr_slam/fr_registration_scan2localmap.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <cstdint>
#include <utility>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>

#include <rclcpp/rclcpp.hpp>

namespace
{

    const rclcpp::Logger kRecoveryLogger =
        rclcpp::get_logger("scan2local_map.recovery");

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
// RegistrationScan2LocalMap
// ============================================================================
//
// This class is the main LiDAR frontend wrapper for:
//
//     Current LiDAR Scan
//              |
//              v
//     Initial Guess Prediction
//       - translation: LiDAR constant-motion model
//       - rotation   : optional IMU relative rotation
//              |
//              v
//        Scan-to-LocalMap
//              |
//              v
//          Quality Gate
//              |
//      +-------+--------+
//      |                |
//   rejected          accepted
//      |                |
//      |                v
//      |         KeyframeDetector
//      |                |
//      |        +-------+-------+
//      |        |               |
//      |      false           true
//      |        |               |
//      |        |               +--> KeyframeManager
//      |        |               |
//      |        |               +--> LocalMap update
//      |        |
//      +--------+
//               |
//               v
//          Commit odometry
//
// ---------------------------------------------------------------------------
// Coordinate convention used in this file:
//
//     T_WL
//
// means:
//
//     LiDAR frame -> World frame
//
// Therefore:
//
//     p_W = T_WL * p_L
//
// The LocalMap is stored in the World frame.
//
// During Scan-to-LocalMap:
//
//     source = current LiDAR scan in current LiDAR frame
//     target = LocalMap in World frame
//
// so the registration result directly represents:
//
//     result.T_target_source = T_WL_current
//
// ---------------------------------------------------------------------------
// Recovery idea used in this version:
//
// If several frames are rejected, the last accepted pose T_WL_ is deliberately
// NOT changed. The LiDAR one-frame motion model is extrapolated for several
// steps.
//
// Recovery V2 additionally adds a coarse point-to-point ICP fallback whenever
// the normal point-to-plane candidate fails the Quality Gate. The coarse pose
// is then refined by the original point-to-plane Scan-to-LocalMap registration.
//
//     T_guess = T_last_accepted * DeltaT^N
//
// while IMU replaces only the rotational part of the initial guess.
//
// IMPORTANT:
//
// When tracking is recovered after several rejected frames, the transform
//
//     T_last_accepted^-1 * T_current
//
// spans MULTIPLE scan intervals. It must therefore NOT overwrite the stored
// one-frame motion model `last_relative_transform_`.
//
// ============================================================================

RegistrationScan2LocalMap::RegistrationScan2LocalMap(
    const LidarRegistrationConfig &registration_config,
    const LocalMapConfig &local_map_config)
    : registration_(registration_config),

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

bool RegistrationScan2LocalMap::AddFrame(
    const pcl::PointCloud<LIDAR_POINT>::ConstPtr &cloud_lidar,
    double timestamp,
    Eigen::Isometry3d &T_WL,
    LidarRegistrationResult &registration_result,
    const Eigen::Quaterniond *imu_relative_rotation)
{
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
        if (!keyframe_manager_.AddKeyframe(
                timestamp,
                T_WL_,
                cloud_lidar))
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
        // 1.2 Add KF0 as the fixed KEYFRAME PoseGraph vertex.
        //
        // V6 backend rule:
        //     every Keyframe enters the graph immediately.
        //
        // Submap lifecycle is irrelevant to graph vertex creation.
        // ---------------------------------------------------------------------
        if (!AddKeyframeToPoseGraph(*first_keyframe))
        {
            std::cerr
                << "RegistrationScan2LocalMap::AddFrame(): "
                << "failed to initialize Keyframe PoseGraph."
                << std::endl;
            return false;
        }

        // Backend visualization map grows from KF0 onward.  This cache is
        // independent from frontend Submap lifecycle and is optional for
        // tracking, so a map-cache failure must not invalidate SLAM.
        if (!UpdateIncrementalGlobalMaps(
                "NEW_KEYFRAME",
                false))
        {
            std::cerr
                << "Incremental global map update failed"
                << " | reason=NEW_KEYFRAME"
                << " | keyframe=" << first_keyframe->id
                << std::endl;
        }

        // ---------------------------------------------------------------------
        // 1.3 Create Active Submap 0 and insert KF0.
        //
        // Submap internally reuses LocalMap for:
        //
        //     transform -> merge -> voxel
        // ---------------------------------------------------------------------
        if (!submap_manager_.AddKeyframe(
                *first_keyframe))
        {
            std::cerr
                << "RegistrationScan2LocalMap::AddFrame(): "
                << "failed to initialize Active Submap."
                << std::endl;
            return false;
        }

        // ---------------------------------------------------------------------
        // 1.4 Register KF0 in the KEYFRAME Scan Context database.
        //
        // This database is independent from Submap finalization. KF0 obviously
        // cannot form a loop yet, so we only store its descriptor here.
        // ---------------------------------------------------------------------
        if (!loop_detector_.AddKeyframe(*first_keyframe))
        {
            std::cerr
                << "Keyframe Scan Context registration failed"
                << " | keyframe=" << first_keyframe->id
                << std::endl;
        }
        else
        {
            std::cout
                << "Keyframe Scan Context registered"
                << " | keyframe=" << first_keyframe->id
                << " | descriptors="
                << loop_detector_.DescriptorCount()
                << std::endl;
        }

        // ---------------------------------------------------------------------
        // 1.5 Prepare point-to-plane target from the current Submap tracking map.
        // ---------------------------------------------------------------------
        if (!registration_.PrepareTarget(
                submap_manager_.GetTrackingMap(),
                prepared_tracking_target_))
        {
            std::cerr
                << "RegistrationScan2LocalMap::AddFrame(): "
                << "failed to prepare first Submap tracking target."
                << std::endl;
            return false;
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
    std::cout
        << "Tracking prediction"
        << " | rejected_frames="
        << consecutive_rejected_frames_
        << " | prediction_steps="
        << prediction_steps
        << " | predicted_translation="
        << (initial_guess.translation() -
            T_WL_previous.translation())
               .norm()
        << " m"
        << " | mode="
        << (consecutive_rejected_frames_ == 0
                ? "TRACKING"
                : "RECOVERY")
        << std::endl;

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

    bool registration_success =
        registration_.Align(
            cloud_lidar,
            prepared_tracking_target_,
            initial_guess,
            result);

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

        const bool coarse_success =
            RunCoarsePointToPointRecovery(
                cloud_lidar,
                submap_manager_.GetTrackingMap(),
                initial_guess,
                T_WL_coarse,
                coarse_fitness);

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

            const bool refined_success =
                registration_.Align(
                    cloud_lidar,
                    prepared_tracking_target_,
                    T_WL_coarse,
                    refined_result);

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
            }
        }
    }

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

    const bool is_keyframe =
        keyframe_detector_.ShouldCreateKeyframe(
            T_WL_current,
            keyframe_translation,
            keyframe_rotation_deg);

    std::cout
        << "Keyframe decision"
        << " | translation=" << keyframe_translation << " m"
        << " | rotation=" << keyframe_rotation_deg << " deg"
        << " | translation_threshold=0.5 m"
        << " | rotation_threshold=5.0 deg"
        << " | keyframe=" << (is_keyframe ? "true" : "false")
        << std::endl;

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
        if (!keyframe_manager_.AddKeyframe(
                timestamp,
                T_WL_current,
                cloud_lidar))
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
        // 9.2 Add the new KEYFRAME vertex + sequential KF odometry edge.
        //
        // This is now the backend graph update point. It happens for every
        // Keyframe and does NOT wait for any Submap to finish.
        // ---------------------------------------------------------------------
        if (!AddKeyframeToPoseGraph(*new_keyframe))
        {
            std::cerr
                << "RegistrationScan2LocalMap::AddFrame(): "
                << "failed to update Keyframe PoseGraph."
                << std::endl;
            return false;
        }

        // Grow the backend map cache immediately for every Keyframe.  In the
        // common case this rebuilds only the one backend block containing the
        // new Keyframe.  Old refinement override blocks remain valid until the
        // next main PoseGraph optimization.
        if (!UpdateIncrementalGlobalMaps(
                "NEW_KEYFRAME",
                false))
        {
            std::cerr
                << "Incremental global map update failed"
                << " | reason=NEW_KEYFRAME"
                << " | keyframe=" << new_keyframe->id
                << std::endl;
        }

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
        if (!submap_manager_.AddKeyframe(
                *new_keyframe))
        {
            std::cerr
                << "RegistrationScan2LocalMap::AddFrame(): "
                << "failed to update SubmapManager."
                << std::endl;
            return false;
        }

        // ---------------------------------------------------------------------
        // 9.4 KEYFRAME-DRIVEN LOOP QUERY.
        //
        // This is the lifecycle fix:
        //     New Keyframe -> Scan Context query NOW
        //
        // It does NOT wait for the current Submap to become finished.
        // ---------------------------------------------------------------------
        if (!loop_detector_.AddKeyframe(*new_keyframe))
        {
            std::cerr
                << "Keyframe Scan Context registration failed"
                << " | keyframe=" << new_keyframe->id
                << std::endl;
        }
        else
        {
            DetectAndVerifyLoopFromKeyframe(
                *new_keyframe,
                current_owner_submap_id);
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

        if (!registration_.PrepareTarget(
                submap_manager_.GetTrackingMap(),
                new_prepared_active_submap))
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
        std::cout
            << "Motion model protected after coarse recovery"
            << " | update=false"
            << " | kept_translation="
            << last_relative_transform_
                   .translation()
                   .norm()
            << " m"
            << std::endl;
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

    std::cout
        << "Motion model gate"
        << " | previous_translation="
        << previous_motion_translation
        << " m"
        << " | current_translation="
        << current_motion_translation
        << " m"
        << " | ratio="
        << motion_translation_ratio
        << " | max_ratio="
        << max_translation_ratio
        << " | update="
        << (motion_model_update_allowed
                ? "true"
                : "false")
        << std::endl;

    if (recovered_from_tracking_loss)
    {
        std::cout
            << "Tracking recovered"
            << " | rejected_frames="
            << consecutive_rejected_frames_
            << " | recovery_prediction_steps="
            << prediction_steps

            // We intentionally keep the previous one-frame LiDAR motion model.
            << " | motion_model=KEEP_PREVIOUS_SINGLE_FRAME"
            << std::endl;
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
        std::cout
            << "Motion model after recovery"
            << " | update=false"
            << " | kept_translation="
            << last_relative_transform_
                   .translation()
                   .norm()
            << " m"
            << std::endl;
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
        std::cout
            << "Motion model protected"
            << " | update=false"
            << " | current_translation="
            << current_motion_translation
            << " m"
            << " | kept_translation="
            << last_relative_transform_
                   .translation()
                   .norm()
            << " m"
            << std::endl;
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
    std::cout
        << "RegistrationScan2LocalMap"
        << " | x=" << T_WL_current.translation().x()
        << " y=" << T_WL_current.translation().y()
        << " z=" << T_WL_current.translation().z()
        << " | keyframe=" << (is_keyframe ? "true" : "false")
        << " | keyframes=" << keyframe_manager_.Size()
        << " | submaps=" << submap_manager_.SubmapCount()
        << " | active_submap=" << submap_manager_.ActiveSubmapId()
        << " | previous_submap=";

    if (submap_manager_.PreviousSubmap() != nullptr)
    {
        std::cout
            << submap_manager_.PreviousSubmapId();
    }
    else
    {
        std::cout
            << "none";
    }

    std::cout
        << " | active_keyframes="
        << submap_manager_.ActiveKeyframeCount()
        << " | target_mode="
        << (submap_manager_.IsTransitionActive()
                ? "TRANSITION"
                : "ACTIVE")
        << " | active_map_points="
        << submap_manager_.ActivePointCount()
        << " | target_points="
        << submap_manager_.TrackingPointCount()
        << " | corr=" << result.correspondences
        << " | rmse=" << result.rmse
        << std::endl;

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
        keyframe_manager_.GetAllKeyframes();

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

    std::cout
        << "Keyframe gravity reference stored"
        << " | keyframe=" << keyframe.id
        << " | gravity_L=["
        << gravity_L_reference.transpose()
        << "]"
        << std::endl;

    std::cout
        << "Keyframe PoseGraph node added"
        << " | keyframe=" << keyframe.id
        << " | fixed=" << (fixed ? "true" : "false")
        << " | T_WK=["
        << T_WK_graph_initial.translation().x() << " "
        << T_WK_graph_initial.translation().y() << " "
        << T_WK_graph_initial.translation().z() << "]"
        << " | frontend_T_WL=["
        << keyframe.T_WL.translation().x() << " "
        << keyframe.T_WL.translation().y() << " "
        << keyframe.T_WL.translation().z() << "]"
        << " | nodes=" << pose_graph_.NodeCount()
        << std::endl;

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

    const Eigen::Matrix<double, 6, 6> information =
        Eigen::Matrix<double, 6, 6>::Identity();

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

    std::cout
        << "Keyframe PoseGraph odometry edge added"
        << " | from=" << previous_keyframe->id
        << " | to=" << keyframe.id
        << " | translation_norm="
        << Z_previous_current.translation().norm()
        << " m"
        << " | rotation="
        << RelativeRotationDeg(
               Eigen::Isometry3d::Identity(),
               Z_previous_current)
        << " deg"
        << " | odom_edges=" << pose_graph_.OdometryEdgeCount()
        << std::endl;

    return true;
}


// ============================================================================
// FindSubmapById()
//
// Read-only backend lookup used by LoopVerifier integration.
// ============================================================================
const Submap *RegistrationScan2LocalMap::FindSubmapById(
    std::size_t submap_id) const
{
    const std::vector<Submap> &submaps =
        submap_manager_.GetAllSubmaps();

    for (const Submap &submap :
         submaps)
    {
        if (submap.id ==
            submap_id)
        {
            return &submap;
        }
    }

    return nullptr;
}


const Keyframe *RegistrationScan2LocalMap::FindKeyframeById(
    std::size_t keyframe_id) const
{
    for (const Keyframe &keyframe :
         keyframe_manager_.GetAllKeyframes())
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
// A Keyframe may appear in two Submaps because of overlap. For geometry
// verification choose a FINISHED/FROZEN Submap containing the candidate KF and
// prefer the one where that KF lies closest to the center of the window.
// ============================================================================
const Submap *
RegistrationScan2LocalMap::FindBestFinishedSubmapForKeyframe(
    std::size_t keyframe_id) const
{
    const Submap *best = nullptr;
    std::size_t best_center_distance =
        std::numeric_limits<std::size_t>::max();

    for (const Submap &submap :
         submap_manager_.GetAllSubmaps())
    {
        if (!submap.finished ||
            !submap.has_frozen_cloud ||
            !submap.cloud_S ||
            submap.cloud_S->empty() ||
            !submap.has_origin_pose ||
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

    const std::vector<LoopCandidate> candidates =
        loop_detector_.Detect(
            current_keyframe.id);

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
    const Submap *best_historical_submap = nullptr;
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

        const Submap *historical_submap =
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
            FindKeyframeById(candidate.candidate_id);

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

        LoopVerificationResult verification;
        bool verification_success = false;
        const char *best_initial_guess_name = "NONE";

        for (const auto &guess_entry : initial_guesses)
        {
            LoopVerificationResult trial;

            // NaN disables LoopVerifier's own absolute-yaw replacement. We
            // already applied the Keyframe->historical-Submap yaw correctly
            // above, including the candidate KF orientation inside H.
            const bool trial_success =
                loop_verifier_.Verify(
                    current_keyframe.cloud,
                    historical_submap->cloud_S,
                    guess_entry.second,
                    std::numeric_limits<double>::quiet_NaN(),
                    trial);

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
                best_initial_guess_name = guess_entry.first;
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
        FindKeyframeById(best_candidate->candidate_id);

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

    const double graph_correction_translation =
        (T_W_L_loop.translation() -
         current_keyframe.T_WL.translation())
            .norm();

    const double graph_correction_rotation =
        RelativeRotationDeg(
            current_keyframe.T_WL,
            T_W_L_loop);

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

        const Eigen::Matrix<double, 6, 6> loop_information =
            Eigen::Matrix<double, 6, 6>::Identity();

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

            if (!pose_graph_.AddLoopEdge(
                    constraint.historical_keyframe_id,
                    constraint.current_keyframe_id,
                    constraint.T_historical_current,
                    loop_information))
            {
                batch_stage_ok = false;
                break;
            }

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

        if (!pose_graph_optimizer_.Optimize(
                pose_graph_,
                optimization_result))
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

        if (!UpdateMapOdomCorrection(
                last_online_loop_current_keyframe_id_))
        {
            std::cerr
                << "Map->odom correction update failed"
                << " | anchor_kf="
                << last_online_loop_current_keyframe_id_
                << std::endl;
        }

        const bool global_map_rebuilt =
            RebuildGlobalMapSnapshots();

        if (!global_map_rebuilt)
        {
            std::cerr
                << "Global map snapshot rebuild failed"
                << " | keyframes=" << keyframe_manager_.Size()
                << " | graph_nodes=" << pose_graph_.NodeCount()
                << std::endl;
        }
        else if (!RebuildPostPgoRefinedMap())
        {
            std::cerr
                << "Post-PGO refined map rebuild skipped/failed"
                << " | global_revision=" << global_map_revision_
                << " | keyframes=" << keyframe_manager_.Size()
                << std::endl;
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
            FindKeyframeById(
                last_online_loop_historical_keyframe_id_);
        const Keyframe *last_current_loop_keyframe =
            FindKeyframeById(
                last_online_loop_current_keyframe_id_);
        const Keyframe *new_historical_keyframe =
            FindKeyframeById(
                edge_historical_keyframe_id);
        const Keyframe *new_current_loop_keyframe =
            FindKeyframeById(
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

    const Eigen::Matrix<double, 6, 6> loop_information =
        Eigen::Matrix<double, 6, 6>::Identity();

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

    if (!pose_graph_optimizer_.Optimize(
            pose_graph_,
            optimization_result))
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
    if (!UpdateMapOdomCorrection(
            current_keyframe.id))
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
    const bool global_map_rebuilt =
        RebuildGlobalMapSnapshots();

    if (!global_map_rebuilt)
    {
        std::cerr
            << "Global map snapshot rebuild failed"
            << " | keyframes=" << keyframe_manager_.Size()
            << " | graph_nodes=" << pose_graph_.NodeCount()
            << std::endl;
    }
    else if (!RebuildPostPgoRefinedMap())
    {
        // Refinement is an OPTIONAL backend product.  A failure here must not
        // invalidate the already-successful PoseGraph optimization or the raw
        // / optimized global-map snapshots.
        std::cerr
            << "Post-PGO refined map rebuild skipped/failed"
            << " | global_revision=" << global_map_revision_
            << " | keyframes=" << keyframe_manager_.Size()
            << std::endl;
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
    const std::vector<Keyframe> &keyframes =
        keyframe_manager_.GetAllKeyframes();

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

    IncrementalGlobalMap::UpdateStats raw_stats;
    IncrementalGlobalMap::UpdateStats optimized_stats;

    const bool raw_ok =
        incremental_global_map_.UpdateRaw(
            keyframes,
            raw_stats);

    const bool optimized_ok =
        incremental_global_map_.UpdateOptimized(
            keyframes,
            optimized_poses,
            optimized_pose_valid,
            clear_refined_overrides,
            optimized_stats);

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
// Post-PGO refinement V3: local-window multi-pose optimization.
//
// Each current-window Keyframe gets its own SE(3) variable.  Per-Keyframe
// point-to-plane registration against one frozen historical LocalMap supplies
// soft geometry anchor edges, while frozen G2O relative poses provide stronger
// consecutive odometry edges.  The temporary graph solution is used only for
// /refined_map and never overwrites the main PoseGraph.
//
// Why V3 exists:
//     V1: each Keyframe independently registered to a LocalMap -> too free.
//     V2: one shared rigid correction for the whole window -> too rigid.
//
// V3 uses a small temporary Keyframe PoseGraph:
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
bool RegistrationScan2LocalMap::RebuildPostPgoRefinedMap()
{
    const std::vector<Keyframe> &keyframes =
        keyframe_manager_.GetAllKeyframes();

    if (keyframes.empty() ||
        pose_graph_.NodeCount() == 0 ||
        global_map_revision_ == 0)
    {
        return false;
    }

    refined_map_revision_ = 0;

    if (refinement_historical_target_debug_)
    {
        refinement_historical_target_debug_->clear();
    }

    if (refinement_current_before_debug_)
    {
        refinement_current_before_debug_->clear();
    }

    if (refinement_current_after_debug_)
    {
        refinement_current_after_debug_->clear();
    }

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

    std::size_t groups_considered = 0;
    std::size_t groups_prepared = 0;
    std::size_t groups_optimized = 0;
    std::size_t groups_accepted = 0;
    std::size_t groups_rejected_overlap = 0;
    std::size_t groups_rejected_quality = 0;
    std::size_t groups_rejected_large_update = 0;
    std::size_t geometry_attempts_total = 0;
    std::size_t geometry_anchors_total = 0;
    std::size_t adjusted_keyframes = 0;

    std::cout
        << "Post-PGO local-window refinement V3 started"
        << " | global_revision=" << global_map_revision_
        << " | keyframes=" << keyframes.size()
        << " | loop_anchors=" << loop_anchors.size()
        << " | local_window=" << refinement_local_window_
        << " | historical_window=" << refinement_historical_keyframe_window_
        << " | historical_radius=" << refinement_historical_radius_ << " m"
        << " | min_geometry_anchors=" << refinement_min_geometry_anchors_
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

        // --------------------------------------------------------------------
        // Current revisit window: one independent SE(3) state per Keyframe.
        // The window ends at the current loop endpoint because those poses are
        // already available when the online loop is accepted.
        // --------------------------------------------------------------------
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

        target_voxel.filter(
            *historical_target_filtered);

        if (historical_target_filtered->size() <
            refinement_min_target_points_)
        {
            ++groups_rejected_quality;
            continue;
        }

        PreparedLidarTarget
            prepared_refinement_target;

        if (!registration_.PrepareTarget(
                historical_target_filtered,
                prepared_refinement_target))
        {
            ++groups_rejected_quality;
            continue;
        }

        ++groups_prepared;

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

        std::vector<bool>
            geometry_anchor_valid(
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

        const Eigen::Matrix<double, 6, 6>
            geometry_information =
                Eigen::Matrix<double, 6, 6>::Identity() *
                refinement_geometry_information_scale_;

        for (std::size_t k = 0;
             k < current_indices.size();
             ++k)
        {
            const std::size_t source_index =
                current_indices[k];

            const Keyframe &source_keyframe =
                keyframes[source_index];

            ++geometry_attempts_total;

            LidarRegistrationResult
                geometry_result;

            const bool align_success =
                registration_.Align(
                    source_keyframe.cloud,
                    prepared_refinement_target,
                    frozen_graph_poses[source_index],
                    geometry_result);

            if (!align_success ||
                !geometry_result.success ||
                !geometry_result.converged ||
                !geometry_result.T_target_source.matrix().allFinite() ||
                !std::isfinite(geometry_result.rmse))
            {
                continue;
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

            if (geometry_result.correspondences <
                    refinement_geometry_min_correspondences_ ||
                geometry_result.rmse >
                    refinement_geometry_max_rmse_ ||
                !std::isfinite(correction_translation) ||
                !std::isfinite(correction_rotation_deg) ||
                correction_translation >
                    refinement_geometry_max_translation_correction_ ||
                correction_rotation_deg >
                    refinement_geometry_max_rotation_correction_deg_)
            {
                continue;
            }

            // Fixed node 0 is Identity in map coordinates.  Therefore the
            // measurement 0 -> k is simply the absolute map pose returned by
            // the point-to-plane registration.
            if (!local_pose_graph.AddLoopEdge(
                    0,
                    local_node_ids[k],
                    geometry_result.T_target_source,
                    geometry_information))
            {
                local_graph_ok = false;
                break;
            }

            geometry_anchor_valid[k] =
                true;

            ++geometry_anchor_count;
            ++geometry_anchors_total;
        }

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

        if (!pose_graph_optimizer_.Optimize(
                local_pose_graph,
                local_optimization_result) ||
            !local_optimization_result.success)
        {
            ++groups_rejected_quality;
            continue;
        }

        ++groups_optimized;

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

        if (!optimized_window_valid ||
            max_pose_delta_t >
                refinement_window_max_translation_update_ ||
            max_pose_delta_R_deg >
                refinement_window_max_rotation_update_deg_)
        {
            ++groups_rejected_large_update;

            std::cout
                << "Post-PGO local-window refinement rejected"
                << " | historical_kf=" << anchor.historical_id
                << " | current_kf=" << anchor.current_id
                << " | geometry_anchors=" << geometry_anchor_count
                << " | chi2_before=" << local_optimization_result.chi2_before
                << " | chi2_after=" << local_optimization_result.chi2_after
                << " | max_delta_t=" << max_pose_delta_t << " m"
                << " | max_delta_R=" << max_pose_delta_R_deg << " deg"
                << " | reason=LARGE_LOCAL_WINDOW_UPDATE"
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

        ++groups_accepted;

        std::cout
            << "Post-PGO local-window refined"
            << " | historical_kf=" << anchor.historical_id
            << " | current_kf=" << anchor.current_id
            << " | current_kfs=" << current_indices.size()
            << " | historical_kfs=" << historical_candidates.size()
            << " | target_points=" << historical_target_filtered->size()
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

    if (!incremental_global_map_.UpdateRefinedOverrides(
            keyframes,
            refined_keyframe_poses_,
            refined_keyframe_pose_was_adjusted_,
            refined_stats))
    {
        return false;
    }

    const pcl::PointCloud<LIDAR_POINT>::ConstPtr refined_map =
        incremental_global_map_.GetRefinedMap();

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
        << " | geometry_attempts=" << geometry_attempts_total
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
bool RegistrationScan2LocalMap::UpdateMapOdomCorrection(
    std::size_t anchor_keyframe_id)
{
    const Keyframe *anchor_keyframe =
        FindKeyframeById(
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
Eigen::Isometry3d RegistrationScan2LocalMap::GetPose() const
{
    return T_WL_;
}

// Return the ACTUAL current registration target.
//
// ACTIVE mode:
//     Active Submap cloud.
//
// TRANSITION mode:
//     voxelized Previous + Active cloud.
//
// The old GetLocalMap() API name is kept so the ROS publisher does not need to
// change.  RViz therefore shows the same map that registration is using.
pcl::PointCloud<LIDAR_POINT>::ConstPtr
RegistrationScan2LocalMap::GetLocalMap() const
{
    return submap_manager_.GetTrackingMap();
}

pcl::PointCloud<LIDAR_POINT>::ConstPtr
RegistrationScan2LocalMap::GetRawKeyframeMap() const
{
    return incremental_global_map_.GetRawMap();
}

pcl::PointCloud<LIDAR_POINT>::ConstPtr
RegistrationScan2LocalMap::GetOptimizedMap() const
{
    return incremental_global_map_.GetOptimizedMap();
}

pcl::PointCloud<LIDAR_POINT>::ConstPtr
RegistrationScan2LocalMap::GetRefinedMap() const
{
    return incremental_global_map_.GetRefinedMap();
}

pcl::PointCloud<LIDAR_POINT>::ConstPtr
RegistrationScan2LocalMap::GetRefinementHistoricalTarget() const
{
    return refinement_historical_target_debug_;
}

pcl::PointCloud<LIDAR_POINT>::ConstPtr
RegistrationScan2LocalMap::GetRefinementCurrentBefore() const
{
    return refinement_current_before_debug_;
}

pcl::PointCloud<LIDAR_POINT>::ConstPtr
RegistrationScan2LocalMap::GetRefinementCurrentAfter() const
{
    return refinement_current_after_debug_;
}

std::size_t RegistrationScan2LocalMap::RefinementDebugRevision() const
{
    return refinement_debug_revision_;
}

std::size_t RegistrationScan2LocalMap::RefinedMapRevision() const
{
    return refined_map_revision_;
}

std::size_t RegistrationScan2LocalMap::GlobalMapRevision() const
{
    return global_map_revision_;
}

bool RegistrationScan2LocalMap::HasMapOdomCorrection() const
{
    return has_map_odom_correction_;
}

Eigen::Isometry3d RegistrationScan2LocalMap::GetMapOdomCorrection() const
{
    return T_map_odom_;
}

Eigen::Isometry3d RegistrationScan2LocalMap::GetCorrectedPose(
    const Eigen::Isometry3d &T_odom_lidar) const
{
    if (!T_odom_lidar.matrix().allFinite())
    {
        return Eigen::Isometry3d::Identity();
    }

    // Before the first loop optimization T_map_odom_ is Identity, so this is
    // intentionally identical to the raw frontend pose.
    return T_map_odom_ *
           T_odom_lidar;
}

std::size_t RegistrationScan2LocalMap::MapOdomRevision() const
{
    return map_odom_revision_;
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

const PoseGraph &
RegistrationScan2LocalMap::GetPoseGraph() const
{
    return pose_graph_;
}

std::size_t
RegistrationScan2LocalMap::PoseGraphNodeCount() const
{
    return pose_graph_.NodeCount();
}

std::size_t
RegistrationScan2LocalMap::PoseGraphEdgeCount() const
{
    return pose_graph_.EdgeCount();
}

std::size_t
RegistrationScan2LocalMap::PoseGraphOdometryEdgeCount() const
{
    return pose_graph_.OdometryEdgeCount();
}

std::size_t
RegistrationScan2LocalMap::PoseGraphLoopEdgeCount() const
{
    return pose_graph_.LoopEdgeCount();
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
    // Reset accepted global pose.
    T_WL_ = Eigen::Isometry3d::Identity();

    // Reset LiDAR constant-motion prediction.
    last_relative_transform_ = Eigen::Isometry3d::Identity();

    // Reset short-term tracking recovery state.
    consecutive_rejected_frames_ = 0;

    // Clear ALL Submaps and their internal LocalMap builders.
    submap_manager_.Clear();

    // Clear backend PoseGraph nodes / edges.
    pose_graph_.Clear();

    // Clear Scan Context descriptor database.
    loop_detector_.Clear();

    // Clear keyframe-level online loop temporal state and multi-loop edge
    // sparsification anchors.
    online_loop_track_ = OnlineLoopTrack();
    has_last_online_loop_edge_ = false;
    last_online_loop_current_keyframe_id_ =
        std::numeric_limits<std::size_t>::max();
    last_online_loop_historical_keyframe_id_ =
        std::numeric_limits<std::size_t>::max();
    last_online_loop_measurement_ =
        Eigen::Isometry3d::Identity();
    pending_first_loop_batch_.clear();

    // Clear cached Submap tracking registration target.
    prepared_tracking_target_ = PreparedLidarTarget();

    // Clear COMPLETE historical keyframe storage.
    keyframe_manager_.Clear();

    // Clear incremental backend global-map block caches.
    incremental_global_map_.Clear();

    if (refinement_historical_target_debug_)
    {
        refinement_historical_target_debug_->clear();
    }

    if (refinement_current_before_debug_)
    {
        refinement_current_before_debug_->clear();
    }

    if (refinement_current_after_debug_)
    {
        refinement_current_after_debug_->clear();
    }

    refined_keyframe_poses_.clear();
    refined_keyframe_pose_was_adjusted_.clear();

    global_map_revision_ = 0;
    refined_map_revision_ = 0;
    refinement_debug_revision_ = 0;

    // Reset backend map-frame <-> frontend odom-frame bridge.
    T_map_odom_ =
        Eigen::Isometry3d::Identity();

    has_map_odom_correction_ =
        false;

    map_odom_revision_ =
        0;

    map_odom_anchor_keyframe_id_ =
        std::numeric_limits<std::size_t>::max();

    // Reset latest-keyframe reference.
    keyframe_detector_.Reset();

    initialized_ = false;
}