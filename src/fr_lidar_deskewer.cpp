#include "fr_slam/fr_lidar_deskewer.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <cstdint>
#include <limits>

// ============================================================================
// Interpolate IMU pose at a specified timestamp
// ============================================================================

bool LidarDeskewer::InterpolatePose(
    const std::vector<IMU_POSE> &imu_poses,
    double timestamp,
    IMU_POSE &output_pose) const
{
    // At least two poses are required for interpolation
    if (imu_poses.size() < 2)
    {
        return false;
    }

    // The requested timestamp must be inside the IMU trajectory
    if (timestamp < imu_poses.front().timestamp ||
        timestamp > imu_poses.back().timestamp)
    {
        return false;
    }

    // Find two adjacent poses:
    //
    // pose_a.timestamp <= timestamp <= pose_b.timestamp
    //
    for (std::size_t i = 0;
         i + 1 < imu_poses.size();
         ++i)
    {
        const IMU_POSE &pose_a =
            imu_poses[i];

        const IMU_POSE &pose_b =
            imu_poses[i + 1];

        if (pose_a.timestamp <= timestamp &&
            timestamp <= pose_b.timestamp)
        {
            const double dt =
                pose_b.timestamp -
                pose_a.timestamp;

            if (dt <= 0.0)
            {
                return false;
            }

            const double alpha =
                (timestamp - pose_a.timestamp) /
                dt;

            output_pose.timestamp =
                timestamp;

            // ----------------------------------------------------
            // Position interpolation
            // ----------------------------------------------------

            output_pose.P_WI =
                (1.0 - alpha) * pose_a.P_WI +
                alpha * pose_b.P_WI;

            // ----------------------------------------------------
            // Velocity interpolation
            // ----------------------------------------------------

            output_pose.V_WI =
                (1.0 - alpha) * pose_a.V_WI +
                alpha * pose_b.V_WI;

            // ----------------------------------------------------
            // Rotation interpolation
            //
            // Quaternion uses SLERP instead of normal
            // linear interpolation.
            // ----------------------------------------------------

            output_pose.Q_WI =
                pose_a.Q_WI.slerp(
                    alpha,
                    pose_b.Q_WI);

            output_pose.Q_WI.normalize();

            return true;
        }
    }

    return false;
}

// ============================================================================
// Convert IMU_POSE into SE(3) transformation
//
// T_WI:
// IMU frame -> World frame
//
// p_W = T_WI * p_I
// ============================================================================

Eigen::Isometry3d LidarDeskewer::PoseToTransform(
    const IMU_POSE &pose) const
{
    Eigen::Isometry3d T_WI =
        Eigen::Isometry3d::Identity();

    T_WI.linear() =
        pose.Q_WI
            .normalized()
            .toRotationMatrix();

    T_WI.translation() =
        pose.P_WI;

    return T_WI;
}

// ============================================================================
// Set LiDAR-IMU extrinsic
//
// T_IL:
// LiDAR frame -> IMU frame
//
// p_I = T_IL * p_L
// ============================================================================

void LidarDeskewer::SetExtrinsic(
    const Eigen::Quaterniond &Q_IL,
    const Eigen::Vector3d &P_IL)
{
    T_IL_.setIdentity();

    T_IL_.linear() =
        Q_IL
            .normalized()
            .toRotationMatrix();

    T_IL_.translation() =
        P_IL;
}

// ============================================================================
// Deskew one LiDAR frame
//
// All LiDAR points are transformed to the LiDAR frame at scan start.
//
// Core equation:
//
// p_Lref =
//     T_WL(t_ref)^(-1)
//     *
//     T_WL(t_i)
//     *
//     p_Li
//
// where:
//
// T_WL = T_WI * T_IL
// ============================================================================

bool LidarDeskewer::Deskew(
    const LIDAR_FRAME &input_frame,
    const std::vector<IMU_POSE> &imu_poses,
    LIDAR_FRAME &output_frame,
    bool use_translation)
{
    // ====================================================================
    // 1. Basic checks
    // ====================================================================

    if (!input_frame.cloud)
    {
        return false;
    }

    if (input_frame.cloud->empty())
    {
        return false;
    }

    if (!input_frame.has_point_time)
    {
        return false;
    }

    if (imu_poses.size() < 2)
    {
        return false;
    }

    // ====================================================================
    // 2. Reference time
    //
    // First version:
    //
    // Deskew all points to scan start.
    // ====================================================================

    const double reference_time =
        input_frame.scan_start_time;

    // ====================================================================
    // 3. Find the actual time range of all LiDAR points
    //
    // point_time =
    //     scan_start_time
    //     +
    //     point.time_offset
    // ====================================================================

    double min_point_time =
        std::numeric_limits<double>::max();

    double max_point_time =
        std::numeric_limits<double>::lowest();

    for (const LIDAR_POINT &point :
         input_frame.cloud->points)
    {
        const double point_time =
            input_frame.scan_start_time +
            point.time_offset;

        min_point_time =
            std::min(
                min_point_time,
                point_time);

        max_point_time =
            std::max(
                max_point_time,
                point_time);
    }

    // ====================================================================
    // 4. Check whether IMU trajectory covers the entire LiDAR frame
    //
    // We also include reference_time in the required range.
    // ====================================================================

    const double required_start_time =
        std::min(
            reference_time,
            min_point_time);

    const double required_end_time =
        std::max(
            reference_time,
            max_point_time);

    if (imu_poses.front().timestamp >
        required_start_time)
    {
        return false;
    }

    if (imu_poses.back().timestamp <
        required_end_time)
    {
        return false;
    }

    // ====================================================================
    // 5. Get IMU pose at reference time
    // ====================================================================

    IMU_POSE reference_pose;

    if (!InterpolatePose(
            imu_poses,
            reference_time,
            reference_pose))
    {
        return false;
    }

    // ====================================================================
    // 6. Convert reference IMU pose into T_WI
    // ====================================================================

    Eigen::Isometry3d T_WI_ref =
        PoseToTransform(reference_pose);

    if (!use_translation)
    {
        T_WI_ref.translation().setZero();
    }

    // ====================================================================
    // 7. Compute LiDAR pose at reference time
    //
    // LiDAR -> IMU -> World
    //
    // T_WL = T_WI * T_IL
    // ====================================================================

    const Eigen::Isometry3d T_WL_ref =
        T_WI_ref *
        T_IL_;

    // World -> reference LiDAR
    //
    // This value is shared by every point, therefore calculate
    // the inverse only once.
    const Eigen::Isometry3d T_Lref_W =
        T_WL_ref.inverse();

    // ====================================================================
    // 8. Create temporary output point cloud
    //
    // Do not modify output_frame until the whole frame succeeds.
    // ====================================================================

    pcl::PointCloud<LIDAR_POINT>::Ptr deskewed_cloud =
        pcl::make_shared<
            pcl::PointCloud<LIDAR_POINT>>();

    deskewed_cloud->reserve(
        input_frame.cloud->size());

    // ====================================================================
    // 9. Deskew every LiDAR point
    // ====================================================================

    for (const LIDAR_POINT &point :
         input_frame.cloud->points)
    {
        // ------------------------------------------------------------
        // Absolute timestamp of this LiDAR point
        // ------------------------------------------------------------

        const double point_time =
            input_frame.scan_start_time +
            point.time_offset;

        // ------------------------------------------------------------
        // Interpolate IMU pose at this point timestamp
        // ------------------------------------------------------------

        IMU_POSE point_pose;

        if (!InterpolatePose(
                imu_poses,
                point_time,
                point_pose))
        {
            return false;
        }

        // ------------------------------------------------------------
        // IMU -> World
        // ------------------------------------------------------------

        Eigen::Isometry3d T_WI_point =
            PoseToTransform(point_pose);

        if (!use_translation)
        {
            T_WI_point.translation().setZero();
        }

        // ------------------------------------------------------------
        // LiDAR -> IMU -> World
        //
        // T_WL(t_i) =
        //     T_WI(t_i)
        //     *
        //     T_IL
        // ------------------------------------------------------------

        const Eigen::Isometry3d T_WL_point =
            T_WI_point *
            T_IL_;

        // ------------------------------------------------------------
        // Original LiDAR point
        // ------------------------------------------------------------

        const Eigen::Vector3d p_L(
            static_cast<double>(point.x),
            static_cast<double>(point.y),
            static_cast<double>(point.z));

        // ------------------------------------------------------------
        // Deskew
        //
        // First:
        //
        // p_W =
        //     T_WL(t_i)
        //     *
        //     p_L
        //
        // Then:
        //
        // p_Lref =
        //     T_WL(t_ref)^-1
        //     *
        //     p_W
        //
        // Therefore:
        //
        // p_Lref =
        //     T_WL(t_ref)^-1
        //     *
        //     T_WL(t_i)
        //     *
        //     p_L
        // ------------------------------------------------------------

        const Eigen::Vector3d p_L_ref =
            T_Lref_W *
            T_WL_point *
            p_L;

        // ------------------------------------------------------------
        // Keep intensity / ring / time_offset,
        // only replace XYZ.
        // ------------------------------------------------------------

        LIDAR_POINT corrected_point =
            point;

        corrected_point.x =
            static_cast<float>(
                p_L_ref.x());

        corrected_point.y =
            static_cast<float>(
                p_L_ref.y());

        corrected_point.z =
            static_cast<float>(
                p_L_ref.z());

        deskewed_cloud->push_back(
            corrected_point);
    }

    // ====================================================================
    // 10. Finish PCL cloud information
    // ====================================================================

    deskewed_cloud->width =
        static_cast<std::uint32_t>(
            deskewed_cloud->size());

    deskewed_cloud->height =
        1;

    deskewed_cloud->is_dense =
        input_frame.cloud->is_dense;

    // ====================================================================
    // 11. Only after all points succeed,
    //     write the final output frame.
    // ====================================================================

    output_frame.scan_start_time =
        input_frame.scan_start_time;

    output_frame.scan_duration =
        input_frame.scan_duration;

    output_frame.frame_id =
        input_frame.frame_id;

    output_frame.has_point_time =
        input_frame.has_point_time;

    output_frame.cloud =
        deskewed_cloud;

    return true;
}