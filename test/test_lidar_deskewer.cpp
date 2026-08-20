#include "fr_slam/fr_lidar_deskewer.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

constexpr double K_PI = 3.14159265358979323846;
constexpr double K_TOLERANCE = 1e-5;

// ============================================================================
// Test 1
// Pure Translation
//
// World point:
//
//      P_W = [5, 0, 0]
//
// LiDAR:
//
//      t = 100.0   P = [0, 0, 0]
//      t = 101.0   P = [1, 0, 0]
//
// No rotation.
//
// Therefore the raw LiDAR measurements are:
//
//      t=0.00 -> [5.00, 0, 0]
//      t=0.25 -> [4.75, 0, 0]
//      t=0.50 -> [4.50, 0, 0]
//      t=0.75 -> [4.25, 0, 0]
//      t=1.00 -> [4.00, 0, 0]
//
// Deskewed result should all be:
//
//      [5, 0, 0]
// ============================================================================

bool TestPureTranslation(
    LidarDeskewer &deskewer)
{
        std::cout
            << "\n==================================================\n"
            << "Test 1: Pure Translation\n"
            << "==================================================\n";

        // ============================================================
        // 1. Create artificial IMU trajectory
        // ============================================================

        std::vector<IMU_POSE> imu_poses;

        IMU_POSE pose_start;

        pose_start.timestamp =
            100.0;

        pose_start.Q_WI =
            Eigen::Quaterniond::Identity();

        pose_start.P_WI =
            Eigen::Vector3d(
                0.0,
                0.0,
                0.0);

        pose_start.V_WI =
            Eigen::Vector3d(
                1.0,
                0.0,
                0.0);

        IMU_POSE pose_end;

        pose_end.timestamp =
            101.0;

        pose_end.Q_WI =
            Eigen::Quaterniond::Identity();

        pose_end.P_WI =
            Eigen::Vector3d(
                1.0,
                0.0,
                0.0);

        pose_end.V_WI =
            Eigen::Vector3d(
                1.0,
                0.0,
                0.0);

        imu_poses.push_back(
            pose_start);

        imu_poses.push_back(
            pose_end);

        // ============================================================
        // 2. Create artificial LiDAR frame
        // ============================================================

        LIDAR_FRAME input_frame;

        input_frame.scan_start_time =
            100.0;

        input_frame.scan_duration =
            1.0;

        input_frame.has_point_time =
            true;

        input_frame.frame_id =
            "translation_test";

        const std::vector<double> time_offsets = {
            0.00,
            0.25,
            0.50,
            0.75,
            1.00};

        for (const double time_offset :
             time_offsets)
        {
                LIDAR_POINT point;

                point.x =
                    static_cast<float>(
                        5.0 -
                        time_offset);

                point.y =
                    0.0f;

                point.z =
                    0.0f;

                point.intensity =
                    1.0f;

                point.ring =
                    0;

                point.time_offset =
                    time_offset;

                input_frame.cloud->push_back(
                    point);
        }

        input_frame.cloud->width =
            static_cast<std::uint32_t>(
                input_frame.cloud->size());

        input_frame.cloud->height =
            1;

        input_frame.cloud->is_dense =
            true;

        // ============================================================
        // 3. Print original points
        // ============================================================

        std::cout
            << "\n================ Original Points ================\n";

        for (const LIDAR_POINT &point :
             input_frame.cloud->points)
        {
                std::cout
                    << "time_offset = "
                    << point.time_offset
                    << " s, point = ["
                    << point.x << ", "
                    << point.y << ", "
                    << point.z << "]\n";
        }

        // ============================================================
        // 4. Deskew
        // ============================================================

        LIDAR_FRAME output_frame;

        const bool success =
            deskewer.Deskew(
                input_frame,
                imu_poses,
                output_frame);

        if (!success)
        {
                std::cerr
                    << "Pure translation deskew failed!\n";

                return false;
        }

        // ============================================================
        // 5. Check result
        // ============================================================

        std::cout
            << "\n================ Deskewed Points ================\n";

        bool all_correct =
            true;

        const Eigen::Vector3d expected(
            5.0,
            0.0,
            0.0);

        for (std::size_t i = 0;
             i < output_frame.cloud->size();
             ++i)
        {
                const LIDAR_POINT &point =
                    output_frame.cloud->points[i];

                const Eigen::Vector3d result(
                    static_cast<double>(point.x),
                    static_cast<double>(point.y),
                    static_cast<double>(point.z));

                const double error =
                    (result -
                     expected)
                        .norm();

                std::cout
                    << "Point "
                    << i
                    << ": ["
                    << point.x << ", "
                    << point.y << ", "
                    << point.z << "]"
                    << " error = "
                    << error
                    << "\n";

                if (error >
                    K_TOLERANCE)
                {
                        all_correct =
                            false;
                }
        }

        if (all_correct)
        {
                std::cout
                    << "\nPURE TRANSLATION TEST PASSED!\n";
        }
        else
        {
                std::cerr
                    << "\nPURE TRANSLATION TEST FAILED!\n";
        }

        return all_correct;
}

// ============================================================================
// Test 2
// Pure Rotation
//
// LiDAR stays at origin.
//
//      t = 200.0 -> yaw = 0 deg
//      t = 201.0 -> yaw = 90 deg
//
// World point:
//
//      P_W = [5, 0, 0]
//
// Deskewed points should all be:
//
//      [5, 0, 0]
// ============================================================================

bool TestPureRotation(
    LidarDeskewer &deskewer)
{
        std::cout
            << "\n==================================================\n"
            << "Test 2: Pure Rotation\n"
            << "==================================================\n";

        std::vector<IMU_POSE> imu_poses;

        IMU_POSE pose_start;

        pose_start.timestamp =
            200.0;

        pose_start.Q_WI =
            Eigen::Quaterniond::Identity();

        pose_start.P_WI =
            Eigen::Vector3d::Zero();

        pose_start.V_WI =
            Eigen::Vector3d::Zero();

        IMU_POSE pose_end;

        pose_end.timestamp =
            201.0;

        pose_end.Q_WI =
            Eigen::Quaterniond(
                Eigen::AngleAxisd(
                    K_PI / 2.0,
                    Eigen::Vector3d::UnitZ()));

        pose_end.Q_WI.normalize();

        pose_end.P_WI =
            Eigen::Vector3d::Zero();

        pose_end.V_WI =
            Eigen::Vector3d::Zero();

        imu_poses.push_back(
            pose_start);

        imu_poses.push_back(
            pose_end);

        LIDAR_FRAME input_frame;

        input_frame.scan_start_time =
            200.0;

        input_frame.scan_duration =
            1.0;

        input_frame.has_point_time =
            true;

        input_frame.frame_id =
            "rotation_test";

        const Eigen::Vector3d world_point(
            5.0,
            0.0,
            0.0);

        const std::vector<double> time_offsets = {
            0.00,
            0.25,
            0.50,
            0.75,
            1.00};

        for (const double time_offset :
             time_offsets)
        {
                const double yaw =
                    time_offset *
                    K_PI /
                    2.0;

                const Eigen::Quaterniond Q_WL(
                    Eigen::AngleAxisd(
                        yaw,
                        Eigen::Vector3d::UnitZ()));

                // ====================================================
                // World -> LiDAR
                //
                // p_L = R_WL^T * p_W
                // ====================================================

                const Eigen::Vector3d point_L =
                    Q_WL.conjugate() *
                    world_point;

                LIDAR_POINT point;

                point.x =
                    static_cast<float>(
                        point_L.x());

                point.y =
                    static_cast<float>(
                        point_L.y());

                point.z =
                    static_cast<float>(
                        point_L.z());

                point.intensity =
                    1.0f;

                point.ring =
                    0;

                point.time_offset =
                    time_offset;

                input_frame.cloud->push_back(
                    point);
        }

        input_frame.cloud->width =
            static_cast<std::uint32_t>(
                input_frame.cloud->size());

        input_frame.cloud->height =
            1;

        input_frame.cloud->is_dense =
            true;

        std::cout
            << "\n================ Original Rotation Points ================\n";

        for (const LIDAR_POINT &point :
             input_frame.cloud->points)
        {
                std::cout
                    << "time_offset = "
                    << point.time_offset
                    << " s, point = ["
                    << point.x << ", "
                    << point.y << ", "
                    << point.z << "]\n";
        }

        LIDAR_FRAME output_frame;

        const bool success =
            deskewer.Deskew(
                input_frame,
                imu_poses,
                output_frame);

        if (!success)
        {
                std::cerr
                    << "Pure rotation deskew failed!\n";

                return false;
        }

        std::cout
            << "\n================ Rotation Deskew Result ================\n";

        bool all_correct =
            true;

        const Eigen::Vector3d expected(
            5.0,
            0.0,
            0.0);

        for (std::size_t i = 0;
             i < output_frame.cloud->size();
             ++i)
        {
                const LIDAR_POINT &point =
                    output_frame.cloud->points[i];

                const Eigen::Vector3d result(
                    static_cast<double>(point.x),
                    static_cast<double>(point.y),
                    static_cast<double>(point.z));

                const double error =
                    (result -
                     expected)
                        .norm();

                std::cout
                    << "Point "
                    << i
                    << ": ["
                    << point.x << ", "
                    << point.y << ", "
                    << point.z << "]"
                    << " error = "
                    << error
                    << "\n";

                if (error >
                    K_TOLERANCE)
                {
                        all_correct =
                            false;
                }
        }

        if (all_correct)
        {
                std::cout
                    << "\nPURE ROTATION TEST PASSED!\n";
        }
        else
        {
                std::cerr
                    << "\nPURE ROTATION TEST FAILED!\n";
        }

        return all_correct;
}

// ============================================================================
// Test 3
// Translation + Rotation
//
//      t = 300.0
//      P = [0,0,0]
//      yaw = 0 deg
//
//              ↓
//
//      t = 301.0
//      P = [1,0,0]
//      yaw = 90 deg
//
// Fixed world point:
//
//      P_W = [5,2,0]
//
// Raw measurement:
//
//      p_L = R_WL^T * (p_W - P_WL)
//
// Deskewed result should all become:
//
//      [5,2,0]
// ============================================================================

bool TestTranslationAndRotation(
    LidarDeskewer &deskewer)
{
        std::cout
            << "\n==================================================\n"
            << "Test 3: Translation + Rotation\n"
            << "==================================================\n";

        std::vector<IMU_POSE> imu_poses;

        // ============================================================
        // Start pose
        // ============================================================

        IMU_POSE pose_start;

        pose_start.timestamp =
            300.0;

        pose_start.Q_WI =
            Eigen::Quaterniond::Identity();

        pose_start.P_WI =
            Eigen::Vector3d(
                0.0,
                0.0,
                0.0);

        pose_start.V_WI =
            Eigen::Vector3d(
                1.0,
                0.0,
                0.0);

        // ============================================================
        // End pose
        // ============================================================

        IMU_POSE pose_end;

        pose_end.timestamp =
            301.0;

        pose_end.Q_WI =
            Eigen::Quaterniond(
                Eigen::AngleAxisd(
                    K_PI / 2.0,
                    Eigen::Vector3d::UnitZ()));

        pose_end.Q_WI.normalize();

        pose_end.P_WI =
            Eigen::Vector3d(
                1.0,
                0.0,
                0.0);

        pose_end.V_WI =
            Eigen::Vector3d(
                1.0,
                0.0,
                0.0);

        imu_poses.push_back(
            pose_start);

        imu_poses.push_back(
            pose_end);

        // ============================================================
        // LiDAR frame
        // ============================================================

        LIDAR_FRAME input_frame;

        input_frame.scan_start_time =
            300.0;

        input_frame.scan_duration =
            1.0;

        input_frame.has_point_time =
            true;

        input_frame.frame_id =
            "se3_test";

        // ============================================================
        // Fixed world point
        // ============================================================

        const Eigen::Vector3d world_point(
            5.0,
            2.0,
            0.0);

        const std::vector<double> time_offsets = {
            0.00,
            0.25,
            0.50,
            0.75,
            1.00};

        // ============================================================
        // Generate raw LiDAR points
        // ============================================================

        for (const double time_offset :
             time_offsets)
        {
                const double alpha =
                    time_offset;

                // ----------------------------------------------------
                // Translation:
                //
                // [0,0,0] -> [1,0,0]
                // ----------------------------------------------------

                const Eigen::Vector3d P_WL(
                    alpha,
                    0.0,
                    0.0);

                // ----------------------------------------------------
                // Rotation:
                //
                // yaw 0 -> 90 deg
                // ----------------------------------------------------

                const double yaw =
                    alpha *
                    K_PI /
                    2.0;

                const Eigen::Quaterniond Q_WL(
                    Eigen::AngleAxisd(
                        yaw,
                        Eigen::Vector3d::UnitZ()));

                // ----------------------------------------------------
                // p_W = R_WL * p_L + P_WL
                //
                // Therefore:
                //
                // p_L =
                // R_WL^T * (p_W - P_WL)
                // ----------------------------------------------------

                const Eigen::Vector3d point_L =
                    Q_WL.conjugate() *
                    (world_point -
                     P_WL);

                LIDAR_POINT point;

                point.x =
                    static_cast<float>(
                        point_L.x());

                point.y =
                    static_cast<float>(
                        point_L.y());

                point.z =
                    static_cast<float>(
                        point_L.z());

                point.intensity =
                    1.0f;

                point.ring =
                    0;

                point.time_offset =
                    time_offset;

                input_frame.cloud->push_back(
                    point);
        }

        input_frame.cloud->width =
            static_cast<std::uint32_t>(
                input_frame.cloud->size());

        input_frame.cloud->height =
            1;

        input_frame.cloud->is_dense =
            true;

        // ============================================================
        // Print original points
        // ============================================================

        std::cout
            << "\n================ Original SE3 Points ================\n";

        for (const LIDAR_POINT &point :
             input_frame.cloud->points)
        {
                std::cout
                    << "time_offset = "
                    << point.time_offset
                    << " s, point = ["
                    << point.x << ", "
                    << point.y << ", "
                    << point.z << "]\n";
        }

        // ============================================================
        // Deskew
        // ============================================================

        LIDAR_FRAME output_frame;

        const bool success =
            deskewer.Deskew(
                input_frame,
                imu_poses,
                output_frame);

        if (!success)
        {
                std::cerr
                    << "Translation + rotation deskew failed!\n";

                return false;
        }

        // ============================================================
        // Check result
        // ============================================================

        std::cout
            << "\n================ SE3 Deskew Result ================\n";

        bool all_correct =
            true;

        const Eigen::Vector3d expected(
            5.0,
            2.0,
            0.0);

        for (std::size_t i = 0;
             i < output_frame.cloud->size();
             ++i)
        {
                const LIDAR_POINT &point =
                    output_frame.cloud->points[i];

                const Eigen::Vector3d result(
                    static_cast<double>(point.x),
                    static_cast<double>(point.y),
                    static_cast<double>(point.z));

                const double error =
                    (result -
                     expected)
                        .norm();

                std::cout
                    << "Point "
                    << i
                    << ": ["
                    << point.x << ", "
                    << point.y << ", "
                    << point.z << "]"
                    << " error = "
                    << error
                    << "\n";

                if (error >
                    K_TOLERANCE)
                {
                        all_correct =
                            false;
                }
        }

        if (all_correct)
        {
                std::cout
                    << "\nTRANSLATION + ROTATION TEST PASSED!\n";
        }
        else
        {
                std::cerr
                    << "\nTRANSLATION + ROTATION TEST FAILED!\n";
        }

        return all_correct;
}

// ============================================================================
// main
// ============================================================================

int main()
{
        LidarDeskewer deskewer;

        // ============================================================
        // Artificial tests:
        //
        // LiDAR frame == IMU frame
        //
        // Therefore:
        //
        // T_IL = Identity
        //
        // We deliberately do NOT use the real Mid-360S extrinsic here,
        // because these tests only verify deskew mathematics.
        // ============================================================

        const Eigen::Quaterniond Q_IL =
            Eigen::Quaterniond::Identity();

        const Eigen::Vector3d P_IL =
            Eigen::Vector3d::Zero();

        deskewer.SetExtrinsic(
            Q_IL,
            P_IL);

        // ============================================================
        // Run tests
        // ============================================================

        const bool translation_ok =
            TestPureTranslation(
                deskewer);

        const bool rotation_ok =
            TestPureRotation(
                deskewer);

        const bool se3_ok =
            TestTranslationAndRotation(
                deskewer);

        // ============================================================
        // Final result
        // ============================================================

        std::cout
            << "\n==================================================\n"
            << "Final Test Result\n"
            << "==================================================\n";

        std::cout
            << "Pure translation:       "
            << (translation_ok ? "PASS" : "FAIL")
            << "\n";

        std::cout
            << "Pure rotation:          "
            << (rotation_ok ? "PASS" : "FAIL")
            << "\n";

        std::cout
            << "Translation + rotation: "
            << (se3_ok ? "PASS" : "FAIL")
            << "\n";

        if (translation_ok &&
            rotation_ok &&
            se3_ok)
        {
                std::cout
                    << "\nALL LIDAR DESKEW TESTS PASSED!\n";

                return 0;
        }

        std::cerr
            << "\nLIDAR DESKEW TEST FAILED!\n";

        return 1;
}