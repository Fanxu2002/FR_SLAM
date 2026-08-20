#include <iostream>
#include <iomanip>
#include <cmath>
#include <memory>
#include <string>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pcl/point_cloud.h>

#include "fr_slam/fr_point_types.hpp"
#include "fr_slam/fr_lidar_registration.hpp"

// ============================================================
// Create a simple 3D environment:
//
//      Wall Y
//        |
//        |
//        |
//        |________ Wall X
//       /
//      / Ground
//
// We deliberately use three approximately orthogonal planes,
// otherwise point-to-plane ICP may be geometrically degenerate.
// ============================================================

pcl::PointCloud<LIDAR_POINT>::Ptr CreateTargetCloud()
{
        pcl::PointCloud<LIDAR_POINT>::Ptr cloud =
            pcl::make_shared<
                pcl::PointCloud<LIDAR_POINT>>();

        // Point spacing
        const double resolution = 0.2;

        // ====================================================
        // 1. Ground plane
        //
        // z = 0
        // ====================================================

        for (double x = -3.0;
             x <= 3.0;
             x += resolution)
        {
                for (double y = -3.0;
                     y <= 3.0;
                     y += resolution)
                {
                        LIDAR_POINT point{};

                        point.x =
                            static_cast<float>(x);

                        point.y =
                            static_cast<float>(y);

                        point.z =
                            0.0f;

                        cloud->push_back(
                            point);
                }
        }

        // ====================================================
        // 2. Vertical wall
        //
        // x = 3
        // ====================================================

        for (double y = -3.0;
             y <= 3.0;
             y += resolution)
        {
                for (double z = 0.2;
                     z <= 3.0;
                     z += resolution)
                {
                        LIDAR_POINT point{};

                        point.x =
                            3.0f;

                        point.y =
                            static_cast<float>(y);

                        point.z =
                            static_cast<float>(z);

                        cloud->push_back(
                            point);
                }
        }

        // ====================================================
        // 3. Another vertical wall
        //
        // y = 3
        // ====================================================

        for (double x = -3.0;
             x <= 3.0;
             x += resolution)
        {
                for (double z = 0.2;
                     z <= 3.0;
                     z += resolution)
                {
                        LIDAR_POINT point{};

                        point.x =
                            static_cast<float>(x);

                        point.y =
                            3.0f;

                        point.z =
                            static_cast<float>(z);

                        cloud->push_back(
                            point);
                }
        }

        cloud->width =
            static_cast<std::uint32_t>(
                cloud->size());

        cloud->height = 1;

        cloud->is_dense = true;

        return cloud;
}

// ============================================================
// Generate source cloud from target cloud.
//
// Registration convention:
//
//      p_target
//          =
//      T_target_source * p_source
//
// Therefore:
//
//      p_source
//          =
//      T_target_source^-1 * p_target
//
// This is VERY IMPORTANT.
// ============================================================

pcl::PointCloud<LIDAR_POINT>::Ptr CreateSourceCloud(
    const pcl::PointCloud<LIDAR_POINT>::ConstPtr &target,
    const Eigen::Isometry3d &T_target_source_true)
{
        pcl::PointCloud<LIDAR_POINT>::Ptr source =
            pcl::make_shared<
                pcl::PointCloud<LIDAR_POINT>>();

        source->reserve(
            target->size());

        const Eigen::Isometry3d T_source_target =
            T_target_source_true.inverse();

        for (const LIDAR_POINT &target_point :
             target->points)
        {
                const Eigen::Vector3d p_target(
                    target_point.x,
                    target_point.y,
                    target_point.z);

                const Eigen::Vector3d p_source =
                    T_source_target *
                    p_target;

                LIDAR_POINT source_point{};

                source_point.x =
                    static_cast<float>(
                        p_source.x());

                source_point.y =
                    static_cast<float>(
                        p_source.y());

                source_point.z =
                    static_cast<float>(
                        p_source.z());

                source->push_back(
                    source_point);
        }

        source->width =
            static_cast<std::uint32_t>(
                source->size());

        source->height = 1;

        source->is_dense = true;

        return source;
}

// ============================================================
// Compute rotation error:
//
//      R_error = R_true^T * R_estimated
//
// Return angle in degrees.
// ============================================================

double RotationErrorDegree(
    const Eigen::Matrix3d &R_true,
    const Eigen::Matrix3d &R_estimated)
{
        Eigen::Matrix3d R_error =
            R_true.transpose() *
            R_estimated;

        Eigen::AngleAxisd angle_axis(
            R_error);

        constexpr double kRadToDeg =
            57.2957795130823208768;

        return std::abs(
                   angle_axis.angle()) *
               kRadToDeg;
}

// ============================================================
// Print SE(3)
// ============================================================

void PrintTransform(
    const std::string &name,
    const Eigen::Isometry3d &T)
{
        std::cout
            << "\n"
            << name
            << "\n";

        std::cout
            << "R =\n"
            << T.rotation()
            << "\n";

        std::cout
            << "t = ["
            << T.translation().x()
            << ", "
            << T.translation().y()
            << ", "
            << T.translation().z()
            << "]"
            << std::endl;
}

// ============================================================
// Run one registration test
// ============================================================

bool RunTest(
    const std::string &test_name,
    LidarRegistration &registration,
    const pcl::PointCloud<LIDAR_POINT>::ConstPtr &target,
    const Eigen::Isometry3d &T_target_source_true)
{
        std::cout
            << "\n\n========================================\n";

        std::cout
            << test_name
            << "\n";

        std::cout
            << "========================================\n";

        // ====================================================
        // Generate source from known ground truth transform
        // ====================================================

        pcl::PointCloud<LIDAR_POINT>::Ptr source =
            CreateSourceCloud(
                target,
                T_target_source_true);

        std::cout
            << "Target points: "
            << target->size()
            << std::endl;

        std::cout
            << "Source points: "
            << source->size()
            << std::endl;

        // ====================================================
        // Initial guess
        //
        // First test deliberately uses Identity.
        // ====================================================

        const Eigen::Isometry3d initial_guess =
            Eigen::Isometry3d::Identity();

        // ====================================================
        // Registration
        // ====================================================

        LidarRegistrationResult result;

        const bool success =
            registration.Align(
                source,
                target,
                initial_guess,
                result);

        if (!success)
        {
                std::cerr
                    << "Registration FAILED!"
                    << std::endl;

                return false;
        }

        // ====================================================
        // Print result
        // ====================================================

        PrintTransform(
            "Ground truth T_target_source:",
            T_target_source_true);

        PrintTransform(
            "Estimated T_target_source:",
            result.T_target_source);

        // ====================================================
        // Translation error
        // ====================================================

        const Eigen::Vector3d translation_error_vector =
            result.T_target_source.translation() -
            T_target_source_true.translation();

        const double translation_error =
            translation_error_vector.norm();

        // ====================================================
        // Rotation error
        // ====================================================

        const double rotation_error_degree =
            RotationErrorDegree(
                T_target_source_true.rotation(),
                result.T_target_source.rotation());

        // ====================================================
        // Result
        // ====================================================

        std::cout
            << "\n----------------------------------------\n";

        std::cout
            << "Converged: "
            << std::boolalpha
            << result.converged
            << "\n";

        std::cout
            << "Iterations: "
            << result.iterations
            << "\n";

        std::cout
            << "Correspondences: "
            << result.correspondences
            << "\n";

        std::cout
            << "RMSE: "
            << result.rmse
            << "\n";

        std::cout
            << "Translation error: "
            << translation_error
            << " m\n";

        std::cout
            << "Rotation error: "
            << rotation_error_degree
            << " deg\n";

        std::cout
            << "----------------------------------------\n";

        // ====================================================
        // PASS thresholds
        //
        // These are deliberately loose for the first test.
        // ====================================================

        const bool translation_pass =
            translation_error < 0.03;

        const bool rotation_pass =
            rotation_error_degree < 0.5;

        const bool pass =
            translation_pass &&
            rotation_pass;

        if (pass)
        {
                std::cout
                    << test_name
                    << " : PASS"
                    << std::endl;
        }
        else
        {
                std::cout
                    << test_name
                    << " : FAIL"
                    << std::endl;
        }

        return pass;
}

// ============================================================
// main
// ============================================================

int main()
{
        std::cout
            << std::fixed
            << std::setprecision(6);

        // ====================================================
        // Registration config
        // ====================================================

        LidarRegistrationConfig config;

        config.max_iterations = 30;

        config.knn = 5;

        config.max_correspondence_distance =
            1.0;

        config.max_plane_fit_error =
            0.05;

        config.max_point_to_plane_distance =
            0.5;

        config.min_correspondences =
            100;

        config.rotation_convergence_threshold =
            1e-6;

        config.translation_convergence_threshold =
            1e-6;

        LidarRegistration registration(
            config);

        // ====================================================
        // Target cloud
        // ====================================================

        pcl::PointCloud<LIDAR_POINT>::Ptr target =
            CreateTargetCloud();

        std::cout
            << "Synthetic target cloud created."
            << std::endl;

        std::cout
            << "Total target points: "
            << target->size()
            << std::endl;

        // ====================================================
        // TEST 1
        //
        // Pure translation
        // ====================================================

        Eigen::Isometry3d T_translation =
            Eigen::Isometry3d::Identity();

        T_translation.translation() =
            Eigen::Vector3d(
                0.10,
                0.05,
                0.02);

        const bool test_translation =
            RunTest(
                "TEST 1 - Pure Translation",
                registration,
                target,
                T_translation);

        // ====================================================
        // TEST 2
        //
        // Pure yaw rotation
        // ====================================================

        constexpr double kDegToRad =
            0.01745329251994329577;

        Eigen::Isometry3d T_rotation =
            Eigen::Isometry3d::Identity();

        T_rotation.linear() =
            Eigen::AngleAxisd(
                2.0 * kDegToRad,
                Eigen::Vector3d::UnitZ())
                .toRotationMatrix();

        const bool test_rotation =
            RunTest(
                "TEST 2 - Pure Rotation",
                registration,
                target,
                T_rotation);

        // ====================================================
        // TEST 3
        //
        // Rotation + translation
        // ====================================================

        Eigen::Isometry3d T_combined =
            Eigen::Isometry3d::Identity();

        const Eigen::AngleAxisd roll(
            1.0 * kDegToRad,
            Eigen::Vector3d::UnitX());

        const Eigen::AngleAxisd pitch(
            -1.5 * kDegToRad,
            Eigen::Vector3d::UnitY());

        const Eigen::AngleAxisd yaw(
            2.0 * kDegToRad,
            Eigen::Vector3d::UnitZ());

        T_combined.linear() =
            (yaw *
             pitch *
             roll)
                .toRotationMatrix();

        T_combined.translation() =
            Eigen::Vector3d(
                0.10,
                -0.05,
                0.03);

        const bool test_combined =
            RunTest(
                "TEST 3 - Rotation + Translation",
                registration,
                target,
                T_combined);

        // ====================================================
        // Final summary
        // ====================================================

        std::cout
            << "\n\n========================================\n";

        std::cout
            << "FINAL TEST SUMMARY"
            << "\n";

        std::cout
            << "========================================\n";

        std::cout
            << "Pure Translation: "
            << (test_translation ? "PASS" : "FAIL")
            << "\n";

        std::cout
            << "Pure Rotation:    "
            << (test_rotation ? "PASS" : "FAIL")
            << "\n";

        std::cout
            << "Combined Motion:  "
            << (test_combined ? "PASS" : "FAIL")
            << "\n";

        std::cout
            << "========================================\n";

        if (test_translation &&
            test_rotation &&
            test_combined)
        {
                std::cout
                    << "ALL REGISTRATION TESTS PASSED."
                    << std::endl;

                return 0;
        }

        std::cerr
            << "SOME REGISTRATION TESTS FAILED."
            << std::endl;

        return 1;
}