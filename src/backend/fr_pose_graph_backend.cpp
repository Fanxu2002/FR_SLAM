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
