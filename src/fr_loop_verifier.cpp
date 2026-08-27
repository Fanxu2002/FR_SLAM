#include "fr_slam/fr_loop_verifier.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>

namespace
{

    constexpr double kPi =
        3.14159265358979323846;

    // ----------------------------------------------------------------------------
    // Convert one project cloud into standard PointXYZ.
    //
    // Loop verification deliberately uses standard PCL PointXYZ so it remains
    // independent from custom-point template registration issues.
    // ----------------------------------------------------------------------------
    pcl::PointCloud<pcl::PointXYZ>::Ptr
    ConvertToXYZ(
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &cloud)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr result(
            new pcl::PointCloud<pcl::PointXYZ>);

        if (!cloud)
        {
            return result;
        }

        result->reserve(
            cloud->size());

        for (const LIDAR_POINT &point :
             cloud->points)
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

            result->push_back(
                xyz);
        }

        result->width =
            static_cast<std::uint32_t>(
                result->size());

        result->height = 1;
        result->is_dense = true;

        return result;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr
    VoxelFilter(
        const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud,
        double leaf_size)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(
            new pcl::PointCloud<pcl::PointXYZ>);

        if (!cloud ||
            cloud->empty())
        {
            return filtered;
        }

        pcl::VoxelGrid<pcl::PointXYZ> voxel;
        voxel.setInputCloud(
            cloud);

        const float leaf =
            static_cast<float>(
                leaf_size);

        voxel.setLeafSize(
            leaf,
            leaf,
            leaf);

        voxel.filter(
            *filtered);

        return filtered;
    }

    double NormalizeAngleRad(
        double angle)
    {
        while (angle > kPi)
        {
            angle -=
                2.0 * kPi;
        }

        while (angle < -kPi)
        {
            angle +=
                2.0 * kPi;
        }

        return angle;
    }

    double YawFromRotation(
        const Eigen::Matrix3d &rotation)
    {
        return std::atan2(
            rotation(1, 0),
            rotation(0, 0));
    }

    double RelativeRotationDeg(
        const Eigen::Isometry3d &T_A,
        const Eigen::Isometry3d &T_B)
    {
        if (!T_A.matrix().allFinite() ||
            !T_B.matrix().allFinite())
        {
            return std::numeric_limits<double>::infinity();
        }

        Eigen::Quaterniond q(
            T_A.rotation().transpose() *
            T_B.rotation());

        if (!q.coeffs().allFinite() ||
            q.norm() < 1.0e-12)
        {
            return std::numeric_limits<double>::infinity();
        }

        q.normalize();

        const double w =
            std::clamp(
                std::abs(q.w()),
                0.0,
                1.0);

        return 2.0 *
               std::acos(w) *
               180.0 /
               kPi;
    }

    // ----------------------------------------------------------------------------
    // Replace only the yaw part of one relative rotation.
    //
    // We preserve the graph-derived tilt component:
    //
    //     R_graph = Rz(yaw_graph) * R_tilt
    //
    // then create:
    //
    //     R_new = Rz(yaw_new) * R_tilt
    //
    // This is safer than multiplying Scan Context yaw on top of the graph yaw,
    // which could accidentally turn an already-correct 180-degree graph yaw into
    // 360 degrees.
    // ----------------------------------------------------------------------------
    Eigen::Isometry3d ReplaceYaw(
        const Eigen::Isometry3d &graph_guess,
        double yaw_new)
    {
        Eigen::Isometry3d result =
            graph_guess;

        const double yaw_graph =
            YawFromRotation(
                graph_guess.rotation());

        const Eigen::Matrix3d Rz_graph =
            Eigen::AngleAxisd(
                yaw_graph,
                Eigen::Vector3d::UnitZ())
                .toRotationMatrix();

        const Eigen::Matrix3d R_tilt =
            Rz_graph.transpose() *
            graph_guess.rotation();

        const Eigen::Matrix3d Rz_new =
            Eigen::AngleAxisd(
                yaw_new,
                Eigen::Vector3d::UnitZ())
                .toRotationMatrix();

        result.linear() =
            Rz_new *
            R_tilt;

        return result;
    }

    // ----------------------------------------------------------------------------
    // MakeScanContextInitialGuess()
    //
    // Scan Context gives us a yaw hypothesis, but it does NOT give us a reliable
    // translation.
    //
    // The PoseGraph translation may already contain tens of meters of odometry
    // drift. Reusing that translation defeats the purpose of Scan Context.
    //
    // Therefore:
    //
    //     GRAPH hypothesis:
    //         rotation    = graph rotation
    //         translation = graph translation
    //
    //     SCAN CONTEXT hypothesis:
    //         rotation    = Scan Context yaw
    //         translation = zero
    //
    // The zero translation is only an ICP starting hypothesis.
    // ICP is still free to estimate the final relative translation.
    // ----------------------------------------------------------------------------
    Eigen::Isometry3d MakeScanContextInitialGuess(
        const Eigen::Isometry3d &graph_guess,
        double scan_context_yaw)
    {
        Eigen::Isometry3d result =
            ReplaceYaw(
                graph_guess,
                scan_context_yaw);

        // IMPORTANT:
        //
        // Do not inherit the potentially badly drifted PoseGraph translation.
        result.translation().setZero();

        return result;
    }

    // ----------------------------------------------------------------------------
    // Explicit geometry quality measurement.
    //
    // After ICP finishes, transform every source point into target coordinates,
    // find its nearest target point, and count inliers inside a strict distance.
    //
    // This gives us metrics that are easier to interpret than relying on the PCL
    // fitness score alone.
    // ----------------------------------------------------------------------------
    bool EvaluateAlignment(
        const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &source,
        const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &target,
        const Eigen::Isometry3d &T_target_source,
        double inlier_distance,
        std::size_t &inliers,
        double &overlap_ratio,
        double &rmse)
    {
        inliers = 0;
        overlap_ratio = 0.0;
        rmse =
            std::numeric_limits<double>::infinity();

        if (!source ||
            !target ||
            source->empty() ||
            target->empty() ||
            !T_target_source.matrix().allFinite() ||
            !std::isfinite(inlier_distance) ||
            inlier_distance <= 0.0)
        {
            return false;
        }

        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
        kdtree.setInputCloud(
            target);

        const double max_squared_distance =
            inlier_distance *
            inlier_distance;

        double squared_error_sum = 0.0;

        std::vector<int> nearest_index(1);
        std::vector<float> nearest_squared_distance(1);

        for (const pcl::PointXYZ &point_source :
             source->points)
        {
            const Eigen::Vector3d p_source(
                static_cast<double>(point_source.x),
                static_cast<double>(point_source.y),
                static_cast<double>(point_source.z));

            const Eigen::Vector3d p_target =
                T_target_source *
                p_source;

            if (!p_target.allFinite())
            {
                continue;
            }

            pcl::PointXYZ query;
            query.x =
                static_cast<float>(
                    p_target.x());
            query.y =
                static_cast<float>(
                    p_target.y());
            query.z =
                static_cast<float>(
                    p_target.z());

            if (kdtree.nearestKSearch(
                    query,
                    1,
                    nearest_index,
                    nearest_squared_distance) <= 0)
            {
                continue;
            }

            const double squared_distance =
                static_cast<double>(
                    nearest_squared_distance[0]);

            if (!std::isfinite(
                    squared_distance) ||
                squared_distance >
                    max_squared_distance)
            {
                continue;
            }

            squared_error_sum +=
                squared_distance;

            ++inliers;
        }

        overlap_ratio =
            static_cast<double>(
                inliers) /
            static_cast<double>(
                source->size());

        if (inliers == 0)
        {
            return true;
        }

        rmse =
            std::sqrt(
                squared_error_sum /
                static_cast<double>(
                    inliers));

        return std::isfinite(rmse);
    }

    // ----------------------------------------------------------------------------
    // Run one ICP hypothesis.
    // ----------------------------------------------------------------------------
    LoopVerificationResult RunHypothesis(
        const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &source,
        const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &target,
        const Eigen::Isometry3d &initial_guess,
        LoopVerifierHypothesis hypothesis,
        const LoopVerifierConfig &config)
    {
        LoopVerificationResult result;
        result.initial_guess =
            initial_guess;
        result.T_target_source =
            initial_guess;
        result.hypothesis =
            hypothesis;
        result.source_points =
            source ? source->size() : 0;
        result.target_points =
            target ? target->size() : 0;

        if (!source ||
            !target ||
            source->size() <
                config.min_cloud_points ||
            target->size() <
                config.min_cloud_points ||
            !initial_guess.matrix().allFinite())
        {
            return result;
        }

        pcl::IterativeClosestPoint<
            pcl::PointXYZ,
            pcl::PointXYZ>
            icp;

        icp.setInputSource(
            source);

        icp.setInputTarget(
            target);

        icp.setMaximumIterations(
            static_cast<int>(
                config.max_iterations));

        icp.setMaxCorrespondenceDistance(
            config.max_correspondence_distance);

        icp.setTransformationEpsilon(
            config.transformation_epsilon);

        icp.setEuclideanFitnessEpsilon(
            config.euclidean_fitness_epsilon);

        pcl::PointCloud<pcl::PointXYZ> aligned;

        icp.align(
            aligned,
            initial_guess
                .matrix()
                .cast<float>());

        result.converged =
            icp.hasConverged();

        if (!result.converged)
        {
            return result;
        }

        const Eigen::Matrix4d final_matrix =
            icp.getFinalTransformation()
                .cast<double>();

        if (!final_matrix.allFinite())
        {
            return result;
        }

        result.T_target_source =
            Eigen::Isometry3d::Identity();

        result.T_target_source.matrix() =
            final_matrix;

        if (!result.T_target_source
                 .matrix()
                 .allFinite())
        {
            return result;
        }

        result.fitness_score =
            icp.getFitnessScore();

        const bool metrics_ok =
            EvaluateAlignment(
                source,
                target,
                result.T_target_source,
                config.verification_inlier_distance,
                result.inliers,
                result.overlap_ratio,
                result.rmse);

        if (!metrics_ok)
        {
            return result;
        }

        result.correction_translation =
            (result.T_target_source.translation() -
             initial_guess.translation())
                .norm();

        result.correction_rotation_deg =
            RelativeRotationDeg(
                initial_guess,
                result.T_target_source);

        result.success = true;

        // ------------------------------------------------------------------------
        // Final geometry gate.
        //
        // V1 checked only inlier count, overlap and RMSE. That was not enough:
        // repeated environments can converge to a geometrically plausible but
        // completely wrong local minimum. We therefore also limit how far ICP is
        // allowed to move away from the selected initial hypothesis.
        // ------------------------------------------------------------------------
        result.accepted =
            result.inliers >=
                config.min_inliers &&
            result.overlap_ratio >=
                config.min_overlap_ratio &&
            std::isfinite(
                result.rmse) &&
            result.rmse <=
                config.max_rmse &&
            std::isfinite(
                result.correction_translation) &&
            result.correction_translation <=
                config.max_correction_translation &&
            std::isfinite(
                result.correction_rotation_deg) &&
            result.correction_rotation_deg <=
                config.max_correction_rotation_deg;

        return result;
    }

    // ----------------------------------------------------------------------------
    // Decide which hypothesis result is more useful.
    //
    // Accepted beats rejected.
    // Then prefer larger verified overlap.
    // Finally prefer lower RMSE.
    // ----------------------------------------------------------------------------
    bool IsBetterResult(
        const LoopVerificationResult &candidate,
        const LoopVerificationResult &current_best)
    {
        if (!candidate.success)
        {
            return false;
        }

        if (!current_best.success)
        {
            return true;
        }

        if (candidate.accepted !=
            current_best.accepted)
        {
            return candidate.accepted;
        }

        if (candidate.overlap_ratio !=
            current_best.overlap_ratio)
        {
            return candidate.overlap_ratio >
                   current_best.overlap_ratio;
        }

        return candidate.rmse <
               current_best.rmse;
    }

} // namespace

LoopVerifier::LoopVerifier(
    const LoopVerifierConfig &config)
    : config_(config)
{
    if (!std::isfinite(
            config_.voxel_leaf_size) ||
        config_.voxel_leaf_size <= 0.0)
    {
        config_.voxel_leaf_size = 0.50;
    }

    if (config_.max_iterations == 0)
    {
        config_.max_iterations = 50;
    }

    if (!std::isfinite(
            config_.max_correspondence_distance) ||
        config_.max_correspondence_distance <= 0.0)
    {
        config_.max_correspondence_distance =
            5.0;
    }

    if (!std::isfinite(
            config_.transformation_epsilon) ||
        config_.transformation_epsilon <= 0.0)
    {
        config_.transformation_epsilon =
            1.0e-6;
    }

    if (!std::isfinite(
            config_.euclidean_fitness_epsilon) ||
        config_.euclidean_fitness_epsilon <= 0.0)
    {
        config_.euclidean_fitness_epsilon =
            1.0e-5;
    }

    if (!std::isfinite(
            config_.verification_inlier_distance) ||
        config_.verification_inlier_distance <= 0.0)
    {
        config_.verification_inlier_distance =
            1.0;
    }

    if (!std::isfinite(
            config_.max_rmse) ||
        config_.max_rmse <= 0.0)
    {
        config_.max_rmse = 0.65;
    }

    if (config_.min_inliers == 0)
    {
        config_.min_inliers = 300;
    }

    if (!std::isfinite(
            config_.min_overlap_ratio) ||
        config_.min_overlap_ratio <= 0.0 ||
        config_.min_overlap_ratio > 1.0)
    {
        config_.min_overlap_ratio =
            0.15;
    }

    if (config_.min_cloud_points == 0)
    {
        config_.min_cloud_points = 300;
    }

    // ------------------------------------------------------------------------
    // Correction gates.
    //
    // These defaults are intentionally loose for the first real-data test.
    // They mainly reject catastrophic hypotheses such as tens of meters of
    // translation correction or ~180 degree rotation correction.
    //
    // NOTE:
    // These two fields must exist in LoopVerifierConfig:
    //
    //     double max_correction_translation = 15.0;
    //     double max_correction_rotation_deg = 45.0;
    // ------------------------------------------------------------------------
    if (!std::isfinite(
            config_.max_correction_translation) ||
        config_.max_correction_translation <= 0.0)
    {
        config_.max_correction_translation =
            15.0;
    }

    if (!std::isfinite(
            config_.max_correction_rotation_deg) ||
        config_.max_correction_rotation_deg <= 0.0)
    {
        config_.max_correction_rotation_deg =
            45.0;
    }
}

bool LoopVerifier::Verify(
    const pcl::PointCloud<LIDAR_POINT>::ConstPtr &source_current,
    const pcl::PointCloud<LIDAR_POINT>::ConstPtr &target_historical,
    const Eigen::Isometry3d &graph_initial_guess,
    double scan_context_yaw_shift_deg,
    LoopVerificationResult &result) const
{
    result =
        LoopVerificationResult();

    if (!source_current ||
        !target_historical ||
        source_current->empty() ||
        target_historical->empty() ||
        !graph_initial_guess
             .matrix()
             .allFinite())
    {
        return false;
    }

    const pcl::PointCloud<pcl::PointXYZ>::Ptr
        source_xyz =
            ConvertToXYZ(
                source_current);

    const pcl::PointCloud<pcl::PointXYZ>::Ptr
        target_xyz =
            ConvertToXYZ(
                target_historical);

    const pcl::PointCloud<pcl::PointXYZ>::Ptr
        source_filtered =
            VoxelFilter(
                source_xyz,
                config_.voxel_leaf_size);

    const pcl::PointCloud<pcl::PointXYZ>::Ptr
        target_filtered =
            VoxelFilter(
                target_xyz,
                config_.voxel_leaf_size);

    if (!source_filtered ||
        !target_filtered ||
        source_filtered->size() <
            config_.min_cloud_points ||
        target_filtered->size() <
            config_.min_cloud_points)
    {
        return false;
    }

    std::vector<
        std::pair<
            LoopVerifierHypothesis,
            Eigen::Isometry3d>>
        hypotheses;

    hypotheses.emplace_back(
        LoopVerifierHypothesis::GraphPose,
        graph_initial_guess);

    if (std::isfinite(
            scan_context_yaw_shift_deg))
    {
        const double yaw_shift_rad =
            NormalizeAngleRad(
                scan_context_yaw_shift_deg *
                kPi /
                180.0);

        const Eigen::Isometry3d positive_yaw_guess =
            MakeScanContextInitialGuess(
                graph_initial_guess,
                yaw_shift_rad);

        const Eigen::Isometry3d negative_yaw_guess =
            MakeScanContextInitialGuess(
                graph_initial_guess,
                -yaw_shift_rad);

        hypotheses.emplace_back(
            LoopVerifierHypothesis::
                ScanContextPositiveYaw,
            positive_yaw_guess);

        hypotheses.emplace_back(
            LoopVerifierHypothesis::
                ScanContextNegativeYaw,
            negative_yaw_guess);
    }

    LoopVerificationResult best;

    for (const auto &entry :
         hypotheses)
    {
        const LoopVerificationResult candidate =
            RunHypothesis(
                source_filtered,
                target_filtered,
                entry.second,
                entry.first,
                config_);

        if (IsBetterResult(
                candidate,
                best))
        {
            best =
                candidate;
        }
    }

    result =
        best;

    return result.success;
}

const LoopVerifierConfig &
LoopVerifier::GetConfig() const
{
    return config_;
}

const char *LoopVerifier::HypothesisName(
    LoopVerifierHypothesis hypothesis)
{
    switch (hypothesis)
    {
    case LoopVerifierHypothesis::GraphPose:
        return "GRAPH";

    case LoopVerifierHypothesis::
        ScanContextPositiveYaw:
        return "SCAN_CONTEXT_POSITIVE_YAW";

    case LoopVerifierHypothesis::
        ScanContextNegativeYaw:
        return "SCAN_CONTEXT_NEGATIVE_YAW";

    default:
        return "UNKNOWN";
    }
}
