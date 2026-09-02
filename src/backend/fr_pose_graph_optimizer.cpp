#include "fr_slam/backend/fr_pose_graph_optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>
#include <Eigen/StdVector>
#include <Eigen/Cholesky>

#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/slam3d/edge_se3.h>
#include <g2o/types/slam3d/vertex_se3.h>

namespace
{
    constexpr double kPi = 3.14159265358979323846;
    constexpr double kRadToDeg = 180.0 / kPi;

    using AlignedVector2d = std::vector<Eigen::Vector2d,
                                        Eigen::aligned_allocator<Eigen::Vector2d>>;

    // ============================================================================
    // Gravity-direction unary factor.
    //
    // State convention:
    //     R_WK maps Keyframe/LiDAR coordinates into backend map coordinates.
    //
    // For a map-frame vertical direction z_W = [0,0,1]^T, the same direction
    // expressed in the Keyframe/LiDAR frame is:
    //
    //     g_K = R_WK^T * z_W
    //
    // The measurement is the immutable frontend value generated from raw T_WL.
    // A pure world-yaw correction leaves g_K unchanged, while an artificial
    // roll/pitch change rotates it.  Therefore this factor protects tilt without
    // preventing the PoseGraph from correcting global yaw.
    // ============================================================================
    class EdgeGravityDirection : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, g2o::VertexSE3>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        EdgeGravityDirection()
        {
            information().setIdentity();
        }

        void computeError() override
        {
            const g2o::VertexSE3 *vertex = static_cast<const g2o::VertexSE3 *>(_vertices[0]);
            if (vertex == nullptr)
            {
                _error.setConstant(
                    std::numeric_limits<double>::quiet_NaN());
                return;
            }

            Eigen::Vector3d gravity_estimated = vertex->estimate().rotation().transpose() *
                                                Eigen::Vector3d::UnitZ();

            const double estimated_norm = gravity_estimated.norm();

            Eigen::Vector3d gravity_measurement = _measurement;

            const double measurement_norm = gravity_measurement.norm();

            if (!gravity_estimated.allFinite() ||
                !gravity_measurement.allFinite() ||
                estimated_norm < 1.0e-12 ||
                measurement_norm < 1.0e-12)
            {
                _error.setConstant(
                    std::numeric_limits<double>::quiet_NaN());
                return;
            }

            gravity_estimated /= estimated_norm;
            gravity_measurement /= measurement_norm;

            // Difference of two unit gravity directions.  For small tilt angles,
            // ||error|| is approximately the angular error in radians.
            _error = gravity_estimated - gravity_measurement;
        }

        bool read(std::istream &) override
        {
            return false;
        }

        bool write(std::ostream &) const override
        {
            return false;
        }
    };

    bool ToG2oId(std::size_t id, int &g2o_id)
    {
        if (id > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        {
            return false;
        }

        g2o_id = static_cast<int>(id);
        return true;
    }

    bool IsFinitePositiveSemidefiniteInformation(const Eigen::Matrix<double, 6, 6> &information)
    {
        if (!information.allFinite())
        {
            return false;
        }

        const double asymmetry =
            (information -
             information.transpose())
                .cwiseAbs()
                .maxCoeff();

        if (!std::isfinite(asymmetry) || asymmetry > 1.0e-8)
        {
            return false;
        }

        const Eigen::Matrix<double, 6, 6> symmetric_information = 0.5 * (information + information.transpose());

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> solver(symmetric_information,
                                                                          Eigen::EigenvaluesOnly);

        if (solver.info() != Eigen::Success)
        {
            return false;
        }

        const Eigen::Matrix<double, 6, 1> eigenvalues = solver.eigenvalues();

        if (!eigenvalues.allFinite())
        {
            return false;
        }

        // g2o requires a valid information matrix.  For the current V2 experiment
        // we deliberately require strict positive definiteness rather than merely
        // checking positive diagonal entries.
        return eigenvalues.minCoeff() > 1.0e-9;
    }

    // ============================================================================
    // RescaleSe3Information()
    //
    // g2o::EdgeSE3 uses a 6D residual ordered as translation followed by rotation.
    // We preserve the base information matrix supplied by PoseGraph and apply a
    // congruence transform
    //
    //     Omega_scaled = D * Omega_base * D
    //
    // where D contains sqrt(weight).  For an Identity base information matrix this
    // is exactly diag([w_t,w_t,w_t,w_r,w_r,w_r]).  Using the congruence transform
    // also preserves symmetry / positive definiteness if non-zero cross terms are
    // introduced later.
    // ============================================================================
    Eigen::Matrix<double, 6, 6> RescaleSe3Information(const Eigen::Matrix<double, 6, 6> &base_information,
                                                      double translation_weight,
                                                      double rotation_weight)
    {
        Eigen::Matrix<double, 6, 6> scale = Eigen::Matrix<double, 6, 6>::Identity();

        const double translation_scale = std::sqrt(translation_weight);

        const double rotation_scale = std::sqrt(rotation_weight);

        scale(0, 0) = translation_scale;
        scale(1, 1) = translation_scale;
        scale(2, 2) = translation_scale;

        scale(3, 3) = rotation_scale;
        scale(4, 4) = rotation_scale;
        scale(5, 5) = rotation_scale;

        return scale * base_information * scale;
    }

    struct TranslationStiffnessDiagnostics
    {
        bool valid = false;

        double direct_world_x = 0.0;
        double direct_world_y = 0.0;
        double direct_world_z = 0.0;

        double effective_world_x = 0.0;
        double effective_world_y = 0.0;
        double effective_world_z = 0.0;

        double relative_world_x = 0.0;
        double relative_world_y = 0.0;
        double relative_world_z = 0.0;

        double weak_eigenvalue = 0.0;
        double weak_axis_world_z_alignment = 0.0;
    };

    TranslationStiffnessDiagnostics
    ComputeTranslationStiffnessDiagnostics(const Eigen::Matrix<double, 6, 6> &information,
                                           const Eigen::Matrix3d &R_WL)
    {
        TranslationStiffnessDiagnostics result;

        if (!information.allFinite() ||
            !R_WL.allFinite())
        {
            return result;
        }

        // ------------------------------------------------------------
        // Full SE(3) information:
        //
        // Omega = [ A  B ]
        //         [ B' C ]
        //
        // order = [tx ty tz rx ry rz]
        // ------------------------------------------------------------
        const Eigen::Matrix3d A = information.block<3, 3>(0, 0);

        const Eigen::Matrix3d B = information.block<3, 3>(0, 3);

        const Eigen::Matrix3d C = information.block<3, 3>(3, 3);

        if (!A.allFinite() ||
            !B.allFinite() ||
            !C.allFinite())
        {
            return result;
        }

        // ------------------------------------------------------------
        // World axes represented in the current / to-Keyframe
        // LiDAR coordinate system.
        //
        // R_WL : LiDAR -> World
        //
        // Therefore:
        //
        //     v_L = R_WL^T * v_W
        // ------------------------------------------------------------
        const Eigen::Vector3d world_x_in_lidar = R_WL.transpose() *
                                                 Eigen::Vector3d::UnitX();

        const Eigen::Vector3d world_y_in_lidar = R_WL.transpose() *
                                                 Eigen::Vector3d::UnitY();

        const Eigen::Vector3d world_z_in_lidar = R_WL.transpose() *
                                                 Eigen::Vector3d::UnitZ();

        // ------------------------------------------------------------
        // Direct translational stiffness.
        //
        // Rotation is assumed fixed here.
        // ------------------------------------------------------------
        result.direct_world_x = world_x_in_lidar.dot(A * world_x_in_lidar);
        result.direct_world_y = world_y_in_lidar.dot(A * world_y_in_lidar);
        result.direct_world_z = world_z_in_lidar.dot(A * world_z_in_lidar);

        // ------------------------------------------------------------
        // Effective translational stiffness after allowing rotation
        // to co-adjust:
        //
        //     S = A - B C^{-1} B^T
        //
        // Do NOT explicitly compute C.inverse().
        // ------------------------------------------------------------
        Eigen::LDLT<Eigen::Matrix3d> ldlt(C);

        if (ldlt.info() != Eigen::Success)
        {
            return result;
        }

        const Eigen::Vector3d diagonal = ldlt.vectorD();

        if (!diagonal.allFinite() || diagonal.minCoeff() <= 1.0e-12)
        {
            return result;
        }

        Eigen::Matrix3d effective_translation_information = A - B * ldlt.solve(B.transpose());
        effective_translation_information = 0.5 * (effective_translation_information + effective_translation_information.transpose());

        if (!effective_translation_information.allFinite())
        {
            return result;
        }

        result.effective_world_x = world_x_in_lidar.dot(effective_translation_information * world_x_in_lidar);
        result.effective_world_y = world_y_in_lidar.dot(effective_translation_information * world_y_in_lidar);
        result.effective_world_z = world_z_in_lidar.dot(effective_translation_information * world_z_in_lidar);

        const double maximum_world_stiffness = std::max(result.effective_world_x,
                                                        std::max(
                                                            result.effective_world_y,
                                                            result.effective_world_z));

        if (!std::isfinite(maximum_world_stiffness) || maximum_world_stiffness <= 1.0e-12)
        {
            return result;
        }

        result.relative_world_x = result.effective_world_x / maximum_world_stiffness;
        result.relative_world_y = result.effective_world_y / maximum_world_stiffness;
        result.relative_world_z = result.effective_world_z / maximum_world_stiffness;

        // ------------------------------------------------------------
        // Find weakest translational direction.
        // ------------------------------------------------------------
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver(effective_translation_information);

        if (eigen_solver.info() != Eigen::Success)
        {
            return result;
        }

        const Eigen::Vector3d eigenvalues = eigen_solver.eigenvalues();
        const Eigen::Matrix3d eigenvectors = eigen_solver.eigenvectors();

        if (!eigenvalues.allFinite() ||
            !eigenvectors.allFinite())
        {
            return result;
        }

        result.weak_eigenvalue = eigenvalues(0);
        const Eigen::Vector3d weak_axis_lidar = eigenvectors.col(0);
        const Eigen::Vector3d weak_axis_world = R_WL * weak_axis_lidar;

        result.weak_axis_world_z_alignment = std::abs(weak_axis_world.normalized().dot(Eigen::Vector3d::UnitZ()));

        result.valid =
            std::isfinite(result.direct_world_x) &&
            std::isfinite(result.direct_world_y) &&
            std::isfinite(result.direct_world_z) &&
            std::isfinite(result.effective_world_x) &&
            std::isfinite(result.effective_world_y) &&
            std::isfinite(result.effective_world_z) &&
            std::isfinite(result.relative_world_x) &&
            std::isfinite(result.relative_world_y) &&
            std::isfinite(result.relative_world_z) &&
            std::isfinite(result.weak_eigenvalue) &&
            std::isfinite(
                result.weak_axis_world_z_alignment);

        return result;
    }

    double ClampUnit(double value)
    {
        return std::max(-1.0, std::min(1.0, value));
    }

    double GravityTiltErrorDeg(const Eigen::Matrix3d &R_WK, const Eigen::Vector3d &gravity_L_reference)
    {
        Eigen::Vector3d gravity_estimated = R_WK.transpose() * Eigen::Vector3d::UnitZ();
        Eigen::Vector3d gravity_reference = gravity_L_reference;
        const double estimated_norm = gravity_estimated.norm();
        const double reference_norm = gravity_reference.norm();

        if (!gravity_estimated.allFinite() || !gravity_reference.allFinite() || estimated_norm < 1.0e-12 || reference_norm < 1.0e-12)
        {
            return std::numeric_limits<double>::infinity();
        }

        gravity_estimated /= estimated_norm;
        gravity_reference /= reference_norm;
        const double cosine = ClampUnit(gravity_estimated.dot(gravity_reference));
        return std::acos(cosine) * kRadToDeg;
    }

    Eigen::Vector3d RotationToRpy(const Eigen::Matrix3d &R)
    {
        Eigen::Vector3d rpy = Eigen::Vector3d::Zero();
        const double sin_pitch = ClampUnit(-R(2, 0));
        rpy.y() = std::asin(sin_pitch);
        rpy.x() = std::atan2(R(2, 1),
                             R(2, 2));
        rpy.z() = std::atan2(R(1, 0),
                             R(0, 0));
        return rpy;
    }

    double WrappedAngleDifferenceDeg(double first,
                                     double second)
    {
        const double difference = first - second;

        return std::abs(std::atan2(std::sin(difference),
                                   std::cos(difference))) *
               kRadToDeg;
    }

    // ============================================================================
    // ComputeXyPcaRatio()
    //
    // ratio = lambda_min / lambda_max for the XY trajectory point set.
    // A genuinely two-dimensional path has a non-zero ratio, while a trajectory
    // collapsed toward one line approaches zero.
    // ============================================================================
    double ComputeXyPcaRatio(const AlignedVector2d &positions)
    {
        if (positions.size() < 3)
        {
            return 0.0;
        }
        Eigen::Vector2d mean = Eigen::Vector2d::Zero();
        for (const Eigen::Vector2d &position : positions)
        {
            if (!position.allFinite())
            {
                return std::numeric_limits<double>::quiet_NaN();
            }
            mean += position;
        }

        mean /= static_cast<double>(positions.size());

        Eigen::Matrix2d covariance = Eigen::Matrix2d::Zero();

        for (const Eigen::Vector2d &position : positions)
        {
            const Eigen::Vector2d delta = position - mean;

            covariance.noalias() += delta * delta.transpose();
        }

        covariance /= static_cast<double>(positions.size());

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(covariance);

        if (solver.info() != Eigen::Success)
        {
            return std::numeric_limits<double>::quiet_NaN();
        }

        const Eigen::Vector2d eigenvalues = solver.eigenvalues();

        const double lambda_min = std::max(0.0, eigenvalues.x());

        const double lambda_max = std::max(0.0, eigenvalues.y());

        if (!std::isfinite(lambda_min) || !std::isfinite(lambda_max) || lambda_max < 1.0e-12)
        {
            return 0.0;
        }

        return lambda_min / lambda_max;
    }

    // ============================================================================
    // ComputeXyPathLength()
    //
    // The graph node vector follows Keyframe insertion order, so consecutive
    // entries represent the backend odometry chain.
    // ============================================================================
    double ComputeXyPathLength(const AlignedVector2d &positions)
    {
        if (positions.size() < 2)
        {
            return 0.0;
        }

        double length = 0.0;

        for (std::size_t i = 1; i < positions.size(); ++i)
        {
            if (!positions[i - 1].allFinite() || !positions[i].allFinite())
            {
                return std::numeric_limits<double>::quiet_NaN();
            }

            length += (positions[i] - positions[i - 1]).norm();
        }
        return length;
    }

    // ============================================================================
    // SignedWrappedAngleDifferenceDeg()
    //
    // Signed wrapped difference first - second in [-180, 180] deg.
    // ============================================================================
    double SignedWrappedAngleDifferenceDeg(double first,
                                           double second)
    {
        const double difference = first - second;

        return std::atan2(
                   std::sin(difference),
                   std::cos(difference)) *
               kRadToDeg;
    }

    // ============================================================================
    // RotationAngleDeg()
    // ============================================================================
    double RotationAngleDeg(const Eigen::Matrix3d &R)
    {
        const Eigen::AngleAxisd angle_axis(R);

        if (!std::isfinite(angle_axis.angle()))
        {
            return std::numeric_limits<double>::quiet_NaN();
        }

        return std::abs(angle_axis.angle()) *
               kRadToDeg;
    }

    // ============================================================================
    // DiagnosticDirectory()
    //
    // fr_slam.launch.py sets FR_SLAM_MAP_DIR to the same directory used by the
    // PCD exporter.  The HOME fallback keeps the diagnostic useful when this
    // executable is started directly without the launch file.
    // ============================================================================
    std::filesystem::path OutputDirectory()
    {
        const char *configured_directory = std::getenv("FR_SLAM_OUTPUT_DIR");

        if (configured_directory != nullptr && configured_directory[0] != '\0')
        {
            return std::filesystem::path(configured_directory);
        }

        const char *home_directory = std::getenv("HOME");

        if (home_directory != nullptr && home_directory[0] != '\0')
        {
            return std::filesystem::path(
                       home_directory) /
                   "ros2_ws" /
                   "src" /
                   "fr_slam" /
                   "output";
        }
        return std::filesystem::path("/tmp/fr_slam_output");
    }

    std::filesystem::path DiagnosticDirectory()
    {
        return OutputDirectory() /
               "diagnostics";
    }

    std::filesystem::path LoopDirectory()
    {
        return OutputDirectory() /
               "loop";
    }

    // ============================================================================
    // OptimizedPose()
    //
    // Read one optimized SE(3) estimate from the temporary g2o graph.
    // ============================================================================
    bool OptimizedPose(g2o::SparseOptimizer &optimizer,
                       std::size_t node_id,
                       Eigen::Isometry3d &T_WK)
    {
        int g2o_id = 0;

        if (!ToG2oId(node_id, g2o_id))
        {
            return false;
        }

        const g2o::VertexSE3 *vertex = dynamic_cast<const g2o::VertexSE3 *>(optimizer.vertex(g2o_id));
        if (vertex == nullptr)
        {
            return false;
        }

        T_WK = vertex->estimate();

        return T_WK.matrix().allFinite();
    }

    // ============================================================================
    // WritePoseGraphDiagnostics()
    //
    // These CSVs diagnose whether the accepted GLOBAL PoseGraph optimization is
    // applying a coherent correction or bending individual odometry segments.
    //
    // Files:
    //   pgo_keyframe_correction.csv
    //   pgo_odom_edge_deformation.csv
    //   pgo_loop_edges.csv
    //
    // IMPORTANT:
    // - This function is diagnostic only.
    // - Failure to write CSV must NEVER reject a valid PoseGraph optimization.
    // ============================================================================
    bool WritePoseGraphDiagnostics(const std::vector<PoseGraphNode> &nodes,
                                   const std::vector<PoseGraphEdge> &edges,
                                   const std::unordered_map<std::size_t, Eigen::Isometry3d> &pose_before_optimization,
                                   g2o::SparseOptimizer &optimizer,
                                   std::filesystem::path &output_directory)
    {
        try
        {
            output_directory = DiagnosticDirectory();

            std::filesystem::create_directories(output_directory);

            // --------------------------------------------------------------------
            // 1. Per-Keyframe correction.
            // --------------------------------------------------------------------
            std::ofstream keyframe_file(output_directory / "pgo_keyframe_correction.csv");

            if (!keyframe_file.is_open())
            {
                return false;
            }

            keyframe_file << std::fixed << std::setprecision(9);

            keyframe_file
                << "kf_id,fixed,"
                << "raw_x,raw_y,raw_z,raw_roll_deg,raw_pitch_deg,raw_yaw_deg,"
                << "opt_x,opt_y,opt_z,opt_roll_deg,opt_pitch_deg,opt_yaw_deg,"
                << "delta_world_x,delta_world_y,delta_world_z,"
                << "delta_local_x,delta_local_y,delta_local_z,"
                << "delta_translation_m,delta_rotation_deg,"
                << "delta_roll_deg,delta_pitch_deg,delta_yaw_deg\n";

            for (const PoseGraphNode &node : nodes)
            {
                const auto before_iterator = pose_before_optimization.find(node.id);
                if (before_iterator ==
                    pose_before_optimization.end())
                {
                    continue;
                }

                Eigen::Isometry3d T_after = Eigen::Isometry3d::Identity();

                if (!OptimizedPose(optimizer, node.id, T_after))
                {
                    continue;
                }

                const Eigen::Isometry3d &T_before = before_iterator->second;

                const Eigen::Isometry3d T_delta_local = T_before.inverse() * T_after;

                const Eigen::Vector3d delta_world = T_after.translation() - T_before.translation();

                const Eigen::Vector3d raw_rpy = RotationToRpy(T_before.rotation());

                const Eigen::Vector3d opt_rpy = RotationToRpy(T_after.rotation());

                keyframe_file
                    << node.id << ","
                    << (node.fixed ? 1 : 0) << ","
                    << T_before.translation().x() << ","
                    << T_before.translation().y() << ","
                    << T_before.translation().z() << ","
                    << raw_rpy.x() * kRadToDeg << ","
                    << raw_rpy.y() * kRadToDeg << ","
                    << raw_rpy.z() * kRadToDeg << ","
                    << T_after.translation().x() << ","
                    << T_after.translation().y() << ","
                    << T_after.translation().z() << ","
                    << opt_rpy.x() * kRadToDeg << ","
                    << opt_rpy.y() * kRadToDeg << ","
                    << opt_rpy.z() * kRadToDeg << ","
                    << delta_world.x() << ","
                    << delta_world.y() << ","
                    << delta_world.z() << ","
                    << T_delta_local.translation().x() << ","
                    << T_delta_local.translation().y() << ","
                    << T_delta_local.translation().z() << ","
                    << T_delta_local.translation().norm() << ","
                    << RotationAngleDeg(
                           T_delta_local.rotation())
                    << ","
                    << SignedWrappedAngleDifferenceDeg(
                           opt_rpy.x(),
                           raw_rpy.x())
                    << ","
                    << SignedWrappedAngleDifferenceDeg(
                           opt_rpy.y(),
                           raw_rpy.y())
                    << ","
                    << SignedWrappedAngleDifferenceDeg(
                           opt_rpy.z(),
                           raw_rpy.z())
                    << "\n";
            }

            keyframe_file.close();

            // --------------------------------------------------------------------
            // 2. Odometry-edge deformation.
            //
            // "deformation" compares the relative pose of each consecutive
            // Keyframe pair before and after G2O. This is the direct diagnostic
            // for local bending of a long straight trajectory.
            // --------------------------------------------------------------------
            std::ofstream odometry_file(output_directory / "pgo_odom_edge_deformation.csv");

            if (!odometry_file.is_open())
            {
                return false;
            }

            odometry_file
                << std::fixed
                << std::setprecision(9);

            odometry_file
                << "from_kf,to_kf,"
                << "measurement_translation_m,measurement_yaw_deg,"
                << "before_translation_m,before_yaw_deg,"
                << "after_translation_m,after_yaw_deg,"

                // Relative-pose deformation expressed in the local relative frame.
                << "deformation_local_x,"
                << "deformation_local_y,"
                << "deformation_local_z,"
                << "deformation_translation_m,"
                << "deformation_rotation_deg,"
                << "deformation_yaw_deg,"

                // Change of the GLOBAL correction from from_kf to to_kf.
                // This is the quantity we need for diagnosing accumulated Z correction.
                << "correction_gradient_world_x,"
                << "correction_gradient_world_y,"
                << "correction_gradient_world_z,"
                << "correction_gradient_world_norm,"

                << "after_measurement_error_translation_m,"
                << "after_measurement_error_rotation_deg\n";

            // --------------------------------------------------------------------
            // 3. Loop-edge measurements and residuals.
            //
            // Keep loop diagnostics separate from general backend diagnostics:
            //
            //     output/loop/pgo_loop_edges.csv
            // --------------------------------------------------------------------
            const std::filesystem::path loop_directory = LoopDirectory();

            std::filesystem::create_directories(loop_directory);

            std::ofstream loop_file(loop_directory / "pgo_loop_edges.csv");

            if (!loop_file.is_open())
            {
                return false;
            }

            loop_file << std::fixed << std::setprecision(9);

            loop_file
                << "from_kf,to_kf,"
                << "measurement_x,measurement_y,measurement_z,"
                << "measurement_roll_deg,measurement_pitch_deg,measurement_yaw_deg,"
                << "before_x,before_y,before_z,before_yaw_deg,"
                << "after_x,after_y,after_z,after_yaw_deg,"
                << "before_error_translation_m,before_error_rotation_deg,"
                << "after_error_translation_m,after_error_rotation_deg\n";

            for (const PoseGraphEdge &edge : edges)
            {
                const auto from_before_iterator = pose_before_optimization.find(edge.from_id);

                const auto to_before_iterator = pose_before_optimization.find(edge.to_id);

                if (from_before_iterator == pose_before_optimization.end() || to_before_iterator == pose_before_optimization.end())
                {
                    continue;
                }

                Eigen::Isometry3d T_W_from_after = Eigen::Isometry3d::Identity();

                Eigen::Isometry3d T_W_to_after = Eigen::Isometry3d::Identity();

                if (!OptimizedPose(optimizer,
                                   edge.from_id,
                                   T_W_from_after) ||
                    !OptimizedPose(optimizer,
                                   edge.to_id,
                                   T_W_to_after))
                {
                    continue;
                }

                const Eigen::Isometry3d T_before_relative = from_before_iterator->second.inverse() * to_before_iterator->second;

                const Eigen::Isometry3d T_after_relative = T_W_from_after.inverse() * T_W_to_after;

                const Eigen::Isometry3d T_deformation = T_before_relative.inverse() * T_after_relative;

                // ------------------------------------------------------------------------
                // Per-node GLOBAL corrections:
                //
                //     delta_from = p_from_after - p_from_before
                //     delta_to   = p_to_after   - p_to_before
                //
                // Difference across this odometry edge:
                //
                //     correction_gradient_world
                //         = delta_to - delta_from
                //
                // For a consecutive odometry chain:
                //
                //     sum_i correction_gradient_world_z(i)
                //       = delta_world_z(last) - delta_world_z(first)
                //
                // Therefore this quantity directly tells us how the global Z correction
                // is distributed along the graph.
                // ------------------------------------------------------------------------
                const Eigen::Vector3d correction_world_from = T_W_from_after.translation() - from_before_iterator->second.translation();

                const Eigen::Vector3d correction_world_to = T_W_to_after.translation() - to_before_iterator->second.translation();

                const Eigen::Vector3d correction_gradient_world = correction_world_to - correction_world_from;

                const Eigen::Isometry3d T_before_measurement_error = edge.T_from_to.inverse() * T_before_relative;

                const Eigen::Isometry3d T_after_measurement_error =
                    edge.T_from_to.inverse() *
                    T_after_relative;

                const Eigen::Vector3d measurement_rpy =
                    RotationToRpy(
                        edge.T_from_to.rotation());

                const Eigen::Vector3d before_rpy =
                    RotationToRpy(
                        T_before_relative.rotation());

                const Eigen::Vector3d after_rpy =
                    RotationToRpy(
                        T_after_relative.rotation());

                if (edge.type ==
                    PoseGraphEdgeType::Odometry)
                {
                    odometry_file
                        << edge.from_id << ","
                        << edge.to_id << ","

                        << edge.T_from_to.translation().norm() << ","
                        << measurement_rpy.z() * kRadToDeg << ","

                        << T_before_relative.translation().norm() << ","
                        << before_rpy.z() * kRadToDeg << ","

                        << T_after_relative.translation().norm() << ","
                        << after_rpy.z() * kRadToDeg << ","

                        // --------------------------------------------------------------
                        // Local relative-pose deformation.
                        // --------------------------------------------------------------
                        << T_deformation.translation().x() << ","
                        << T_deformation.translation().y() << ","
                        << T_deformation.translation().z() << ","
                        << T_deformation.translation().norm() << ","

                        << RotationAngleDeg(
                               T_deformation.rotation())
                        << ","

                        << SignedWrappedAngleDifferenceDeg(
                               after_rpy.z(),
                               before_rpy.z())
                        << ","

                        // --------------------------------------------------------------
                        // GLOBAL correction gradient across this odometry edge.
                        //
                        // This is the most important quantity for the current Z-drift
                        // investigation.
                        // --------------------------------------------------------------
                        << correction_gradient_world.x() << ","
                        << correction_gradient_world.y() << ","
                        << correction_gradient_world.z() << ","
                        << correction_gradient_world.norm() << ","

                        << T_after_measurement_error.translation().norm() << ","

                        << RotationAngleDeg(
                               T_after_measurement_error.rotation())

                        << "\n";
                }
                else if (edge.type ==
                         PoseGraphEdgeType::Loop)
                {
                    loop_file
                        << edge.from_id << ","
                        << edge.to_id << ","
                        << edge.T_from_to.translation().x() << ","
                        << edge.T_from_to.translation().y() << ","
                        << edge.T_from_to.translation().z() << ","
                        << measurement_rpy.x() * kRadToDeg << ","
                        << measurement_rpy.y() * kRadToDeg << ","
                        << measurement_rpy.z() * kRadToDeg << ","
                        << T_before_relative.translation().x() << ","
                        << T_before_relative.translation().y() << ","
                        << T_before_relative.translation().z() << ","
                        << before_rpy.z() * kRadToDeg << ","
                        << T_after_relative.translation().x() << ","
                        << T_after_relative.translation().y() << ","
                        << T_after_relative.translation().z() << ","
                        << after_rpy.z() * kRadToDeg << ","
                        << T_before_measurement_error.translation().norm() << ","
                        << RotationAngleDeg(
                               T_before_measurement_error.rotation())
                        << ","
                        << T_after_measurement_error.translation().norm() << ","
                        << RotationAngleDeg(
                               T_after_measurement_error.rotation())
                        << "\n";
                }
            }

            odometry_file.close();
            loop_file.close();

            return true;
        }
        catch (const std::exception &exception)
        {
            std::cerr
                << "PoseGraph diagnostic CSV exception"
                << " | what=" << exception.what()
                << std::endl;

            return false;
        }
    }

} // namespace

PoseGraphOptimizer::PoseGraphOptimizer(
    const PoseGraphOptimizerConfig &config)
    : config_(config)
{
}

bool PoseGraphOptimizer::Optimize(
    PoseGraph &pose_graph,
    PoseGraphOptimizationResult &result) const
{
    result = PoseGraphOptimizationResult();

    const std::vector<PoseGraphNode> &nodes =
        pose_graph.GetNodes();

    const std::vector<PoseGraphEdge> &edges =
        pose_graph.GetEdges();

    if (nodes.size() < 2 || edges.empty())
    {
        std::cerr
            << "PoseGraphOptimizer: graph is too small"
            << " | nodes=" << nodes.size()
            << " | edges=" << edges.size()
            << std::endl;
        return false;
    }

    bool has_fixed_node = false;

    for (const PoseGraphNode &node : nodes)
    {
        if (node.fixed)
        {
            has_fixed_node = true;
            break;
        }
    }

    if (!has_fixed_node)
    {
        std::cerr
            << "PoseGraphOptimizer: no fixed Keyframe vertex."
            << std::endl;
        return false;
    }

    if (config_.use_gravity_direction_prior &&
        (!std::isfinite(config_.gravity_information_scale) ||
         config_.gravity_information_scale <= 0.0))
    {
        std::cerr
            << "PoseGraphOptimizer: invalid gravity information scale"
            << " | scale="
            << config_.gravity_information_scale
            << std::endl;
        return false;
    }

    if (config_.use_gravity_hard_guard &&
        (!std::isfinite(config_.max_gravity_tilt_error_deg) ||
         config_.max_gravity_tilt_error_deg <= 0.0))
    {
        std::cerr
            << "PoseGraphOptimizer: invalid gravity hard-guard threshold"
            << " | max_tilt="
            << config_.max_gravity_tilt_error_deg
            << std::endl;
        return false;
    }

    if (config_.use_trajectory_shape_guard &&
        (!std::isfinite(config_.min_xy_pca_ratio_scale) ||
         config_.min_xy_pca_ratio_scale <= 0.0 ||
         config_.min_xy_pca_ratio_scale > 1.0 ||
         !std::isfinite(config_.min_path_length_ratio) ||
         !std::isfinite(config_.max_path_length_ratio) ||
         config_.min_path_length_ratio <= 0.0 ||
         config_.max_path_length_ratio <=
             config_.min_path_length_ratio))
    {
        std::cerr
            << "PoseGraphOptimizer: invalid trajectory shape guard config"
            << " | pca_scale="
            << config_.min_xy_pca_ratio_scale
            << " | min_length_ratio="
            << config_.min_path_length_ratio
            << " | max_length_ratio="
            << config_.max_path_length_ratio
            << std::endl;
        return false;
    }

    if (config_.use_global_information_v1 &&
        (config_.global_information_min_nodes < 2 ||
         !std::isfinite(config_.global_odom_translation_weight) ||
         config_.global_odom_translation_weight <= 0.0 ||
         !std::isfinite(config_.global_odom_rotation_weight) ||
         config_.global_odom_rotation_weight <= 0.0 ||
         !std::isfinite(config_.global_loop_translation_weight) ||
         config_.global_loop_translation_weight <= 0.0 ||
         !std::isfinite(config_.global_loop_rotation_weight) ||
         config_.global_loop_rotation_weight <= 0.0))
    {
        std::cerr
            << "PoseGraphOptimizer: invalid Global Information Matrix V1 config"
            << " | min_nodes="
            << config_.global_information_min_nodes
            << " | odom_t="
            << config_.global_odom_translation_weight
            << " | odom_r="
            << config_.global_odom_rotation_weight
            << " | loop_t="
            << config_.global_loop_translation_weight
            << " | loop_r="
            << config_.global_loop_rotation_weight
            << std::endl;
        return false;
    }

    typedef g2o::BlockSolver<
        g2o::BlockSolverTraits<6, 6>>
        BlockSolverType;

    typedef g2o::LinearSolverEigen<
        BlockSolverType::PoseMatrixType>
        LinearSolverType;

    std::unique_ptr<LinearSolverType> linear_solver =
        std::make_unique<LinearSolverType>();

    std::unique_ptr<BlockSolverType> block_solver =
        std::make_unique<BlockSolverType>(
            std::move(linear_solver));

    g2o::OptimizationAlgorithmLevenberg *algorithm =
        new g2o::OptimizationAlgorithmLevenberg(
            std::move(block_solver));

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(algorithm);
    optimizer.setVerbose(config_.verbose);

    std::unordered_map<std::size_t, Eigen::Isometry3d>
        pose_before_optimization;

    pose_before_optimization.reserve(nodes.size());

    // ------------------------------------------------------------------------
    // 1. Keyframe vertices.
    // ------------------------------------------------------------------------
    for (const PoseGraphNode &node : nodes)
    {
        if (!node.T_WK.matrix().allFinite())
        {
            return false;
        }

        int g2o_id = 0;
        if (!ToG2oId(node.id, g2o_id))
        {
            return false;
        }

        g2o::VertexSE3 *vertex =
            new g2o::VertexSE3();

        vertex->setId(g2o_id);
        vertex->setEstimate(node.T_WK);
        vertex->setFixed(node.fixed);

        if (!optimizer.addVertex(vertex))
        {
            delete vertex;
            return false;
        }

        pose_before_optimization.emplace(
            node.id,
            node.T_WK);
    }

    // ------------------------------------------------------------------------
    // 2. Odometry and Loop edges.
    // ------------------------------------------------------------------------
    const bool apply_global_information_v1 =
        config_.use_global_information_v1 &&
        nodes.size() >=
            config_.global_information_min_nodes &&
        pose_graph.LoopEdgeCount() > 0;

    std::ofstream odometry_information_file;

    bool odometry_information_file_open =
        false;

    if (nodes.size() >= 100 &&
        pose_graph.LoopEdgeCount() > 0)
    {
        try
        {
            const std::filesystem::path directory =
                DiagnosticDirectory();

            std::filesystem::create_directories(
                directory);

            odometry_information_file.open(
                directory /
                "pgo_odom_information.csv");

            if (odometry_information_file.is_open())
            {
                odometry_information_file
                    << std::fixed
                    << std::setprecision(9);

                odometry_information_file
                    << "from_kf,to_kf,"
                    << "info_tx,info_ty,info_tz,"
                    << "info_rx,info_ry,info_rz,"
                    << "direct_world_x,"
                    << "direct_world_y,"
                    << "direct_world_z,"
                    << "effective_world_x,"
                    << "effective_world_y,"
                    << "effective_world_z,"
                    << "relative_world_x,"
                    << "relative_world_y,"
                    << "relative_world_z,"
                    << "weak_eigenvalue,"
                    << "weak_axis_world_z_alignment\n";

                odometry_information_file_open =
                    true;
            }
        }
        catch (...)
        {
            odometry_information_file_open =
                false;
        }
    }

    for (const PoseGraphEdge &edge_data : edges)
    {
        int from_id = 0;
        int to_id = 0;

        if (!ToG2oId(edge_data.from_id, from_id) ||
            !ToG2oId(edge_data.to_id, to_id))
        {
            return false;
        }

        g2o::HyperGraph::Vertex *from_vertex =
            optimizer.vertex(from_id);

        g2o::HyperGraph::Vertex *to_vertex =
            optimizer.vertex(to_id);

        if (from_vertex == nullptr ||
            to_vertex == nullptr ||
            !edge_data.T_from_to.matrix().allFinite() ||
            !IsFinitePositiveSemidefiniteInformation(
                edge_data.information))
        {
            return false;
        }

        g2o::EdgeSE3 *edge =
            new g2o::EdgeSE3();

        edge->setVertex(0, from_vertex);
        edge->setVertex(1, to_vertex);
        edge->setMeasurement(edge_data.T_from_to);

        Eigen::Matrix<double, 6, 6> information =
            edge_data.information;

        if (apply_global_information_v1)
        {
            if (edge_data.type ==
                PoseGraphEdgeType::Odometry)
            {
                information =
                    RescaleSe3Information(
                        edge_data.information,
                        config_.global_odom_translation_weight,
                        config_.global_odom_rotation_weight);
            }
            else if (edge_data.type ==
                     PoseGraphEdgeType::Loop)
            {
                information =
                    RescaleSe3Information(
                        edge_data.information,
                        config_.global_loop_translation_weight,
                        config_.global_loop_rotation_weight);
            }
        }

        if (!IsFinitePositiveSemidefiniteInformation(
                information))
        {
            delete edge;
            return false;
        }

        if (edge_data.type ==
                PoseGraphEdgeType::Odometry &&
            odometry_information_file_open)
        {
            const g2o::VertexSE3 *to_vertex_se3 =
                dynamic_cast<const g2o::VertexSE3 *>(
                    to_vertex);

            if (to_vertex_se3 != nullptr)
            {
                const Eigen::Matrix3d R_WL_to =
                    to_vertex_se3
                        ->estimate()
                        .rotation();

                const TranslationStiffnessDiagnostics
                    stiffness =
                        ComputeTranslationStiffnessDiagnostics(
                            information,
                            R_WL_to);

                if (stiffness.valid)
                {
                    odometry_information_file
                        << edge_data.from_id << ","
                        << edge_data.to_id << ","

                        << information(0, 0) << ","
                        << information(1, 1) << ","
                        << information(2, 2) << ","

                        << information(3, 3) << ","
                        << information(4, 4) << ","
                        << information(5, 5) << ","

                        << stiffness.direct_world_x << ","
                        << stiffness.direct_world_y << ","
                        << stiffness.direct_world_z << ","

                        << stiffness.effective_world_x << ","
                        << stiffness.effective_world_y << ","
                        << stiffness.effective_world_z << ","

                        << stiffness.relative_world_x << ","
                        << stiffness.relative_world_y << ","
                        << stiffness.relative_world_z << ","

                        << stiffness.weak_eigenvalue << ","
                        << stiffness.weak_axis_world_z_alignment
                        << "\n";
                }
            }
        }

        edge->setInformation(
            information);

        if (edge_data.type == PoseGraphEdgeType::Loop &&
            config_.use_huber_for_loop_edges)
        {
            g2o::RobustKernelHuber *robust_kernel =
                new g2o::RobustKernelHuber();

            robust_kernel->setDelta(
                config_.loop_huber_delta);

            edge->setRobustKernel(
                robust_kernel);
        }

        if (!optimizer.addEdge(edge))
        {
            delete edge;
            return false;
        }
    }

    // ------------------------------------------------------------------------
    // 3. Gravity-direction unary factors.
    //
    // They are only added for graph nodes that carry the immutable reference
    // generated when the raw frontend Keyframe was first committed.
    // ------------------------------------------------------------------------
    if (config_.use_gravity_direction_prior)
    {
        for (const PoseGraphNode &node : nodes)
        {
            if (!node.has_gravity_reference)
            {
                continue;
            }

            if (!node.gravity_L_reference.allFinite() ||
                node.gravity_L_reference.norm() < 1.0e-9)
            {
                return false;
            }

            int g2o_id = 0;
            if (!ToG2oId(node.id, g2o_id))
            {
                return false;
            }

            g2o::HyperGraph::Vertex *vertex =
                optimizer.vertex(g2o_id);

            if (vertex == nullptr)
            {
                return false;
            }

            EdgeGravityDirection *gravity_edge =
                new EdgeGravityDirection();

            gravity_edge->setVertex(
                0,
                vertex);

            gravity_edge->setMeasurement(
                node.gravity_L_reference.normalized());

            Eigen::Matrix3d gravity_information =
                Eigen::Matrix3d::Identity() *
                config_.gravity_information_scale;

            gravity_edge->setInformation(
                gravity_information);

            if (!optimizer.addEdge(
                    gravity_edge))
            {
                delete gravity_edge;
                return false;
            }

            ++result.gravity_edges;
        }
    }

    // ------------------------------------------------------------------------
    // 4. Optimize.
    // ------------------------------------------------------------------------
    if (!optimizer.initializeOptimization())
    {
        std::cerr
            << "PoseGraphOptimizer: initializeOptimization failed."
            << std::endl;
        return false;
    }

    optimizer.computeActiveErrors();
    result.chi2_before = optimizer.activeChi2();

    result.iterations =
        optimizer.optimize(config_.max_iterations);

    if (result.iterations <= 0)
    {
        std::cerr
            << "PoseGraphOptimizer: g2o performed no iteration."
            << std::endl;
        return false;
    }

    optimizer.computeActiveErrors();
    result.chi2_after = optimizer.activeChi2();

    if (!std::isfinite(result.chi2_before) ||
        !std::isfinite(result.chi2_after))
    {
        return false;
    }

    if (result.chi2_after > result.chi2_before + 1.0e-9)
    {
        std::cerr
            << "PoseGraphOptimizer: chi2 increased"
            << " | before=" << result.chi2_before
            << " | after=" << result.chi2_after
            << std::endl;
        return false;
    }

    // ------------------------------------------------------------------------
    // 5. VALIDATE ALL optimized poses BEFORE committing any pose.
    //
    // This keeps optimization transactional.  A gravity-guard failure cannot
    // leave PoseGraph half updated.
    // ------------------------------------------------------------------------
    std::size_t max_translation_update_node_id = 0;
    std::size_t max_rotation_update_node_id = 0;
    std::size_t large_rotation_update_count = 0;

    constexpr double large_rotation_threshold_deg =
        20.0;

    double gravity_tilt_error_sum_deg =
        0.0;

    AlignedVector2d xy_positions_before;
    AlignedVector2d xy_positions_after;

    xy_positions_before.reserve(nodes.size());
    xy_positions_after.reserve(nodes.size());

    for (const PoseGraphNode &node : nodes)
    {
        int g2o_id = 0;
        if (!ToG2oId(node.id, g2o_id))
        {
            return false;
        }

        g2o::VertexSE3 *vertex =
            dynamic_cast<g2o::VertexSE3 *>(
                optimizer.vertex(g2o_id));

        if (vertex == nullptr)
        {
            return false;
        }

        const Eigen::Isometry3d T_WK_optimized =
            vertex->estimate();

        if (!T_WK_optimized.matrix().allFinite())
        {
            return false;
        }

        const auto old_pose_iterator =
            pose_before_optimization.find(node.id);

        if (old_pose_iterator ==
            pose_before_optimization.end())
        {
            return false;
        }

        xy_positions_before.emplace_back(
            old_pose_iterator->second.translation().x(),
            old_pose_iterator->second.translation().y());

        xy_positions_after.emplace_back(
            T_WK_optimized.translation().x(),
            T_WK_optimized.translation().y());

        const Eigen::Isometry3d T_old_new =
            old_pose_iterator->second.inverse() *
            T_WK_optimized;

        const double translation_update =
            T_old_new.translation().norm();

        const Eigen::AngleAxisd rotation_update(
            T_old_new.rotation());

        const double rotation_update_deg =
            std::abs(rotation_update.angle()) *
            kRadToDeg;

        if (translation_update >
            result.max_translation_update)
        {
            result.max_translation_update =
                translation_update;

            max_translation_update_node_id =
                node.id;
        }

        if (rotation_update_deg >
            result.max_rotation_update_deg)
        {
            result.max_rotation_update_deg =
                rotation_update_deg;

            max_rotation_update_node_id =
                node.id;
        }

        const Eigen::Vector3d old_rpy =
            RotationToRpy(
                old_pose_iterator->second.rotation());

        const Eigen::Vector3d new_rpy =
            RotationToRpy(
                T_WK_optimized.rotation());

        const double roll_update_deg =
            WrappedAngleDifferenceDeg(
                new_rpy.x(),
                old_rpy.x());

        const double pitch_update_deg =
            WrappedAngleDifferenceDeg(
                new_rpy.y(),
                old_rpy.y());

        const double yaw_update_deg =
            WrappedAngleDifferenceDeg(
                new_rpy.z(),
                old_rpy.z());

        if (roll_update_deg >
            result.max_roll_update_deg)
        {
            result.max_roll_update_deg =
                roll_update_deg;
            result.max_roll_update_keyframe_id =
                node.id;
        }

        if (pitch_update_deg >
            result.max_pitch_update_deg)
        {
            result.max_pitch_update_deg =
                pitch_update_deg;
            result.max_pitch_update_keyframe_id =
                node.id;
        }

        if (yaw_update_deg >
            result.max_yaw_update_deg)
        {
            result.max_yaw_update_deg =
                yaw_update_deg;
            result.max_yaw_update_keyframe_id =
                node.id;
        }

        if (rotation_update_deg >
            large_rotation_threshold_deg)
        {
            ++large_rotation_update_count;
        }

        if (node.has_gravity_reference)
        {
            ++result.gravity_reference_nodes;

            const double gravity_tilt_error_deg =
                GravityTiltErrorDeg(
                    T_WK_optimized.rotation(),
                    node.gravity_L_reference);

            if (!std::isfinite(
                    gravity_tilt_error_deg))
            {
                result.gravity_guard_passed = false;
                return false;
            }

            gravity_tilt_error_sum_deg +=
                gravity_tilt_error_deg;

            if (gravity_tilt_error_deg >
                result.max_gravity_tilt_error_deg)
            {
                result.max_gravity_tilt_error_deg =
                    gravity_tilt_error_deg;

                result.worst_gravity_keyframe_id =
                    node.id;
            }
        }
    }

    if (result.gravity_reference_nodes > 0)
    {
        result.mean_gravity_tilt_error_deg =
            gravity_tilt_error_sum_deg /
            static_cast<double>(
                result.gravity_reference_nodes);
    }

    result.optimized_nodes = nodes.size();
    result.odometry_edges =
        pose_graph.OdometryEdgeCount();
    result.loop_edges =
        pose_graph.LoopEdgeCount();

    result.xy_pca_ratio_before =
        ComputeXyPcaRatio(
            xy_positions_before);

    result.xy_pca_ratio_after =
        ComputeXyPcaRatio(
            xy_positions_after);

    result.path_length_before =
        ComputeXyPathLength(
            xy_positions_before);

    result.path_length_after =
        ComputeXyPathLength(
            xy_positions_after);

    if (result.path_length_before > 1.0e-9)
    {
        result.path_length_ratio =
            result.path_length_after /
            result.path_length_before;
    }
    else
    {
        result.path_length_ratio = 1.0;
    }

    if (config_.use_trajectory_shape_guard)
    {
        const bool shape_metrics_finite =
            std::isfinite(result.xy_pca_ratio_before) &&
            std::isfinite(result.xy_pca_ratio_after) &&
            std::isfinite(result.path_length_before) &&
            std::isfinite(result.path_length_after) &&
            std::isfinite(result.path_length_ratio);

        bool pca_guard_ok =
            shape_metrics_finite;

        // Only activate PCA collapse protection if the pre-optimization path
        // was genuinely two-dimensional.  A naturally straight path should
        // not be rejected for remaining straight.
        if (pca_guard_ok &&
            result.xy_pca_ratio_before > 0.02)
        {
            const double minimum_allowed_pca_ratio =
                result.xy_pca_ratio_before *
                config_.min_xy_pca_ratio_scale;

            pca_guard_ok =
                result.xy_pca_ratio_after >=
                minimum_allowed_pca_ratio;
        }

        const bool length_guard_ok =
            shape_metrics_finite &&
            result.path_length_ratio >=
                config_.min_path_length_ratio &&
            result.path_length_ratio <=
                config_.max_path_length_ratio;

        if (!pca_guard_ok ||
            !length_guard_ok)
        {
            result.trajectory_shape_guard_passed =
                false;

            std::cerr
                << "PoseGraph trajectory shape guard"
                << " | accepted=false"
                << " | reason="
                << (!pca_guard_ok
                        ? "XY_PCA_COLLAPSE"
                        : "PATH_LENGTH_CHANGE")
                << " | xy_pca_before="
                << result.xy_pca_ratio_before
                << " | xy_pca_after="
                << result.xy_pca_ratio_after
                << " | min_pca_scale="
                << config_.min_xy_pca_ratio_scale
                << " | path_length_ratio="
                << result.path_length_ratio
                << " | allowed_length=["
                << config_.min_path_length_ratio
                << ","
                << config_.max_path_length_ratio
                << "]"
                << " | action=NO_POSE_COMMIT"
                << std::endl;

            return false;
        }
    }

    result.trajectory_shape_guard_passed = true;

    if (config_.use_gravity_hard_guard &&
        result.gravity_reference_nodes > 0 &&
        result.max_gravity_tilt_error_deg >
            config_.max_gravity_tilt_error_deg)
    {
        result.gravity_guard_passed = false;

        std::cerr
            << "PoseGraph gravity guard"
            << " | accepted=false"
            << " | reason=MAX_TILT_ERROR"
            << " | max_tilt_error="
            << result.max_gravity_tilt_error_deg << " deg"
            << " | hard_limit="
            << config_.max_gravity_tilt_error_deg << " deg"
            << " | worst_kf="
            << result.worst_gravity_keyframe_id
            << " | action=NO_POSE_COMMIT"
            << std::endl;

        return false;
    }

    result.gravity_guard_passed = true;

    // ------------------------------------------------------------------------
    // Compact runtime summary.
    //
    // Detailed information-matrix / per-edge / per-keyframe diagnostics are
    // written to CSV. Keep stdout concise during normal rosbag runs.
    // ------------------------------------------------------------------------
    std::cout
        << "PoseGraph optimization accepted"
        << " | nodes=" << result.optimized_nodes
        << " | odom_edges=" << result.odometry_edges
        << " | loop_edges=" << result.loop_edges
        << " | iterations=" << result.iterations
        << " | chi2=" << result.chi2_before
        << "->" << result.chi2_after
        << " | information_v1="
        << (apply_global_information_v1
                ? "ON"
                : "OFF")
        << " | max_translation="
        << result.max_translation_update
        << "m@kf" << max_translation_update_node_id
        << " | max_rotation="
        << result.max_rotation_update_deg
        << "deg@kf" << max_rotation_update_node_id
        << " | max_rpy=["
        << result.max_roll_update_deg << ","
        << result.max_pitch_update_deg << ","
        << result.max_yaw_update_deg << "]deg"
        << " | large_rotation_nodes="
        << large_rotation_update_count
        << " | path_length_ratio="
        << result.path_length_ratio
        << " | max_gravity_tilt="
        << result.max_gravity_tilt_error_deg
        << "deg"
        << std::endl;

    // ------------------------------------------------------------------------
    // 5.5 Accepted GLOBAL-PGO diagnostics.
    //
    // The same PoseGraphOptimizer is also reused by the 16-Keyframe local
    // refinement graph.  We intentionally save CSV only for a large graph so
    // the local refinement pass cannot overwrite the global-PGO diagnostics.
    // The current dataset reaches first loop closure at >500 Keyframes.
    // ------------------------------------------------------------------------
    if (nodes.size() >=
            config_.global_information_min_nodes &&
        pose_graph.LoopEdgeCount() > 0)
    {
        std::filesystem::path
            diagnostic_directory;

        const bool diagnostic_saved =
            WritePoseGraphDiagnostics(
                nodes,
                edges,
                pose_before_optimization,
                optimizer,
                diagnostic_directory);

        if (!diagnostic_saved)
        {
            std::cerr
                << "PoseGraph diagnostic CSV write failed"
                << " | directory="
                << diagnostic_directory.string()
                << std::endl;
        }
    }

    // ------------------------------------------------------------------------
    // 6. COMMIT only after every validation passes.
    // ------------------------------------------------------------------------
    for (const PoseGraphNode &node : nodes)
    {
        int g2o_id = 0;
        if (!ToG2oId(node.id, g2o_id))
        {
            return false;
        }

        const g2o::VertexSE3 *vertex =
            dynamic_cast<const g2o::VertexSE3 *>(
                optimizer.vertex(g2o_id));

        if (vertex == nullptr)
        {
            return false;
        }

        const Eigen::Isometry3d T_WK_optimized =
            vertex->estimate();

        if (!pose_graph.SetNodePose(
                node.id,
                T_WK_optimized))
        {
            return false;
        }
    }

    result.success = true;
    if (odometry_information_file.is_open())
    {
        odometry_information_file.close();
    }
    return true;
}
