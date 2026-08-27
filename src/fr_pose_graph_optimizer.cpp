#include "fr_slam/fr_pose_graph_optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>
#include <Eigen/StdVector>

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
constexpr double kPi =
    3.14159265358979323846;

constexpr double kRadToDeg =
    180.0 / kPi;

using AlignedVector2d =
    std::vector<
        Eigen::Vector2d,
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
class EdgeGravityDirection
    : public g2o::BaseUnaryEdge<
          3,
          Eigen::Vector3d,
          g2o::VertexSE3>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeGravityDirection()
    {
        information().setIdentity();
    }

    void computeError() override
    {
        const g2o::VertexSE3 *vertex =
            static_cast<const g2o::VertexSE3 *>(
                _vertices[0]);

        if (vertex == nullptr)
        {
            _error.setConstant(
                std::numeric_limits<double>::quiet_NaN());
            return;
        }

        Eigen::Vector3d gravity_estimated =
            vertex->estimate().rotation().transpose() *
            Eigen::Vector3d::UnitZ();

        const double estimated_norm =
            gravity_estimated.norm();

        Eigen::Vector3d gravity_measurement =
            _measurement;

        const double measurement_norm =
            gravity_measurement.norm();

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
        _error =
            gravity_estimated -
            gravity_measurement;
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

bool ToG2oId(
    std::size_t id,
    int &g2o_id)
{
    if (id > static_cast<std::size_t>(
                 std::numeric_limits<int>::max()))
    {
        return false;
    }

    g2o_id = static_cast<int>(id);
    return true;
}

bool IsFinitePositiveSemidefiniteInformation(
    const Eigen::Matrix<double, 6, 6> &information)
{
    if (!information.allFinite())
    {
        return false;
    }

    for (int i = 0; i < 6; ++i)
    {
        if (information(i, i) <= 0.0)
        {
            return false;
        }
    }

    return true;
}

double ClampUnit(
    double value)
{
    return std::max(
        -1.0,
        std::min(
            1.0,
            value));
}

double GravityTiltErrorDeg(
    const Eigen::Matrix3d &R_WK,
    const Eigen::Vector3d &gravity_L_reference)
{
    Eigen::Vector3d gravity_estimated =
        R_WK.transpose() *
        Eigen::Vector3d::UnitZ();

    Eigen::Vector3d gravity_reference =
        gravity_L_reference;

    const double estimated_norm =
        gravity_estimated.norm();

    const double reference_norm =
        gravity_reference.norm();

    if (!gravity_estimated.allFinite() ||
        !gravity_reference.allFinite() ||
        estimated_norm < 1.0e-12 ||
        reference_norm < 1.0e-12)
    {
        return std::numeric_limits<double>::infinity();
    }

    gravity_estimated /= estimated_norm;
    gravity_reference /= reference_norm;

    const double cosine =
        ClampUnit(
            gravity_estimated.dot(
                gravity_reference));

    return std::acos(cosine) *
           kRadToDeg;
}

Eigen::Vector3d RotationToRpy(
    const Eigen::Matrix3d &R)
{
    Eigen::Vector3d rpy =
        Eigen::Vector3d::Zero();

    const double sin_pitch =
        ClampUnit(-R(2, 0));

    rpy.y() =
        std::asin(sin_pitch);

    rpy.x() =
        std::atan2(
            R(2, 1),
            R(2, 2));

    rpy.z() =
        std::atan2(
            R(1, 0),
            R(0, 0));

    return rpy;
}

double WrappedAngleDifferenceDeg(
    double first,
    double second)
{
    const double difference =
        first - second;

    return std::abs(
               std::atan2(
                   std::sin(difference),
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
double ComputeXyPcaRatio(
    const AlignedVector2d &positions)
{
    if (positions.size() < 3)
    {
        return 0.0;
    }

    Eigen::Vector2d mean =
        Eigen::Vector2d::Zero();

    for (const Eigen::Vector2d &position : positions)
    {
        if (!position.allFinite())
        {
            return std::numeric_limits<double>::quiet_NaN();
        }

        mean += position;
    }

    mean /= static_cast<double>(positions.size());

    Eigen::Matrix2d covariance =
        Eigen::Matrix2d::Zero();

    for (const Eigen::Vector2d &position : positions)
    {
        const Eigen::Vector2d delta =
            position - mean;

        covariance.noalias() +=
            delta * delta.transpose();
    }

    covariance /= static_cast<double>(positions.size());

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(covariance);

    if (solver.info() != Eigen::Success)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const Eigen::Vector2d eigenvalues =
        solver.eigenvalues();

    const double lambda_min =
        std::max(0.0, eigenvalues.x());

    const double lambda_max =
        std::max(0.0, eigenvalues.y());

    if (!std::isfinite(lambda_min) ||
        !std::isfinite(lambda_max) ||
        lambda_max < 1.0e-12)
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
double ComputeXyPathLength(
    const AlignedVector2d &positions)
{
    if (positions.size() < 2)
    {
        return 0.0;
    }

    double length = 0.0;

    for (std::size_t i = 1;
         i < positions.size();
         ++i)
    {
        if (!positions[i - 1].allFinite() ||
            !positions[i].allFinite())
        {
            return std::numeric_limits<double>::quiet_NaN();
        }

        length +=
            (positions[i] - positions[i - 1]).norm();
    }

    return length;
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
        edge->setInformation(edge_data.information);

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

            Eigen::Vector3d rotation_axis =
                rotation_update.axis();

            if (!rotation_axis.allFinite())
            {
                rotation_axis =
                    Eigen::Vector3d::Zero();
            }

            std::cout
                << "Large G2O rotation update"
                << " | kf=" << node.id
                << " | translation="
                << translation_update << " m"
                << " | rotation="
                << rotation_update_deg << " deg"
                << " | axis=["
                << rotation_axis.transpose()
                << "]"
                << " | droll="
                << roll_update_deg << " deg"
                << " | dpitch="
                << pitch_update_deg << " deg"
                << " | dyaw="
                << yaw_update_deg << " deg"
                << std::endl;
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

    std::cout
        << "G2O maximum update detail"
        << " | translation_kf="
        << max_translation_update_node_id
        << " | translation="
        << result.max_translation_update << " m"
        << " | rotation_kf="
        << max_rotation_update_node_id
        << " | rotation="
        << result.max_rotation_update_deg << " deg"
        << " | large_rotation_nodes="
        << large_rotation_update_count
        << " | threshold="
        << large_rotation_threshold_deg << " deg"
        << std::endl;

    std::cout
        << "PoseGraph RPY diagnostics"
        << " | max_droll="
        << result.max_roll_update_deg << " deg"
        << " | roll_kf="
        << result.max_roll_update_keyframe_id
        << " | max_dpitch="
        << result.max_pitch_update_deg << " deg"
        << " | pitch_kf="
        << result.max_pitch_update_keyframe_id
        << " | max_dyaw="
        << result.max_yaw_update_deg << " deg"
        << " | yaw_kf="
        << result.max_yaw_update_keyframe_id
        << std::endl;

    std::cout
        << "PoseGraph gravity diagnostics"
        << " | gravity_edges="
        << result.gravity_edges
        << " | gravity_reference_nodes="
        << result.gravity_reference_nodes
        << " | mean_tilt_error="
        << result.mean_gravity_tilt_error_deg << " deg"
        << " | max_tilt_error="
        << result.max_gravity_tilt_error_deg << " deg"
        << " | worst_kf="
        << result.worst_gravity_keyframe_id
        << " | hard_limit="
        << config_.max_gravity_tilt_error_deg << " deg"
        << std::endl;

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

    std::cout
        << "PoseGraph trajectory shape diagnostics"
        << " | xy_pca_before="
        << result.xy_pca_ratio_before
        << " | xy_pca_after="
        << result.xy_pca_ratio_after
        << " | pca_scale="
        << (result.xy_pca_ratio_before > 1.0e-12
                ? result.xy_pca_ratio_after /
                      result.xy_pca_ratio_before
                : 1.0)
        << " | path_length_before="
        << result.path_length_before << " m"
        << " | path_length_after="
        << result.path_length_after << " m"
        << " | path_length_ratio="
        << result.path_length_ratio
        << std::endl;

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

    std::cout
        << "PoseGraph trajectory shape guard"
        << " | accepted=true"
        << " | xy_pca_before="
        << result.xy_pca_ratio_before
        << " | xy_pca_after="
        << result.xy_pca_ratio_after
        << " | path_length_ratio="
        << result.path_length_ratio
        << std::endl;

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

    std::cout
        << "PoseGraph gravity guard"
        << " | accepted=true"
        << " | max_tilt_error="
        << result.max_gravity_tilt_error_deg << " deg"
        << " | hard_limit="
        << config_.max_gravity_tilt_error_deg << " deg"
        << std::endl;

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

    return true;
}
