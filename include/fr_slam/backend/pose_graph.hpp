#pragma once

#include <cstddef>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

// ============================================================================
// Keyframe Pose Graph
//
// Canonical state convention:
//
//     X_i = T_WK_i
//
// where K_i is the LiDAR frame of Keyframe i and T_AB maps B -> A.
//
// Submaps are deliberately NOT PoseGraph vertices. They remain auxiliary
// geometry containers for Scan-to-LocalMap and loop ICP verification.
// ============================================================================

enum class PoseGraphEdgeType
{
    Odometry,
    Loop
};

struct PoseGraphNode
{
    std::size_t id = 0;

    // Canonical backend state: Keyframe -> World.
    Eigen::Isometry3d T_WK =
        Eigen::Isometry3d::Identity();

    // Transitional compatibility mirror for older source files that still
    // refer to PoseGraphNode::T_WS. It carries exactly the same transform as
    // T_WK and is updated together by AddNode()/SetNodePose().
    //
    // New backend code must use T_WK. This member can be removed after all
    // legacy Submap-graph consumers have been migrated.
    Eigen::Isometry3d T_WS =
        Eigen::Isometry3d::Identity();

    bool fixed = false;

    // Immutable frontend gravity/up-direction reference expressed in the
    // Keyframe LiDAR frame.  It is generated from the raw frontend pose:
    //
    //     g_L_ref = R_WL(raw)^T * UnitZ
    //
    // A pure world-yaw correction leaves this vector unchanged, while an
    // artificial roll/pitch tilt changes it.  The backend optimizer therefore
    // uses it as a gravity-direction prior without forcing roll/pitch to zero.
    Eigen::Vector3d gravity_L_reference =
        Eigen::Vector3d::UnitZ();

    bool has_gravity_reference = false;
};

struct PoseGraphEdge
{
    std::size_t from_id = 0;
    std::size_t to_id = 0;

    // Measurement convention:
    //
    //     Z_ij = T_Ki_Kj = T_WKi^-1 * T_WKj
    //
    // It maps the `to` Keyframe into the `from` Keyframe.
    Eigen::Isometry3d T_from_to =
        Eigen::Isometry3d::Identity();

    Eigen::Matrix<double, 6, 6> information =
        Eigen::Matrix<double, 6, 6>::Identity();

    PoseGraphEdgeType type =
        PoseGraphEdgeType::Odometry;
};

class PoseGraph
{
public:
    bool AddNode(
        std::size_t id,
        const Eigen::Isometry3d &T_WK,
        bool fixed = false);

    // Backend optimizer write-back hook. This updates only the graph estimate;
    // it does not modify the running Scan-to-LocalMap frontend state.
    bool SetNodePose(
        std::size_t id,
        const Eigen::Isometry3d &T_WK);

    // Store one immutable gravity/up-direction reference for a Keyframe.
    // The reference is intentionally separate from T_WK so later G2O
    // corrections cannot overwrite the physical roll/pitch reference.
    bool SetNodeGravityReference(
        std::size_t id,
        const Eigen::Vector3d &gravity_L_reference);

    bool AddOdometryEdge(
        std::size_t from_id,
        std::size_t to_id,
        const Eigen::Isometry3d &T_from_to,
        const Eigen::Matrix<double, 6, 6> &information);

    bool AddLoopEdge(
        std::size_t from_id,
        std::size_t to_id,
        const Eigen::Isometry3d &T_from_to,
        const Eigen::Matrix<double, 6, 6> &information);

    // Remove the newest matching loop edge.  This is used as a transactional
    // rollback when a newly inserted loop makes the optimizer violate the
    // gravity hard guard.
    bool RemoveLoopEdge(
        std::size_t from_id,
        std::size_t to_id);

    bool HasNode(
        std::size_t id) const;

    const PoseGraphNode *GetNode(
        std::size_t id) const;

    const std::vector<PoseGraphNode> &GetNodes() const;
    const std::vector<PoseGraphEdge> &GetEdges() const;

    std::size_t NodeCount() const;
    std::size_t EdgeCount() const;
    std::size_t OdometryEdgeCount() const;
    std::size_t LoopEdgeCount() const;

    void Clear();

private:
    bool AddEdge(
        std::size_t from_id,
        std::size_t to_id,
        const Eigen::Isometry3d &T_from_to,
        const Eigen::Matrix<double, 6, 6> &information,
        PoseGraphEdgeType type);

private:
    std::vector<PoseGraphNode> nodes_;
    std::vector<PoseGraphEdge> edges_;

    std::unordered_map<std::size_t, std::size_t>
        node_index_by_id_;
};
