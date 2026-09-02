#include "fr_slam/fr_pose_graph.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>


bool PoseGraph::AddNode(
    std::size_t id,
    const Eigen::Isometry3d &T_WK,
    bool fixed)
{
    if (HasNode(id))
    {
        return false;
    }

    if (!T_WK.matrix().allFinite())
    {
        return false;
    }

    PoseGraphNode node;
    node.id = id;
    node.T_WK = T_WK;
    node.T_WS = T_WK;
    node.fixed = fixed;

    node_index_by_id_[id] =
        nodes_.size();

    nodes_.push_back(node);

    return true;
}


bool PoseGraph::SetNodePose(
    std::size_t id,
    const Eigen::Isometry3d &T_WK)
{
    if (!T_WK.matrix().allFinite())
    {
        return false;
    }

    const auto iterator =
        node_index_by_id_.find(id);

    if (iterator == node_index_by_id_.end())
    {
        return false;
    }

    const std::size_t index =
        iterator->second;

    if (index >= nodes_.size())
    {
        return false;
    }

    nodes_[index].T_WK = T_WK;
    nodes_[index].T_WS = T_WK;

    return true;
}


bool PoseGraph::SetNodeGravityReference(
    std::size_t id,
    const Eigen::Vector3d &gravity_L_reference)
{
    if (!gravity_L_reference.allFinite())
    {
        return false;
    }

    const double norm =
        gravity_L_reference.norm();

    if (!std::isfinite(norm) ||
        norm < 1.0e-9)
    {
        return false;
    }

    const auto iterator =
        node_index_by_id_.find(id);

    if (iterator == node_index_by_id_.end())
    {
        return false;
    }

    const std::size_t index =
        iterator->second;

    if (index >= nodes_.size())
    {
        return false;
    }

    nodes_[index].gravity_L_reference =
        gravity_L_reference / norm;

    nodes_[index].has_gravity_reference = true;

    return true;
}


bool PoseGraph::AddOdometryEdge(
    std::size_t from_id,
    std::size_t to_id,
    const Eigen::Isometry3d &T_from_to,
    const Eigen::Matrix<double, 6, 6> &information)
{
    return AddEdge(
        from_id,
        to_id,
        T_from_to,
        information,
        PoseGraphEdgeType::Odometry);
}


bool PoseGraph::AddLoopEdge(
    std::size_t from_id,
    std::size_t to_id,
    const Eigen::Isometry3d &T_from_to,
    const Eigen::Matrix<double, 6, 6> &information)
{
    return AddEdge(
        from_id,
        to_id,
        T_from_to,
        information,
        PoseGraphEdgeType::Loop);
}


bool PoseGraph::RemoveLoopEdge(
    std::size_t from_id,
    std::size_t to_id)
{
    for (auto iterator = edges_.rbegin();
         iterator != edges_.rend();
         ++iterator)
    {
        if (iterator->type != PoseGraphEdgeType::Loop)
        {
            continue;
        }

        if (iterator->from_id != from_id ||
            iterator->to_id != to_id)
        {
            continue;
        }

        edges_.erase(std::next(iterator).base());
        return true;
    }

    return false;
}


bool PoseGraph::HasNode(
    std::size_t id) const
{
    return node_index_by_id_.find(id) !=
           node_index_by_id_.end();
}


const PoseGraphNode *PoseGraph::GetNode(
    std::size_t id) const
{
    const auto iterator =
        node_index_by_id_.find(id);

    if (iterator == node_index_by_id_.end())
    {
        return nullptr;
    }

    const std::size_t index =
        iterator->second;

    if (index >= nodes_.size())
    {
        return nullptr;
    }

    return &nodes_[index];
}


const std::vector<PoseGraphNode> &
PoseGraph::GetNodes() const
{
    return nodes_;
}


const std::vector<PoseGraphEdge> &
PoseGraph::GetEdges() const
{
    return edges_;
}


std::size_t PoseGraph::NodeCount() const
{
    return nodes_.size();
}


std::size_t PoseGraph::EdgeCount() const
{
    return edges_.size();
}


std::size_t PoseGraph::OdometryEdgeCount() const
{
    return static_cast<std::size_t>(
        std::count_if(
            edges_.begin(),
            edges_.end(),
            [](const PoseGraphEdge &edge)
            {
                return edge.type ==
                       PoseGraphEdgeType::Odometry;
            }));
}


std::size_t PoseGraph::LoopEdgeCount() const
{
    return static_cast<std::size_t>(
        std::count_if(
            edges_.begin(),
            edges_.end(),
            [](const PoseGraphEdge &edge)
            {
                return edge.type ==
                       PoseGraphEdgeType::Loop;
            }));
}


void PoseGraph::Clear()
{
    nodes_.clear();
    edges_.clear();
    node_index_by_id_.clear();
}


bool PoseGraph::AddEdge(
    std::size_t from_id,
    std::size_t to_id,
    const Eigen::Isometry3d &T_from_to,
    const Eigen::Matrix<double, 6, 6> &information,
    PoseGraphEdgeType type)
{
    if (from_id == to_id)
    {
        return false;
    }

    if (!HasNode(from_id) ||
        !HasNode(to_id))
    {
        return false;
    }

    if (!T_from_to.matrix().allFinite())
    {
        return false;
    }

    if (!information.allFinite())
    {
        return false;
    }

    PoseGraphEdge edge;
    edge.from_id = from_id;
    edge.to_id = to_id;
    edge.T_from_to = T_from_to;
    edge.information = information;
    edge.type = type;

    edges_.push_back(edge);

    return true;
}
