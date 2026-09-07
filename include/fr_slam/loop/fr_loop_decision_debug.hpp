#pragma once

#include <algorithm>
#include <cstddef>
#include <mutex>
#include <utility>
#include <vector>

// ============================================================================
// Loop decision RViz debug bridge.
//
// Purpose:
//   The PoseGraph itself only contains loop factors that were actually
//   committed.  That makes RViz red loop lines sparse by design and hides
//   useful evidence such as LOOP_EDGE_SPACING / TRACK_ONLY observations.
//
//   This lightweight in-process snapshot records verified loop observations
//   that were intentionally NOT inserted into the PoseGraph, so the ROS node
//   can visualize them without changing graph semantics.
//
// Visualization semantics used by fr_slam_node.cpp:
//   TRACK_ONLY_SPACING       -> yellow thin line; strict cycle/new-cluster
//                               verified, omitted only by graph spacing.
//   TRACK_ONLY_CYCLE_RELAXED -> yellow thin line; geometry/graph/temporal
//                               verified and within V22.8 relaxed TRACK cycle
//                               gate, but NEVER eligible for a PGO factor.
//   CYCLE_INCONSISTENT       -> orange thin line
//   NEW_CLUSTER_PENDING      -> orange thin line
//
// IMPORTANT:
//   These edges are debug evidence only.  They are never PoseGraph factors.
// ============================================================================
namespace fr_slam_debug
{

enum class LoopDecisionDebugKind
{
    TRACK_ONLY_SPACING = 0,
    CYCLE_INCONSISTENT = 1,
    NEW_CLUSTER_PENDING = 2,
    TRACK_ONLY_CYCLE_RELAXED = 3
};

struct LoopDecisionDebugEdge
{
    std::size_t historical_keyframe_id = 0;
    std::size_t current_keyframe_id = 0;
    LoopDecisionDebugKind kind =
        LoopDecisionDebugKind::TRACK_ONLY_SPACING;
};

struct LoopDecisionDebugSnapshot
{
    std::size_t revision = 0;
    std::vector<LoopDecisionDebugEdge> edges;
};

inline std::mutex &LoopDecisionDebugMutex()
{
    static std::mutex mutex;
    return mutex;
}

inline LoopDecisionDebugSnapshot &MutableLoopDecisionDebugSnapshot()
{
    static LoopDecisionDebugSnapshot snapshot;
    return snapshot;
}

inline void RecordLoopDecisionDebugEdge(
    const std::size_t historical_keyframe_id,
    const std::size_t current_keyframe_id,
    const LoopDecisionDebugKind kind)
{
    std::lock_guard<std::mutex> lock(
        LoopDecisionDebugMutex());

    LoopDecisionDebugSnapshot &snapshot =
        MutableLoopDecisionDebugSnapshot();

    const auto existing =
        std::find_if(
            snapshot.edges.begin(),
            snapshot.edges.end(),
            [historical_keyframe_id,
             current_keyframe_id,
             kind](const LoopDecisionDebugEdge &edge)
            {
                return
                    edge.historical_keyframe_id == historical_keyframe_id &&
                    edge.current_keyframe_id == current_keyframe_id &&
                    edge.kind == kind;
            });

    if (existing == snapshot.edges.end())
    {
        LoopDecisionDebugEdge edge;
        edge.historical_keyframe_id =
            historical_keyframe_id;
        edge.current_keyframe_id =
            current_keyframe_id;
        edge.kind = kind;

        snapshot.edges.push_back(edge);

        // Keep the debug history bounded.  This is far above the number of
        // edges expected in normal bags, while preventing unbounded growth in
        // very long runs.
        constexpr std::size_t kMaxDebugEdges = 4096;
        if (snapshot.edges.size() > kMaxDebugEdges)
        {
            snapshot.edges.erase(
                snapshot.edges.begin(),
                snapshot.edges.begin() +
                    static_cast<std::ptrdiff_t>(
                        snapshot.edges.size() - kMaxDebugEdges));
        }

        ++snapshot.revision;
    }
}

inline void RemoveLoopDecisionDebugEdge(
    const std::size_t historical_keyframe_id,
    const std::size_t current_keyframe_id)
{
    std::lock_guard<std::mutex> lock(
        LoopDecisionDebugMutex());

    LoopDecisionDebugSnapshot &snapshot =
        MutableLoopDecisionDebugSnapshot();

    const std::size_t old_size =
        snapshot.edges.size();

    snapshot.edges.erase(
        std::remove_if(
            snapshot.edges.begin(),
            snapshot.edges.end(),
            [historical_keyframe_id,
             current_keyframe_id](const LoopDecisionDebugEdge &edge)
            {
                return
                    edge.historical_keyframe_id == historical_keyframe_id &&
                    edge.current_keyframe_id == current_keyframe_id;
            }),
        snapshot.edges.end());

    if (snapshot.edges.size() != old_size)
    {
        ++snapshot.revision;
    }
}

inline LoopDecisionDebugSnapshot GetLoopDecisionDebugSnapshot()
{
    std::lock_guard<std::mutex> lock(
        LoopDecisionDebugMutex());

    return MutableLoopDecisionDebugSnapshot();
}

} // namespace fr_slam_debug
