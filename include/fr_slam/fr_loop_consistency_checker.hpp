#pragma once

#include <cstddef>
#include <deque>
#include <limits>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "fr_slam/fr_pose_graph.hpp"

// ============================================================================
// LoopConsistencyChecker
//
// This class is deliberately placed AFTER LoopVerifier.
//
// LoopVerifier answers:
//
//     "Can these two frozen Submap clouds be geometrically aligned?"
//
// LoopConsistencyChecker answers:
//
//     "Is that geometry-valid proposal also consistent with nearby loop
//      proposals and with the local PoseGraph trajectory?"
//
// Two checks are used:
//
//     1. Temporal consistency
//     2. Cycle consistency
//
// The class does NOT run ICP and does NOT optimize the PoseGraph.
// ============================================================================

struct LoopConsistencyConfig
{
        // ------------------------------------------------------------------------
        // Temporal consistency.
        //
        // Example of a consistent sequence:
        //
        //     current 46 -> historical 3
        //     current 47 -> historical 4
        //     current 48 -> historical 5
        //
        // The gaps are intentionally allowed to be slightly larger than one
        // because Scan Context/Submap overlap does not guarantee exact +1 matches.
        // ------------------------------------------------------------------------
        bool use_temporal_consistency = true;

        std::size_t temporal_max_current_gap = 2;

        std::size_t temporal_max_historical_gap = 3;

        // Includes the current proposal itself.
        //
        // Value 3 means:
        //
        //     two previous supporting proposals
        //     +
        //     current proposal.
        //
        // Example:
        //
        //     current 17 -> historical 4    PENDING
        //     current 18 -> historical 5    supporting proposal
        //     current 19 -> historical 6    third consistent proposal
        //
        // Only after this sequence can temporal consistency pass.
        std::size_t temporal_min_consistent_proposals = 3;

        // ------------------------------------------------------------------------
        // Cycle consistency.
        //
        // A short cycle is constructed from:
        //
        //     previous geometry-valid loop
        //     + historical-side PoseGraph motion
        //     + current-side PoseGraph motion
        //     + current geometry-valid loop
        //
        // The resulting transform should be close to Identity.
        // ------------------------------------------------------------------------
        bool use_cycle_consistency = true;

        double max_cycle_translation_error = 2.0;

        double max_cycle_rotation_error_deg = 10.0;

        // Number of recent geometry-valid proposals retained for support.
        std::size_t max_history_size = 20;
};

// ============================================================================
// Geometry-valid loop proposal.
//
// Transform convention:
//
//     T_AB maps B -> A
//
// Therefore:
//
//     T_historical_current
//
// maps coordinates:
//
//     current Submap -> historical Submap
// ============================================================================
struct LoopConsistencyProposal
{
        std::size_t current_id = 0;

        std::size_t historical_id = 0;

        Eigen::Isometry3d T_historical_current =
            Eigen::Isometry3d::Identity();

        double rmse =
            std::numeric_limits<double>::infinity();

        double overlap_ratio = 0.0;
};

struct LoopConsistencyResult
{
        bool valid = false;

        // Temporal diagnostics.
        bool temporal_available = false;
        bool temporal_consistent = false;
        std::size_t temporal_support = 0;

        // Cycle diagnostics.
        bool cycle_available = false;
        bool cycle_consistent = false;

        double cycle_translation_error =
            std::numeric_limits<double>::infinity();

        double cycle_rotation_error_deg =
            std::numeric_limits<double>::infinity();

        // Final decision.
        bool accepted = false;
};

class LoopConsistencyChecker
{
public:
        explicit LoopConsistencyChecker(
            const LoopConsistencyConfig &config =
                LoopConsistencyConfig());

        // Check one geometry-valid proposal.
        //
        // NOTE:
        //     This method changes internal history, therefore it is intentionally
        //     non-const.
        LoopConsistencyResult Check(
            const PoseGraph &pose_graph,
            const LoopConsistencyProposal &proposal);

        void Reset();

        const LoopConsistencyConfig &GetConfig() const;

private:
        bool IsTemporalNeighbour(
            const LoopConsistencyProposal &previous,
            const LoopConsistencyProposal &current) const;

        bool EvaluateCycle(
            const PoseGraph &pose_graph,
            const LoopConsistencyProposal &reference,
            const LoopConsistencyProposal &current,
            double &translation_error,
            double &rotation_error_deg) const;

        static std::size_t IdGap(
            std::size_t a,
            std::size_t b);

        static double RotationAngleDeg(
            const Eigen::Matrix3d &rotation);

private:
        LoopConsistencyConfig config_;

        // IMPORTANT:
        //     These are geometry-valid proposals, not necessarily finally accepted
        //     loop edges. The first good proposal can therefore remain PENDING and
        //     still provide support for the next proposal.
        std::deque<LoopConsistencyProposal> history_;
};
