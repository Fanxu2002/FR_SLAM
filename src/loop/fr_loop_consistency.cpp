#include "fr_slam/fr_loop_consistency.hpp"

#include <algorithm>
#include <cmath>

namespace
{

        constexpr double kPi =
            3.14159265358979323846;

} // namespace

LoopConsistencyChecker::LoopConsistencyChecker(
    const LoopConsistencyConfig &config)
    : config_(config)
{
        if (config_.temporal_max_current_gap == 0)
        {
                config_.temporal_max_current_gap = 2;
        }

        if (config_.temporal_max_historical_gap == 0)
        {
                config_.temporal_max_historical_gap = 3;
        }

        if (config_.temporal_min_consistent_proposals < 2)
        {
                config_.temporal_min_consistent_proposals = 3;
        }

        if (!std::isfinite(
                config_.max_cycle_translation_error) ||
            config_.max_cycle_translation_error <= 0.0)
        {
                config_.max_cycle_translation_error = 2.0;
        }

        if (!std::isfinite(
                config_.max_cycle_rotation_error_deg) ||
            config_.max_cycle_rotation_error_deg <= 0.0)
        {
                config_.max_cycle_rotation_error_deg = 10.0;
        }

        if (config_.max_history_size == 0)
        {
                config_.max_history_size = 20;
        }
}

std::size_t LoopConsistencyChecker::IdGap(
    std::size_t a,
    std::size_t b)
{
        return a > b
                   ? a - b
                   : b - a;
}

double LoopConsistencyChecker::RotationAngleDeg(
    const Eigen::Matrix3d &rotation)
{
        if (!rotation.allFinite())
        {
                return std::numeric_limits<double>::infinity();
        }

        Eigen::Quaterniond quaternion(
            rotation);

        if (!quaternion.coeffs().allFinite() ||
            quaternion.norm() < 1.0e-12)
        {
                return std::numeric_limits<double>::infinity();
        }

        quaternion.normalize();

        const double w =
            std::clamp(
                std::abs(
                    quaternion.w()),
                0.0,
                1.0);

        return 2.0 *
               std::acos(w) *
               180.0 /
               kPi;
}

bool LoopConsistencyChecker::IsTemporalNeighbour(
    const LoopConsistencyProposal &previous,
    const LoopConsistencyProposal &current) const
{
        // Multiple Top-K candidates from the same current Submap must never
        // support one another.
        if (previous.current_id ==
            current.current_id)
        {
                return false;
        }

        const std::size_t current_gap =
            IdGap(
                previous.current_id,
                current.current_id);

        const std::size_t historical_gap =
            IdGap(
                previous.historical_id,
                current.historical_id);

        if (current_gap >
            config_.temporal_max_current_gap)
        {
                return false;
        }

        if (historical_gap >
            config_.temporal_max_historical_gap)
        {
                return false;
        }

        return true;
}

// ============================================================================
// EvaluateCycle()
//
// Transform convention:
//
//     T_AB maps B -> A.
//
// Previous geometry-valid loop:
//
//     reference.T_historical_current
//       = T_hPrev_cPrev
//
// Current proposal:
//
//     current.T_historical_current
//       = T_hNow_cNow
//
// PoseGraph node states are T_WS. Therefore:
//
//     T_cPrev_cNow
//       = T_WScPrev^-1 * T_WScNow
//
//     T_hNow_hPrev
//       = T_WShNow^-1 * T_WShPrev
//
// Using the previous loop and the two short trajectory segments, predict the
// current loop:
//
//     T_predicted_hNow_cNow
//       = T_hNow_hPrev
//       * T_hPrev_cPrev
//       * T_cPrev_cNow
//
// Then:
//
//     error
//       = T_predicted^-1 * T_measured
//       ~= Identity
// ============================================================================
bool LoopConsistencyChecker::EvaluateCycle(
    const PoseGraph &pose_graph,
    const LoopConsistencyProposal &reference,
    const LoopConsistencyProposal &current,
    double &translation_error,
    double &rotation_error_deg) const
{
        translation_error =
            std::numeric_limits<double>::infinity();

        rotation_error_deg =
            std::numeric_limits<double>::infinity();

        const PoseGraphNode *historical_previous =
            pose_graph.GetNode(
                reference.historical_id);

        const PoseGraphNode *current_previous =
            pose_graph.GetNode(
                reference.current_id);

        const PoseGraphNode *historical_now =
            pose_graph.GetNode(
                current.historical_id);

        const PoseGraphNode *current_now =
            pose_graph.GetNode(
                current.current_id);

        if (historical_previous == nullptr ||
            current_previous == nullptr ||
            historical_now == nullptr ||
            current_now == nullptr)
        {
                return false;
        }

        if (!historical_previous->T_WS.matrix().allFinite() ||
            !current_previous->T_WS.matrix().allFinite() ||
            !historical_now->T_WS.matrix().allFinite() ||
            !current_now->T_WS.matrix().allFinite() ||
            !reference.T_historical_current.matrix().allFinite() ||
            !current.T_historical_current.matrix().allFinite())
        {
                return false;
        }

        // current_now -> current_previous
        const Eigen::Isometry3d T_current_previous_current_now =
            current_previous->T_WS.inverse() *
            current_now->T_WS;

        // historical_previous -> historical_now
        const Eigen::Isometry3d T_historical_now_historical_previous =
            historical_now->T_WS.inverse() *
            historical_previous->T_WS;

        // current_now -> historical_now
        const Eigen::Isometry3d predicted =
            T_historical_now_historical_previous *
            reference.T_historical_current *
            T_current_previous_current_now;

        if (!predicted.matrix().allFinite())
        {
                return false;
        }

        const Eigen::Isometry3d error =
            predicted.inverse() *
            current.T_historical_current;

        if (!error.matrix().allFinite())
        {
                return false;
        }

        translation_error =
            error.translation().norm();

        rotation_error_deg =
            RotationAngleDeg(
                error.rotation());

        return std::isfinite(
                   translation_error) &&
               std::isfinite(
                   rotation_error_deg);
}

LoopConsistencyResult
LoopConsistencyChecker::Check(
    const PoseGraph &pose_graph,
    const LoopConsistencyProposal &proposal)
{
        LoopConsistencyResult result;

        if (!proposal.T_historical_current
                 .matrix()
                 .allFinite())
        {
                return result;
        }

        if (!pose_graph.HasNode(
                proposal.current_id) ||
            !pose_graph.HasNode(
                proposal.historical_id))
        {
                return result;
        }

        result.valid = true;

        // ------------------------------------------------------------------------
        // Search recent geometry-valid proposals for temporal support.
        // ------------------------------------------------------------------------
        std::size_t support = 0;

        const LoopConsistencyProposal *cycle_reference =
            nullptr;

        for (auto iterator =
                 history_.rbegin();
             iterator != history_.rend();
             ++iterator)
        {
                const LoopConsistencyProposal &previous =
                    *iterator;

                if (!IsTemporalNeighbour(
                        previous,
                        proposal))
                {
                        continue;
                }

                ++support;

                // The most recent temporal neighbour is used as the short-cycle
                // reference because its local odometry segments are normally shortest.
                if (cycle_reference == nullptr)
                {
                        cycle_reference =
                            &previous;
                }
        }

        result.temporal_support =
            support;

        result.temporal_available =
            support > 0;

        if (!config_.use_temporal_consistency)
        {
                result.temporal_consistent = true;
        }
        else
        {
                // +1 includes the current proposal itself.
                result.temporal_consistent =
                    (support + 1) >=
                    config_
                        .temporal_min_consistent_proposals;
        }

        // ------------------------------------------------------------------------
        // Cycle consistency.
        // ------------------------------------------------------------------------
        if (!config_.use_cycle_consistency)
        {
                result.cycle_consistent = true;
        }
        else if (cycle_reference != nullptr)
        {
                result.cycle_available =
                    EvaluateCycle(
                        pose_graph,
                        *cycle_reference,
                        proposal,
                        result.cycle_translation_error,
                        result.cycle_rotation_error_deg);

                if (result.cycle_available)
                {
                        result.cycle_consistent =
                            result.cycle_translation_error <=
                                config_
                                    .max_cycle_translation_error &&
                            result.cycle_rotation_error_deg <=
                                config_
                                    .max_cycle_rotation_error_deg;
                }
        }

        result.accepted =
            result.temporal_consistent &&
            result.cycle_consistent;

        // ------------------------------------------------------------------------
        // Decide whether this proposal is allowed to enter consistency history.
        //
        // We need to distinguish:
        //
        // 1. First geometry-valid proposal:
        //
        //        no temporal neighbour yet
        //
        //    This proposal must be stored as a SEED, otherwise a later proposal
        //    can never obtain temporal / cycle support.
        //
        // 2. Proposal with a valid and consistent cycle:
        //
        //        safe to keep in history.
        //
        // 3. Proposal with an available but inconsistent cycle:
        //
        //        this is an explicit geometric inconsistency.
        //
        //    DO NOT store it, otherwise a rejected false loop could support later
        //    false proposals.
        // ------------------------------------------------------------------------
        bool store_proposal = false;

        // ------------------------------------------------------------------------
        // Case A:
        // No previous temporal neighbour exists.
        //
        // This is the first proposal of a possible loop sequence.
        // Keep it as a seed.
        // ------------------------------------------------------------------------
        if (!result.temporal_available)
        {
                store_proposal = true;
        }
        // ------------------------------------------------------------------------
        // Case B:
        // Cycle consistency checking is disabled.
        //
        // Temporal consistency needs historical proposals, so geometry-valid
        // proposals may continue to build the temporal sequence.
        // ------------------------------------------------------------------------
        else if (!config_.use_cycle_consistency)
        {
                store_proposal = true;
        }
        // ------------------------------------------------------------------------
        // Case C:
        // A temporal neighbour exists AND cycle consistency is enabled.
        //
        // Only a successfully evaluated, cycle-consistent proposal may enter
        // history.
        // ------------------------------------------------------------------------
        else if (result.cycle_available &&
                 result.cycle_consistent)
        {
                store_proposal = true;
        }

        // Explicit cycle failure:
        //
        //     cycle_available == true
        //     cycle_consistent == false
        //
        // reaches here with:
        //
        //     store_proposal == false
        //
        // and is deliberately NOT written into history.

        if (store_proposal)
        {
                history_.push_back(
                    proposal);

                while (history_.size() >
                       config_.max_history_size)
                {
                        history_.pop_front();
                }
        }

        return result;
}

void LoopConsistencyChecker::Reset()
{
        history_.clear();
}

const LoopConsistencyConfig &
LoopConsistencyChecker::GetConfig() const
{
        return config_;
}
