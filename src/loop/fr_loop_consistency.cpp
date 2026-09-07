#include "fr_slam/loop/fr_loop_consistency.hpp"

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
    // ------------------------------------------------------------------------
    // Current-side Keyframe IDs must move forward in time.
    //
    // history_ contains older geometry-valid proposals, therefore:
    //
    //     previous.current_id < current.current_id
    //
    // must hold.
    //
    // This is deliberately NOT an absolute-ID comparison.
    // ------------------------------------------------------------------------
    if (current.current_id <=
        previous.current_id)
    {
        return false;
    }

    const std::size_t current_gap =
        current.current_id -
        previous.current_id;

    if (current_gap >
        config_.temporal_max_current_gap)
    {
        return false;
    }

    // ------------------------------------------------------------------------
    // Historical-side motion may be either:
    //
    //     Forward:
    //         20 -> 21 -> 22
    //
    // or:
    //
    //     Reverse:
    //         80 -> 79 -> 78
    //
    // Therefore the absolute gap is still used here.
    //
    // Direction consistency itself is checked while constructing the
    // temporal support chain in Check().
    // ------------------------------------------------------------------------
    const std::size_t historical_gap =
        IdGap(
            previous.historical_id,
            current.historical_id);

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
    // Direction-aware temporal support chain.
    //
    // We build ONE continuous proposal chain backwards from the current
    // geometry-valid proposal.
    //
    // Current-side progression is always forward in time:
    //
    //     current_id increases.
    //
    // Historical-side progression may be:
    //
    //     Forward:
    //         20 -> 21 -> 22
    //
    // or:
    //
    //     Reverse:
    //         80 -> 79 -> 78
    //
    // but the direction must remain consistent inside one support chain.
    //
    // A zero historical step:
    //
    //     20 -> 20
    //
    // is allowed because several neighbouring current Keyframes may align best
    // with the same historical Keyframe / local region.
    // ------------------------------------------------------------------------
    std::size_t support = 0;

    const LoopConsistencyProposal *cycle_reference =
        nullptr;

    // The newer end of the chain currently being constructed.
    //
    // Initially:
    //
    //     chain_newer = current proposal.
    //
    // Every accepted historical proposal becomes the new older end.
    const LoopConsistencyProposal *chain_newer =
        &proposal;

    // Historical traversal direction:
    //
    //      0 : not determined yet
    //     +1 : historical IDs increase as current IDs increase
    //     -1 : historical IDs decrease as current IDs increase
    int historical_direction = 0;

    for (auto iterator =
             history_.rbegin();
         iterator != history_.rend();
         ++iterator)
    {
        const LoopConsistencyProposal &previous =
            *iterator;

        // IMPORTANT:
        //
        // Compare against the last accepted member of the chain,
        // NOT always against the current proposal.
        //
        // This guarantees:
        //
        //     A -> B -> C
        //
        // is actually a locally continuous chain.
        if (!IsTemporalNeighbour(
                previous,
                *chain_newer))
        {
            continue;
        }

        // --------------------------------------------------------------------
        // Determine historical-side direction for this chronological step:
        //
        //     previous -> chain_newer
        //
        // Examples:
        //
        //     20 -> 21 : +1
        //     21 -> 20 : -1
        //     20 -> 20 :  0
        // --------------------------------------------------------------------
        int step_direction = 0;

        if (chain_newer->historical_id >
            previous.historical_id)
        {
            step_direction = +1;
        }
        else if (chain_newer->historical_id <
                 previous.historical_id)
        {
            step_direction = -1;
        }

        // --------------------------------------------------------------------
        // Once a non-zero traversal direction has been established,
        // an opposite historical progression is NOT allowed to support the
        // same temporal chain.
        // --------------------------------------------------------------------
        if (historical_direction != 0 &&
            step_direction != 0 &&
            step_direction != historical_direction)
        {
            continue;
        }

        // First non-zero step determines the track direction.
        if (historical_direction == 0 &&
            step_direction != 0)
        {
            historical_direction =
                step_direction;
        }

        ++support;

        // The first accepted previous proposal is the temporally closest
        // reference to the current proposal, therefore it remains the preferred
        // short-cycle reference.
        if (cycle_reference == nullptr)
        {
            cycle_reference =
                &previous;
        }

        // Extend the chain backwards.
        chain_newer =
            &previous;
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
