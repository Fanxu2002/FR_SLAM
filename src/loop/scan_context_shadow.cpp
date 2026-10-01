#include "fr_slam/loop/scan_context_shadow.hpp"

#include <algorithm>
#include <cmath>

ScanContextShadowDetector::ScanContextShadowDetector(
    const ScanContextShadowConfig &config)
    : config_(config),
      scan_context_(config.scan_context)
{
    if (config_.min_keyframe_id_separation == 0)
    {
        config_.min_keyframe_id_separation = 1;
    }

    if (!std::isfinite(config_.min_time_separation_sec) ||
        config_.min_time_separation_sec < 0.0)
    {
        config_.min_time_separation_sec = 10.0;
    }

    if (!std::isfinite(config_.max_scan_context_distance) ||
        config_.max_scan_context_distance <= 0.0)
    {
        config_.max_scan_context_distance = 0.40;
    }

    if (!std::isfinite(config_.max_candidate_distance) ||
        config_.max_candidate_distance <= 0.0)
    {
        config_.max_candidate_distance = 5.0;
    }

    if (config_.max_candidates == 0)
    {
        config_.max_candidates = 10;
    }

    config_.scan_context =
        scan_context_.GetConfig();
}

const ScanContextShadowDetector::DescriptorEntry *
ScanContextShadowDetector::FindDescriptor(
    std::size_t keyframe_id) const
{
    for (const DescriptorEntry &entry : database_)
    {
        if (entry.keyframe_id == keyframe_id)
        {
            return &entry;
        }
    }

    return nullptr;
}

bool ScanContextShadowDetector::AddKeyframe(
    const Keyframe &keyframe)
{
    if (!keyframe.cloud ||
        keyframe.cloud->empty() ||
        !std::isfinite(keyframe.timestamp) ||
        !keyframe.T_WL.matrix().allFinite())
    {
        return false;
    }

    if (FindDescriptor(keyframe.id) != nullptr)
    {
        return true;
    }

    const ScanContextDescriptor descriptor =
        scan_context_.MakeDescriptor(
            keyframe.cloud);

    if (!descriptor.valid)
    {
        return false;
    }

    DescriptorEntry entry;
    entry.keyframe_id = keyframe.id;
    entry.timestamp = keyframe.timestamp;
    entry.T_WL = keyframe.T_WL;
    entry.descriptor = descriptor;

    database_.push_back(entry);

    return true;
}

std::vector<ScanContextShadowCandidate>
ScanContextShadowDetector::Detect(
    std::size_t current_keyframe_id,
    ScanContextShadowDiagnostics *diagnostics) const
{
    std::vector<ScanContextShadowCandidate> candidates;

    // ================================================================
    // POSE SUPPLEMENT DETECTOR V1
    //
    // Track one pose-nearest long-history candidate independently from
    // the ordinary SC ranking.
    // ================================================================
    bool have_pose_supplement = false;

    ScanContextShadowCandidate
        pose_supplement_candidate;

    if (diagnostics != nullptr)
    {
        *diagnostics = ScanContextShadowDiagnostics();
        diagnostics->database_entries =
            database_.size();
    }

    const DescriptorEntry *current =
        FindDescriptor(current_keyframe_id);

    if (current == nullptr ||
        !current->descriptor.valid ||
        !current->T_WL.matrix().allFinite())
    {
        return candidates;
    }

    for (const DescriptorEntry &history : database_)
    {
        if (history.keyframe_id >= current_keyframe_id)
        {
            continue;
        }

        const std::size_t id_gap =
            current_keyframe_id -
            history.keyframe_id;

        if (id_gap <
            config_.min_keyframe_id_separation)
        {
            continue;
        }

        const double time_separation =
            current->timestamp -
            history.timestamp;

        if (!std::isfinite(time_separation) ||
            time_separation <
                config_.min_time_separation_sec)
        {
            continue;
        }

        if (diagnostics != nullptr)
        {
            ++diagnostics->separation_eligible;
        }

        const ScanContextMatch match =
            scan_context_.Compare(
                history.descriptor,
                current->descriptor);

        if (!match.valid ||
            !std::isfinite(match.distance))
        {
            continue;
        }

        if (diagnostics != nullptr)
        {
            ++diagnostics->valid_matches;
        }

        const double pose_distance =
            (current->T_WL.translation() -
             history.T_WL.translation())
                .norm();

        ScanContextShadowCandidate candidate;

        candidate.current_id =
            current_keyframe_id;

        candidate.candidate_id =
            history.keyframe_id;

        candidate.pose_distance =
            pose_distance;

        candidate.time_separation_sec =
            time_separation;

        candidate.scan_context_distance =
            match.distance;

        candidate.scan_context_similarity =
            match.similarity;

        candidate.raw_cosine_similarity =
            match.raw_cosine_similarity;

        candidate.sector_coverage_ratio =
            match.sector_coverage_ratio;

        candidate.cell_coverage_ratio =
            match.cell_coverage_ratio;

        candidate.compared_sectors =
            match.compared_sectors;

        candidate.sector_shift =
            match.sector_shift;

        candidate.yaw_shift_deg =
            match.yaw_shift_deg;

        // ============================================================
        // POSE SUPPLEMENT DETECTOR V1
        //
        // This runs BEFORE the normal SC-distance gate.
        //
        // Therefore an old location that is physically close in the
        // current frontend trajectory can survive even when repeated-row
        // Scan Context aliases push it outside the SC Top-K.
        //
        // We retain the REAL Scan Context diagnostics/yaw here; pose is
        // only used to propose the historical KF.
        // ============================================================
        if (config_.enable_pose_supplement &&
            id_gap >=
                config_
                    .pose_supplement_min_keyframe_id_separation &&
            std::isfinite(pose_distance) &&
            pose_distance <=
                config_
                    .pose_supplement_max_distance)
        {
            const bool better_pose_candidate =
                !have_pose_supplement ||
                pose_distance <
                    pose_supplement_candidate
                        .pose_distance;

            if (better_pose_candidate)
            {
                pose_supplement_candidate =
                    candidate;

                pose_supplement_candidate
                    .pose_supplement =
                        true;

                have_pose_supplement =
                    true;
            }
        }

        if (diagnostics != nullptr &&
            (!diagnostics->has_best_match ||
             candidate.scan_context_distance <
                 diagnostics->best_match
                     .scan_context_distance))
        {
            diagnostics->has_best_match = true;
            diagnostics->best_match =
                candidate;
        }

        if (match.distance >
            config_.max_scan_context_distance)
        {
            continue;
        }

        if (config_.use_pose_distance_gate &&
            (!std::isfinite(pose_distance) ||
             pose_distance >
                 config_.max_candidate_distance))
        {
            continue;
        }

        candidates.push_back(candidate);
    }

    std::sort(
        candidates.begin(),
        candidates.end(),
        [](const ScanContextShadowCandidate &a,
           const ScanContextShadowCandidate &b)
        {
            if (a.scan_context_distance !=
                b.scan_context_distance)
            {
                return a.scan_context_distance <
                       b.scan_context_distance;
            }

            return a.pose_distance <
                   b.pose_distance;
        });

    if (candidates.size() >
        config_.max_candidates)
    {
        candidates.resize(
            config_.max_candidates);
    }

    // ================================================================
    // Append AT MOST one pose supplement AFTER normal SC Top-K has
    // already been selected.
    //
    // If the same historical KF is already inside SC Top-K, do nothing.
    // ================================================================
    if (have_pose_supplement)
    {
        const bool already_in_sc_topk =
            std::any_of(
                candidates.begin(),
                candidates.end(),
                [&pose_supplement_candidate](
                    const ScanContextShadowCandidate &candidate)
                {
                    return
                        candidate.candidate_id ==
                        pose_supplement_candidate
                            .candidate_id;
                });

        if (!already_in_sc_topk)
        {
            candidates.push_back(
                pose_supplement_candidate);
        }
    }

    if (diagnostics != nullptr)
    {
        diagnostics->accepted_candidates =
            candidates.size();
    }

    return candidates;
}

std::size_t
ScanContextShadowDetector::DescriptorCount() const
{
    return database_.size();
}

void ScanContextShadowDetector::Clear()
{
    database_.clear();
}
