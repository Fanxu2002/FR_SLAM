#include "fr_slam/fr_loop_detector.hpp"

#include <algorithm>
#include <cmath>

LoopDetector::LoopDetector(
    const LoopDetectorConfig &config)
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
        config_.max_scan_context_distance = 0.30;
    }

    if (!std::isfinite(config_.max_candidate_distance) ||
        config_.max_candidate_distance <= 0.0)
    {
        config_.max_candidate_distance = 5.0;
    }

    if (config_.max_candidates == 0)
    {
        config_.max_candidates = 1;
    }

    config_.scan_context =
        scan_context_.GetConfig();
}

const LoopDetector::DescriptorEntry *
LoopDetector::FindDescriptor(
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

bool LoopDetector::AddKeyframe(
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

bool LoopDetector::HasDescriptor(
    std::size_t keyframe_id) const
{
    return FindDescriptor(keyframe_id) != nullptr;
}

std::size_t LoopDetector::DescriptorCount() const
{
    return database_.size();
}

std::vector<LoopCandidate>
LoopDetector::Detect(
    std::size_t current_keyframe_id) const
{
    std::vector<LoopCandidate> candidates;

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
        if (history.keyframe_id == current_keyframe_id)
        {
            continue;
        }

        // The database is chronological in normal operation. Keep the test
        // explicit so this function is still safe if IDs are ever sparse.
        if (history.keyframe_id > current_keyframe_id)
        {
            continue;
        }

        const std::size_t id_gap =
            current_keyframe_id - history.keyframe_id;

        if (id_gap < config_.min_keyframe_id_separation)
        {
            continue;
        }

        const double time_separation =
            current->timestamp - history.timestamp;

        if (!std::isfinite(time_separation) ||
            time_separation < config_.min_time_separation_sec)
        {
            continue;
        }

        const ScanContextMatch match =
            scan_context_.Compare(
                history.descriptor,
                current->descriptor);

        if (!match.valid ||
            !std::isfinite(match.distance) ||
            match.distance > config_.max_scan_context_distance)
        {
            continue;
        }

        double pose_distance =
            std::numeric_limits<double>::infinity();

        if (history.T_WL.matrix().allFinite())
        {
            pose_distance =
                (current->T_WL.translation() -
                 history.T_WL.translation())
                    .norm();
        }

        if (config_.use_pose_distance_gate)
        {
            if (!std::isfinite(pose_distance) ||
                pose_distance > config_.max_candidate_distance)
            {
                continue;
            }
        }

        LoopCandidate candidate;
        candidate.current_id = current_keyframe_id;
        candidate.candidate_id = history.keyframe_id;
        candidate.distance = pose_distance;
        candidate.time_separation_sec = time_separation;
        candidate.scan_context_distance = match.distance;
        candidate.scan_context_similarity = match.similarity;
        candidate.sector_shift = match.sector_shift;
        candidate.yaw_shift_deg = match.yaw_shift_deg;

        candidates.push_back(candidate);
    }

    std::sort(
        candidates.begin(),
        candidates.end(),
        [](const LoopCandidate &a,
           const LoopCandidate &b)
        {
            if (a.scan_context_distance !=
                b.scan_context_distance)
            {
                return a.scan_context_distance <
                       b.scan_context_distance;
            }

            return a.distance < b.distance;
        });

    if (candidates.size() > config_.max_candidates)
    {
        candidates.resize(config_.max_candidates);
    }

    return candidates;
}

void LoopDetector::Clear()
{
    database_.clear();
}
