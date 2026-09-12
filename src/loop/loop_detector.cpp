#include "fr_slam/loop/loop_detector.hpp"

#include <cmath>

LoopDetector::LoopDetector(
    const LoopDetectorConfig &config)
    : config_(config)
{
    if (config_.min_keyframe_id_separation == 0)
    {
        config_.min_keyframe_id_separation = 30;
    }

    if (!std::isfinite(config_.min_time_separation_sec) ||
        config_.min_time_separation_sec < 0.0)
    {
        config_.min_time_separation_sec = 10.0;
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
}

const LoopDetectorConfig &LoopDetector::GetConfig() const
{
    return config_;
}

void LoopDetector::Clear()
{
    // No descriptor/database state remains. BTC owns retrieval state inside
    // the backend adapter and is rebuilt naturally with a new SLAM process.
}
