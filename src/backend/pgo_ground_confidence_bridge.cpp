#include "fr_slam/backend/pgo_ground_confidence_bridge.hpp"

#include <cmath>
#include <iostream>
#include <mutex>
#include <unordered_map>

namespace fr_slam
{

namespace
{

std::mutex
    g_pgo_ground_confidence_mutex;

bool
    g_has_latest_frame_record = false;

PgoGroundConfidenceRecord
    g_latest_frame_record;

std::unordered_map<
    std::size_t,
    PgoGroundConfidenceRecord>
    g_record_by_keyframe;

} // namespace


void PublishPgoGroundConfidenceFrame(
    const PgoGroundConfidenceRecord &record)
{
    if (!std::isfinite(record.timestamp))
    {
        return;
    }

    std::lock_guard<std::mutex>
        lock(g_pgo_ground_confidence_mutex);

    g_latest_frame_record =
        record;

    g_has_latest_frame_record =
        true;
}


bool BindPgoGroundConfidenceKeyframe(
    double timestamp,
    std::size_t keyframe_id,
    double maximum_time_error_s)
{
    if (!std::isfinite(timestamp) ||
        !std::isfinite(maximum_time_error_s) ||
        maximum_time_error_s < 0.0)
    {
        return false;
    }

    std::lock_guard<std::mutex>
        lock(g_pgo_ground_confidence_mutex);

    if (!g_has_latest_frame_record)
    {
        return false;
    }

    const double time_error =
        std::abs(
            g_latest_frame_record.timestamp -
            timestamp);

    if (!std::isfinite(time_error) ||
        time_error > maximum_time_error_s)
    {
        return false;
    }

    g_record_by_keyframe[keyframe_id] =
        g_latest_frame_record;

    if (keyframe_id < 3U ||
        (keyframe_id % 100U) == 0U)
    {
        std::cout
            << "PGO_GROUND_CONF_BIND_OK_V3"
            << " | kf="
            << keyframe_id
            << " | dt="
            << time_error
            << " | registry_size="
            << g_record_by_keyframe.size()
            << " | segmentation_valid="
            << (g_latest_frame_record
                    .segmentation_valid
                    ? 1
                    : 0)
            << " | constraint_valid="
            << (g_latest_frame_record
                    .support_constraint_valid
                    ? 1
                    : 0)
            << " | conf="
            << g_latest_frame_record.confidence
            << std::endl;
    }

    return true;
}


bool GetPgoGroundConfidenceKeyframe(
    std::size_t keyframe_id,
    PgoGroundConfidenceRecord &record)
{
    std::lock_guard<std::mutex>
        lock(g_pgo_ground_confidence_mutex);

    const auto iterator =
        g_record_by_keyframe.find(
            keyframe_id);

    if (iterator ==
        g_record_by_keyframe.end())
    {
        return false;
    }

    record =
        iterator->second;

    return true;
}


std::size_t PgoGroundConfidenceRegistrySize()
{
    std::lock_guard<std::mutex>
        lock(g_pgo_ground_confidence_mutex);

    return
        g_record_by_keyframe.size();
}

} // namespace fr_slam
