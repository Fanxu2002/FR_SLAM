#pragma once

#include <cstddef>

namespace fr_slam
{

struct PgoGroundConfidenceRecord
{
    double timestamp = 0.0;

    bool segmentation_valid = false;
    bool support_constraint_valid = false;

    double confidence = 0.0;

    bool anchor_valid = false;
    double anchor_error_m = 0.0;
    double anchor_tolerance_m = 0.0;
};


void PublishPgoGroundConfidenceFrame(
    const PgoGroundConfidenceRecord &record);


bool BindPgoGroundConfidenceKeyframe(
    double timestamp,
    std::size_t keyframe_id,
    double maximum_time_error_s);


bool GetPgoGroundConfidenceKeyframe(
    std::size_t keyframe_id,
    PgoGroundConfidenceRecord &record);


std::size_t PgoGroundConfidenceRegistrySize();

} // namespace fr_slam
