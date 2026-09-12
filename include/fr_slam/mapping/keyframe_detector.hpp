#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>


// ============================================================================
// Decide whether an accepted LiDAR pose should become a keyframe.
//
// IMPORTANT:
//     This class only makes the decision.
//
// It does NOT:
//     - store keyframes,
//     - update LocalMap,
//     - perform registration.
//
// Current rule:
//
//     translation >= threshold
//
// OR
//
//     rotation >= threshold
//
// The comparison is always made against the LAST KEYFRAME pose,
// not against the previous ordinary LiDAR frame.
// ============================================================================
class KeyframeDetector
{
public:
    KeyframeDetector(
        double translation_threshold,
        double rotation_threshold_deg);

    // Evaluate current accepted pose relative to the latest keyframe.
    //
    // translation:
    //     relative translation distance in meters.
    //
    // rotation_deg:
    //     relative rotation angle in degrees.
    bool ShouldCreateKeyframe(
        const Eigen::Isometry3d &T_WL,
        double &translation,
        double &rotation_deg) const;

    // Commit a pose as the latest keyframe reference.
    void SetLastKeyframePose(
        const Eigen::Isometry3d &T_WL);

    bool HasKeyframe() const;

    // Reset detector state. The next accepted frame becomes a keyframe.
    void Reset();

private:
    // Unit: meter.
    double translation_threshold_;

    // Unit: radian.
    double rotation_threshold_rad_;

    bool has_keyframe_ = false;

    // Global pose of latest keyframe.
    Eigen::Isometry3d last_keyframe_pose_ =
        Eigen::Isometry3d::Identity();
};
