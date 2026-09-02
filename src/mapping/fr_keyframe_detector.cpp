#include "fr_slam/fr_keyframe_detector.hpp"

#include <cmath>


namespace
{
constexpr double kPi = 3.14159265358979323846;
}


KeyframeDetector::KeyframeDetector(
    double translation_threshold,
    double rotation_threshold_deg)
    : translation_threshold_(translation_threshold),
      rotation_threshold_rad_(
          rotation_threshold_deg * kPi / 180.0)
{
}


bool KeyframeDetector::ShouldCreateKeyframe(
    const Eigen::Isometry3d &T_WL,
    double &translation,
    double &rotation_deg) const
{
    // ------------------------------------------------------------------------
    // The first accepted frame is always a keyframe.
    // ------------------------------------------------------------------------
    if (!has_keyframe_)
    {
        translation = 0.0;
        rotation_deg = 0.0;
        return true;
    }

    // ------------------------------------------------------------------------
    // Relative pose:
    //
    //     T_KF_L = T_W_KF^-1 * T_W_L
    //
    // Current LiDAR -> latest keyframe LiDAR.
    // ------------------------------------------------------------------------
    const Eigen::Isometry3d T_KF_L =
        last_keyframe_pose_.inverse() * T_WL;

    // Translation distance from latest keyframe.
    translation = T_KF_L.translation().norm();

    // Rotation angle from latest keyframe.
    const Eigen::AngleAxisd angle_axis(T_KF_L.rotation());
    const double rotation_rad = std::abs(angle_axis.angle());

    rotation_deg = rotation_rad * 180.0 / kPi;

    // OR logic: translation OR rotation can create a keyframe.
    return
        translation >= translation_threshold_ ||
        rotation_rad >= rotation_threshold_rad_;
}


void KeyframeDetector::SetLastKeyframePose(
    const Eigen::Isometry3d &T_WL)
{
    last_keyframe_pose_ = T_WL;
    has_keyframe_ = true;
}


bool KeyframeDetector::HasKeyframe() const
{
    return has_keyframe_;
}


void KeyframeDetector::Reset()
{
    has_keyframe_ = false;
    last_keyframe_pose_ = Eigen::Isometry3d::Identity();
}
