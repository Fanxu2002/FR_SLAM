#pragma once

#include <cstddef>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pcl/point_cloud.h>

#include "fr_slam/fr_keyframe.hpp"
#include "fr_slam/fr_point_types.hpp"

// ============================================================================
// Historical keyframe manager.
//
// Responsibility:
//     Save and manage ALL historical keyframes.
//
// It does NOT decide whether a frame should become a keyframe.
// That responsibility belongs to KeyframeDetector.
//
// KeyframeManager != LocalMap
//
// KeyframeManager:
//     KF0, KF1, KF2, ... full historical sequence.
//
// LocalMap:
//     only recent keyframes used by Scan-to-LocalMap.
// ============================================================================
class KeyframeManager
{
public:
    KeyframeManager() = default;

    // Create and store one historical keyframe.
    bool AddKeyframe(
        double timestamp,
        const Eigen::Isometry3d &T_WL,
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &cloud_lidar,
        const Eigen::Matrix<double, 6, 6> *odom_information = nullptr);

    // Number of ALL historical keyframes.
    std::size_t Size() const;

    bool Empty() const;

    // Returns nullptr when empty.
    const Keyframe *Latest() const;

    // Returns nullptr when index is invalid.
    const Keyframe *GetKeyframe(std::size_t index) const;

    // Read-only access to complete historical keyframe sequence.
    const std::vector<Keyframe> &GetAllKeyframes() const;

    // Clear history and restart IDs from zero.
    void Clear();

private:
    std::vector<Keyframe> keyframes_;
    std::size_t next_keyframe_id_ = 0;
};
