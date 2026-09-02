#include "fr_slam/mapping/fr_keyframe_manager.hpp"

#include <cmath>
#include <utility>

bool KeyframeManager::AddKeyframe(
    double timestamp,
    const Eigen::Isometry3d &T_WL,
    const pcl::PointCloud<LIDAR_POINT>::ConstPtr &cloud_lidar,
    const Eigen::Matrix<double, 6, 6> *odom_information)
{
    // ------------------------------------------------------------------------
    // 1. Validate input cloud.
    // ------------------------------------------------------------------------
    if (!cloud_lidar || cloud_lidar->empty())
    {
        return false;
    }

    // ------------------------------------------------------------------------
    // 2. Validate timestamp and pose.
    // ------------------------------------------------------------------------
    if (!std::isfinite(timestamp) || !T_WL.matrix().allFinite())
    {
        return false;
    }

    // ------------------------------------------------------------------------
    // 3. Create a new keyframe.
    // ------------------------------------------------------------------------
    Keyframe keyframe;
    keyframe.id = next_keyframe_id_;
    keyframe.timestamp = timestamp;
    keyframe.T_WL = T_WL;
    if (odom_information != nullptr &&
        odom_information->allFinite())
    {
        bool information_valid = true;

        for (int i = 0;
             i < 6;
             ++i)
        {
            if (!std::isfinite(
                    (*odom_information)(i, i)) ||
                (*odom_information)(i, i) <= 0.0)
            {
                information_valid = false;
                break;
            }
        }

        if (information_valid)
        {
            keyframe.odom_information =
                *odom_information;

            keyframe.has_odom_information =
                true;
        }
    }

    // ------------------------------------------------------------------------
    // 4. Deep-copy current LiDAR cloud.
    //
    // A keyframe is a historical snapshot, so it should own independent
    // point-cloud data instead of sharing the current scan pointer.
    // ------------------------------------------------------------------------
    keyframe.cloud =
        pcl::make_shared<pcl::PointCloud<LIDAR_POINT>>(*cloud_lidar);

    // ------------------------------------------------------------------------
    // 5. Store keyframe and consume ID only after successful insertion.
    // ------------------------------------------------------------------------
    keyframes_.push_back(std::move(keyframe));
    ++next_keyframe_id_;

    return true;
}

std::size_t KeyframeManager::Size() const
{
    return keyframes_.size();
}

bool KeyframeManager::Empty() const
{
    return keyframes_.empty();
}

const Keyframe *KeyframeManager::Latest() const
{
    if (keyframes_.empty())
    {
        return nullptr;
    }

    return &keyframes_.back();
}

const Keyframe *KeyframeManager::GetKeyframe(std::size_t index) const
{
    if (index >= keyframes_.size())
    {
        return nullptr;
    }

    return &keyframes_[index];
}

const std::vector<Keyframe> &KeyframeManager::GetAllKeyframes() const
{
    return keyframes_;
}

void KeyframeManager::Clear()
{
    keyframes_.clear();
    next_keyframe_id_ = 0;
}
