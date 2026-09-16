#include "fr_slam/frontend/lio_map_odom_bridge.hpp"

#include <cmath>

namespace fr_slam
{

LioMapOdomBridge::LioMapOdomBridge()
{
    Reset();
}

void LioMapOdomBridge::Reset()
{
    T_map_odom_.setIdentity();
    revision_ = 0;
    has_correction_ = false;
}

bool LioMapOdomBridge::Update(
    const Eigen::Isometry3d &T_map_odom,
    const std::size_t revision)
{
    if (revision == 0 ||
        !T_map_odom.matrix().allFinite())
    {
        return false;
    }

    const Eigen::Matrix3d R =
        T_map_odom.rotation();

    const double determinant =
        R.determinant();

    if (!std::isfinite(determinant) ||
        std::abs(determinant - 1.0) > 1.0e-3)
    {
        return false;
    }

    if (revision < revision_)
    {
        return false;
    }

    T_map_odom_ =
        T_map_odom;

    revision_ =
        revision;

    has_correction_ =
        true;

    return true;
}

bool LioMapOdomBridge::HasCorrection() const
{
    return has_correction_;
}

std::size_t LioMapOdomBridge::Revision() const
{
    return revision_;
}

const Eigen::Isometry3d &
LioMapOdomBridge::TMapOdom() const
{
    return T_map_odom_;
}

Eigen::Isometry3d LioMapOdomBridge::ToMap(
    const Eigen::Isometry3d &T_odom_L) const
{
    if (!T_odom_L.matrix().allFinite())
    {
        return Eigen::Isometry3d::Identity();
    }

    return T_map_odom_ *
           T_odom_L;
}

}
