#pragma once

#include <cstddef>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace fr_slam
{

class LioMapOdomBridge
{
public:
    LioMapOdomBridge();

    void Reset();

    bool Update(
        const Eigen::Isometry3d &T_map_odom,
        std::size_t revision);

    bool HasCorrection() const;

    std::size_t Revision() const;

    const Eigen::Isometry3d &TMapOdom() const;

    Eigen::Isometry3d ToMap(
        const Eigen::Isometry3d &T_odom_L) const;

private:
    Eigen::Isometry3d T_map_odom_ =
        Eigen::Isometry3d::Identity();

    std::size_t revision_ = 0;

    bool has_correction_ = false;
};

}
