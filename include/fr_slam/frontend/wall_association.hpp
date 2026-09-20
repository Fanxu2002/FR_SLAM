#pragma once

#include <cstddef>
#include <limits>
#include <memory>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "fr_slam/frontend/multi_plane_extractor.hpp"

namespace fr_slam
{

struct ActiveWallAssociation
{
    std::size_t persistent_wall_id =
        std::numeric_limits<std::size_t>::max();

    Eigen::Vector3d reference_normal_A =
        Eigen::Vector3d::UnitX();

    double reference_d_A = 0.0;

    Eigen::Vector3d observed_normal_A =
        Eigen::Vector3d::UnitX();

    double observed_d_A = 0.0;

    Eigen::Vector3d observed_center_A =
        Eigen::Vector3d::Zero();

    double observed_radius_m = 0.0;

    double quality = 0.0;

    double normal_difference_deg =
        std::numeric_limits<double>::quiet_NaN();

    double plane_distance_difference_m =
        std::numeric_limits<double>::quiet_NaN();
};

struct WallAssociationResult
{
    std::vector<ActiveWallAssociation>
        active_static_walls;

    std::size_t raw_wall_candidates = 0;

    std::size_t physical_wall_candidates = 0;

    std::size_t persistent_walls = 0;

    std::size_t active_static_wall_count = 0;
};

class WallAssociation
{
public:
    WallAssociation();
    ~WallAssociation();

    WallAssociation(
        WallAssociation &&other) noexcept;

    WallAssociation &operator=(
        WallAssociation &&other) noexcept;

    WallAssociation(
        const WallAssociation &) = delete;

    WallAssociation &operator=(
        const WallAssociation &) = delete;

    void Reset();

    WallAssociationResult Update(
        const MultiPlaneExtractionResult &planes,
        const Eigen::Isometry3d &T_AL,
        std::size_t frame_index);

private:
    struct Impl;

    std::unique_ptr<Impl> impl_;
};

} // namespace fr_slam
