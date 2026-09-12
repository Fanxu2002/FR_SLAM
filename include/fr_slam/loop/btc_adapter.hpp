#pragma once

#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pcl/point_cloud.h>

#include "fr_slam/common/point_types.hpp"
#include "btc_core.hpp"

namespace fr_slam_btc
{

struct OfficialBtcSubmapResult
{
    bool input_valid = false;
    bool newly_processed = false;
    bool queried = false;
    bool has_candidate = false;

    std::size_t current_submap_id = 0;
    std::size_t historical_submap_id = 0;

    int internal_frame_id = -1;
    std::size_t input_points = 0;
    std::size_t btc_descriptor_count = 0;
    std::size_t matched_triangle_pairs = 0;

    double score = 0.0;

    // p_H = R_H_C * p_C + t_H_C
    Eigen::Vector3d t_H_C = Eigen::Vector3d::Zero();
    Eigen::Matrix3d R_H_C = Eigen::Matrix3d::Identity();

    double roll_deg = 0.0;
    double pitch_deg = 0.0;
    double yaw_deg = 0.0;
};

class OfficialBtcAdapter
{
public:
    OfficialBtcAdapter(
        const std::string &config_profile,
        int skip_near_num,
        int proj_plane_num,
        std::size_t window_size,
        std::size_t window_stride);

    bool IsProcessedSubmap(
        std::size_t submap_id) const;

    OfficialBtcSubmapResult ProcessSubmap(
        std::size_t submap_id,
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &cloud_S);

    std::size_t ProcessedSubmapCount() const;

    // FR-SLAM V31.11
    bool ConfigurePendingRescue(
        std::size_t historical_anchor_kf,
        std::size_t historical_gap)
    {
        std::lock_guard<std::mutex> lock(
            mutex_);

        if (!manager_ ||
            internal_to_submap_id_.empty())
        {
            return false;
        }

        const std::size_t min_anchor =
            historical_anchor_kf >
                    historical_gap
                ? historical_anchor_kf -
                      historical_gap
                : 0;

        const std::size_t max_anchor =
            historical_anchor_kf +
            historical_gap;

        constexpr std::size_t
            kWindowAnchorOffset = 3;

        int internal_min = -1;
        int internal_max = -1;

        for (std::size_t internal_id = 0;
             internal_id <
                 internal_to_submap_id_.size();
             ++internal_id)
        {
            const std::size_t endpoint_kf =
                internal_to_submap_id_[
                    internal_id];

            if (endpoint_kf <
                kWindowAnchorOffset)
            {
                continue;
            }

            const std::size_t anchor_kf =
                endpoint_kf -
                kWindowAnchorOffset;

            if (anchor_kf < min_anchor ||
                anchor_kf > max_anchor)
            {
                continue;
            }

            const int internal_frame =
                static_cast<int>(
                    internal_id);

            if (internal_min < 0)
            {
                internal_min =
                    internal_frame;
            }

            internal_max =
                internal_frame;
        }

        if (internal_min < 0 ||
            internal_max < internal_min)
        {
            manager_->
                ClearPendingRescueRange();

            return false;
        }

        manager_->
            SetPendingRescueRange(
                internal_min,
                internal_max);

        return true;
    }

    void ClearPendingRescue()
    {
        std::lock_guard<std::mutex> lock(
            mutex_);

        if (manager_)
        {
            manager_->
                ClearPendingRescueRange();
        }
    }

    const ConfigSetting &Config() const
    {
        return config_;
    }

private:

    // Runtime FR-SLAM/BTC integration settings.
    // Official descriptor parameters still come from BTC's own profile YAML.
    std::size_t btc_window_size_ = 7;
    std::size_t btc_window_stride_ = 4;

    static pcl::PointCloud<pcl::PointXYZI>::Ptr
    ConvertSubmapCloud(
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &cloud_S);

    static void RotationToRpyDeg(
        const Eigen::Matrix3d &R,
        double &roll_deg,
        double &pitch_deg,
        double &yaw_deg);

    ConfigSetting config_;
    std::unique_ptr<BtcDescManager> manager_;

    // internal BTC frame -> FR-SLAM finished Submap ID
    std::vector<std::size_t> internal_to_submap_id_;

    mutable std::mutex mutex_;
};

} // namespace fr_slam_btc
