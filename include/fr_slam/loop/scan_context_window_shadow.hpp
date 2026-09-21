#pragma once

#include "fr_slam/loop/scan_context.hpp"

#include <cstddef>
#include <limits>
#include <vector>

#include <Eigen/Geometry>

#include <pcl/point_cloud.h>

struct ScanContextWindowCandidate
{
    // Query 7-KF window.
    std::size_t query_first_kf = 0;
    std::size_t query_last_kf = 0;
    std::size_t query_anchor_kf = 0;

    // Historical database 7-KF window.
    std::size_t historical_first_kf = 0;
    std::size_t historical_last_kf = 0;
    std::size_t historical_anchor_kf = 0;

    double time_separation_sec =
        std::numeric_limits<double>::infinity();

    double anchor_pose_distance =
        std::numeric_limits<double>::infinity();

    double scan_context_distance =
        std::numeric_limits<double>::infinity();

    double scan_context_similarity = 0.0;
    double raw_cosine_similarity = 0.0;

    double sector_coverage_ratio = 0.0;
    double cell_coverage_ratio = 0.0;

    std::size_t compared_sectors = 0;
    std::size_t sector_shift = 0;

    double yaw_shift_deg = 0.0;
};

struct ScanContextWindowConfig
{
    // Historical window must be sufficiently old.
    // Applied using last-KF IDs:
    //
    // query_last_kf - historical_last_kf >= separation
    //
    std::size_t min_keyframe_id_separation = 30;

    double min_time_separation_sec = 10.0;

    // Same SC candidate gate as the old baseline.
    double max_scan_context_distance = 0.40;

    // Diagnostic only by default.
    bool use_pose_distance_gate = false;
    double max_candidate_distance = 5.0;

    std::size_t max_candidates = 10;

    ScanContextConfig scan_context;
};

struct ScanContextWindowDiagnostics
{
    std::size_t database_entries = 0;
    std::size_t separation_eligible = 0;
    std::size_t valid_matches = 0;
    std::size_t accepted_candidates = 0;

    bool query_descriptor_valid = false;

    // Best SC match BEFORE max distance gating.
    bool has_best_match = false;
    ScanContextWindowCandidate best_match;
};

class ScanContextWindowShadow
{
public:
    explicit ScanContextWindowShadow(
        const ScanContextWindowConfig &config =
            ScanContextWindowConfig());

    // Add one BTC-style full database window:
    //
    // [0..6], [4..10], [8..14], ...
    //
    // cloud_A must already be accumulated into the anchor-KF frame.
    bool AddDatabaseWindow(
        std::size_t first_kf,
        std::size_t last_kf,
        std::size_t anchor_kf,
        double anchor_timestamp,
        const Eigen::Isometry3d &T_WA,
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &cloud_A);

    // Query one trailing BTC-style window:
    //
    // current 306 -> [300..306]
    // current 307 -> [301..307]
    //
    // This function NEVER modifies the database.
    std::vector<ScanContextWindowCandidate> QueryWindow(
        std::size_t first_kf,
        std::size_t last_kf,
        std::size_t anchor_kf,
        double anchor_timestamp,
        const Eigen::Isometry3d &T_WA,
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &cloud_A,
        ScanContextWindowDiagnostics *diagnostics = nullptr) const;

    bool HasDatabaseWindow(
        std::size_t last_kf) const;

    std::size_t DescriptorCount() const;

    void Clear();

private:
    struct WindowEntry
    {
        std::size_t first_kf = 0;
        std::size_t last_kf = 0;
        std::size_t anchor_kf = 0;

        double anchor_timestamp = 0.0;

        Eigen::Isometry3d T_WA =
            Eigen::Isometry3d::Identity();

        ScanContextDescriptor descriptor;
    };

    const WindowEntry *FindDatabaseWindow(
        std::size_t last_kf) const;

private:
    ScanContextWindowConfig config_;
    ScanContext scan_context_;

    std::vector<WindowEntry> database_;
};
