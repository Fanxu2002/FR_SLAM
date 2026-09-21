#include "fr_slam/loop/scan_context_window_shadow.hpp"

#include <algorithm>
#include <cmath>

ScanContextWindowShadow::ScanContextWindowShadow(
    const ScanContextWindowConfig &config)
    : config_(config),
      scan_context_(config.scan_context)
{
    if (config_.min_keyframe_id_separation == 0)
    {
        config_.min_keyframe_id_separation = 1;
    }

    if (!std::isfinite(config_.min_time_separation_sec) ||
        config_.min_time_separation_sec < 0.0)
    {
        config_.min_time_separation_sec = 10.0;
    }

    if (!std::isfinite(config_.max_scan_context_distance) ||
        config_.max_scan_context_distance <= 0.0)
    {
        config_.max_scan_context_distance = 0.40;
    }

    if (!std::isfinite(config_.max_candidate_distance) ||
        config_.max_candidate_distance <= 0.0)
    {
        config_.max_candidate_distance = 5.0;
    }

    if (config_.max_candidates == 0)
    {
        config_.max_candidates = 10;
    }

    config_.scan_context =
        scan_context_.GetConfig();
}

const ScanContextWindowShadow::WindowEntry *
ScanContextWindowShadow::FindDatabaseWindow(
    std::size_t last_kf) const
{
    for (const WindowEntry &entry : database_)
    {
        if (entry.last_kf == last_kf)
        {
            return &entry;
        }
    }

    return nullptr;
}

bool ScanContextWindowShadow::HasDatabaseWindow(
    std::size_t last_kf) const
{
    return FindDatabaseWindow(last_kf) != nullptr;
}

bool ScanContextWindowShadow::AddDatabaseWindow(
    std::size_t first_kf,
    std::size_t last_kf,
    std::size_t anchor_kf,
    double anchor_timestamp,
    const Eigen::Isometry3d &T_WA,
    const pcl::PointCloud<LIDAR_POINT>::ConstPtr &cloud_A)
{
    if (FindDatabaseWindow(last_kf) != nullptr)
    {
        return true;
    }

    if (!cloud_A ||
        cloud_A->empty() ||
        !std::isfinite(anchor_timestamp) ||
        !T_WA.matrix().allFinite())
    {
        return false;
    }

    const ScanContextDescriptor descriptor =
        scan_context_.MakeDescriptor(cloud_A);

    if (!descriptor.valid)
    {
        return false;
    }

    WindowEntry entry;

    entry.first_kf = first_kf;
    entry.last_kf = last_kf;
    entry.anchor_kf = anchor_kf;

    entry.anchor_timestamp =
        anchor_timestamp;

    entry.T_WA =
        T_WA;

    entry.descriptor =
        descriptor;

    database_.push_back(
        std::move(entry));

    return true;
}

std::vector<ScanContextWindowCandidate>
ScanContextWindowShadow::QueryWindow(
    std::size_t first_kf,
    std::size_t last_kf,
    std::size_t anchor_kf,
    double anchor_timestamp,
    const Eigen::Isometry3d &T_WA,
    const pcl::PointCloud<LIDAR_POINT>::ConstPtr &cloud_A,
    ScanContextWindowDiagnostics *diagnostics) const
{
    std::vector<ScanContextWindowCandidate> candidates;

    if (diagnostics != nullptr)
    {
        *diagnostics =
            ScanContextWindowDiagnostics();

        diagnostics->database_entries =
            database_.size();
    }

    if (!cloud_A ||
        cloud_A->empty() ||
        !std::isfinite(anchor_timestamp) ||
        !T_WA.matrix().allFinite())
    {
        return candidates;
    }

    const ScanContextDescriptor query_descriptor =
        scan_context_.MakeDescriptor(
            cloud_A);

    if (!query_descriptor.valid)
    {
        return candidates;
    }

    if (diagnostics != nullptr)
    {
        diagnostics->query_descriptor_valid =
            true;
    }

    for (const WindowEntry &history : database_)
    {
        if (history.last_kf >= last_kf)
        {
            continue;
        }

        const std::size_t id_gap =
            last_kf -
            history.last_kf;

        if (id_gap <
            config_.min_keyframe_id_separation)
        {
            continue;
        }

        const double time_separation =
            anchor_timestamp -
            history.anchor_timestamp;

        if (!std::isfinite(time_separation) ||
            time_separation <
                config_.min_time_separation_sec)
        {
            continue;
        }

        if (diagnostics != nullptr)
        {
            ++diagnostics->separation_eligible;
        }

        const ScanContextMatch match =
            scan_context_.Compare(
                history.descriptor,
                query_descriptor);

        if (!match.valid ||
            !std::isfinite(match.distance))
        {
            continue;
        }

        if (diagnostics != nullptr)
        {
            ++diagnostics->valid_matches;
        }

        const double anchor_pose_distance =
            (T_WA.translation() -
             history.T_WA.translation())
                .norm();

        ScanContextWindowCandidate candidate;

        candidate.query_first_kf =
            first_kf;

        candidate.query_last_kf =
            last_kf;

        candidate.query_anchor_kf =
            anchor_kf;

        candidate.historical_first_kf =
            history.first_kf;

        candidate.historical_last_kf =
            history.last_kf;

        candidate.historical_anchor_kf =
            history.anchor_kf;

        candidate.time_separation_sec =
            time_separation;

        candidate.anchor_pose_distance =
            anchor_pose_distance;

        candidate.scan_context_distance =
            match.distance;

        candidate.scan_context_similarity =
            match.similarity;

        candidate.raw_cosine_similarity =
            match.raw_cosine_similarity;

        candidate.sector_coverage_ratio =
            match.sector_coverage_ratio;

        candidate.cell_coverage_ratio =
            match.cell_coverage_ratio;

        candidate.compared_sectors =
            match.compared_sectors;

        candidate.sector_shift =
            match.sector_shift;

        candidate.yaw_shift_deg =
            match.yaw_shift_deg;

        // Record best raw match BEFORE SC threshold.
        if (diagnostics != nullptr &&
            (!diagnostics->has_best_match ||
             candidate.scan_context_distance <
                 diagnostics->best_match
                     .scan_context_distance))
        {
            diagnostics->has_best_match =
                true;

            diagnostics->best_match =
                candidate;
        }

        if (match.distance >
            config_.max_scan_context_distance)
        {
            continue;
        }

        if (config_.use_pose_distance_gate &&
            (!std::isfinite(anchor_pose_distance) ||
             anchor_pose_distance >
                 config_.max_candidate_distance))
        {
            continue;
        }

        candidates.push_back(
            candidate);
    }

    std::sort(
        candidates.begin(),
        candidates.end(),
        [](const ScanContextWindowCandidate &a,
           const ScanContextWindowCandidate &b)
        {
            if (a.scan_context_distance !=
                b.scan_context_distance)
            {
                return a.scan_context_distance <
                       b.scan_context_distance;
            }

            return a.anchor_pose_distance <
                   b.anchor_pose_distance;
        });

    if (candidates.size() >
        config_.max_candidates)
    {
        candidates.resize(
            config_.max_candidates);
    }

    if (diagnostics != nullptr)
    {
        diagnostics->accepted_candidates =
            candidates.size();
    }

    return candidates;
}

std::size_t
ScanContextWindowShadow::DescriptorCount() const
{
    return database_.size();
}

void ScanContextWindowShadow::Clear()
{
    database_.clear();
}
