#pragma once

#include "fr_slam/mapping/keyframe.hpp"
#include "fr_slam/loop/scan_context.hpp"

#include <cstddef>
#include <limits>
#include <vector>

struct ScanContextShadowCandidate
{
    std::size_t current_id = 0;
    std::size_t candidate_id = 0;

    double pose_distance =
        std::numeric_limits<double>::infinity();

    double time_separation_sec =
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

    // ================================================================
    // POSE SUPPLEMENT RETRIEVAL V1
    //
    // False:
    //     ordinary Scan Context Top-K candidate.
    //
    // True:
    //     one extra long-history candidate selected only by frontend
    //     pose proximity.
    //
    // IMPORTANT:
    //     this candidate is NOT allowed to become the ordinary
    //     single-frame geometry winner.  It exists only to seed the
    //     causal TrackRecovery rescue path.
    // ================================================================
    bool pose_supplement = false;
};

struct ScanContextShadowConfig
{
    std::size_t min_keyframe_id_separation = 30;
    double min_time_separation_sec = 10.0;

    double max_scan_context_distance = 0.40;

    bool use_pose_distance_gate = false;
    double max_candidate_distance = 5.0;

    std::size_t max_candidates = 10;

    // ================================================================
    // POSE SUPPLEMENT RETRIEVAL V1
    //
    // Keep normal SC Top-K unchanged and append AT MOST one additional
    // long-history pose-nearest candidate.
    //
    // It is only a retrieval supplement, never direct loop acceptance.
    // ================================================================
    bool enable_pose_supplement = false;

    std::size_t
        pose_supplement_min_keyframe_id_separation = 100;

    double
        pose_supplement_max_distance = 2.0;

    ScanContextConfig scan_context;
};

struct ScanContextShadowDiagnostics
{
    std::size_t database_entries = 0;
    std::size_t separation_eligible = 0;
    std::size_t valid_matches = 0;
    std::size_t accepted_candidates = 0;

    bool has_best_match = false;
    ScanContextShadowCandidate best_match;
};

class ScanContextShadowDetector
{
public:
    explicit ScanContextShadowDetector(
        const ScanContextShadowConfig &config =
            ScanContextShadowConfig());

    bool AddKeyframe(
        const Keyframe &keyframe);

    std::vector<ScanContextShadowCandidate> Detect(
        std::size_t current_keyframe_id,
        ScanContextShadowDiagnostics *diagnostics = nullptr) const;

    std::size_t DescriptorCount() const;

    void Clear();

private:
    struct DescriptorEntry
    {
        std::size_t keyframe_id = 0;
        double timestamp = 0.0;

        Eigen::Isometry3d T_WL =
            Eigen::Isometry3d::Identity();

        ScanContextDescriptor descriptor;
    };

    const DescriptorEntry *FindDescriptor(
        std::size_t keyframe_id) const;

private:
    ScanContextShadowConfig config_;
    ScanContext scan_context_;

    std::vector<DescriptorEntry> database_;
};
