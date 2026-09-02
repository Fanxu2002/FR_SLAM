#pragma once

#include "fr_slam/mapping/fr_keyframe.hpp"
#include "fr_slam/loop/fr_scan_context.hpp"

#include <cstddef>
#include <limits>
#include <vector>

// ============================================================================
// LoopCandidate
//
// V3 meaning:
//     current_id   = current KEYFRAME id
//     candidate_id = historical KEYFRAME id
//
// Candidate retrieval is keyframe-driven. Geometry verification is performed
// later against a historical Submap selected around candidate_id.
// ============================================================================
struct LoopCandidate
{
    std::size_t current_id = 0;
    std::size_t candidate_id = 0;

    // Frontend pose distance between the two Keyframes.
    // Diagnostic by default; optional hard gate when configured.
    double distance =
        std::numeric_limits<double>::infinity();

    double time_separation_sec =
        std::numeric_limits<double>::infinity();

    // Scan Context distance: smaller is better.
    double scan_context_distance =
        std::numeric_limits<double>::infinity();

    // Scan Context similarity: larger is better.
    double scan_context_similarity = 0.0;

    std::size_t sector_shift = 0;
    double yaw_shift_deg = 0.0;
};

struct LoopDetectorConfig
{
    // Keyframe-level temporal exclusion.
    // With the current ~0.5 m keyframe rule, 30 KFs is already far enough to
    // avoid comparing with the immediate local trajectory while still allowing
    // a real loop to be queried well before a whole new Submap is finished.
    std::size_t min_keyframe_id_separation = 30;

    // Additional time exclusion so a stationary / slow-moving robot does not
    // create a trivial local loop merely because many KFs were produced.
    double min_time_separation_sec = 10.0;

    // Scan Context candidate gate.
    double max_scan_context_distance = 0.30;

    // Pose distance remains optional because loop closure must survive drift.
    bool use_pose_distance_gate = false;
    double max_candidate_distance = 5.0;

    std::size_t max_candidates = 5;

    ScanContextConfig scan_context;
};

// ============================================================================
// LoopDetector V3 -- KEYFRAME descriptor database
//
// Every accepted Keyframe can immediately enter this database:
//
//     New Keyframe
//          |
//          +--> Make Scan Context descriptor
//          |
//          +--> compare against sufficiently old Keyframes
//          |
//          v
//     Top-K historical Keyframe candidates
//
// This intentionally has NO dependency on Submap::finished.
// ============================================================================
class LoopDetector
{
public:
    explicit LoopDetector(
        const LoopDetectorConfig &config =
            LoopDetectorConfig());

    bool AddKeyframe(
        const Keyframe &keyframe);

    bool HasDescriptor(
        std::size_t keyframe_id) const;

    std::size_t DescriptorCount() const;

    std::vector<LoopCandidate> Detect(
        std::size_t current_keyframe_id) const;

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
    LoopDetectorConfig config_;
    ScanContext scan_context_;
    std::vector<DescriptorEntry> database_;
};
