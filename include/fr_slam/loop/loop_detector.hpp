#pragma once

#include "fr_slam/mapping/keyframe.hpp"

#include <cstddef>
#include <limits>

// ============================================================================
// LoopCandidate
//
// Candidate identity used by the BTC -> geometry -> consistency -> PGO chain.
// Retrieval is provided by BTC; this type only carries the physical Keyframe
// endpoints and retrieval diagnostics into the FR-SLAM backend.
// ============================================================================
struct LoopCandidate
{
    std::size_t current_id = 0;
    std::size_t candidate_id = 0;

    double distance =
        std::numeric_limits<double>::infinity();

    double time_separation_sec =
        std::numeric_limits<double>::infinity();

    double retrieval_score = 0.0;
    std::size_t matched_descriptor_pairs = 0;
};

struct LoopDetectorConfig
{
    // Master switch for the complete backend loop-closure chain.
    bool enabled = true;

    // Keyframe-level temporal exclusion applied to BTC candidates.
    std::size_t min_keyframe_id_separation = 30;
    double min_time_separation_sec = 10.0;

    // Optional frontend-pose diagnostic gate. Keep disabled by default because
    // loop closure must remain usable when frontend drift is large.
    bool use_pose_distance_gate = false;
    double max_candidate_distance = 5.0;

    // Safety cap if a future BTC adapter returns more than one candidate.
    std::size_t max_candidates = 10;
};

// ============================================================================
// LoopDetector
//
// Historical name retained to avoid a broad public-API change.  Place
// recognition has been removed from this class; BTC is now the only global
// retrieval source.  The class is therefore only a small configuration holder.
// ============================================================================
class LoopDetector
{
public:
    explicit LoopDetector(
        const LoopDetectorConfig &config = LoopDetectorConfig());

    const LoopDetectorConfig &GetConfig() const;

    void Clear();

private:
    LoopDetectorConfig config_;
};
