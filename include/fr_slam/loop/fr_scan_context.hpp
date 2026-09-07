#pragma once

#include "fr_slam/common/fr_point_types.hpp"

#include <cstddef>
#include <limits>

#include <Eigen/Core>

#include <pcl/point_cloud.h>

// ============================================================================
// Scan Context V3
//
// Purpose:
//
//     Convert one finished Submap cloud, expressed in the Submap's own S frame,
//     into a compact polar descriptor:
//
//         rows    -> Rings
//         columns -> Sectors
//
// Each occupied cell stores the maximum relative height observed in that
// polar bin.
//
// This module is intentionally independent from PoseGraph / LoopDetector.
// It only knows how to:
//
//     1. Build a descriptor.
//     2. Compare two descriptors with circular sector shifts.
//
// Circular sector shifting makes the comparison approximately yaw invariant
// and also gives a coarse yaw-shift estimate for the future LoopVerifier.
//
// IMPORTANT:
//     This is candidate retrieval only.
//     It does NOT prove a loop closure and does NOT create a PoseGraph edge.
// ============================================================================

struct ScanContextConfig
{
    // Standard lightweight resolution for a first implementation.
    std::size_t num_rings = 20;
    std::size_t num_sectors = 60;

    // Ignore very near points around the sensor / Submap origin.
    double min_radius = 1.0;

    // Maximum radial range represented by the descriptor.
    //
    // Keep this consistent with the preprocessing range.  The current Livox
    // and Hesai profiles both retain points up to 30 m.  Using 80 m here made
    // most rings empty and was one cause of sparse one-column matches.
    double max_radius = 30.0;

    // Minimum number of valid points required to accept a descriptor.
    std::size_t min_valid_points = 100;

    // A descriptor made from only a handful of angular/radial bins is not
    // discriminative, even if it contains many duplicate points.
    std::size_t min_occupied_sectors = 6;
    std::size_t min_occupied_cells = 12;

    // Minimum Jaccard-style angular coverage:
    //
    //     common occupied sectors / union occupied sectors
    //
    // A comparison must explain a meaningful fraction of the occupied
    // angular support in BOTH descriptors.  This prevents the old failure
    // mode where only a few coincident sparse columns produced
    // similarity == 1.0.
    double min_sector_coverage_ratio = 0.20;

    // Penalize unmatched occupied sectors/cells after the raw cosine score.
    // Coverage is measured with true intersection-over-union (IoU/Jaccard)
    // ratios for both sectors and cells.
    //
    // 0 disables the penalty; 1 applies the full missing-coverage penalty.
    double coverage_penalty_weight = 0.50;

    // Empty Scan Context cells are exactly 0.
    //
    // Occupied cells therefore need a strictly positive value.
    // We first subtract the minimum z of the current cloud, then add this
    // epsilon:
    //
    //     value = z - min_z + occupied_height_epsilon
    //
    // This avoids the ambiguity between:
    //
    //     empty cell = 0
    //
    // and:
    //
    //     occupied cell whose shifted relative height is exactly 0.
    //
    // Keep this value very small.  A large offset (for example 1.0 m)
    // behaves like a common DC component and makes cosine similarity depend
    // too strongly on occupancy while weakening the actual height pattern.
    double occupied_height_epsilon = 1.0e-3;
};

struct ScanContextDescriptor
{
    Eigen::MatrixXf matrix;

    std::size_t valid_points = 0;

    std::size_t occupied_sectors = 0;
    std::size_t occupied_cells = 0;

    bool valid = false;
};

struct ScanContextMatch
{
    // Scan Context distance:
    //
    //     distance = 1 - similarity
    //
    // Smaller is better.
    double distance =
        std::numeric_limits<double>::infinity();

    // Coverage-adjusted similarity used by candidate ranking/gating.
    //
    // Larger is better. Usually lies in [0, 1] for our non-negative
    // descriptors.
    double similarity = 0.0;

    // Mean cosine similarity before the occupancy-coverage penalty.
    double raw_cosine_similarity = 0.0;

    // Jaccard / IoU angular coverage:
    //
    //     common occupied sectors / union occupied sectors
    double sector_coverage_ratio = 0.0;

    // Jaccard / IoU cell coverage:
    //
    //     common occupied cells / union occupied cells
    double cell_coverage_ratio = 0.0;

    std::size_t compared_sectors = 0;

    // Number of sector columns by which the QUERY descriptor is indexed
    // relative to the REFERENCE descriptor during the best match.
    //
    // In Compare(reference, query), comparison is:
    //
    //     reference.col(sector)
    //
    // versus:
    //
    //     query.col((sector + sector_shift) % num_sectors)
    //
    std::size_t sector_shift = 0;

    // Coarse yaw displacement corresponding to sector_shift.
    //
    // With 60 sectors:
    //
    //     one sector = 6 degrees
    //
    // We deliberately call this "yaw_shift", not a final relative yaw pose.
    // The exact sign convention for the future registration initial guess
    // will be calibrated when LoopVerifier is connected.
    double yaw_shift_deg = 0.0;

    bool valid = false;
};

class ScanContext
{
public:
    explicit ScanContext(
        const ScanContextConfig &config =
            ScanContextConfig());

    // Build one descriptor from a frozen Submap-local cloud:
    //
    //     cloud_S
    //
    // Input coordinates must already be expressed in the Submap frame.
    ScanContextDescriptor MakeDescriptor(
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &cloud_S) const;

    // Compare:
    //
    //     reference descriptor
    //
    // against:
    //
    //     query descriptor
    //
    // using exhaustive circular sector shifting.
    //
    // V1 intentionally uses exhaustive shifts because the database is still
    // small and this keeps the implementation easy to verify.
    ScanContextMatch Compare(
        const ScanContextDescriptor &reference,
        const ScanContextDescriptor &query) const;

    const ScanContextConfig &GetConfig() const;

private:
    // Similarity of two sector columns.
    //
    // Returns false when either sector column is empty.
    bool SectorCosineSimilarity(
        const Eigen::VectorXf &reference_sector,
        const Eigen::VectorXf &query_sector,
        double &similarity) const;

private:
    ScanContextConfig config_;
};
