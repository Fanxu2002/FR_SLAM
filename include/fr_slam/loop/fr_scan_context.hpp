#pragma once

#include "fr_slam/common/fr_point_types.hpp"

#include <cstddef>
#include <limits>

#include <Eigen/Core>

#include <pcl/point_cloud.h>

// ============================================================================
// Scan Context V1
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
    // For the current MID-360 based pipeline, 80 m is a safe first value.
    // We can tune this from real data later.
    double max_radius = 80.0;

    // Minimum number of valid points required to accept a descriptor.
    std::size_t min_valid_points = 100;

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
    //     occupied cell whose original z was <= 0.
    double occupied_height_epsilon = 1.0;
};

struct ScanContextDescriptor
{
    Eigen::MatrixXf matrix;

    std::size_t valid_points = 0;

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

    // Mean cosine similarity of valid sector-column pairs.
    //
    // Larger is better. Usually lies in [0, 1] for our non-negative
    // descriptors.
    double similarity = 0.0;

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
