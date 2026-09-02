#include "fr_slam/loop/fr_scan_context.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace
{

constexpr double kPi =
    3.14159265358979323846;

} // namespace

ScanContext::ScanContext(
    const ScanContextConfig &config)
    : config_(config)
{
    if (config_.num_rings == 0)
    {
        config_.num_rings = 20;
    }

    if (config_.num_sectors == 0)
    {
        config_.num_sectors = 60;
    }

    if (!std::isfinite(config_.min_radius) ||
        config_.min_radius < 0.0)
    {
        config_.min_radius = 1.0;
    }

    if (!std::isfinite(config_.max_radius) ||
        config_.max_radius <= config_.min_radius)
    {
        config_.max_radius = 80.0;
    }

    if (config_.min_valid_points == 0)
    {
        config_.min_valid_points = 100;
    }

    if (!std::isfinite(
            config_.occupied_height_epsilon) ||
        config_.occupied_height_epsilon <= 0.0)
    {
        config_.occupied_height_epsilon = 1.0;
    }
}

ScanContextDescriptor
ScanContext::MakeDescriptor(
    const pcl::PointCloud<LIDAR_POINT>::ConstPtr &cloud_S) const
{
    ScanContextDescriptor descriptor;

    descriptor.matrix =
        Eigen::MatrixXf::Zero(
            static_cast<Eigen::Index>(
                config_.num_rings),
            static_cast<Eigen::Index>(
                config_.num_sectors));

    if (!cloud_S ||
        cloud_S->empty())
    {
        return descriptor;
    }

    // ------------------------------------------------------------------------
    // Pass 1:
    // determine the minimum z among points that can actually contribute to
    // Scan Context.
    //
    // We later encode:
    //
    //     cell_value = max(z - min_z + epsilon)
    //
    // so every occupied cell is strictly positive while empty cells remain 0.
    // ------------------------------------------------------------------------
    double min_z =
        std::numeric_limits<double>::infinity();

    std::size_t valid_points = 0;

    for (const LIDAR_POINT &point :
         cloud_S->points)
    {
        if (!std::isfinite(point.x) ||
            !std::isfinite(point.y) ||
            !std::isfinite(point.z))
        {
            continue;
        }

        const double x =
            static_cast<double>(point.x);

        const double y =
            static_cast<double>(point.y);

        const double radius =
            std::sqrt(
                x * x +
                y * y);

        if (!std::isfinite(radius) ||
            radius < config_.min_radius ||
            radius > config_.max_radius)
        {
            continue;
        }

        min_z =
            std::min(
                min_z,
                static_cast<double>(
                    point.z));

        ++valid_points;
    }

    descriptor.valid_points =
        valid_points;

    if (valid_points <
            config_.min_valid_points ||
        !std::isfinite(min_z))
    {
        return descriptor;
    }

    // ------------------------------------------------------------------------
    // Pass 2:
    // polar binning.
    //
    // Ring:
    //
    //     min_radius -------------------- max_radius
    //          | ring0 | ring1 | ... | ringN |
    //
    // Sector:
    //
    //     atan2(y, x) -> [0, 2*pi)
    //
    // Each cell stores maximum relative height.
    // ------------------------------------------------------------------------
    const double radial_span =
        config_.max_radius -
        config_.min_radius;

    for (const LIDAR_POINT &point :
         cloud_S->points)
    {
        if (!std::isfinite(point.x) ||
            !std::isfinite(point.y) ||
            !std::isfinite(point.z))
        {
            continue;
        }

        const double x =
            static_cast<double>(point.x);

        const double y =
            static_cast<double>(point.y);

        const double z =
            static_cast<double>(point.z);

        const double radius =
            std::sqrt(
                x * x +
                y * y);

        if (!std::isfinite(radius) ||
            radius < config_.min_radius ||
            radius > config_.max_radius)
        {
            continue;
        }

        double angle =
            std::atan2(
                y,
                x);

        if (angle < 0.0)
        {
            angle +=
                2.0 * kPi;
        }

        const double normalized_radius =
            (radius -
             config_.min_radius) /
            radial_span;

        const double normalized_angle =
            angle /
            (2.0 * kPi);

        std::size_t ring_index =
            static_cast<std::size_t>(
                normalized_radius *
                static_cast<double>(
                    config_.num_rings));

        std::size_t sector_index =
            static_cast<std::size_t>(
                normalized_angle *
                static_cast<double>(
                    config_.num_sectors));

        // radius == max_radius can numerically land exactly on num_rings.
        if (ring_index >=
            config_.num_rings)
        {
            ring_index =
                config_.num_rings - 1;
        }

        if (sector_index >=
            config_.num_sectors)
        {
            sector_index =
                config_.num_sectors - 1;
        }

        const float height_value =
            static_cast<float>(
                z -
                min_z +
                config_
                    .occupied_height_epsilon);

        float &cell =
            descriptor.matrix(
                static_cast<Eigen::Index>(
                    ring_index),
                static_cast<Eigen::Index>(
                    sector_index));

        if (height_value >
            cell)
        {
            cell =
                height_value;
        }
    }

    descriptor.valid =
        descriptor.matrix.maxCoeff() > 0.0f;

    return descriptor;
}

bool ScanContext::SectorCosineSimilarity(
    const Eigen::VectorXf &reference_sector,
    const Eigen::VectorXf &query_sector,
    double &similarity) const
{
    similarity = 0.0;

    if (reference_sector.size() == 0 ||
        reference_sector.size() !=
            query_sector.size())
    {
        return false;
    }

    const double reference_norm =
        static_cast<double>(
            reference_sector.norm());

    const double query_norm =
        static_cast<double>(
            query_sector.norm());

    // Empty sector columns are ignored, matching the usual Scan Context idea
    // of comparing only sectors that contain useful geometry on both sides.
    constexpr double min_norm =
        1e-6;

    if (reference_norm <= min_norm ||
        query_norm <= min_norm)
    {
        return false;
    }

    similarity =
        static_cast<double>(
            reference_sector.dot(
                query_sector)) /
        (reference_norm *
         query_norm);

    if (!std::isfinite(similarity))
    {
        similarity = 0.0;
        return false;
    }

    // Protect against tiny floating-point overshoot.
    similarity =
        std::max(
            -1.0,
            std::min(
                1.0,
                similarity));

    return true;
}

ScanContextMatch ScanContext::Compare(
    const ScanContextDescriptor &reference,
    const ScanContextDescriptor &query) const
{
    ScanContextMatch best_match;

    if (!reference.valid ||
        !query.valid)
    {
        return best_match;
    }

    if (reference.matrix.rows() !=
            static_cast<Eigen::Index>(
                config_.num_rings) ||
        reference.matrix.cols() !=
            static_cast<Eigen::Index>(
                config_.num_sectors) ||
        query.matrix.rows() !=
            reference.matrix.rows() ||
        query.matrix.cols() !=
            reference.matrix.cols())
    {
        return best_match;
    }

    // ------------------------------------------------------------------------
    // Exhaustive yaw search.
    //
    // For every circular sector shift:
    //
    //     reference.col(sector)
    //
    // is compared with:
    //
    //     query.col((sector + shift) % num_sectors)
    //
    // The score is the average cosine similarity over sector pairs that are
    // non-empty in BOTH descriptors.
    // ------------------------------------------------------------------------
    for (std::size_t shift = 0;
         shift < config_.num_sectors;
         ++shift)
    {
        double similarity_sum = 0.0;
        std::size_t compared_sectors = 0;

        for (std::size_t sector = 0;
             sector < config_.num_sectors;
             ++sector)
        {
            const std::size_t query_sector_index =
                (sector + shift) %
                config_.num_sectors;

            const Eigen::VectorXf reference_sector =
                reference.matrix.col(
                    static_cast<Eigen::Index>(
                        sector));

            const Eigen::VectorXf query_sector =
                query.matrix.col(
                    static_cast<Eigen::Index>(
                        query_sector_index));

            double sector_similarity = 0.0;

            if (!SectorCosineSimilarity(
                    reference_sector,
                    query_sector,
                    sector_similarity))
            {
                continue;
            }

            similarity_sum +=
                sector_similarity;

            ++compared_sectors;
        }

        // A shift supported by too little common angular coverage is not a
        // trustworthy Scan Context comparison.
        const std::size_t minimum_compared_sectors =
            std::max<std::size_t>(
                3,
                config_.num_sectors / 10);

        if (compared_sectors <
            minimum_compared_sectors)
        {
            continue;
        }

        const double similarity =
            similarity_sum /
            static_cast<double>(
                compared_sectors);

        const double distance =
            1.0 -
            similarity;

        if (!std::isfinite(distance))
        {
            continue;
        }

        if (!best_match.valid ||
            distance <
                best_match.distance)
        {
            best_match.valid = true;
            best_match.distance =
                distance;
            best_match.similarity =
                similarity;
            best_match.sector_shift =
                shift;
            best_match.yaw_shift_deg =
                static_cast<double>(
                    shift) *
                360.0 /
                static_cast<double>(
                    config_.num_sectors);
        }
    }

    return best_match;
}

const ScanContextConfig &
ScanContext::GetConfig() const
{
    return config_;
}
