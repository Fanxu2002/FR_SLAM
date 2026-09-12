#pragma once

#include "fr_slam/common/point_types.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <pcl/point_cloud.h>

// ============================================================================
// FR-SLAM native BTC-style descriptor (V25)
//
// Purpose:
//   * the sole global loop proposal path;
//   * triangle geometry provides rigid-transform invariant global structure;
//   * a compact binary local-geometry signature at each triangle vertex
//     suppresses ambiguous triangle matches;
//   * matched triangle vertices directly produce a full SE(3) hypothesis that
//     is ONLY used as an ICP seed.  This module never accepts a loop edge.
//
// Integration philosophy:
//   BTC = proposal + full 6DoF initial guess
//   Existing FR-SLAM prescore / ICP / temporal / cycle / PGO guards
//         = final authority
//
// This is a dependency-light FR-SLAM implementation of the BTC principle.  It
// intentionally avoids the official ROS1/OpenCV/Ceres runtime dependencies so
// it can live inside the current ROS2 Humble backend without changing the rest
// of the project build graph.
// ============================================================================
namespace fr_slam_btc
{
    struct Config
    {
        // Descriptor cloud simplification.
        double voxel_size_m = 0.45;

        // Uniformly distributed keypoints selected by deterministic farthest-point
        // sampling after voxelization.
        std::size_t max_keypoints = 72;

        // Binary local geometry around every keypoint.
        double binary_radius_m = 2.4;
        double binary_height_extent_m = 1.6;

        // Triangle generation.
        std::size_t triangle_neighbor_count = 7;
        std::size_t max_triangles = 320;
        double triangle_min_side_m = 0.8;
        double triangle_max_side_m = 8.0;
        double triangle_min_area_m2 = 0.18;

        // Rough descriptor matching.  Deliberately recall-oriented; the existing
        // FR-SLAM geometry stack still decides whether a loop is real.
        double side_abs_tolerance_m = 0.30;
        double side_rel_tolerance = 0.08;
        double min_binary_similarity = 0.45;
        std::size_t max_raw_triangle_matches = 80;

        // Triangle-consensus pose hypothesis.
        double inlier_vertex_residual_m = 0.75;
        std::size_t min_raw_triangle_matches = 3;
        std::size_t min_inlier_triangle_matches = 2;
    };

    struct BinarySignature
    {
        std::uint64_t occupied_once = 0;
        std::uint64_t occupied_twice = 0;
    };

    struct Keypoint
    {
        Eigen::Vector3d point = Eigen::Vector3d::Zero();
        BinarySignature binary;
    };

    struct Triangle
    {
        std::array<Eigen::Vector3d, 3> vertices;
        std::array<BinarySignature, 3> binary;
        Eigen::Vector3d sorted_sides = Eigen::Vector3d::Zero();
        Eigen::Vector3d center = Eigen::Vector3d::Zero();
        double area_m2 = 0.0;
    };

    struct Descriptor
    {
        bool valid = false;
        std::size_t input_points = 0;
        std::size_t voxel_points = 0;
        std::vector<Keypoint> keypoints;
        std::vector<Triangle> triangles;
    };

    struct MatchResult
    {
        bool valid = false;
        std::size_t raw_triangle_matches = 0;
        std::size_t inlier_triangle_matches = 0;
        double mean_binary_similarity = 0.0;
        double mean_vertex_residual_m =
            std::numeric_limits<double>::infinity();
        double similarity = 0.0;

        // Query -> Reference.
        Eigen::Isometry3d T_reference_query =
            Eigen::Isometry3d::Identity();
    };

    namespace detail
    {
        struct VoxelKey
        {
            std::int64_t x = 0;
            std::int64_t y = 0;
            std::int64_t z = 0;

            bool operator==(const VoxelKey &other) const
            {
                return x == other.x &&
                       y == other.y &&
                       z == other.z;
            }
        };

        struct VoxelKeyHash
        {
            std::size_t operator()(const VoxelKey &key) const
            {
                const std::uint64_t ux =
                    static_cast<std::uint64_t>(key.x * 73856093LL);
                const std::uint64_t uy =
                    static_cast<std::uint64_t>(key.y * 19349663LL);
                const std::uint64_t uz =
                    static_cast<std::uint64_t>(key.z * 83492791LL);

                return static_cast<std::size_t>(ux ^ uy ^ uz);
            }
        };

        struct VoxelAccumulator
        {
            Eigen::Vector3d sum = Eigen::Vector3d::Zero();
            std::size_t count = 0;
        };

        inline std::size_t Popcount64(std::uint64_t value)
        {
#if defined(__GNUG__) || defined(__clang__)
            return static_cast<std::size_t>(
                __builtin_popcountll(
                    static_cast<unsigned long long>(value)));
#else
            std::size_t count = 0;
            while (value != 0)
            {
                value &= (value - 1);
                ++count;
            }
            return count;
#endif
        }

        inline double BinarySimilarity(
            const BinarySignature &lhs,
            const BinarySignature &rhs)
        {
            const std::uint64_t lhs_union =
                lhs.occupied_once | rhs.occupied_once;
            const std::uint64_t lhs_intersection =
                lhs.occupied_once & rhs.occupied_once;

            const std::uint64_t rhs_union =
                lhs.occupied_twice | rhs.occupied_twice;
            const std::uint64_t rhs_intersection =
                lhs.occupied_twice & rhs.occupied_twice;

            const std::size_t union_count =
                Popcount64(lhs_union) +
                Popcount64(rhs_union);

            if (union_count == 0)
            {
                return 0.0;
            }

            const std::size_t intersection_count =
                Popcount64(lhs_intersection) +
                Popcount64(rhs_intersection);

            return static_cast<double>(intersection_count) /
                   static_cast<double>(union_count);
        }

        inline std::vector<Eigen::Vector3d> VoxelCentroids(
            const pcl::PointCloud<LIDAR_POINT>::ConstPtr &cloud,
            double voxel_size_m)
        {
            std::vector<Eigen::Vector3d> centroids;

            if (!cloud || cloud->empty() ||
                !std::isfinite(voxel_size_m) ||
                voxel_size_m <= 0.0)
            {
                return centroids;
            }

            std::unordered_map<
                VoxelKey,
                VoxelAccumulator,
                VoxelKeyHash>
                voxels;

            voxels.reserve(cloud->size());

            const double inv_leaf = 1.0 / voxel_size_m;

            for (const LIDAR_POINT &point : cloud->points)
            {
                if (!std::isfinite(point.x) ||
                    !std::isfinite(point.y) ||
                    !std::isfinite(point.z))
                {
                    continue;
                }

                VoxelKey key;
                key.x = static_cast<std::int64_t>(
                    std::floor(static_cast<double>(point.x) * inv_leaf));
                key.y = static_cast<std::int64_t>(
                    std::floor(static_cast<double>(point.y) * inv_leaf));
                key.z = static_cast<std::int64_t>(
                    std::floor(static_cast<double>(point.z) * inv_leaf));

                VoxelAccumulator &accumulator = voxels[key];
                accumulator.sum += Eigen::Vector3d(
                    static_cast<double>(point.x),
                    static_cast<double>(point.y),
                    static_cast<double>(point.z));
                ++accumulator.count;
            }

            centroids.reserve(voxels.size());

            for (const auto &entry : voxels)
            {
                if (entry.second.count == 0)
                {
                    continue;
                }

                const Eigen::Vector3d centroid =
                    entry.second.sum /
                    static_cast<double>(entry.second.count);

                if (centroid.allFinite())
                {
                    centroids.push_back(centroid);
                }
            }

            std::sort(
                centroids.begin(),
                centroids.end(),
                [](const Eigen::Vector3d &lhs,
                   const Eigen::Vector3d &rhs)
                {
                    if (lhs.x() != rhs.x())
                    {
                        return lhs.x() < rhs.x();
                    }
                    if (lhs.y() != rhs.y())
                    {
                        return lhs.y() < rhs.y();
                    }
                    return lhs.z() < rhs.z();
                });

            return centroids;
        }

        inline std::vector<std::size_t> FarthestPointIndices(
            const std::vector<Eigen::Vector3d> &points,
            std::size_t max_points)
        {
            std::vector<std::size_t> selected;

            if (points.empty() || max_points == 0)
            {
                return selected;
            }

            if (points.size() <= max_points)
            {
                selected.resize(points.size());
                for (std::size_t i = 0; i < points.size(); ++i)
                {
                    selected[i] = i;
                }
                return selected;
            }

            Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
            for (const Eigen::Vector3d &point : points)
            {
                centroid += point;
            }
            centroid /= static_cast<double>(points.size());

            std::size_t first_index = 0;
            double first_distance_sq = -1.0;

            for (std::size_t i = 0; i < points.size(); ++i)
            {
                const double distance_sq =
                    (points[i] - centroid).squaredNorm();

                if (distance_sq > first_distance_sq)
                {
                    first_distance_sq = distance_sq;
                    first_index = i;
                }
            }

            selected.reserve(max_points);
            selected.push_back(first_index);

            std::vector<double> min_distance_sq(
                points.size(),
                std::numeric_limits<double>::infinity());

            while (selected.size() < max_points)
            {
                const std::size_t newest = selected.back();

                for (std::size_t i = 0; i < points.size(); ++i)
                {
                    const double distance_sq =
                        (points[i] - points[newest]).squaredNorm();

                    if (distance_sq < min_distance_sq[i])
                    {
                        min_distance_sq[i] = distance_sq;
                    }
                }

                std::size_t best_index = 0;
                double best_distance_sq = -1.0;

                for (std::size_t i = 0; i < points.size(); ++i)
                {
                    const bool already_selected =
                        std::find(
                            selected.begin(),
                            selected.end(),
                            i) != selected.end();

                    if (!already_selected &&
                        min_distance_sq[i] > best_distance_sq)
                    {
                        best_distance_sq = min_distance_sq[i];
                        best_index = i;
                    }
                }

                if (best_distance_sq <= 0.0 ||
                    !std::isfinite(best_distance_sq))
                {
                    break;
                }

                selected.push_back(best_index);
            }

            return selected;
        }

        inline BinarySignature BuildBinarySignature(
            const Eigen::Vector3d &keypoint,
            const std::vector<Eigen::Vector3d> &support_points,
            const Config &config)
        {
            BinarySignature signature;

            std::array<std::uint16_t, 64> counts;
            counts.fill(0);

            const double radius =
                std::max(0.1, config.binary_radius_m);
            const double height_extent =
                std::max(0.2, config.binary_height_extent_m);

            for (const Eigen::Vector3d &point : support_points)
            {
                const Eigen::Vector3d delta =
                    point - keypoint;

                const double radial =
                    std::sqrt(
                        delta.x() * delta.x() +
                        delta.y() * delta.y());

                if (!std::isfinite(radial) ||
                    radial <= 1.0e-6 ||
                    radial > radius ||
                    std::abs(delta.z()) > height_extent)
                {
                    continue;
                }

                std::size_t radial_bin =
                    static_cast<std::size_t>(
                        std::floor(
                            8.0 * radial / radius));

                if (radial_bin >= 8)
                {
                    radial_bin = 7;
                }

                const double normalized_height =
                    (delta.z() + height_extent) /
                    (2.0 * height_extent);

                std::size_t height_bin =
                    static_cast<std::size_t>(
                        std::floor(8.0 * normalized_height));

                if (height_bin >= 8)
                {
                    height_bin = 7;
                }

                const std::size_t bit_index =
                    height_bin * 8 + radial_bin;

                if (bit_index < counts.size() &&
                    counts[bit_index] <
                        std::numeric_limits<std::uint16_t>::max())
                {
                    ++counts[bit_index];
                }
            }

            for (std::size_t bit = 0; bit < counts.size(); ++bit)
            {
                if (counts[bit] >= 1)
                {
                    signature.occupied_once |=
                        (std::uint64_t(1) << bit);
                }

                if (counts[bit] >= 2)
                {
                    signature.occupied_twice |=
                        (std::uint64_t(1) << bit);
                }
            }

            return signature;
        }

        inline bool EstimateRigidTransform(
            const std::vector<Eigen::Vector3d> &query_points,
            const std::vector<Eigen::Vector3d> &reference_points,
            Eigen::Isometry3d &T_reference_query)
        {
            if (query_points.size() != reference_points.size() ||
                query_points.size() < 3)
            {
                return false;
            }

            Eigen::Vector3d query_centroid = Eigen::Vector3d::Zero();
            Eigen::Vector3d reference_centroid = Eigen::Vector3d::Zero();

            for (std::size_t i = 0; i < query_points.size(); ++i)
            {
                if (!query_points[i].allFinite() ||
                    !reference_points[i].allFinite())
                {
                    return false;
                }

                query_centroid += query_points[i];
                reference_centroid += reference_points[i];
            }

            const double inverse_count =
                1.0 / static_cast<double>(query_points.size());

            query_centroid *= inverse_count;
            reference_centroid *= inverse_count;

            Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();

            for (std::size_t i = 0; i < query_points.size(); ++i)
            {
                covariance +=
                    (query_points[i] - query_centroid) *
                    (reference_points[i] - reference_centroid).transpose();
            }

            Eigen::JacobiSVD<Eigen::Matrix3d> svd(
                covariance,
                Eigen::ComputeFullU | Eigen::ComputeFullV);

            Eigen::Matrix3d U = svd.matrixU();
            Eigen::Matrix3d V = svd.matrixV();

            if (!U.allFinite() || !V.allFinite())
            {
                return false;
            }

            Eigen::Matrix3d R = V * U.transpose();

            if (R.determinant() < 0.0)
            {
                V.col(2) *= -1.0;
                R = V * U.transpose();
            }

            const Eigen::Vector3d t =
                reference_centroid -
                R * query_centroid;

            if (!R.allFinite() || !t.allFinite())
            {
                return false;
            }

            T_reference_query = Eigen::Isometry3d::Identity();
            T_reference_query.linear() = R;
            T_reference_query.translation() = t;

            return T_reference_query.matrix().allFinite();
        }

        inline std::uint64_t TriangleIndexKey(
            std::size_t a,
            std::size_t b,
            std::size_t c)
        {
            std::array<std::size_t, 3> ids = {{a, b, c}};
            std::sort(ids.begin(), ids.end());

            // 21 bits per index is vastly more than this module needs.
            return (static_cast<std::uint64_t>(ids[0]) << 42) |
                   (static_cast<std::uint64_t>(ids[1]) << 21) |
                   static_cast<std::uint64_t>(ids[2]);
        }

        inline Triangle MakeCanonicalTriangle(
            const std::vector<Keypoint> &keypoints,
            std::size_t i0,
            std::size_t i1,
            std::size_t i2)
        {
            const std::array<std::size_t, 3> original = {{i0, i1, i2}};

            const Eigen::Vector3d &p0 = keypoints[i0].point;
            const Eigen::Vector3d &p1 = keypoints[i1].point;
            const Eigen::Vector3d &p2 = keypoints[i2].point;

            const double d01 = (p0 - p1).norm();
            const double d12 = (p1 - p2).norm();
            const double d20 = (p2 - p0).norm();

            // Each vertex is labeled by the length of the side opposite it.  This
            // gives a stable correspondence for non-degenerate triangles regardless of
            // point insertion order.
            std::array<std::pair<double, std::size_t>, 3> opposite = {{std::make_pair(d12, original[0]),
                                                                       std::make_pair(d20, original[1]),
                                                                       std::make_pair(d01, original[2])}};

            std::sort(
                opposite.begin(),
                opposite.end(),
                [](const std::pair<double, std::size_t> &lhs,
                   const std::pair<double, std::size_t> &rhs)
                {
                    if (lhs.first != rhs.first)
                    {
                        return lhs.first < rhs.first;
                    }
                    return lhs.second < rhs.second;
                });

            Triangle triangle;

            for (std::size_t vertex = 0; vertex < 3; ++vertex)
            {
                const std::size_t keypoint_index =
                    opposite[vertex].second;

                triangle.vertices[vertex] =
                    keypoints[keypoint_index].point;
                triangle.binary[vertex] =
                    keypoints[keypoint_index].binary;
            }

            std::array<double, 3> sides = {{d01, d12, d20}};
            std::sort(sides.begin(), sides.end());

            triangle.sorted_sides = Eigen::Vector3d(
                sides[0], sides[1], sides[2]);

            triangle.center =
                (p0 + p1 + p2) / 3.0;

            triangle.area_m2 =
                0.5 *
                ((p1 - p0).cross(p2 - p0)).norm();

            return triangle;
        }
    } // namespace detail

    class Engine
    {
    public:
        explicit Engine(const Config &config = Config())
            : config_(config)
        {
        }

        const Config &GetConfig() const
        {
            return config_;
        }

        Descriptor Build(
            const pcl::PointCloud<LIDAR_POINT>::ConstPtr &cloud) const
        {
            Descriptor descriptor;

            if (!cloud || cloud->empty())
            {
                return descriptor;
            }

            descriptor.input_points = cloud->size();

            const std::vector<Eigen::Vector3d> voxel_points =
                detail::VoxelCentroids(
                    cloud,
                    config_.voxel_size_m);

            descriptor.voxel_points = voxel_points.size();

            if (voxel_points.size() < 8)
            {
                return descriptor;
            }

            const std::vector<std::size_t> selected_indices =
                detail::FarthestPointIndices(
                    voxel_points,
                    config_.max_keypoints);

            descriptor.keypoints.reserve(selected_indices.size());

            for (const std::size_t index : selected_indices)
            {
                if (index >= voxel_points.size())
                {
                    continue;
                }

                Keypoint keypoint;
                keypoint.point = voxel_points[index];
                keypoint.binary =
                    detail::BuildBinarySignature(
                        keypoint.point,
                        voxel_points,
                        config_);

                descriptor.keypoints.push_back(keypoint);
            }

            if (descriptor.keypoints.size() < 6)
            {
                return descriptor;
            }

            struct TriangleCandidate
            {
                Triangle triangle;
                double score = 0.0;
            };

            std::vector<TriangleCandidate> triangle_candidates;
            triangle_candidates.reserve(
                descriptor.keypoints.size() *
                config_.triangle_neighbor_count);

            std::unordered_set<std::uint64_t> seen_triangles;

            for (std::size_t i = 0;
                 i < descriptor.keypoints.size();
                 ++i)
            {
                std::vector<std::pair<double, std::size_t>> neighbors;
                neighbors.reserve(descriptor.keypoints.size() - 1);

                for (std::size_t j = 0;
                     j < descriptor.keypoints.size();
                     ++j)
                {
                    if (i == j)
                    {
                        continue;
                    }

                    const double distance =
                        (descriptor.keypoints[i].point -
                         descriptor.keypoints[j].point)
                            .norm();

                    if (std::isfinite(distance) &&
                        distance >= config_.triangle_min_side_m &&
                        distance <= config_.triangle_max_side_m)
                    {
                        neighbors.emplace_back(distance, j);
                    }
                }

                std::sort(
                    neighbors.begin(),
                    neighbors.end(),
                    [](const std::pair<double, std::size_t> &lhs,
                       const std::pair<double, std::size_t> &rhs)
                    {
                        if (lhs.first != rhs.first)
                        {
                            return lhs.first < rhs.first;
                        }
                        return lhs.second < rhs.second;
                    });

                if (neighbors.size() > config_.triangle_neighbor_count)
                {
                    neighbors.resize(config_.triangle_neighbor_count);
                }

                for (std::size_t a = 0; a < neighbors.size(); ++a)
                {
                    for (std::size_t b = a + 1; b < neighbors.size(); ++b)
                    {
                        const std::size_t j = neighbors[a].second;
                        const std::size_t k = neighbors[b].second;

                        const std::uint64_t key =
                            detail::TriangleIndexKey(i, j, k);

                        if (!seen_triangles.insert(key).second)
                        {
                            continue;
                        }

                        const Triangle triangle =
                            detail::MakeCanonicalTriangle(
                                descriptor.keypoints,
                                i,
                                j,
                                k);

                        if (!triangle.sorted_sides.allFinite() ||
                            triangle.sorted_sides.x() <
                                config_.triangle_min_side_m ||
                            triangle.sorted_sides.z() >
                                config_.triangle_max_side_m ||
                            !std::isfinite(triangle.area_m2) ||
                            triangle.area_m2 <
                                config_.triangle_min_area_m2)
                        {
                            continue;
                        }

                        TriangleCandidate candidate;
                        candidate.triangle = triangle;

                        // Prefer large, non-degenerate triangles; they yield more
                        // stable 6DoF seeds than tiny local triangles.
                        candidate.score =
                            triangle.area_m2 +
                            0.05 * triangle.sorted_sides.sum();

                        triangle_candidates.push_back(candidate);
                    }
                }
            }

            std::sort(
                triangle_candidates.begin(),
                triangle_candidates.end(),
                [](const TriangleCandidate &lhs,
                   const TriangleCandidate &rhs)
                {
                    return lhs.score > rhs.score;
                });

            if (triangle_candidates.size() > config_.max_triangles)
            {
                triangle_candidates.resize(config_.max_triangles);
            }

            descriptor.triangles.reserve(triangle_candidates.size());

            for (const TriangleCandidate &candidate : triangle_candidates)
            {
                descriptor.triangles.push_back(candidate.triangle);
            }

            descriptor.valid =
                descriptor.keypoints.size() >= 6 &&
                descriptor.triangles.size() >=
                    config_.min_raw_triangle_matches;

            return descriptor;
        }

        MatchResult Match(
            const Descriptor &query,
            const Descriptor &reference) const
        {
            MatchResult result;

            if (!query.valid ||
                !reference.valid ||
                query.triangles.empty() ||
                reference.triangles.empty())
            {
                return result;
            }

            struct RawMatch
            {
                const Triangle *query = nullptr;
                const Triangle *reference = nullptr;
                double binary_similarity = 0.0;
                double side_error =
                    std::numeric_limits<double>::infinity();
            };

            std::vector<RawMatch> raw_matches;

            for (const Triangle &query_triangle : query.triangles)
            {
                for (const Triangle &reference_triangle : reference.triangles)
                {
                    bool side_compatible = true;
                    double normalized_side_error = 0.0;

                    for (int side = 0; side < 3; ++side)
                    {
                        const double q = query_triangle.sorted_sides(side);
                        const double r = reference_triangle.sorted_sides(side);

                        const double tolerance =
                            config_.side_abs_tolerance_m +
                            config_.side_rel_tolerance *
                                std::max(q, r);

                        const double error = std::abs(q - r);

                        if (!std::isfinite(error) ||
                            error > tolerance)
                        {
                            side_compatible = false;
                            break;
                        }

                        normalized_side_error +=
                            error /
                            std::max(0.1, tolerance);
                    }

                    if (!side_compatible)
                    {
                        continue;
                    }

                    double binary_similarity = 0.0;

                    for (int vertex = 0; vertex < 3; ++vertex)
                    {
                        binary_similarity +=
                            detail::BinarySimilarity(
                                query_triangle.binary[vertex],
                                reference_triangle.binary[vertex]);
                    }

                    binary_similarity /= 3.0;

                    if (!std::isfinite(binary_similarity) ||
                        binary_similarity <
                            config_.min_binary_similarity)
                    {
                        continue;
                    }

                    RawMatch match;
                    match.query = &query_triangle;
                    match.reference = &reference_triangle;
                    match.binary_similarity = binary_similarity;
                    match.side_error = normalized_side_error / 3.0;

                    raw_matches.push_back(match);
                }
            }

            std::sort(
                raw_matches.begin(),
                raw_matches.end(),
                [](const RawMatch &lhs,
                   const RawMatch &rhs)
                {
                    if (lhs.binary_similarity != rhs.binary_similarity)
                    {
                        return lhs.binary_similarity > rhs.binary_similarity;
                    }
                    return lhs.side_error < rhs.side_error;
                });

            if (raw_matches.size() > config_.max_raw_triangle_matches)
            {
                raw_matches.resize(config_.max_raw_triangle_matches);
            }

            result.raw_triangle_matches = raw_matches.size();

            if (raw_matches.size() < config_.min_raw_triangle_matches)
            {
                return result;
            }

            std::size_t best_hypothesis_index =
                std::numeric_limits<std::size_t>::max();
            std::vector<std::size_t> best_inlier_indices;
            double best_binary_sum = -1.0;
            double best_residual_sum =
                std::numeric_limits<double>::infinity();

            for (std::size_t hypothesis_index = 0;
                 hypothesis_index < raw_matches.size();
                 ++hypothesis_index)
            {
                const RawMatch &hypothesis_match =
                    raw_matches[hypothesis_index];

                std::vector<Eigen::Vector3d> query_points;
                std::vector<Eigen::Vector3d> reference_points;

                query_points.reserve(3);
                reference_points.reserve(3);

                for (int vertex = 0; vertex < 3; ++vertex)
                {
                    query_points.push_back(
                        hypothesis_match.query->vertices[vertex]);
                    reference_points.push_back(
                        hypothesis_match.reference->vertices[vertex]);
                }

                Eigen::Isometry3d hypothesis =
                    Eigen::Isometry3d::Identity();

                if (!detail::EstimateRigidTransform(
                        query_points,
                        reference_points,
                        hypothesis))
                {
                    continue;
                }

                std::vector<std::size_t> inlier_indices;
                double binary_sum = 0.0;
                double residual_sum = 0.0;

                for (std::size_t match_index = 0;
                     match_index < raw_matches.size();
                     ++match_index)
                {
                    const RawMatch &match = raw_matches[match_index];

                    double vertex_residual_sum = 0.0;

                    for (int vertex = 0; vertex < 3; ++vertex)
                    {
                        const Eigen::Vector3d transformed =
                            hypothesis * match.query->vertices[vertex];

                        vertex_residual_sum +=
                            (transformed -
                             match.reference->vertices[vertex])
                                .norm();
                    }

                    const double mean_vertex_residual =
                        vertex_residual_sum / 3.0;

                    if (std::isfinite(mean_vertex_residual) &&
                        mean_vertex_residual <=
                            config_.inlier_vertex_residual_m)
                    {
                        inlier_indices.push_back(match_index);
                        binary_sum += match.binary_similarity;
                        residual_sum += mean_vertex_residual;
                    }
                }

                const bool better =
                    inlier_indices.size() > best_inlier_indices.size() ||
                    (inlier_indices.size() == best_inlier_indices.size() &&
                     (binary_sum > best_binary_sum ||
                      (binary_sum == best_binary_sum &&
                       residual_sum < best_residual_sum)));

                if (better)
                {
                    best_hypothesis_index = hypothesis_index;
                    best_inlier_indices = inlier_indices;
                    best_binary_sum = binary_sum;
                    best_residual_sum = residual_sum;
                }
            }

            if (best_hypothesis_index ==
                    std::numeric_limits<std::size_t>::max() ||
                best_inlier_indices.size() <
                    config_.min_inlier_triangle_matches)
            {
                return result;
            }

            std::vector<Eigen::Vector3d> refine_query_points;
            std::vector<Eigen::Vector3d> refine_reference_points;
            refine_query_points.reserve(best_inlier_indices.size() * 3);
            refine_reference_points.reserve(best_inlier_indices.size() * 3);

            for (const std::size_t match_index : best_inlier_indices)
            {
                const RawMatch &match = raw_matches[match_index];

                for (int vertex = 0; vertex < 3; ++vertex)
                {
                    refine_query_points.push_back(
                        match.query->vertices[vertex]);
                    refine_reference_points.push_back(
                        match.reference->vertices[vertex]);
                }
            }

            Eigen::Isometry3d refined = Eigen::Isometry3d::Identity();

            if (!detail::EstimateRigidTransform(
                    refine_query_points,
                    refine_reference_points,
                    refined))
            {
                return result;
            }

            std::size_t final_inliers = 0;
            double final_binary_sum = 0.0;
            double final_residual_sum = 0.0;

            for (const RawMatch &match : raw_matches)
            {
                double vertex_residual_sum = 0.0;

                for (int vertex = 0; vertex < 3; ++vertex)
                {
                    const Eigen::Vector3d transformed =
                        refined * match.query->vertices[vertex];

                    vertex_residual_sum +=
                        (transformed -
                         match.reference->vertices[vertex])
                            .norm();
                }

                const double mean_vertex_residual =
                    vertex_residual_sum / 3.0;

                if (std::isfinite(mean_vertex_residual) &&
                    mean_vertex_residual <=
                        config_.inlier_vertex_residual_m)
                {
                    ++final_inliers;
                    final_binary_sum += match.binary_similarity;
                    final_residual_sum += mean_vertex_residual;
                }
            }

            if (final_inliers < config_.min_inlier_triangle_matches)
            {
                return result;
            }

            result.inlier_triangle_matches = final_inliers;
            result.mean_binary_similarity =
                final_binary_sum /
                static_cast<double>(final_inliers);
            result.mean_vertex_residual_m =
                final_residual_sum /
                static_cast<double>(final_inliers);

            const double support_score =
                std::min(
                    1.0,
                    static_cast<double>(final_inliers) / 10.0);

            result.similarity =
                0.70 * support_score +
                0.30 * std::max(
                           0.0,
                           std::min(1.0, result.mean_binary_similarity));

            result.T_reference_query = refined;
            result.valid =
                result.T_reference_query.matrix().allFinite() &&
                std::isfinite(result.similarity) &&
                std::isfinite(result.mean_vertex_residual_m);

            return result;
        }

    private:
        Config config_;
    };
} // namespace fr_slam_btc
