#include "fr_slam/fr_ground_segmenter.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>

namespace
{

struct GroundGridCell
{
    std::vector<double> z_values;

    double surface_z =
        std::numeric_limits<double>::quiet_NaN();

    double low_roughness_m =
        std::numeric_limits<double>::quiet_NaN();

    bool surface_valid = false;
    bool is_ground = false;
    bool is_support = false;
};


struct SeedComponent
{
    std::vector<std::pair<int, int>> keys;

    std::set<std::size_t> angular_sectors;

    double median_surface_z =
        std::numeric_limits<double>::quiet_NaN();
};


struct HeightFieldModel
{
    bool valid = false;

    // z = a*x + b*y + c
    double a = 0.0;
    double b = 0.0;
    double c = 0.0;

    double slope_deg = 0.0;
    double rmse_m = 0.0;

    std::size_t support_cells = 0;
};


bool IsFinitePoint(
    const pcl::PointXYZ &point)
{
    return
        std::isfinite(point.x) &&
        std::isfinite(point.y) &&
        std::isfinite(point.z);
}


double DegreesToRadians(
    double degrees)
{
    return
        degrees *
        3.14159265358979323846 /
        180.0;
}


double RadiansToDegrees(
    double radians)
{
    return
        radians *
        180.0 /
        3.14159265358979323846;
}


double HorizontalRange(
    const pcl::PointXYZ &point)
{
    return
        std::hypot(
            static_cast<double>(point.x),
            static_cast<double>(point.y));
}


std::pair<int, int> ComputeGridKey(
    const pcl::PointXYZ &point,
    double grid_size_m)
{
    const int grid_x =
        static_cast<int>(
            std::floor(
                static_cast<double>(point.x) /
                grid_size_m));

    const int grid_y =
        static_cast<int>(
            std::floor(
                static_cast<double>(point.y) /
                grid_size_m));

    return
        std::make_pair(
            grid_x,
            grid_y);
}


Eigen::Vector2d GridCellCenter(
    const std::pair<int, int> &key,
    double grid_size_m)
{
    return Eigen::Vector2d(
        (static_cast<double>(key.first) + 0.5) *
            grid_size_m,
        (static_cast<double>(key.second) + 0.5) *
            grid_size_m);
}


double Median(
    std::vector<double> values)
{
    if (values.empty())
    {
        return
            std::numeric_limits<double>::quiet_NaN();
    }

    const std::size_t middle =
        values.size() / 2;

    std::nth_element(
        values.begin(),
        values.begin() +
            static_cast<std::ptrdiff_t>(middle),
        values.end());

    double median =
        values[middle];

    if ((values.size() % 2U) == 0U)
    {
        const double lower =
            *std::max_element(
                values.begin(),
                values.begin() +
                    static_cast<std::ptrdiff_t>(middle));

        median =
            0.5 *
            (lower + median);
    }

    return median;
}


bool RobustLowSurfaceStatistics(
    std::vector<double> z_values,
    double low_surface_fraction,
    double &surface_z,
    double &roughness_m)
{
    surface_z =
        std::numeric_limits<double>::quiet_NaN();

    roughness_m =
        std::numeric_limits<double>::quiet_NaN();

    if (z_values.empty())
    {
        return false;
    }

    std::sort(
        z_values.begin(),
        z_values.end());

    const double safe_fraction =
        std::clamp(
            low_surface_fraction,
            0.05,
            1.0);

    std::size_t low_count =
        static_cast<std::size_t>(
            std::ceil(
                safe_fraction *
                static_cast<double>(
                    z_values.size())));

    low_count =
        std::max<std::size_t>(
            1,
            low_count);

    low_count =
        std::min(
            low_count,
            z_values.size());

    std::vector<double> low_values(
        z_values.begin(),
        z_values.begin() +
            static_cast<std::ptrdiff_t>(low_count));

    surface_z =
        Median(
            low_values);

    if (!std::isfinite(surface_z))
    {
        return false;
    }

    std::vector<double> absolute_deviations;

    absolute_deviations.reserve(
        low_values.size());

    for (const double z :
         low_values)
    {
        absolute_deviations.push_back(
            std::abs(
                z -
                surface_z));
    }

    const double mad =
        Median(
            std::move(
                absolute_deviations));

    if (!std::isfinite(mad))
    {
        return false;
    }

    // Robust sigma estimate for a locally coherent low layer.
    roughness_m =
        1.4826 * mad;

    return
        std::isfinite(roughness_m);
}


bool NeighborSurfaceCompatible(
    const GroundGridCell &cell_a,
    const GroundGridCell &cell_b,
    int dx,
    int dy,
    const fr_slam::GroundSegmentationConfig &config)
{
    if (!cell_a.surface_valid ||
        !cell_b.surface_valid)
    {
        return false;
    }

    const double delta_z =
        std::abs(
            cell_b.surface_z -
            cell_a.surface_z);

    if (!std::isfinite(delta_z) ||
        delta_z >
            config.maximum_neighbor_height_jump_m)
    {
        return false;
    }

    const double horizontal_distance =
        config.grid_size_m *
        std::sqrt(
            static_cast<double>(
                dx * dx +
                dy * dy));

    if (!std::isfinite(horizontal_distance) ||
        horizontal_distance <= 1.0e-9)
    {
        return false;
    }

    const double slope =
        std::atan2(
            delta_z,
            horizontal_distance);

    const double maximum_slope =
        DegreesToRadians(
            config.maximum_local_slope_deg);

    return
        std::isfinite(slope) &&
        slope <= maximum_slope;
}


std::size_t ComputeAngularSector(
    const Eigen::Vector2d &center,
    std::size_t sector_count)
{
    if (sector_count == 0)
    {
        return 0;
    }

    double angle =
        std::atan2(
            center.y(),
            center.x());

    if (angle < 0.0)
    {
        angle +=
            2.0 *
            3.14159265358979323846;
    }

    const double normalized =
        angle /
        (2.0 *
         3.14159265358979323846);

    std::size_t sector =
        static_cast<std::size_t>(
            std::floor(
                normalized *
                static_cast<double>(
                    sector_count)));

    if (sector >= sector_count)
    {
        sector =
            sector_count - 1;
    }

    return sector;
}


bool IsNearSeedCandidate(
    const std::pair<int, int> &key,
    const GroundGridCell &cell,
    const fr_slam::GroundSegmentationConfig &config)
{
    if (!cell.surface_valid)
    {
        return false;
    }

    const Eigen::Vector2d center =
        GridCellCenter(
            key,
            config.grid_size_m);

    const double range =
        center.norm();

    return
        std::isfinite(range) &&
        range >= config.seed_minimum_range_m &&
        range <= config.seed_maximum_range_m;
}


std::vector<SeedComponent> BuildSeedComponents(
    const std::map<
        std::pair<int, int>,
        GroundGridCell> &grid_map,
    const fr_slam::GroundSegmentationConfig &config)
{
    std::vector<SeedComponent> components;

    std::set<std::pair<int, int>> visited;

    for (const auto &entry :
         grid_map)
    {
        const std::pair<int, int> &start_key =
            entry.first;

        if (visited.find(start_key) !=
            visited.end())
        {
            continue;
        }

        if (!IsNearSeedCandidate(
                start_key,
                entry.second,
                config))
        {
            continue;
        }

        SeedComponent component;

        std::queue<std::pair<int, int>> queue;

        queue.push(
            start_key);

        visited.insert(
            start_key);

        std::vector<double> component_heights;

        while (!queue.empty())
        {
            const std::pair<int, int> current_key =
                queue.front();

            queue.pop();

            const auto current_iterator =
                grid_map.find(
                    current_key);

            if (current_iterator ==
                grid_map.end())
            {
                continue;
            }

            const GroundGridCell &current_cell =
                current_iterator->second;

            component.keys.push_back(
                current_key);

            component_heights.push_back(
                current_cell.surface_z);

            const Eigen::Vector2d current_center =
                GridCellCenter(
                    current_key,
                    config.grid_size_m);

            component.angular_sectors.insert(
                ComputeAngularSector(
                    current_center,
                    config.seed_angular_sector_count));

            for (int dx = -1;
                 dx <= 1;
                 ++dx)
            {
                for (int dy = -1;
                     dy <= 1;
                     ++dy)
                {
                    if (dx == 0 &&
                        dy == 0)
                    {
                        continue;
                    }

                    const std::pair<int, int> neighbor_key =
                        std::make_pair(
                            current_key.first + dx,
                            current_key.second + dy);

                    if (visited.find(neighbor_key) !=
                        visited.end())
                    {
                        continue;
                    }

                    const auto neighbor_iterator =
                        grid_map.find(
                            neighbor_key);

                    if (neighbor_iterator ==
                        grid_map.end())
                    {
                        continue;
                    }

                    if (!IsNearSeedCandidate(
                            neighbor_key,
                            neighbor_iterator->second,
                            config))
                    {
                        continue;
                    }

                    if (!NeighborSurfaceCompatible(
                            current_cell,
                            neighbor_iterator->second,
                            dx,
                            dy,
                            config))
                    {
                        continue;
                    }

                    visited.insert(
                        neighbor_key);

                    queue.push(
                        neighbor_key);
                }
            }
        }

        component.median_surface_z =
            Median(
                std::move(
                    component_heights));

        components.push_back(
            std::move(
                component));
    }

    return components;
}


bool SelectBestSeedComponent(
    const std::vector<SeedComponent> &components,
    const fr_slam::GroundSegmentationConfig &config,
    SeedComponent &best_component)
{
    bool found = false;

    double best_score =
        -std::numeric_limits<double>::infinity();

    std::vector<double> component_medians;

    for (const SeedComponent &component :
         components)
    {
        if (component.keys.size() <
            config.minimum_seed_component_cells)
        {
            continue;
        }

        if (!std::isfinite(
                component.median_surface_z))
        {
            continue;
        }

        component_medians.push_back(
            component.median_surface_z);
    }

    const double median_of_component_medians =
        Median(
            component_medians);

    for (const SeedComponent &component :
         components)
    {
        if (component.keys.size() <
            config.minimum_seed_component_cells)
        {
            continue;
        }

        if (component.angular_sectors.size() <
            config.minimum_seed_angular_sectors)
        {
            continue;
        }

        if (!std::isfinite(
                component.median_surface_z))
        {
            continue;
        }

        const double cell_score =
            static_cast<double>(
                component.keys.size());

        const double angular_score =
            2.0 *
            static_cast<double>(
                component.angular_sectors.size());

        double lower_surface_bonus =
            0.0;

        if (std::isfinite(
                median_of_component_medians) &&
            component.median_surface_z <=
                median_of_component_medians)
        {
            lower_surface_bonus =
                3.0;
        }

        const double score =
            cell_score +
            angular_score +
            lower_surface_bonus;

        if (!found ||
            score > best_score)
        {
            best_component =
                component;

            best_score =
                score;

            found =
                true;
        }
    }

    return found;
}


bool FitHeightFieldSamples(
    const std::vector<Eigen::Vector3d> &samples,
    HeightFieldModel &model)
{
    model =
        HeightFieldModel();

    if (samples.size() < 3)
    {
        return false;
    }

    Eigen::MatrixXd A(
        static_cast<Eigen::Index>(
            samples.size()),
        3);

    Eigen::VectorXd z(
        static_cast<Eigen::Index>(
            samples.size()));

    for (std::size_t i = 0;
         i < samples.size();
         ++i)
    {
        const Eigen::Vector3d &sample =
            samples[i];

        if (!sample.allFinite())
        {
            return false;
        }

        A(
            static_cast<Eigen::Index>(i),
            0) = sample.x();

        A(
            static_cast<Eigen::Index>(i),
            1) = sample.y();

        A(
            static_cast<Eigen::Index>(i),
            2) = 1.0;

        z(
            static_cast<Eigen::Index>(i)) =
            sample.z();
    }

    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(
        A);

    if (qr.rank() < 3)
    {
        return false;
    }

    const Eigen::Vector3d parameters =
        qr.solve(
            z);

    if (!parameters.allFinite())
    {
        return false;
    }

    const Eigen::VectorXd residual =
        A * parameters -
        z;

    const double squared_error =
        residual.squaredNorm();

    if (!std::isfinite(squared_error))
    {
        return false;
    }

    model.a =
        parameters.x();

    model.b =
        parameters.y();

    model.c =
        parameters.z();

    model.rmse_m =
        std::sqrt(
            squared_error /
            static_cast<double>(
                samples.size()));

    model.slope_deg =
        RadiansToDegrees(
            std::atan(
                std::hypot(
                    model.a,
                    model.b)));

    model.support_cells =
        samples.size();

    model.valid =
        std::isfinite(model.a) &&
        std::isfinite(model.b) &&
        std::isfinite(model.c) &&
        std::isfinite(model.rmse_m) &&
        std::isfinite(model.slope_deg);

    return
        model.valid;
}


bool BuildLocalPredictionModel(
    const std::map<
        std::pair<int, int>,
        GroundGridCell> &grid_map,
    const std::pair<int, int> &candidate_key,
    const fr_slam::GroundSegmentationConfig &config,
    HeightFieldModel &model)
{
    model =
        HeightFieldModel();

    const int radius =
        std::max(
            1,
            config.prediction_neighbor_radius_cells);

    std::vector<Eigen::Vector3d> samples;

    for (int dx = -radius;
         dx <= radius;
         ++dx)
    {
        for (int dy = -radius;
             dy <= radius;
             ++dy)
        {
            if (dx == 0 &&
                dy == 0)
            {
                continue;
            }

            const std::pair<int, int> key =
                std::make_pair(
                    candidate_key.first + dx,
                    candidate_key.second + dy);

            const auto iterator =
                grid_map.find(
                    key);

            if (iterator ==
                grid_map.end())
            {
                continue;
            }

            const GroundGridCell &cell =
                iterator->second;

            if (!cell.surface_valid ||
                !cell.is_ground)
            {
                continue;
            }

            const Eigen::Vector2d center =
                GridCellCenter(
                    key,
                    config.grid_size_m);

            samples.emplace_back(
                center.x(),
                center.y(),
                cell.surface_z);
        }
    }

    if (samples.size() <
        config.minimum_prediction_support_cells)
    {
        return false;
    }

    if (!FitHeightFieldSamples(
            samples,
            model))
    {
        return false;
    }

    if (model.slope_deg >
        config.maximum_predicted_surface_slope_deg)
    {
        model.valid = false;
        return false;
    }

    return true;
}


bool CandidateMatchesLocalSurface(
    const std::pair<int, int> &candidate_key,
    const GroundGridCell &candidate_cell,
    const HeightFieldModel &model,
    const fr_slam::GroundSegmentationConfig &config)
{
    if (!candidate_cell.surface_valid ||
        !model.valid)
    {
        return false;
    }

    const Eigen::Vector2d center =
        GridCellCenter(
            candidate_key,
            config.grid_size_m);

    const double predicted_z =
        model.a * center.x() +
        model.b * center.y() +
        model.c;

    const double residual =
        std::abs(
            candidate_cell.surface_z -
            predicted_z);

    if (!std::isfinite(residual))
    {
        return false;
    }

    const double adaptive_tolerance =
        std::clamp(
            config.surface_prediction_base_tolerance_m +
                config.surface_prediction_roughness_scale *
                    std::max(
                        0.0,
                        candidate_cell.low_roughness_m) +
                config.surface_prediction_rmse_scale *
                    std::max(
                        0.0,
                        model.rmse_m),
            config.surface_prediction_base_tolerance_m,
            config.maximum_surface_prediction_tolerance_m);

    return
        residual <=
        adaptive_tolerance;
}



bool FitRobustHeightFieldModel(
    const std::vector<Eigen::Vector3d> &samples,
    std::size_t minimum_cells,
    double base_residual_m,
    double mad_scale,
    double maximum_residual_m,
    double minimum_inlier_ratio,
    HeightFieldModel &final_model,
    std::vector<Eigen::Vector3d> &inlier_samples,
    double &inlier_ratio)
{
    final_model =
        HeightFieldModel();

    inlier_samples.clear();

    inlier_ratio = 0.0;

    if (samples.size() <
        std::max<std::size_t>(
            3,
            minimum_cells))
    {
        return false;
    }

    HeightFieldModel initial_model;

    if (!FitHeightFieldSamples(
            samples,
            initial_model))
    {
        return false;
    }

    std::vector<double> absolute_residuals;

    absolute_residuals.reserve(
        samples.size());

    for (const Eigen::Vector3d &sample :
         samples)
    {
        const double predicted_z =
            initial_model.a * sample.x() +
            initial_model.b * sample.y() +
            initial_model.c;

        absolute_residuals.push_back(
            std::abs(
                sample.z() -
                predicted_z));
    }

    const double median_absolute_residual =
        Median(
            absolute_residuals);

    if (!std::isfinite(
            median_absolute_residual))
    {
        return false;
    }

    const double robust_sigma =
        1.4826 *
        median_absolute_residual;

    const double safe_base =
        std::max(
            1.0e-4,
            base_residual_m);

    const double safe_maximum =
        std::max(
            safe_base,
            maximum_residual_m);

    const double residual_threshold =
        std::clamp(
            std::max(
                safe_base,
                mad_scale *
                    robust_sigma),
            safe_base,
            safe_maximum);

    inlier_samples.reserve(
        samples.size());

    for (const Eigen::Vector3d &sample :
         samples)
    {
        const double predicted_z =
            initial_model.a * sample.x() +
            initial_model.b * sample.y() +
            initial_model.c;

        const double residual =
            std::abs(
                sample.z() -
                predicted_z);

        if (std::isfinite(residual) &&
            residual <= residual_threshold)
        {
            inlier_samples.push_back(
                sample);
        }
    }

    inlier_ratio =
        static_cast<double>(
            inlier_samples.size()) /
        static_cast<double>(
            samples.size());

    if (inlier_samples.size() <
            std::max<std::size_t>(
                3,
                minimum_cells) ||
        !std::isfinite(inlier_ratio) ||
        inlier_ratio <
            minimum_inlier_ratio)
    {
        return false;
    }

    return
        FitHeightFieldSamples(
            inlier_samples,
            final_model);
}


double HeightFieldPointResidual(
    const HeightFieldModel &model,
    const Eigen::Vector3d &sample)
{
    if (!model.valid ||
        !sample.allFinite())
    {
        return
            std::numeric_limits<double>::infinity();
    }

    const double numerator =
        std::abs(
            -model.a * sample.x() -
            model.b * sample.y() +
            sample.z() -
            model.c);

    const double denominator =
        std::sqrt(
            model.a * model.a +
            model.b * model.b +
            1.0);

    if (!std::isfinite(numerator) ||
        !std::isfinite(denominator) ||
        denominator < 1.0e-9)
    {
        return
            std::numeric_limits<double>::infinity();
    }

    return
        numerator /
        denominator;
}


bool HeightFieldToPlane(
    const HeightFieldModel &model,
    Eigen::Vector3d &normal_L,
    double &plane_d,
    double &distance_m,
    double &tilt_deg)
{
    if (!model.valid)
    {
        return false;
    }

    normal_L =
        Eigen::Vector3d(
            -model.a,
            -model.b,
            1.0);

    const double normal_norm =
        normal_L.norm();

    if (!normal_L.allFinite() ||
        !std::isfinite(normal_norm) ||
        normal_norm < 1.0e-9)
    {
        return false;
    }

    normal_L /=
        normal_norm;

    plane_d =
        -model.c /
        normal_norm;

    distance_m =
        std::abs(
            plane_d);

    tilt_deg =
        RadiansToDegrees(
            std::acos(
                std::clamp(
                    normal_L.z(),
                    -1.0,
                    1.0)));

    return
        normal_L.allFinite() &&
        std::isfinite(plane_d) &&
        std::isfinite(distance_m) &&
        std::isfinite(tilt_deg);
}


double NormalAngleDeg(
    const Eigen::Vector3d &normal_a,
    const Eigen::Vector3d &normal_b)
{
    if (!normal_a.allFinite() ||
        !normal_b.allFinite() ||
        normal_a.norm() < 1.0e-9 ||
        normal_b.norm() < 1.0e-9)
    {
        return
            std::numeric_limits<double>::infinity();
    }

    const double cosine =
        std::clamp(
            normal_a.normalized().dot(
                normal_b.normalized()),
            -1.0,
            1.0);

    return
        RadiansToDegrees(
            std::acos(
                cosine));
}


bool IsInsideSupportCorridor(
    const Eigen::Vector2d &center,
    const fr_slam::GroundSegmentationConfig &config)
{
    return
        center.x() >=
            -std::abs(
                config.support_corridor_rear_m) &&
        center.x() <=
            std::abs(
                config.support_corridor_forward_m) &&
        std::abs(
            center.y()) <=
            std::abs(
                config.support_corridor_half_width_m);
}


bool IsInsideSupportPrimaryStrip(
    const Eigen::Vector2d &center,
    const fr_slam::GroundSegmentationConfig &config)
{
    return
        center.x() >=
            -std::abs(
                config.support_corridor_rear_m) &&
        center.x() <=
            std::abs(
                config.support_corridor_forward_m) &&
        std::abs(
            center.y()) <=
            std::abs(
                config.support_primary_half_width_m);
}


bool BuildSupportComponentLocalModel(
    const std::map<
        std::pair<int, int>,
        GroundGridCell> &grid_map,
    const std::pair<int, int> &center_key,
    const fr_slam::GroundSegmentationConfig &config,
    HeightFieldModel &model)
{
    model =
        HeightFieldModel();

    const int radius =
        std::max(
            1,
            config.support_component_local_model_radius_cells);

    std::vector<Eigen::Vector3d> samples;

    for (int dx = -radius;
         dx <= radius;
         ++dx)
    {
        for (int dy = -radius;
             dy <= radius;
             ++dy)
        {
            const std::pair<int, int> key =
                std::make_pair(
                    center_key.first + dx,
                    center_key.second + dy);

            const auto iterator =
                grid_map.find(
                    key);

            if (iterator ==
                grid_map.end())
            {
                continue;
            }

            const GroundGridCell &cell =
                iterator->second;

            if (!cell.surface_valid ||
                !cell.is_ground)
            {
                continue;
            }

            const Eigen::Vector2d center =
                GridCellCenter(
                    key,
                    config.grid_size_m);

            const double range =
                center.norm();

            if (!std::isfinite(range) ||
                range <
                    config.local_plane_minimum_range_m ||
                range >
                    config.support_maximum_range_m)
            {
                continue;
            }

            samples.emplace_back(
                center.x(),
                center.y(),
                cell.surface_z);
        }
    }

    if (samples.size() <
        std::max<std::size_t>(
            3,
            config.minimum_support_component_local_cells))
    {
        return false;
    }

    return
        FitHeightFieldSamples(
            samples,
            model);
}


bool SupportCellsGeometricallyCompatible(
    const std::pair<int, int> &key_a,
    const GroundGridCell &cell_a,
    const std::pair<int, int> &key_b,
    const GroundGridCell &cell_b,
    const std::map<
        std::pair<int, int>,
        HeightFieldModel> &local_models,
    const fr_slam::GroundSegmentationConfig &config,
    double normal_threshold_scale,
    double residual_threshold_scale,
    double fallback_slope_threshold_scale,
    double height_jump_threshold_scale)
{
    if (!cell_a.surface_valid ||
        !cell_b.surface_valid ||
        !cell_a.is_ground ||
        !cell_b.is_ground)
    {
        return false;
    }

    const auto model_a_iterator =
        local_models.find(
            key_a);

    const auto model_b_iterator =
        local_models.find(
            key_b);

    const bool model_a_valid =
        model_a_iterator != local_models.end() &&
        model_a_iterator->second.valid;

    const bool model_b_valid =
        model_b_iterator != local_models.end() &&
        model_b_iterator->second.valid;

    if (model_a_valid &&
        model_b_valid)
    {
        Eigen::Vector3d normal_a =
            Eigen::Vector3d::UnitZ();

        Eigen::Vector3d normal_b =
            Eigen::Vector3d::UnitZ();

        double plane_d = 0.0;
        double distance_m = 0.0;
        double tilt_deg = 0.0;

        if (!HeightFieldToPlane(
                model_a_iterator->second,
                normal_a,
                plane_d,
                distance_m,
                tilt_deg) ||
            !HeightFieldToPlane(
                model_b_iterator->second,
                normal_b,
                plane_d,
                distance_m,
                tilt_deg))
        {
            return false;
        }

        const double normal_change_deg =
            NormalAngleDeg(
                normal_a,
                normal_b);

        if (!std::isfinite(normal_change_deg) ||
            normal_change_deg >
                config.support_component_maximum_normal_change_deg *
                    std::clamp(
                        normal_threshold_scale,
                        0.1,
                        1.0))
        {
            return false;
        }

        const Eigen::Vector2d center_a =
            GridCellCenter(
                key_a,
                config.grid_size_m);

        const Eigen::Vector2d center_b =
            GridCellCenter(
                key_b,
                config.grid_size_m);

        const Eigen::Vector3d sample_a(
            center_a.x(),
            center_a.y(),
            cell_a.surface_z);

        const Eigen::Vector3d sample_b(
            center_b.x(),
            center_b.y(),
            cell_b.surface_z);

        const double residual_ab =
            HeightFieldPointResidual(
                model_a_iterator->second,
                sample_b);

        const double residual_ba =
            HeightFieldPointResidual(
                model_b_iterator->second,
                sample_a);

        return
            std::isfinite(residual_ab) &&
            std::isfinite(residual_ba) &&
            residual_ab <=
                config.support_component_maximum_model_residual_m *
                    std::clamp(
                        residual_threshold_scale,
                        0.1,
                        1.0) &&
            residual_ba <=
                config.support_component_maximum_model_residual_m *
                    std::clamp(
                        residual_threshold_scale,
                        0.1,
                        1.0);
    }

    const Eigen::Vector2d center_a =
        GridCellCenter(
            key_a,
            config.grid_size_m);

    const Eigen::Vector2d center_b =
        GridCellCenter(
            key_b,
            config.grid_size_m);

    const double horizontal_distance =
        (center_b - center_a).norm();

    const double delta_z =
        std::abs(
            cell_b.surface_z -
            cell_a.surface_z);

    if (!std::isfinite(horizontal_distance) ||
        horizontal_distance < 1.0e-9 ||
        !std::isfinite(delta_z) ||
        delta_z >
            config.support_component_fallback_maximum_height_jump_m *
                std::clamp(
                    height_jump_threshold_scale,
                    0.1,
                    1.0))
    {
        return false;
    }

    const double slope_deg =
        RadiansToDegrees(
            std::atan2(
                delta_z,
                horizontal_distance));

    return
        std::isfinite(slope_deg) &&
        slope_deg <=
            config.support_component_fallback_maximum_slope_deg *
                std::clamp(
                    fallback_slope_threshold_scale,
                    0.1,
                    1.0);
}


struct SupportCandidate
{
    std::vector<std::pair<int, int>> keys;

    HeightFieldModel model;

    Eigen::Vector3d normal_L =
        Eigen::Vector3d::UnitZ();

    double plane_d = 0.0;
    double distance_m = 0.0;
    double tilt_deg = 0.0;

    std::size_t center_strip_cells = 0;

    std::size_t fit_inliers = 0;
    double fit_inlier_ratio = 0.0;
    double fit_rmse_m = 0.0;

    double mean_center_proximity = 0.0;
    double score =
        -std::numeric_limits<double>::infinity();
};


std::vector<std::vector<std::pair<int, int>>>
BuildSupportCorridorComponents(
    const std::map<
        std::pair<int, int>,
        GroundGridCell> &grid_map,
    const fr_slam::GroundSegmentationConfig &config,
    std::size_t &corridor_cell_count,
    std::size_t &gap_link_count)
{
    corridor_cell_count = 0;
    gap_link_count = 0;

    std::set<std::pair<int, int>> eligible_keys;

    std::map<
        std::pair<int, int>,
        HeightFieldModel>
        local_models;

    for (const auto &entry :
         grid_map)
    {
        const GroundGridCell &cell =
            entry.second;

        if (!cell.surface_valid ||
            !cell.is_ground)
        {
            continue;
        }

        const Eigen::Vector2d center =
            GridCellCenter(
                entry.first,
                config.grid_size_m);

        const double range =
            center.norm();

        if (!std::isfinite(range) ||
            range <
                config.local_plane_minimum_range_m ||
            range >
                config.support_maximum_range_m ||
            !IsInsideSupportCorridor(
                center,
                config))
        {
            continue;
        }

        eligible_keys.insert(
            entry.first);

        HeightFieldModel local_model;

        if (BuildSupportComponentLocalModel(
                grid_map,
                entry.first,
                config,
                local_model))
        {
            local_models[
                entry.first] =
                local_model;
        }
    }

    corridor_cell_count =
        eligible_keys.size();

    std::vector<std::vector<std::pair<int, int>>> components;

    std::set<std::pair<int, int>> visited;

    const int gap_radius =
        std::clamp(
            config.support_component_gap_radius_cells,
            1,
            2);

    for (const std::pair<int, int> &start_key :
         eligible_keys)
    {
        if (visited.find(
                start_key) !=
            visited.end())
        {
            continue;
        }

        std::vector<std::pair<int, int>> component;

        std::queue<std::pair<int, int>> queue;

        queue.push(
            start_key);

        visited.insert(
            start_key);

        while (!queue.empty())
        {
            const std::pair<int, int> current_key =
                queue.front();

            queue.pop();

            component.push_back(
                current_key);

            const auto current_iterator =
                grid_map.find(
                    current_key);

            if (current_iterator ==
                grid_map.end())
            {
                continue;
            }

            // ----------------------------------------------------
            // Pass 1: ordinary 8-neighbor connectivity.
            // ----------------------------------------------------
            for (int dx = -1;
                 dx <= 1;
                 ++dx)
            {
                for (int dy = -1;
                     dy <= 1;
                     ++dy)
                {
                    if (dx == 0 &&
                        dy == 0)
                    {
                        continue;
                    }

                    const std::pair<int, int> neighbor_key =
                        std::make_pair(
                            current_key.first + dx,
                            current_key.second + dy);

                    if (eligible_keys.find(
                            neighbor_key) ==
                            eligible_keys.end() ||
                        visited.find(
                            neighbor_key) !=
                            visited.end())
                    {
                        continue;
                    }

                    const auto neighbor_iterator =
                        grid_map.find(
                            neighbor_key);

                    if (neighbor_iterator ==
                        grid_map.end())
                    {
                        continue;
                    }

                    if (!SupportCellsGeometricallyCompatible(
                            current_key,
                            current_iterator->second,
                            neighbor_key,
                            neighbor_iterator->second,
                            local_models,
                            config,
                            1.0,
                            1.0,
                            1.0,
                            1.0))
                    {
                        continue;
                    }

                    visited.insert(
                        neighbor_key);

                    queue.push(
                        neighbor_key);
                }
            }

            if (gap_radius < 2)
            {
                continue;
            }

            // ----------------------------------------------------
            // Pass 2: bridge exactly ONE missing grid cell.
            //
            // Only cardinal and true diagonal two-cell offsets are
            // considered. The midpoint must be absent from eligible_keys;
            // therefore this can reconnect a sampling hole, but it cannot
            // jump across an existing incompatible shoulder/curb cell.
            // Geometry thresholds are stricter than direct connectivity.
            // ----------------------------------------------------
            for (int dx = -2;
                 dx <= 2;
                 dx += 2)
            {
                for (int dy = -2;
                     dy <= 2;
                     dy += 2)
                {
                    if (dx == 0 &&
                        dy == 0)
                    {
                        continue;
                    }

                    const int abs_dx =
                        std::abs(dx);

                    const int abs_dy =
                        std::abs(dy);

                    if (std::max(abs_dx, abs_dy) != 2)
                    {
                        continue;
                    }

                    const std::pair<int, int> midpoint_key =
                        std::make_pair(
                            current_key.first + dx / 2,
                            current_key.second + dy / 2);

                    // If the midpoint is already a valid corridor Ground cell,
                    // do not leap over it. A present-but-incompatible midpoint
                    // is useful evidence of a genuine surface boundary.
                    if (eligible_keys.find(
                            midpoint_key) !=
                        eligible_keys.end())
                    {
                        continue;
                    }

                    const std::pair<int, int> gap_neighbor_key =
                        std::make_pair(
                            current_key.first + dx,
                            current_key.second + dy);

                    if (eligible_keys.find(
                            gap_neighbor_key) ==
                            eligible_keys.end() ||
                        visited.find(
                            gap_neighbor_key) !=
                            visited.end())
                    {
                        continue;
                    }

                    const auto gap_neighbor_iterator =
                        grid_map.find(
                            gap_neighbor_key);

                    if (gap_neighbor_iterator ==
                        grid_map.end())
                    {
                        continue;
                    }

                    if (!SupportCellsGeometricallyCompatible(
                            current_key,
                            current_iterator->second,
                            gap_neighbor_key,
                            gap_neighbor_iterator->second,
                            local_models,
                            config,
                            config.support_component_gap_normal_scale,
                            config.support_component_gap_residual_scale,
                            config.support_component_gap_fallback_slope_scale,
                            config.support_component_gap_height_jump_scale))
                    {
                        continue;
                    }

                    visited.insert(
                        gap_neighbor_key);

                    queue.push(
                        gap_neighbor_key);

                    ++gap_link_count;
                }
            }
        }

        if (!component.empty())
        {
            components.push_back(
                std::move(
                    component));
        }
    }

    return components;
}

double GaussianScore(
    double value,
    double sigma)
{
    const double safe_sigma =
        std::max(
            1.0e-6,
            std::abs(
                sigma));

    const double normalized =
        value /
        safe_sigma;

    return
        std::exp(
            -0.5 *
            normalized *
            normalized);
}


struct ClearanceAnchorStats
{
    bool valid = false;
    double median_m = 0.0;
    double robust_sigma_m = 0.0;
    double tolerance_m = 0.0;
};


ClearanceAnchorStats ComputeClearanceAnchorStats(
    const std::deque<double> &history,
    const fr_slam::GroundSegmentationConfig &config)
{
    ClearanceAnchorStats stats;

    if (history.empty())
    {
        return stats;
    }

    std::vector<double> values;

    values.reserve(
        history.size());

    for (const double value :
         history)
    {
        if (std::isfinite(value))
        {
            values.push_back(
                value);
        }
    }

    if (values.empty())
    {
        return stats;
    }

    stats.median_m =
        Median(
            values);

    if (!std::isfinite(
            stats.median_m))
    {
        return ClearanceAnchorStats();
    }

    std::vector<double> absolute_deviations;

    absolute_deviations.reserve(
        values.size());

    for (const double value :
         values)
    {
        absolute_deviations.push_back(
            std::abs(
                value -
                stats.median_m));
    }

    const double mad =
        Median(
            std::move(
                absolute_deviations));

    if (!std::isfinite(mad))
    {
        return ClearanceAnchorStats();
    }

    stats.robust_sigma_m =
        1.4826 * mad;

    const double base_tolerance_m =
        std::max(
            1.0e-4,
            config.support_clearance_anchor_base_tolerance_m);

    const double maximum_tolerance_m =
        std::max(
            base_tolerance_m,
            config.support_clearance_anchor_maximum_tolerance_m);

    stats.tolerance_m =
        std::clamp(
            std::max(
                base_tolerance_m,
                config.support_clearance_anchor_mad_scale *
                    stats.robust_sigma_m),
            base_tolerance_m,
            maximum_tolerance_m);

    const std::size_t required_bootstrap_samples =
        std::min(
            std::max<std::size_t>(
                1,
                config.support_clearance_bootstrap_samples),
            std::max<std::size_t>(
                1,
                config.support_clearance_history_size));

    stats.valid =
        values.size() >=
            required_bootstrap_samples &&
        std::isfinite(
            stats.robust_sigma_m) &&
        std::isfinite(
            stats.tolerance_m);

    return stats;
}


double LinearQuality(
    double value,
    double minimum_value,
    double target_value)
{
    if (!std::isfinite(value))
    {
        return 0.0;
    }

    if (!(target_value > minimum_value))
    {
        return
            value >= minimum_value ?
                1.0 :
                0.0;
    }

    return
        std::clamp(
            (value - minimum_value) /
                (target_value - minimum_value),
            0.0,
            1.0);
}


bool EvaluateSupportCandidate(
    const std::vector<std::pair<int, int>> &component,
    const std::map<
        std::pair<int, int>,
        GroundGridCell> &grid_map,
    const fr_slam::GroundSegmentationConfig &config,
    bool has_previous_support_plane,
    const Eigen::Vector3d &previous_support_normal_L,
    double previous_support_distance_m,
    SupportCandidate &candidate)
{
    candidate =
        SupportCandidate();

    if (component.size() <
        std::max<std::size_t>(
            3,
            config.minimum_support_candidate_cells))
    {
        return false;
    }

    std::vector<Eigen::Vector3d> samples;
    std::vector<Eigen::Vector3d> center_strip_samples;

    samples.reserve(
        component.size());

    center_strip_samples.reserve(
        component.size());

    double center_proximity_sum = 0.0;

    for (const std::pair<int, int> &key :
         component)
    {
        const auto iterator =
            grid_map.find(
                key);

        if (iterator ==
            grid_map.end())
        {
            continue;
        }

        const GroundGridCell &cell =
            iterator->second;

        if (!cell.surface_valid ||
            !cell.is_ground)
        {
            continue;
        }

        const Eigen::Vector2d center =
            GridCellCenter(
                key,
                config.grid_size_m);

        const Eigen::Vector3d sample(
            center.x(),
            center.y(),
            cell.surface_z);

        samples.push_back(
            sample);

        if (IsInsideSupportPrimaryStrip(
                center,
                config))
        {
            ++candidate.center_strip_cells;

            center_strip_samples.push_back(
                sample);
        }

        center_proximity_sum +=
            GaussianScore(
                std::abs(
                    center.y()),
                config.support_center_gaussian_sigma_m);
    }

    if (candidate.center_strip_cells <
        config.minimum_support_center_strip_cells ||
        samples.size() <
            std::max<std::size_t>(
                3,
                config.minimum_support_candidate_cells))
    {
        return false;
    }

    // V3.3 keeps the V3.2 rule: fit the hypothesis from the PRIMARY CENTER STRIP
    // whenever enough center samples exist. Therefore a large road shoulder
    // inside the wider corridor cannot dominate the initial plane simply by
    // contributing more cells.
    const std::vector<Eigen::Vector3d> *fit_samples =
        &samples;

    std::size_t minimum_fit_cells =
        std::max<std::size_t>(
            3,
            config.minimum_support_candidate_cells);

    if (center_strip_samples.size() >=
        std::max<std::size_t>(
            3,
            config.minimum_support_center_strip_cells))
    {
        fit_samples =
            &center_strip_samples;

        minimum_fit_cells =
            std::max<std::size_t>(
                3,
                config.minimum_support_center_strip_cells);
    }

    std::vector<Eigen::Vector3d> inliers;
    double inlier_ratio = 0.0;

    if (!FitRobustHeightFieldModel(
            *fit_samples,
            minimum_fit_cells,
            config.support_plane_base_residual_m,
            config.support_plane_mad_scale,
            config.support_plane_maximum_residual_m,
            config.minimum_support_plane_inlier_ratio,
            candidate.model,
            inliers,
            inlier_ratio))
    {
        return false;
    }

    if (!HeightFieldToPlane(
            candidate.model,
            candidate.normal_L,
            candidate.plane_d,
            candidate.distance_m,
            candidate.tilt_deg))
    {
        return false;
    }

    candidate.keys =
        component;

    candidate.fit_inliers =
        inliers.size();

    candidate.fit_inlier_ratio =
        inlier_ratio;

    candidate.fit_rmse_m =
        candidate.model.rmse_m;

    candidate.mean_center_proximity =
        center_proximity_sum /
        static_cast<double>(
            samples.size());

    const double center_strip_score =
        std::clamp(
            static_cast<double>(
                candidate.center_strip_cells) /
                static_cast<double>(
                    std::max<std::size_t>(
                        1,
                        config.support_recovery_minimum_center_strip_cells)),
            0.0,
            1.0);

    const double area_score =
        std::clamp(
            static_cast<double>(
                candidate.keys.size()) /
                static_cast<double>(
                    std::max<std::size_t>(
                        1,
                        config.support_score_area_saturation_cells)),
            0.0,
            1.0);

    const double fit_score =
        std::clamp(
            candidate.fit_inlier_ratio,
            0.0,
            1.0);

    const double rmse_score =
        std::exp(
            -candidate.fit_rmse_m /
            std::max(
                1.0e-4,
                config.support_plane_base_residual_m));

    candidate.score =
        config.support_score_center_strip_weight *
            center_strip_score +
        config.support_score_center_proximity_weight *
            std::clamp(
                candidate.mean_center_proximity,
                0.0,
                1.0) +
        config.support_score_area_weight *
            area_score +
        config.support_score_fit_weight *
            fit_score +
        config.support_score_rmse_weight *
            rmse_score;

    if (has_previous_support_plane)
    {
        const double normal_change_deg =
            NormalAngleDeg(
                previous_support_normal_L,
                candidate.normal_L);

        const double distance_change_m =
            std::abs(
                candidate.distance_m -
                previous_support_distance_m);

        candidate.score +=
            config.support_score_temporal_normal_weight *
                GaussianScore(
                    normal_change_deg,
                    config.support_score_temporal_normal_sigma_deg) +
            config.support_score_temporal_distance_weight *
                GaussianScore(
                    distance_change_m,
                    config.support_score_temporal_distance_sigma_m);
    }

    return
        std::isfinite(
            candidate.score);
}


std::size_t CountSupportCenterStripCells(
    const std::vector<std::pair<int, int>> &component,
    const std::map<
        std::pair<int, int>,
        GroundGridCell> &grid_map,
    const fr_slam::GroundSegmentationConfig &config)
{
    std::size_t count = 0;

    for (const std::pair<int, int> &key :
         component)
    {
        const auto iterator =
            grid_map.find(
                key);

        if (iterator ==
                grid_map.end() ||
            !iterator->second.surface_valid ||
            !iterator->second.is_ground)
        {
            continue;
        }

        const Eigen::Vector2d center =
            GridCellCenter(
                key,
                config.grid_size_m);

        if (IsInsideSupportPrimaryStrip(
                center,
                config))
        {
            ++count;
        }
    }

    return count;
}


bool SelectBestSupportCandidate(
    const std::vector<std::vector<std::pair<int, int>>> &components,
    const std::map<
        std::pair<int, int>,
        GroundGridCell> &grid_map,
    const fr_slam::GroundSegmentationConfig &config,
    bool has_previous_support_plane,
    const Eigen::Vector3d &previous_support_normal_L,
    double previous_support_distance_m,
    SupportCandidate &best_candidate,
    std::size_t &candidate_count,
    std::size_t &selected_candidate_index,
    std::size_t &rejected_small_count,
    std::size_t &rejected_no_center_count,
    std::size_t &rejected_fit_count)
{
    candidate_count = 0;
    selected_candidate_index = 0;
    rejected_small_count = 0;
    rejected_no_center_count = 0;
    rejected_fit_count = 0;

    bool found = false;
    double best_score =
        -std::numeric_limits<double>::infinity();

    const std::size_t minimum_candidate_cells =
        std::max<std::size_t>(
            3,
            config.minimum_support_candidate_cells);

    for (const std::vector<std::pair<int, int>> &component :
         components)
    {
        if (component.size() <
            minimum_candidate_cells)
        {
            ++rejected_small_count;
            continue;
        }

        const std::size_t center_strip_cells =
            CountSupportCenterStripCells(
                component,
                grid_map,
                config);

        if (center_strip_cells <
            config.minimum_support_center_strip_cells)
        {
            ++rejected_no_center_count;
            continue;
        }

        SupportCandidate candidate;

        if (!EvaluateSupportCandidate(
                component,
                grid_map,
                config,
                has_previous_support_plane,
                previous_support_normal_L,
                previous_support_distance_m,
                candidate))
        {
            ++rejected_fit_count;
            continue;
        }

        ++candidate_count;

        if (!found ||
            candidate.score >
                best_score)
        {
            best_candidate =
                candidate;

            best_score =
                candidate.score;

            selected_candidate_index =
                candidate_count;

            found = true;
        }
    }

    return found;
}

bool CellMatchesSupportModel(
    const std::pair<int, int> &key,
    const GroundGridCell &cell,
    const HeightFieldModel &model,
    const fr_slam::GroundSegmentationConfig &config)
{
    if (!cell.surface_valid ||
        !cell.is_ground ||
        !model.valid)
    {
        return false;
    }

    const Eigen::Vector2d center =
        GridCellCenter(
            key,
            config.grid_size_m);

    const double range =
        center.norm();

    if (!std::isfinite(range) ||
        range <
            config.local_plane_minimum_range_m ||
        range >
            config.support_maximum_range_m)
    {
        return false;
    }

    const Eigen::Vector3d sample(
        center.x(),
        center.y(),
        cell.surface_z);

    const double residual =
        HeightFieldPointResidual(
            model,
            sample);

    const double adaptive_tolerance =
        std::clamp(
            config.support_cell_base_residual_m +
                config.support_cell_roughness_scale *
                    std::max(
                        0.0,
                        cell.low_roughness_m),
            config.support_cell_base_residual_m,
            config.support_cell_maximum_residual_m);

    return
        std::isfinite(residual) &&
        residual <=
            adaptive_tolerance;
}


std::set<std::pair<int, int>>
GrowConnectedSupportRegion(
    const std::vector<std::pair<int, int>> &seed_keys,
    const std::map<
        std::pair<int, int>,
        GroundGridCell> &grid_map,
    const HeightFieldModel &model,
    const fr_slam::GroundSegmentationConfig &config)
{
    std::set<std::pair<int, int>> support_keys;

    std::queue<std::pair<int, int>> queue;

    for (const std::pair<int, int> &key :
         seed_keys)
    {
        const auto iterator =
            grid_map.find(
                key);

        if (iterator ==
                grid_map.end() ||
            !CellMatchesSupportModel(
                key,
                iterator->second,
                model,
                config))
        {
            continue;
        }

        if (support_keys.insert(
                key).second)
        {
            queue.push(
                key);
        }
    }

    while (!queue.empty())
    {
        const std::pair<int, int> current_key =
            queue.front();

        queue.pop();

        for (int dx = -1;
             dx <= 1;
             ++dx)
        {
            for (int dy = -1;
                 dy <= 1;
                 ++dy)
            {
                if (dx == 0 &&
                    dy == 0)
                {
                    continue;
                }

                const std::pair<int, int> neighbor_key =
                    std::make_pair(
                        current_key.first + dx,
                        current_key.second + dy);

                if (support_keys.find(
                        neighbor_key) !=
                    support_keys.end())
                {
                    continue;
                }

                const auto neighbor_iterator =
                    grid_map.find(
                        neighbor_key);

                if (neighbor_iterator ==
                    grid_map.end())
                {
                    continue;
                }

                if (!CellMatchesSupportModel(
                        neighbor_key,
                        neighbor_iterator->second,
                        model,
                        config))
                {
                    continue;
                }

                support_keys.insert(
                    neighbor_key);

                queue.push(
                    neighbor_key);
            }
        }
    }

    return support_keys;
}


void ClearSupportMarks(
    std::map<
        std::pair<int, int>,
        GroundGridCell> &grid_map)
{
    for (auto &entry :
         grid_map)
    {
        entry.second.is_support = false;
    }
}


bool BuildVehicleSupportSurface(
    std::map<
        std::pair<int, int>,
        GroundGridCell> &grid_map,
    const fr_slam::GroundSegmentationConfig &config,
    bool has_previous_support_plane,
    const Eigen::Vector3d &previous_support_normal_L,
    double previous_support_distance_m,
    bool &has_pending_support_plane,
    Eigen::Vector3d &pending_support_normal_L,
    double &pending_support_distance_m,
    std::size_t &pending_support_consecutive_frames,
    fr_slam::GroundSegmentationResult &result)
{
    result.support_plane_valid = false;
    result.support_temporal_gate_passed = false;
    result.support_recovery_pending = false;
    result.support_recovery_pending_count = 0;
    result.support_reinitialized = false;

    result.support_component_count_total = 0;
    result.support_component_gap_links = 0;
    result.support_component_rejected_small = 0;
    result.support_component_rejected_no_center = 0;
    result.support_component_rejected_fit = 0;

    result.support_normal_change_deg =
        std::numeric_limits<double>::quiet_NaN();

    result.support_distance_change_m =
        std::numeric_limits<double>::quiet_NaN();

    ClearSupportMarks(
        grid_map);

    std::size_t corridor_cell_count = 0;
    std::size_t gap_link_count = 0;

    const std::vector<std::vector<std::pair<int, int>>> components =
        BuildSupportCorridorComponents(
            grid_map,
            config,
            corridor_cell_count,
            gap_link_count);

    result.support_corridor_cells =
        corridor_cell_count;

    result.support_component_count_total =
        components.size();

    result.support_component_gap_links =
        gap_link_count;

    if (corridor_cell_count <
        config.minimum_support_corridor_cells)
    {
        has_pending_support_plane = false;
        pending_support_consecutive_frames = 0;
        return false;
    }

    SupportCandidate best_candidate;

    if (!SelectBestSupportCandidate(
            components,
            grid_map,
            config,
            has_previous_support_plane,
            previous_support_normal_L,
            previous_support_distance_m,
            best_candidate,
            result.support_candidate_count,
            result.support_selected_candidate_index,
            result.support_component_rejected_small,
            result.support_component_rejected_no_center,
            result.support_component_rejected_fit))
    {
        has_pending_support_plane = false;
        pending_support_consecutive_frames = 0;
        return false;
    }

    result.support_selected_component_cells =
        best_candidate.keys.size();

    result.support_selected_center_strip_cells =
        best_candidate.center_strip_cells;

    result.support_selected_score =
        best_candidate.score;

    // First grow only the connected region consistent with the selected
    // hypothesis. This prevents a geometrically similar but disconnected
    // surface from being merged into the vehicle support cloud.
    std::set<std::pair<int, int>> support_keys =
        GrowConnectedSupportRegion(
            best_candidate.keys,
            grid_map,
            best_candidate.model,
            config);

    if (support_keys.size() <
        config.minimum_support_plane_cells)
    {
        has_pending_support_plane = false;
        pending_support_consecutive_frames = 0;
        return false;
    }

    std::vector<Eigen::Vector3d> support_samples;

    support_samples.reserve(
        support_keys.size());

    for (const std::pair<int, int> &key :
         support_keys)
    {
        const auto iterator =
            grid_map.find(
                key);

        if (iterator ==
            grid_map.end())
        {
            continue;
        }

        const Eigen::Vector2d center =
            GridCellCenter(
                key,
                config.grid_size_m);

        support_samples.emplace_back(
            center.x(),
            center.y(),
            iterator->second.surface_z);
    }

    HeightFieldModel final_model;
    std::vector<Eigen::Vector3d> final_inliers;
    double final_inlier_ratio = 0.0;

    if (!FitRobustHeightFieldModel(
            support_samples,
            config.minimum_support_plane_cells,
            config.support_plane_base_residual_m,
            config.support_plane_mad_scale,
            config.support_plane_maximum_residual_m,
            config.minimum_support_plane_inlier_ratio,
            final_model,
            final_inliers,
            final_inlier_ratio))
    {
        has_pending_support_plane = false;
        pending_support_consecutive_frames = 0;
        return false;
    }

    Eigen::Vector3d final_normal_L =
        Eigen::Vector3d::UnitZ();

    double final_plane_d = 0.0;
    double final_distance_m = 0.0;
    double final_tilt_deg = 0.0;

    if (!HeightFieldToPlane(
            final_model,
            final_normal_L,
            final_plane_d,
            final_distance_m,
            final_tilt_deg))
    {
        has_pending_support_plane = false;
        pending_support_consecutive_frames = 0;
        return false;
    }

    bool direct_temporal_pass =
        true;

    if (has_previous_support_plane)
    {
        result.support_normal_change_deg =
            NormalAngleDeg(
                previous_support_normal_L,
                final_normal_L);

        result.support_distance_change_m =
            std::abs(
                final_distance_m -
                previous_support_distance_m);

        direct_temporal_pass =
            std::isfinite(
                result.support_normal_change_deg) &&
            std::isfinite(
                result.support_distance_change_m) &&
            result.support_normal_change_deg <=
                config.maximum_support_normal_change_deg &&
            result.support_distance_change_m <=
                config.maximum_support_distance_change_m;
    }

    bool accepted_by_recovery =
        false;

    if (!direct_temporal_pass)
    {
        const bool recovery_geometry_good =
            best_candidate.center_strip_cells >=
                config.support_recovery_minimum_center_strip_cells &&
            final_inlier_ratio >=
                config.support_recovery_minimum_inlier_ratio &&
            final_model.rmse_m <=
                config.support_recovery_maximum_rmse_m;

        if (!recovery_geometry_good)
        {
            has_pending_support_plane = false;
            pending_support_consecutive_frames = 0;

            ClearSupportMarks(
                grid_map);

            return false;
        }

        bool same_pending_candidate =
            false;

        if (has_pending_support_plane)
        {
            const double pending_normal_change_deg =
                NormalAngleDeg(
                    pending_support_normal_L,
                    final_normal_L);

            const double pending_distance_change_m =
                std::abs(
                    pending_support_distance_m -
                    final_distance_m);

            same_pending_candidate =
                std::isfinite(
                    pending_normal_change_deg) &&
                std::isfinite(
                    pending_distance_change_m) &&
                pending_normal_change_deg <=
                    config.support_recovery_maximum_candidate_normal_change_deg &&
                pending_distance_change_m <=
                    config.support_recovery_maximum_candidate_distance_change_m;
        }

        if (same_pending_candidate)
        {
            ++pending_support_consecutive_frames;

            // Slowly track the stable pending surface rather than freezing the
            // very first rejected frame.
            pending_support_normal_L =
                (pending_support_normal_L +
                 final_normal_L)
                    .normalized();

            pending_support_distance_m =
                0.5 *
                (pending_support_distance_m +
                 final_distance_m);
        }
        else
        {
            has_pending_support_plane =
                true;

            pending_support_normal_L =
                final_normal_L;

            pending_support_distance_m =
                final_distance_m;

            pending_support_consecutive_frames =
                1;
        }

        result.support_recovery_pending =
            true;

        result.support_recovery_pending_count =
            pending_support_consecutive_frames;

        if (pending_support_consecutive_frames <
            std::max<std::size_t>(
                1,
                config.support_recovery_required_consecutive_frames))
        {
            ClearSupportMarks(
                grid_map);

            return false;
        }

        accepted_by_recovery =
            true;

        result.support_reinitialized =
            true;

        has_pending_support_plane =
            false;

        pending_support_consecutive_frames =
            0;
    }
    else
    {
        has_pending_support_plane =
            false;

        pending_support_consecutive_frames =
            0;
    }

    // Recompute the connected support set once with the final model.
    support_keys =
        GrowConnectedSupportRegion(
            best_candidate.keys,
            grid_map,
            final_model,
            config);

    ClearSupportMarks(
        grid_map);

    result.support_ground_cells = 0;

    for (const std::pair<int, int> &key :
         support_keys)
    {
        auto iterator =
            grid_map.find(
                key);

        if (iterator ==
            grid_map.end())
        {
            continue;
        }

        if (!CellMatchesSupportModel(
                key,
                iterator->second,
                final_model,
                config))
        {
            continue;
        }

        iterator->second.is_support =
            true;

        ++result.support_ground_cells;
    }

    if (result.support_ground_cells <
        config.minimum_support_plane_cells)
    {
        ClearSupportMarks(
            grid_map);

        result.support_ground_cells = 0;

        return false;
    }

    result.support_plane_cells =
        support_samples.size();

    result.support_plane_inliers =
        final_inliers.size();

    result.support_plane_inlier_ratio =
        final_inlier_ratio;

    result.support_plane_rmse_m =
        final_model.rmse_m;

    result.support_ground_normal_L =
        final_normal_L;

    result.support_ground_plane_d =
        final_plane_d;

    result.support_ground_distance_m =
        final_distance_m;

    result.support_ground_tilt_deg =
        final_tilt_deg;

    result.support_temporal_gate_passed =
        direct_temporal_pass ||
        accepted_by_recovery;

    result.support_plane_valid =
        result.support_ground_normal_L.allFinite() &&
        std::isfinite(
            result.support_ground_plane_d) &&
        std::isfinite(
            result.support_ground_distance_m) &&
        std::isfinite(
            result.support_ground_tilt_deg) &&
        std::isfinite(
            result.support_plane_rmse_m);

    return
        result.support_plane_valid;
}


bool FitRobustLocalGroundPlane(
    const std::map<
        std::pair<int, int>,
        GroundGridCell> &grid_map,
    const fr_slam::GroundSegmentationConfig &config,
    fr_slam::GroundSegmentationResult &result)
{
    result.local_plane_valid =
        false;

    std::vector<Eigen::Vector3d> samples;

    for (const auto &entry :
         grid_map)
    {
        const std::pair<int, int> &key =
            entry.first;

        const GroundGridCell &cell =
            entry.second;

        if (!cell.surface_valid ||
            !cell.is_ground)
        {
            continue;
        }

        const Eigen::Vector2d center =
            GridCellCenter(
                key,
                config.grid_size_m);

        const double range =
            center.norm();

        if (!std::isfinite(range) ||
            range <
                config.local_plane_minimum_range_m ||
            range >
                config.local_plane_maximum_range_m)
        {
            continue;
        }

        samples.emplace_back(
            center.x(),
            center.y(),
            cell.surface_z);
    }

    result.local_plane_cells =
        samples.size();

    if (samples.size() <
        config.minimum_local_plane_cells)
    {
        return false;
    }

    HeightFieldModel initial_model;

    if (!FitHeightFieldSamples(
            samples,
            initial_model))
    {
        return false;
    }

    std::vector<double> absolute_residuals;

    absolute_residuals.reserve(
        samples.size());

    for (const Eigen::Vector3d &sample :
         samples)
    {
        const double predicted_z =
            initial_model.a * sample.x() +
            initial_model.b * sample.y() +
            initial_model.c;

        absolute_residuals.push_back(
            std::abs(
                sample.z() -
                predicted_z));
    }

    const double median_absolute_residual =
        Median(
            absolute_residuals);

    if (!std::isfinite(
            median_absolute_residual))
    {
        return false;
    }

    const double robust_sigma =
        1.4826 *
        median_absolute_residual;

    const double residual_threshold =
        std::clamp(
            std::max(
                config.local_plane_base_residual_m,
                config.local_plane_mad_scale *
                    robust_sigma),
            config.local_plane_base_residual_m,
            config.local_plane_maximum_residual_m);

    std::vector<Eigen::Vector3d> inlier_samples;

    inlier_samples.reserve(
        samples.size());

    for (const Eigen::Vector3d &sample :
         samples)
    {
        const double predicted_z =
            initial_model.a * sample.x() +
            initial_model.b * sample.y() +
            initial_model.c;

        const double residual =
            std::abs(
                sample.z() -
                predicted_z);

        if (std::isfinite(residual) &&
            residual <=
                residual_threshold)
        {
            inlier_samples.push_back(
                sample);
        }
    }

    result.local_plane_inliers =
        inlier_samples.size();

    result.local_plane_inlier_ratio =
        static_cast<double>(
            inlier_samples.size()) /
        static_cast<double>(
            samples.size());

    if (inlier_samples.size() <
            config.minimum_local_plane_cells ||
        !std::isfinite(
            result.local_plane_inlier_ratio) ||
        result.local_plane_inlier_ratio <
            config.minimum_local_plane_inlier_ratio)
    {
        return false;
    }

    HeightFieldModel final_model;

    if (!FitHeightFieldSamples(
            inlier_samples,
            final_model))
    {
        return false;
    }

    // Height field:
    //
    //     z = a*x + b*y + c
    //
    // Plane form:
    //
    //     -a*x - b*y + z - c = 0
    Eigen::Vector3d normal_L(
        -final_model.a,
        -final_model.b,
        1.0);

    const double normal_norm =
        normal_L.norm();

    if (!normal_L.allFinite() ||
        !std::isfinite(normal_norm) ||
        normal_norm < 1.0e-9)
    {
        return false;
    }

    normal_L /=
        normal_norm;

    const double plane_d =
        -final_model.c /
        normal_norm;

    const double normal_z =
        std::clamp(
            normal_L.z(),
            -1.0,
            1.0);

    result.local_ground_normal_L =
        normal_L;

    result.local_ground_plane_d =
        plane_d;

    result.local_ground_distance_m =
        std::abs(
            plane_d);

    result.local_ground_tilt_deg =
        RadiansToDegrees(
            std::acos(
                normal_z));

    result.local_plane_rmse_m =
        final_model.rmse_m;

    result.local_plane_valid =
        result.local_ground_normal_L.allFinite() &&
        std::isfinite(
            result.local_ground_plane_d) &&
        std::isfinite(
            result.local_ground_distance_m) &&
        std::isfinite(
            result.local_ground_tilt_deg) &&
        std::isfinite(
            result.local_plane_rmse_m);

    return
        result.local_plane_valid;
}

}  // namespace


namespace fr_slam
{

GroundSegmenter::GroundSegmenter()
    : config_()
{
}


GroundSegmenter::GroundSegmenter(
    const GroundSegmentationConfig &config)
    : config_(config)
{
}


void GroundSegmenter::SetConfig(
    const GroundSegmentationConfig &config)
{
    config_ =
        config;

    // Configuration changes invalidate all temporal / learned state.
    Reset();
}


const GroundSegmentationConfig &
GroundSegmenter::GetConfig() const
{
    return
        config_;
}


void GroundSegmenter::Reset()
{
    has_previous_support_plane_ = false;

    previous_support_normal_L_ =
        Eigen::Vector3d::UnitZ();

    previous_support_distance_m_ = 0.0;

    has_pending_support_plane_ = false;

    pending_support_normal_L_ =
        Eigen::Vector3d::UnitZ();

    pending_support_distance_m_ = 0.0;

    pending_support_consecutive_frames_ = 0;

    support_clearance_history_m_.clear();

    support_constraint_was_valid_ = false;

    has_trusted_support_plane_ = false;

    trusted_support_normal_L_ =
        Eigen::Vector3d::UnitZ();

    trusted_support_distance_m_ = 0.0;
}


void GroundSegmenter::EvaluateSupportConstraint(
    GroundSegmentationResult &result)
{
    result.support_constraint_valid = false;
    result.support_constraint_confidence = 0.0;
    result.support_constraint_rejection_mask =
        SUPPORT_CONSTRAINT_REJECT_NONE;

    result.support_clearance_sample_accepted = false;

    result.support_trusted_normal_change_deg =
        std::numeric_limits<double>::quiet_NaN();

    result.support_trusted_distance_change_m =
        std::numeric_limits<double>::quiet_NaN();

    result.support_clearance_error_m =
        std::numeric_limits<double>::quiet_NaN();

    auto update_anchor_diagnostics =
        [&]()
        {
            const ClearanceAnchorStats stats =
                ComputeClearanceAnchorStats(
                    support_clearance_history_m_,
                    config_);

            result.support_clearance_anchor_valid =
                stats.valid;

            result.support_clearance_anchor_m =
                stats.median_m;

            result.support_clearance_anchor_sigma_m =
                stats.robust_sigma_m;

            result.support_clearance_anchor_tolerance_m =
                stats.tolerance_m;

            result.support_clearance_history_samples =
                support_clearance_history_m_.size();

            if (result.support_plane_valid &&
                stats.valid &&
                std::isfinite(
                    result.support_ground_distance_m))
            {
                result.support_clearance_error_m =
                    std::abs(
                        result.support_ground_distance_m -
                        stats.median_m);
            }
        };

    update_anchor_diagnostics();

    if (!result.support_plane_valid)
    {
        result.support_constraint_rejection_mask |=
            SUPPORT_CONSTRAINT_REJECT_NO_SUPPORT;

        support_constraint_was_valid_ = false;

        return;
    }

    if (has_trusted_support_plane_)
    {
        result.support_trusted_normal_change_deg =
            NormalAngleDeg(
                trusted_support_normal_L_,
                result.support_ground_normal_L);

        result.support_trusted_distance_change_m =
            std::abs(
                trusted_support_distance_m_ -
                result.support_ground_distance_m);
    }

    // ------------------------------------------------------------
    // 1. Bootstrap the rigid LiDAR-to-support clearance anchor.
    // Only strong geometry can contribute.  After a few samples, a
    // provisional median gate prevents an accidental shoulder sequence from
    // dragging the bootstrap history far away from the dominant clearance.
    // ------------------------------------------------------------
    const bool bootstrap_geometry_good =
        result.support_temporal_gate_passed &&
        result.support_ground_points >=
            config_.support_clearance_bootstrap_minimum_points &&
        result.support_ground_cells >=
            config_.support_clearance_bootstrap_minimum_cells &&
        result.support_selected_center_strip_cells >=
            config_.support_clearance_bootstrap_minimum_center_strip_cells &&
        result.support_plane_inlier_ratio >=
            config_.support_clearance_bootstrap_minimum_inlier_ratio &&
        result.support_plane_rmse_m <=
            config_.support_clearance_bootstrap_maximum_rmse_m &&
        result.support_selected_score >=
            config_.support_clearance_bootstrap_minimum_selected_score &&
        std::isfinite(
            result.support_ground_distance_m);

    if (!result.support_clearance_anchor_valid &&
        bootstrap_geometry_good)
    {
        bool provisional_distance_good =
            true;

        if (support_clearance_history_m_.size() >=
            config_.support_clearance_bootstrap_provisional_gate_samples &&
            !support_clearance_history_m_.empty())
        {
            std::vector<double> bootstrap_values(
                support_clearance_history_m_.begin(),
                support_clearance_history_m_.end());

            const double provisional_median_m =
                Median(
                    std::move(
                        bootstrap_values));

            provisional_distance_good =
                std::isfinite(
                    provisional_median_m) &&
                std::abs(
                    result.support_ground_distance_m -
                    provisional_median_m) <=
                    config_.support_clearance_bootstrap_maximum_deviation_m;
        }

        if (provisional_distance_good)
        {
            support_clearance_history_m_.push_back(
                result.support_ground_distance_m);

            result.support_clearance_sample_accepted =
                true;

            while (support_clearance_history_m_.size() >
                   std::max<std::size_t>(
                       1,
                       config_.support_clearance_history_size))
            {
                support_clearance_history_m_.pop_front();
            }

            update_anchor_diagnostics();
        }
    }

    // ------------------------------------------------------------
    // 2. Hard safety gates.  These are deliberately independent of the
    // confidence score: a high average confidence can never override a hard
    // physical inconsistency.
    // ------------------------------------------------------------
    if (result.support_selected_score <
        config_.support_constraint_minimum_selected_score)
    {
        result.support_constraint_rejection_mask |=
            SUPPORT_CONSTRAINT_REJECT_LOW_SCORE;
    }

    if (result.support_ground_points <
        config_.support_constraint_minimum_points)
    {
        result.support_constraint_rejection_mask |=
            SUPPORT_CONSTRAINT_REJECT_LOW_POINTS;
    }

    if (result.support_ground_cells <
        config_.support_constraint_minimum_cells)
    {
        result.support_constraint_rejection_mask |=
            SUPPORT_CONSTRAINT_REJECT_LOW_CELLS;
    }

    if (result.support_selected_center_strip_cells <
        config_.support_constraint_minimum_center_strip_cells)
    {
        result.support_constraint_rejection_mask |=
            SUPPORT_CONSTRAINT_REJECT_LOW_CENTER;
    }

    if (result.support_plane_inlier_ratio <
        config_.support_constraint_minimum_inlier_ratio)
    {
        result.support_constraint_rejection_mask |=
            SUPPORT_CONSTRAINT_REJECT_LOW_INLIER;
    }

    if (result.support_plane_rmse_m >
        config_.support_constraint_maximum_rmse_m)
    {
        result.support_constraint_rejection_mask |=
            SUPPORT_CONSTRAINT_REJECT_HIGH_RMSE;
    }

    if (!result.support_temporal_gate_passed)
    {
        result.support_constraint_rejection_mask |=
            SUPPORT_CONSTRAINT_REJECT_TEMPORAL;
    }

    if (!result.support_clearance_anchor_valid)
    {
        result.support_constraint_rejection_mask |=
            SUPPORT_CONSTRAINT_REJECT_BOOTSTRAP;
    }
    else if (!std::isfinite(
                 result.support_clearance_error_m) ||
             result.support_clearance_error_m >
                 result.support_clearance_anchor_tolerance_m)
    {
        result.support_constraint_rejection_mask |=
            SUPPORT_CONSTRAINT_REJECT_ANCHOR;
    }

    if (has_trusted_support_plane_)
    {
        const bool trusted_jump =
            !std::isfinite(
                result.support_trusted_normal_change_deg) ||
            !std::isfinite(
                result.support_trusted_distance_change_m) ||
            result.support_trusted_normal_change_deg >
                config_.support_constraint_maximum_trusted_normal_change_deg ||
            result.support_trusted_distance_change_m >
                config_.support_constraint_maximum_trusted_distance_change_m;

        if (trusted_jump)
        {
            result.support_constraint_rejection_mask |=
                SUPPORT_CONSTRAINT_REJECT_TRUSTED_JUMP;
        }
    }

    // ------------------------------------------------------------
    // 3. Continuous confidence [0,1].  This will later be useful for
    // converting the trusted support constraint into a soft ICP information
    // weight rather than an on/off hard correction.
    // ------------------------------------------------------------
    const double score_quality =
        LinearQuality(
            result.support_selected_score,
            config_.support_constraint_minimum_selected_score,
            14.0);

    const double point_quality =
        LinearQuality(
            static_cast<double>(
                result.support_ground_points),
            static_cast<double>(
                config_.support_constraint_minimum_points),
            300.0);

    const double cell_quality =
        LinearQuality(
            static_cast<double>(
                result.support_ground_cells),
            static_cast<double>(
                config_.support_constraint_minimum_cells),
            90.0);

    const double center_quality =
        LinearQuality(
            static_cast<double>(
                result.support_selected_center_strip_cells),
            static_cast<double>(
                config_.support_constraint_minimum_center_strip_cells),
            7.0);

    const double fit_quality =
        LinearQuality(
            result.support_plane_inlier_ratio,
            config_.support_constraint_minimum_inlier_ratio,
            1.0);

    const double rmse_quality =
        GaussianScore(
            result.support_plane_rmse_m,
            std::max(
                0.010,
                0.60 *
                    config_.support_constraint_maximum_rmse_m));

    double anchor_quality = 0.0;

    if (result.support_clearance_anchor_valid &&
        std::isfinite(
            result.support_clearance_error_m))
    {
        anchor_quality =
            GaussianScore(
                result.support_clearance_error_m,
                std::max(
                    0.020,
                    0.50 *
                        result.support_clearance_anchor_tolerance_m));
    }

    double temporal_normal_quality = 1.0;
    double temporal_distance_quality = 1.0;

    if (has_trusted_support_plane_)
    {
        temporal_normal_quality =
            GaussianScore(
                result.support_trusted_normal_change_deg,
                2.5);

        temporal_distance_quality =
            GaussianScore(
                result.support_trusted_distance_change_m,
                0.035);
    }
    else if (has_previous_support_plane_)
    {
        temporal_normal_quality =
            GaussianScore(
                result.support_normal_change_deg,
                2.5);

        temporal_distance_quality =
            GaussianScore(
                result.support_distance_change_m,
                0.035);
    }

    result.support_constraint_confidence =
        std::clamp(
            0.08 * score_quality +
            0.14 * point_quality +
            0.10 * cell_quality +
            0.10 * center_quality +
            0.15 * fit_quality +
            0.10 * rmse_quality +
            0.18 * anchor_quality +
            0.08 * temporal_normal_quality +
            0.07 * temporal_distance_quality,
            0.0,
            1.0);

    const double confidence_threshold =
        support_constraint_was_valid_ ?
            config_.support_constraint_keep_confidence :
            config_.support_constraint_enter_confidence;

    if (result.support_constraint_confidence <
        confidence_threshold)
    {
        result.support_constraint_rejection_mask |=
            SUPPORT_CONSTRAINT_REJECT_LOW_CONFIDENCE;
    }

    result.support_constraint_valid =
        result.support_constraint_rejection_mask ==
            SUPPORT_CONSTRAINT_REJECT_NONE;

    support_constraint_was_valid_ =
        result.support_constraint_valid;

    if (result.support_constraint_valid)
    {
        has_trusted_support_plane_ = true;

        trusted_support_normal_L_ =
            result.support_ground_normal_L;

        trusted_support_distance_m_ =
            result.support_ground_distance_m;

        // Update the learned clearance only from very consistent trusted
        // frames.  This lets suspension / calibration drift move the anchor
        // slowly, while a shoulder cannot drag it.
        if (!result.support_clearance_sample_accepted &&
            std::isfinite(
                result.support_clearance_error_m) &&
            result.support_clearance_error_m <=
                config_.support_clearance_history_update_maximum_error_m)
        {
            support_clearance_history_m_.push_back(
                result.support_ground_distance_m);

            result.support_clearance_sample_accepted =
                true;

            while (support_clearance_history_m_.size() >
                   std::max<std::size_t>(
                       1,
                       config_.support_clearance_history_size))
            {
                support_clearance_history_m_.pop_front();
            }

            update_anchor_diagnostics();
        }
    }
}


GroundSegmentationResult
GroundSegmenter::Segment(
    const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud)
{
    GroundSegmentationResult result;

    result.support_normal_change_deg =
        std::numeric_limits<double>::quiet_NaN();

    result.support_distance_change_m =
        std::numeric_limits<double>::quiet_NaN();

    result.ground_cloud =
        pcl::PointCloud<pcl::PointXYZ>::Ptr(
            new pcl::PointCloud<pcl::PointXYZ>);

    result.nonground_cloud =
        pcl::PointCloud<pcl::PointXYZ>::Ptr(
            new pcl::PointCloud<pcl::PointXYZ>);

    result.support_ground_cloud =
        pcl::PointCloud<pcl::PointXYZ>::Ptr(
            new pcl::PointCloud<pcl::PointXYZ>);

    if (!cloud ||
        cloud->empty())
    {
        return result;
    }

    result.input_points =
        cloud->size();

    if (!(config_.grid_size_m > 0.0) ||
        !(config_.maximum_range_m >
          config_.minimum_range_m) ||
        !(config_.seed_maximum_range_m >
          config_.seed_minimum_range_m) ||
        config_.maximum_cell_low_roughness_m < 0.0 ||
        config_.maximum_surface_prediction_tolerance_m <= 0.0)
    {
        return result;
    }

    // ============================================================
    // 1. Build XY grid.
    // ============================================================
    std::map<
        std::pair<int, int>,
        GroundGridCell>
        grid_map;

    for (const pcl::PointXYZ &point :
         cloud->points)
    {
        if (!IsFinitePoint(point))
        {
            continue;
        }

        const double range =
            HorizontalRange(
                point);

        if (range <
                config_.minimum_range_m ||
            range >
                config_.maximum_range_m)
        {
            continue;
        }

        ++result.valid_points;

        const std::pair<int, int> key =
            ComputeGridKey(
                point,
                config_.grid_size_m);

        grid_map[key]
            .z_values
            .push_back(
                static_cast<double>(
                    point.z));
    }

    if (grid_map.empty())
    {
        return result;
    }

    result.grid_cells =
        grid_map.size();

    // ============================================================
    // 2. Robust low-surface + low-layer roughness for each cell.
    //
    // Vertical structures tend to have a large low-layer spread inside one
    // XY cell.  A real slope may change absolute z, but remains locally thin.
    // ============================================================
    for (auto &entry :
         grid_map)
    {
        GroundGridCell &cell =
            entry.second;

        if (cell.z_values.size() <
            config_.minimum_points_per_cell)
        {
            continue;
        }

        double surface_z =
            std::numeric_limits<double>::quiet_NaN();

        double roughness_m =
            std::numeric_limits<double>::quiet_NaN();

        if (!RobustLowSurfaceStatistics(
                cell.z_values,
                config_.low_surface_fraction,
                surface_z,
                roughness_m))
        {
            continue;
        }

        cell.surface_z =
            surface_z;

        cell.low_roughness_m =
            roughness_m;

        if (roughness_m >
            config_.maximum_cell_low_roughness_m)
        {
            ++result.rejected_rough_surface_cells;
            continue;
        }

        cell.surface_valid =
            true;

        ++result.valid_surface_cells;
    }

    // ============================================================
    // 3. Count near-field seed candidates.
    // ============================================================
    for (const auto &entry :
         grid_map)
    {
        if (IsNearSeedCandidate(
                entry.first,
                entry.second,
                config_))
        {
            ++result.near_seed_candidate_cells;
        }
    }

    if (result.near_seed_candidate_cells == 0)
    {
        return result;
    }

    // ============================================================
    // 4. Build and choose near-field connected support component.
    // ============================================================
    const std::vector<SeedComponent> components =
        BuildSeedComponents(
            grid_map,
            config_);

    SeedComponent best_component;

    if (!SelectBestSeedComponent(
            components,
            config_,
            best_component))
    {
        return result;
    }

    result.seed_component_cells =
        best_component.keys.size();

    result.seed_component_angular_sectors =
        best_component.angular_sectors.size();

    // ============================================================
    // 5. Initialize BFS from the selected support component.
    // ============================================================
    std::queue<std::pair<int, int>> ground_queue;

    for (const std::pair<int, int> &key :
         best_component.keys)
    {
        auto iterator =
            grid_map.find(
                key);

        if (iterator ==
            grid_map.end())
        {
            continue;
        }

        iterator->second.is_ground =
            true;

        ground_queue.push(
            key);
    }

    if (ground_queue.empty())
    {
        return result;
    }

    // ============================================================
    // 6. V3 propagation.
    //
    // Preferred path:
    //     already accepted local ground cells
    //          -> fit z = a*x + b*y + c
    //          -> predict candidate z
    //          -> check only residual to that local terrain trend
    //
    // Fallback path:
    //     only when support is still insufficient for a local model.
    // ============================================================
    while (!ground_queue.empty())
    {
        const std::pair<int, int> current_key =
            ground_queue.front();

        ground_queue.pop();

        const auto current_iterator =
            grid_map.find(
                current_key);

        if (current_iterator ==
            grid_map.end())
        {
            continue;
        }

        const GroundGridCell &current_cell =
            current_iterator->second;

        if (!current_cell.surface_valid)
        {
            continue;
        }

        for (int dx = -1;
             dx <= 1;
             ++dx)
        {
            for (int dy = -1;
                 dy <= 1;
                 ++dy)
            {
                if (dx == 0 &&
                    dy == 0)
                {
                    continue;
                }

                const std::pair<int, int> neighbor_key =
                    std::make_pair(
                        current_key.first + dx,
                        current_key.second + dy);

                auto neighbor_iterator =
                    grid_map.find(
                        neighbor_key);

                if (neighbor_iterator ==
                    grid_map.end())
                {
                    continue;
                }

                GroundGridCell &neighbor_cell =
                    neighbor_iterator->second;

                if (!neighbor_cell.surface_valid ||
                    neighbor_cell.is_ground)
                {
                    continue;
                }

                // Hard one-step jump protection remains active even in V3.
                const double hard_delta_z =
                    std::abs(
                        neighbor_cell.surface_z -
                        current_cell.surface_z);

                if (!std::isfinite(hard_delta_z) ||
                    hard_delta_z >
                        config_.maximum_neighbor_height_jump_m)
                {
                    continue;
                }

                HeightFieldModel local_model;

                const bool local_model_valid =
                    BuildLocalPredictionModel(
                        grid_map,
                        neighbor_key,
                        config_,
                        local_model);

                bool accept_neighbor =
                    false;

                bool accepted_by_prediction =
                    false;

                if (local_model_valid)
                {
                    accept_neighbor =
                        CandidateMatchesLocalSurface(
                            neighbor_key,
                            neighbor_cell,
                            local_model,
                            config_);

                    accepted_by_prediction =
                        accept_neighbor;
                }
                else
                {
                    accept_neighbor =
                        NeighborSurfaceCompatible(
                            current_cell,
                            neighbor_cell,
                            dx,
                            dy,
                            config_);
                }

                if (!accept_neighbor)
                {
                    continue;
                }

                neighbor_cell.is_ground =
                    true;

                ground_queue.push(
                    neighbor_key);

                if (accepted_by_prediction)
                {
                    ++result.predicted_ground_cells;
                }
                else
                {
                    ++result.fallback_ground_cells;
                }
            }
        }
    }

    // ============================================================
    // 7. Count Ground Cells + roughness diagnostics.
    // ============================================================
    double roughness_sum =
        0.0;

    for (const auto &entry :
         grid_map)
    {
        const GroundGridCell &cell =
            entry.second;

        if (!cell.is_ground)
        {
            continue;
        }

        ++result.ground_cells;

        if (std::isfinite(
                cell.low_roughness_m))
        {
            roughness_sum +=
                cell.low_roughness_m;

            result.maximum_ground_cell_roughness_m =
                std::max(
                    result.maximum_ground_cell_roughness_m,
                    cell.low_roughness_m);
        }
    }

    if (result.ground_cells == 0)
    {
        return result;
    }

    result.mean_ground_cell_roughness_m =
        roughness_sum /
        static_cast<double>(
            result.ground_cells);

    // ============================================================
    // 8. Ground V3.3 Gap-Tolerant Multi-Hypothesis Support Surface selection.
    //
    // Ground-like terrain remains untouched. The selector only marks a
    // subset of Ground Cells as the vehicle support surface.
    // ============================================================
    const bool support_valid =
        BuildVehicleSupportSurface(
            grid_map,
            config_,
            has_previous_support_plane_,
            previous_support_normal_L_,
            previous_support_distance_m_,
            has_pending_support_plane_,
            pending_support_normal_L_,
            pending_support_distance_m_,
            pending_support_consecutive_frames_,
            result);

    // V4.0 intentionally does NOT update the temporal reference here.
    // We first count the actual support points and run the trusted constraint
    // gate.  A geometrically plausible but low-confidence shoulder therefore
    // cannot become the next temporal reference merely because it passed the
    // broad V3.3 support-plane gate.
    (void)support_valid;

    // ============================================================
    // 9. Point-level classification with adaptive upper threshold.
    // ============================================================
    result.ground_cloud->reserve(
        cloud->size());

    result.nonground_cloud->reserve(
        cloud->size());

    result.support_ground_cloud->reserve(
        cloud->size());

    for (const pcl::PointXYZ &point :
         cloud->points)
    {
        if (!IsFinitePoint(point))
        {
            continue;
        }

        const double range =
            HorizontalRange(
                point);

        if (range <
                config_.minimum_range_m ||
            range >
                config_.maximum_range_m)
        {
            result.nonground_cloud->push_back(
                point);

            continue;
        }

        const std::pair<int, int> key =
            ComputeGridKey(
                point,
                config_.grid_size_m);

        const auto iterator =
            grid_map.find(
                key);

        if (iterator ==
                grid_map.end() ||
            !iterator->second.surface_valid ||
            !iterator->second.is_ground)
        {
            result.nonground_cloud->push_back(
                point);

            continue;
        }

        const GroundGridCell &cell =
            iterator->second;

        const double height_difference =
            static_cast<double>(
                point.z) -
            cell.surface_z;

        const double adaptive_upper_threshold =
            std::clamp(
                config_.point_height_base_threshold_m +
                    config_.point_height_roughness_scale *
                        std::max(
                            0.0,
                            cell.low_roughness_m),
                config_.point_height_minimum_threshold_m,
                config_.point_height_maximum_threshold_m);

        if (height_difference >=
                -config_.point_below_surface_tolerance_m &&
            height_difference <=
                adaptive_upper_threshold)
        {
            result.ground_cloud->push_back(
                point);

            if (cell.is_support)
            {
                result.support_ground_cloud->push_back(
                    point);
            }
        }
        else
        {
            result.nonground_cloud->push_back(
                point);
        }
    }

    result.ground_cloud->width =
        static_cast<std::uint32_t>(
            result.ground_cloud->size());

    result.ground_cloud->height = 1;
    result.ground_cloud->is_dense = true;

    result.nonground_cloud->width =
        static_cast<std::uint32_t>(
            result.nonground_cloud->size());

    result.nonground_cloud->height = 1;
    result.nonground_cloud->is_dense = true;

    result.support_ground_cloud->width =
        static_cast<std::uint32_t>(
            result.support_ground_cloud->size());

    result.support_ground_cloud->height = 1;
    result.support_ground_cloud->is_dense = true;

    result.ground_points =
        result.ground_cloud->size();

    result.nonground_points =
        result.nonground_cloud->size();

    result.support_ground_points =
        result.support_ground_cloud->size();

    result.success =
        result.ground_points > 0 &&
        result.ground_cells > 0;

    // ============================================================
    // 10. Ground V4.0 trusted support-constraint gate.
    //
    // This is deliberately AFTER point classification so the gate sees the
    // actual number of support points that a future ICP residual would use.
    // ============================================================
    EvaluateSupportConstraint(
        result);

    // Only trusted constraints, or high-quality bootstrap samples accepted by
    // the clearance learner, are allowed to move the short-term support
    // reference.  This blocks gradual reference drift toward a road shoulder.
    if (result.support_plane_valid &&
        (result.support_constraint_valid ||
         result.support_clearance_sample_accepted))
    {
        has_previous_support_plane_ =
            true;

        previous_support_normal_L_ =
            result.support_ground_normal_L;

        previous_support_distance_m_ =
            result.support_ground_distance_m;
    }

    // ============================================================
    // 11. Fit general local Ground plane to Ground GRID-SURFACE cells.
    // ============================================================
    if (result.success)
    {
        FitRobustLocalGroundPlane(
            grid_map,
            config_,
            result);
    }

    return result;
}

}  // namespace fr_slam
