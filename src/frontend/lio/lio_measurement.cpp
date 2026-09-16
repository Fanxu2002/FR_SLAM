#include "fr_slam/frontend/lio_measurement.hpp"

#include <cmath>
#include <iostream>
#include <vector>

#include "fr_slam/frontend/lio_math.hpp"

namespace
{

        bool MeasurementStateIsFinite(
            const LioState &state)
        {
                return std::isfinite(state.timestamp) &&
                       state.Q_WI.coeffs().allFinite() &&
                       state.Q_WI.norm() > 1.0e-12 &&
                       state.P_WI.allFinite() &&
                       state.V_WI.allFinite() &&
                       state.gyro_bias.allFinite() &&
                       state.accel_bias.allFinite() &&
                       state.gravity_W.allFinite() &&
                       state.gravity_W.norm() > 1.0e-12 &&
                       state.Q_IL.coeffs().allFinite() &&
                       state.Q_IL.norm() > 1.0e-12 &&
                       state.P_IL.allFinite();
        }

        bool LidarPointIsFinite(
            const LIDAR_POINT &point)
        {
                return std::isfinite(static_cast<double>(point.x)) &&
                       std::isfinite(static_cast<double>(point.y)) &&
                       std::isfinite(static_cast<double>(point.z));
        }

} // namespace

LioMeasurementBuilder::LioMeasurementBuilder(
    const LioMeasurementConfig &config)
    : config_(config)
{
}

bool LioMeasurementBuilder::SetConfig(
    const LioMeasurementConfig &config)
{
        if (!ConfigIsValid(config))
        {
                return false;
        }

        config_ = config;
        return true;
}

const LioMeasurementConfig &
LioMeasurementBuilder::Config() const
{
        return config_;
}

bool LioMeasurementBuilder::ConfigIsValid(
    const LioMeasurementConfig &config) const
{
        return config.knn > 0 &&
               std::isfinite(config.max_correspondence_distance) &&
               config.max_correspondence_distance > 0.0 &&
               std::isfinite(config.max_point_to_plane_distance) &&
               config.max_point_to_plane_distance > 0.0 &&
               config.min_correspondences > 0 &&
               (!config.enable_huber_loss ||
                (std::isfinite(config.huber_delta) &&
                 config.huber_delta > 0.0)) &&
               std::isfinite(config.point_to_plane_noise_std) &&
               config.point_to_plane_noise_std > 0.0;
}

bool LioMeasurementBuilder::Build(
    const LioState &state,
    const pcl::PointCloud<LIDAR_POINT>::ConstPtr &deskewed_scan_L,
    const PreparedLidarTarget &target,
    LioMeasurementResult &result) const
{
        result = LioMeasurementResult();

        if (!ConfigIsValid(config_) ||
            !MeasurementStateIsFinite(state) ||
            !deskewed_scan_L ||
            deskewed_scan_L->empty() ||
            !target.ready ||
            !target.cloud ||
            target.cloud->empty() ||
            !target.kdtree ||
            target.planes.size() != target.cloud->size())
        {
                return false;
        }

        const Eigen::Matrix3d R_WI =
            state.Q_WI.normalized().toRotationMatrix();

        const Eigen::Matrix3d R_IL =
            state.Q_IL.normalized().toRotationMatrix();

        const double max_correspondence_distance_squared =
            config_.max_correspondence_distance *
            config_.max_correspondence_distance;

        const double measurement_variance =
            config_.point_to_plane_noise_std *
            config_.point_to_plane_noise_std;

        const double inverse_measurement_variance =
            1.0 / measurement_variance;

        std::vector<int> neighbor_indices(
            static_cast<std::size_t>(config_.knn));

        std::vector<float> neighbor_squared_distances(
            static_cast<std::size_t>(config_.knn));

        double squared_error_sum = 0.0;
        double robust_squared_error_sum = 0.0;
        double robust_weight_sum = 0.0;

        for (const LIDAR_POINT &point :
             deskewed_scan_L->points)
        {
                if (!LidarPointIsFinite(point))
                {
                        continue;
                }

                const Eigen::Vector3d p_L(
                    static_cast<double>(point.x),
                    static_cast<double>(point.y),
                    static_cast<double>(point.z));

                const Eigen::Vector3d q_I =
                    R_IL * p_L +
                    state.P_IL;

                const Eigen::Vector3d p_W =
                    R_WI * q_I +
                    state.P_WI;

                if (!p_W.allFinite())
                {
                        continue;
                }

                LIDAR_POINT query_point = point;

                query_point.x =
                    static_cast<float>(p_W.x());

                query_point.y =
                    static_cast<float>(p_W.y());

                query_point.z =
                    static_cast<float>(p_W.z());

                const int found =
                    target.kdtree->nearestKSearch(
                        query_point,
                        config_.knn,
                        neighbor_indices,
                        neighbor_squared_distances);

                if (found <= 0)
                {
                        continue;
                }

                int plane_index = -1;

                for (int j = 0;
                     j < found;
                     ++j)
                {
                        const std::size_t j_index =
                            static_cast<std::size_t>(j);

                        const double distance_squared =
                            static_cast<double>(
                                neighbor_squared_distances[j_index]);

                        if (!std::isfinite(distance_squared))
                        {
                                continue;
                        }

                        if (distance_squared >
                            max_correspondence_distance_squared)
                        {
                                break;
                        }

                        const int candidate_index =
                            neighbor_indices[j_index];

                        if (candidate_index < 0)
                        {
                                continue;
                        }

                        const std::size_t candidate =
                            static_cast<std::size_t>(
                                candidate_index);

                        if (candidate >=
                            target.planes.size())
                        {
                                continue;
                        }

                        if (target.planes[candidate].state !=
                            TargetPlane::State::Valid)
                        {
                                continue;
                        }

                        plane_index =
                            candidate_index;

                        break;
                }

                if (plane_index < 0)
                {
                        continue;
                }

                const TargetPlane &plane =
                    target.planes[static_cast<std::size_t>(
                        plane_index)];

                if (!plane.point.allFinite() ||
                    !plane.normal.allFinite())
                {
                        continue;
                }

                Eigen::Vector3d normal =
                    plane.normal;

                const double normal_norm =
                    normal.norm();

                if (!std::isfinite(normal_norm) ||
                    normal_norm < 1.0e-12)
                {
                        continue;
                }

                normal /= normal_norm;

                const double residual =
                    normal.dot(
                        p_W -
                        plane.point);

                if (!std::isfinite(residual))
                {
                        continue;
                }

                const double absolute_residual =
                    std::abs(residual);

                if (absolute_residual >
                    config_.max_point_to_plane_distance)
                {
                        continue;
                }

                Eigen::Matrix<double, 1, Ieskf::STATE_DIM> H =
                    Eigen::Matrix<double, 1, Ieskf::STATE_DIM>::Zero();

                H.block<1, 3>(
                    0,
                    LioStateIndex::ROTATION) =
                    -normal.transpose() *
                    R_WI *
                    Skew(q_I);

                H.block<1, 3>(
                    0,
                    LioStateIndex::POSITION) =
                    normal.transpose();

                H.block<1, 3>(
                    0,
                    LioStateIndex::EXTRINSIC_ROTATION) =
                    -normal.transpose() *
                    R_WI *
                    R_IL *
                    Skew(p_L);

                H.block<1, 3>(
                    0,
                    LioStateIndex::EXTRINSIC_POSITION) =
                    normal.transpose() *
                    R_WI;

                if (!H.allFinite())
                {
                        continue;
                }

                double robust_weight = 1.0;

                if (config_.enable_huber_loss &&
                    absolute_residual >
                        config_.huber_delta)
                {
                        robust_weight =
                            config_.huber_delta /
                            absolute_residual;

                        ++result.downweighted_correspondences;
                }

                if (!std::isfinite(robust_weight) ||
                    robust_weight <= 0.0)
                {
                        continue;
                }

                const double information_weight =
                    robust_weight *
                    inverse_measurement_variance;

                result.information.noalias() +=
                    information_weight *
                    H.transpose() *
                    H;

                result.gradient.noalias() +=
                    information_weight *
                    H.transpose() *
                    residual;

                squared_error_sum +=
                    residual *
                    residual;

                robust_squared_error_sum +=
                    robust_weight *
                    residual *
                    residual;

                robust_weight_sum +=
                    robust_weight;

                ++result.correspondences;
        }

        if (result.correspondences > 0)
        {
                result.rmse =
                    std::sqrt(
                        squared_error_sum /
                        static_cast<double>(
                            result.correspondences));
        }

        if (robust_weight_sum > 0.0)
        {
                result.robust_rmse =
                    std::sqrt(
                        robust_squared_error_sum /
                        robust_weight_sum);
        }

        result.robust_weight_sum =
            robust_weight_sum;

        result.information =
            0.5 *
            (result.information +
             result.information.transpose());

        if (!result.information.allFinite() ||
            !result.gradient.allFinite() ||
            result.correspondences <
                config_.min_correspondences)
        {
                result.success = false;
                return false;
        }

        result.success = true;
        return true;
}
