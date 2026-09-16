#pragma once

#include <cstddef>
#include <limits>

#include <Eigen/Core>

#include <pcl/point_cloud.h>

#include "fr_slam/common/point_types.hpp"
#include "fr_slam/frontend/ieskf.hpp"
#include "fr_slam/frontend/lio_state.hpp"
#include "fr_slam/lidar/lidar_registration.hpp"


struct LioMeasurementConfig
{
        // Number of nearby target points queried for each transformed scan point.
        int knn = 5;

        // Source-to-target Euclidean correspondence gate [m].
        double max_correspondence_distance = 1.0;

        // Hard point-to-plane residual gate [m].
        double max_point_to_plane_distance = 0.5;

        // Minimum accepted point-to-plane constraints required by one
        // IESKF linearization.
        std::size_t min_correspondences = 50;

        // Robust kernel.
        bool enable_huber_loss = true;
        double huber_delta = 0.20;

        // Standard deviation of one point-to-plane measurement [m].
        //
        // This is measurement noise, not the hard residual gate above.
        // It must be configured explicitly before running the LiDAR update.
        double point_to_plane_noise_std =
            std::numeric_limits<double>::quiet_NaN();
};


struct LioMeasurementResult
{
        bool success = false;

        std::size_t correspondences = 0;
        std::size_t downweighted_correspondences = 0;

        double rmse =
            std::numeric_limits<double>::infinity();

        double robust_rmse =
            std::numeric_limits<double>::infinity();

        double robust_weight_sum = 0.0;

        // LiDAR information contribution:
        //
        //     Lambda_L = sum_i alpha_i H_i^T H_i
        //     eta_L    = sum_i alpha_i H_i^T r_i
        //
        // with
        //
        //     alpha_i = robust_weight_i / sigma_i^2
        //
        // The IESKF update later combines this with the IMU prior.
        Ieskf::StateMatrix information =
            Ieskf::StateMatrix::Zero();

        Ieskf::StateVector gradient =
            Ieskf::StateVector::Zero();
};


class LioMeasurementBuilder
{
public:
        explicit LioMeasurementBuilder(
            const LioMeasurementConfig &config);

        bool SetConfig(
            const LioMeasurementConfig &config);

        const LioMeasurementConfig &Config() const;

        // Build one complete LiDAR linearization at the supplied state.
        //
        // The deskewed scan is expressed in the reference LiDAR frame.
        // The prepared target is expressed in the same World/Map frame as
        // state.P_WI / state.Q_WI.
        //
        // Each IESKF iteration calls Build() again:
        //
        //     transform
        //       -> KNN
        //       -> cached-plane selection
        //       -> residual
        //       -> 23-DOF Jacobian
        //
        // The target plane PCA itself is cached by PreparedLidarTarget because
        // the target map is fixed during one LiDAR update.
        bool Build(
            const LioState &state,
            const pcl::PointCloud<LIDAR_POINT>::ConstPtr &deskewed_scan_L,
            const PreparedLidarTarget &target,
            LioMeasurementResult &result) const;


private:
        bool ConfigIsValid(
            const LioMeasurementConfig &config) const;


private:
        LioMeasurementConfig config_;
};
