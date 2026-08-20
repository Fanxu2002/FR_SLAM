#pragma once
#include <Eigen/Core>
struct PreprocessorConfig
{

        // Range filter config
        double range_min = 1.0;
        double range_max = 30;

        // Passthrough filter config
        double ROI_max_x = 30.0;
        double ROI_max_y = 15.0;
        double ROI_max_z = 10.0;
        double ROI_min_x = -30.0;
        double ROI_min_y = -15.0;
        double ROI_min_z = -2.0;

        // cropbox filter config
        float cropbox_min_x = -0.15;
        float cropbox_min_y = -0.15;
        float cropbox_min_z = -0.15;
        float cropbox_max_x = 0.15;
        float cropbox_max_y = 0.15;
        float cropbox_max_z = 0.15;

        // VoxelGrid filter config
        float voxel_leaf_size = 0.3f;
        unsigned int voxel_min_points = 1;

        // SOR filter config
        int sor_mean_k = 50;
        double sor_stddev_mul_thresh = 1.0;

        // ROR filter config
        double ror_RadiusSearch = 0.3;
        int ror_MinNeighborsInRadius = 1;

        // Enable / Disable
        bool enable_cropbox = true;
        bool enable_SOR = true;
        bool enable_voxel = true;
        bool enable_passthrough = true;
        bool enable_ROI = true;
        bool enable_ROR = true;
};
