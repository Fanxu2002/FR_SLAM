#include "fr_slam/fr_lidar_preprocessor.hpp"

#include <cstdint>
#include <iostream>

#include <Eigen/Core>

#include <pcl/PointIndices.h>

// ============================================================================
// Set LiDAR-IMU extrinsic used by the internal LidarDeskewer
//
// T_IL:
// LiDAR frame -> IMU frame
// ============================================================================

void PreProcessor::SetDeskewExtrinsic(
    const Eigen::Quaterniond &Q_IL,
    const Eigen::Vector3d &P_IL)
{
        deskewer_.SetExtrinsic(
            Q_IL,
            P_IL);
}

// ============================================================================
// Deskew one LiDAR frame
//
// PreProcessor owns the LidarDeskewer, but the actual deskew algorithm remains
// inside LidarDeskewer.
//
// This keeps the responsibility clear:
//
// PreProcessor:
//     controls the preprocessing pipeline
//
// LidarDeskewer:
//     performs motion compensation mathematics
// ============================================================================

bool PreProcessor::Deskew(
    const LIDAR_FRAME &lidar_frame,
    const std::vector<IMU_POSE> &imu_poses,
    LIDAR_FRAME &output_frame,
    bool use_translation)
{
        return deskewer_.Deskew(
            lidar_frame,
            imu_poses,
            output_frame,
            use_translation);
}

// ============================================================================
// Complete LiDAR preprocessing pipeline
//
// Raw LiDAR
//     -> Deskew
//     -> Remove NaN / Range / ROI / CropBox
//     -> SOR
//     -> ROR
//     -> VoxelGrid
//
// The returned frame is ready for registration.
// ============================================================================

LIDAR_FRAME PreProcessor::Process(
    const LIDAR_FRAME &lidar_frame,
    const std::vector<IMU_POSE> &imu_poses,
    bool use_translation)
{
        // ============================================================
        // 1. Deskew
        // ============================================================

        LIDAR_FRAME deskewed_frame;

        if (!Deskew(
                lidar_frame,
                imu_poses,
                deskewed_frame,
                use_translation))
        {
                std::cerr
                    << "PreProcessor::Process(): deskew failed."
                    << std::endl;

                return LIDAR_FRAME();
        }

        // ============================================================
        // 2. Basic preprocessing
        // ============================================================

        LIDAR_FRAME clean_frame =
            preprocess(
                deskewed_frame);

        if (!clean_frame.cloud ||
            clean_frame.cloud->empty())
        {
                std::cerr
                    << "PreProcessor::Process(): "
                    << "cloud is empty after preprocess."
                    << std::endl;

                return LIDAR_FRAME();
        }

        // ============================================================
        // 3. Statistical outlier removal
        //
        // If SOR is disabled in config, SOR() simply returns input.
        // ============================================================

        LIDAR_FRAME sor_frame =
            SOR(
                clean_frame);

        if (!sor_frame.cloud ||
            sor_frame.cloud->empty())
        {
                std::cerr
                    << "PreProcessor::Process(): "
                    << "cloud is empty after SOR."
                    << std::endl;

                return LIDAR_FRAME();
        }

        // ============================================================
        // 4. Radius outlier removal
        //
        // If ROR is disabled in config, ROR() simply returns input.
        // ============================================================

        LIDAR_FRAME ror_frame =
            ROR(
                sor_frame);

        if (!ror_frame.cloud ||
            ror_frame.cloud->empty())
        {
                std::cerr
                    << "PreProcessor::Process(): "
                    << "cloud is empty after ROR."
                    << std::endl;

                return LIDAR_FRAME();
        }

        // ============================================================
        // 5. Voxel downsample
        //
        // If voxel is disabled in config, VoxelGrid() returns input.
        // ============================================================

        LIDAR_FRAME voxel_frame =
            VoxelGrid(
                ror_frame);

        if (!voxel_frame.cloud ||
            voxel_frame.cloud->empty())
        {
                std::cerr
                    << "PreProcessor::Process(): "
                    << "cloud is empty after VoxelGrid."
                    << std::endl;

                return LIDAR_FRAME();
        }

        return voxel_frame;
}

// ============================================================================
// Basic preprocessing
// ============================================================================

LIDAR_FRAME PreProcessor::preprocess(
    const LIDAR_FRAME &lidar_frame)
{
        pcl::PointCloud<LIDAR_POINT>::Ptr original_pointcloud;
        pcl::PointCloud<LIDAR_POINT>::Ptr final_pointcloud;

        final_pointcloud =
            pcl::make_shared<
                pcl::PointCloud<LIDAR_POINT>>();

        if (!lidar_frame.cloud)
        {
                std::cerr
                    << "Input point cloud is nullptr!"
                    << std::endl;

                return LIDAR_FRAME();
        }

        if (lidar_frame.cloud->empty())
        {
                std::cerr
                    << "Input point cloud is empty!"
                    << std::endl;

                return LIDAR_FRAME();
        }

        original_pointcloud =
            lidar_frame.cloud;

        // ============================================================
        // 1. Remove NaN
        // ============================================================

        pcl::Indices valid_indices;

        pcl::removeNaNFromPointCloud<LIDAR_POINT>(
            *original_pointcloud,
            *final_pointcloud,
            valid_indices);

        std::cout
            << "There are "
            << original_pointcloud->size()
            << " points in the original cloud;\n"
            << std::endl;

        std::cout
            << "After removing NaN value, there are "
            << final_pointcloud->size()
            << " points in the cloud;\n"
            << std::endl;

        // ============================================================
        // 2. Range filter:
        //
        // range_min^2 <= x^2 + y^2 + z^2 <= range_max^2
        // ============================================================

        if (config_.enable_ROI)
        {
                pcl::PointCloud<LIDAR_POINT>::Ptr range_cloud =
                    pcl::make_shared<
                        pcl::PointCloud<LIDAR_POINT>>();

                range_cloud->reserve(
                    final_pointcloud->size());

                const double min_range_square =
                    config_.range_min *
                    config_.range_min;

                const double max_range_square =
                    config_.range_max *
                    config_.range_max;

                for (const LIDAR_POINT &point :
                     final_pointcloud->points)
                {
                        const double range_square =
                            static_cast<double>(point.x) *
                                static_cast<double>(point.x) +
                            static_cast<double>(point.y) *
                                static_cast<double>(point.y) +
                            static_cast<double>(point.z) *
                                static_cast<double>(point.z);

                        if (range_square >= min_range_square &&
                            range_square <= max_range_square)
                        {
                                range_cloud->push_back(
                                    point);
                        }
                }

                range_cloud->width =
                    static_cast<std::uint32_t>(
                        range_cloud->size());

                range_cloud->height =
                    1;

                range_cloud->is_dense =
                    true;

                final_pointcloud =
                    range_cloud;
        }

        // ============================================================
        // 3. PassThrough ROI
        // ============================================================

        if (config_.enable_passthrough)
        {
                pcl::PassThrough<LIDAR_POINT>::Ptr passthrough =
                    pcl::make_shared<
                        pcl::PassThrough<LIDAR_POINT>>();

                passthrough->setInputCloud(
                    final_pointcloud);

                passthrough->setFilterFieldName(
                    "x");

                passthrough->setFilterLimits(
                    config_.ROI_min_x,
                    config_.ROI_max_x);

                passthrough->filter(
                    *final_pointcloud);

                passthrough->setInputCloud(
                    final_pointcloud);

                passthrough->setFilterFieldName(
                    "y");

                passthrough->setFilterLimits(
                    config_.ROI_min_y,
                    config_.ROI_max_y);

                passthrough->filter(
                    *final_pointcloud);

                passthrough->setInputCloud(
                    final_pointcloud);

                passthrough->setFilterFieldName(
                    "z");

                passthrough->setFilterLimits(
                    config_.ROI_min_z,
                    config_.ROI_max_z);

                passthrough->filter(
                    *final_pointcloud);
        }

        // ============================================================
        // 4. CropBox
        //
        // Usually used to remove points on the vehicle / sensor body.
        // ============================================================

        if (config_.enable_cropbox)
        {
                pcl::CropBox<LIDAR_POINT>::Ptr cropbox =
                    pcl::make_shared<
                        pcl::CropBox<LIDAR_POINT>>();

                const Eigen::Vector4f max_range(
                    config_.cropbox_max_x,
                    config_.cropbox_max_y,
                    config_.cropbox_max_z,
                    1.0f);

                const Eigen::Vector4f min_range(
                    config_.cropbox_min_x,
                    config_.cropbox_min_y,
                    config_.cropbox_min_z,
                    1.0f);

                cropbox->setInputCloud(
                    final_pointcloud);

                cropbox->setMax(
                    max_range);

                cropbox->setMin(
                    min_range);

                cropbox->setNegative(
                    true);

                cropbox->filter(
                    *final_pointcloud);

                std::cout
                    << "There are "
                    << final_pointcloud->size()
                    << " points after cropbox filtering!\n"
                    << std::endl;
        }

        // ============================================================
        // 5. Build output frame
        // ============================================================

        LIDAR_FRAME output_frame;

        output_frame.frame_id =
            lidar_frame.frame_id;

        output_frame.cloud =
            final_pointcloud;

        output_frame.has_point_time =
            lidar_frame.has_point_time;

        output_frame.scan_start_time =
            lidar_frame.scan_start_time;

        output_frame.scan_duration =
            lidar_frame.scan_duration;

        return output_frame;
}

// ============================================================================
// VoxelGrid
// ============================================================================

LIDAR_FRAME PreProcessor::VoxelGrid(
    const LIDAR_FRAME &lidar_frame) const
{
        if (!config_.enable_voxel)
        {
                return lidar_frame;
        }

        if (!lidar_frame.cloud)
        {
                std::cerr
                    << "Input point cloud is nullptr!"
                    << std::endl;

                return LIDAR_FRAME();
        }

        if (lidar_frame.cloud->empty())
        {
                std::cerr
                    << "Input point cloud is empty!"
                    << std::endl;

                return LIDAR_FRAME();
        }

        pcl::PointCloud<LIDAR_POINT>::Ptr final_pointcloud =
            pcl::make_shared<
                pcl::PointCloud<LIDAR_POINT>>();

        pcl::VoxelGrid<LIDAR_POINT>::Ptr voxel_grid =
            pcl::make_shared<
                pcl::VoxelGrid<LIDAR_POINT>>();

        voxel_grid->setInputCloud(
            lidar_frame.cloud);

        voxel_grid->setDownsampleAllData(
            false);

        voxel_grid->setLeafSize(
            config_.voxel_leaf_size,
            config_.voxel_leaf_size,
            config_.voxel_leaf_size);

        voxel_grid->setMinimumPointsNumberPerVoxel(
            config_.voxel_min_points);

        voxel_grid->filter(
            *final_pointcloud);

        LIDAR_FRAME output_frame;

        output_frame.frame_id =
            lidar_frame.frame_id;

        output_frame.cloud =
            final_pointcloud;

        output_frame.has_point_time =
            lidar_frame.has_point_time;

        output_frame.scan_start_time =
            lidar_frame.scan_start_time;

        output_frame.scan_duration =
            lidar_frame.scan_duration;

        return output_frame;
}

// ============================================================================
// Statistical Outlier Removal
// ============================================================================

LIDAR_FRAME PreProcessor::SOR(
    const LIDAR_FRAME &lidar_frame) const
{
        if (!config_.enable_SOR)
        {
                return lidar_frame;
        }

        if (!lidar_frame.cloud)
        {
                std::cerr
                    << "Input point cloud is nullptr!"
                    << std::endl;

                return LIDAR_FRAME();
        }

        if (lidar_frame.cloud->empty())
        {
                std::cerr
                    << "Input point cloud is empty!"
                    << std::endl;

                return LIDAR_FRAME();
        }

        pcl::PointCloud<LIDAR_POINT>::Ptr final_pointcloud =
            pcl::make_shared<
                pcl::PointCloud<LIDAR_POINT>>();

        pcl::StatisticalOutlierRemoval<LIDAR_POINT>::Ptr sor =
            pcl::make_shared<
                pcl::StatisticalOutlierRemoval<LIDAR_POINT>>();

        sor->setInputCloud(
            lidar_frame.cloud);

        sor->setMeanK(
            config_.sor_mean_k);

        sor->setStddevMulThresh(
            config_.sor_stddev_mul_thresh);

        sor->filter(
            *final_pointcloud);

        std::cout
            << "There are "
            << final_pointcloud->size()
            << " points after SOR filtering!\n"
            << std::endl;

        LIDAR_FRAME output_frame;

        output_frame.frame_id =
            lidar_frame.frame_id;

        output_frame.cloud =
            final_pointcloud;

        output_frame.has_point_time =
            lidar_frame.has_point_time;

        output_frame.scan_start_time =
            lidar_frame.scan_start_time;

        output_frame.scan_duration =
            lidar_frame.scan_duration;

        return output_frame;
}

// ============================================================================
// Radius Outlier Removal
// ============================================================================

LIDAR_FRAME PreProcessor::ROR(
    const LIDAR_FRAME &lidar_frame) const
{
        if (!config_.enable_ROR)
        {
                return lidar_frame;
        }

        if (!lidar_frame.cloud)
        {
                std::cerr
                    << "Input point cloud is nullptr!"
                    << std::endl;

                return LIDAR_FRAME();
        }

        if (lidar_frame.cloud->empty())
        {
                std::cerr
                    << "Input point cloud is empty!"
                    << std::endl;

                return LIDAR_FRAME();
        }

        pcl::PointCloud<LIDAR_POINT>::Ptr final_pointcloud =
            pcl::make_shared<
                pcl::PointCloud<LIDAR_POINT>>();

        pcl::RadiusOutlierRemoval<LIDAR_POINT>::Ptr ror =
            pcl::make_shared<
                pcl::RadiusOutlierRemoval<LIDAR_POINT>>();

        ror->setInputCloud(
            lidar_frame.cloud);

        ror->setRadiusSearch(
            config_.ror_RadiusSearch);

        ror->setMinNeighborsInRadius(
            config_.ror_MinNeighborsInRadius);

        ror->setNegative(
            false);

        ror->filter(
            *final_pointcloud);

        std::cout
            << "There are "
            << final_pointcloud->size()
            << " points after ROR filtering!\n"
            << std::endl;

        LIDAR_FRAME output_frame;

        output_frame.frame_id =
            lidar_frame.frame_id;

        output_frame.cloud =
            final_pointcloud;

        output_frame.has_point_time =
            lidar_frame.has_point_time;

        output_frame.scan_start_time =
            lidar_frame.scan_start_time;

        output_frame.scan_duration =
            lidar_frame.scan_duration;

        return output_frame;
}