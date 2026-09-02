#include "fr_slam/fr_lidar_preprocessor.hpp"
#include "fr_slam/fr_ground_icp_input_bridge.hpp"

#include <chrono>
#include <cstdint>
#include <iostream>

#include <Eigen/Core>

#include <pcl/PointIndices.h>

namespace
{
    using PreprocessorClock = std::chrono::steady_clock;

    thread_local PreprocessorTiming g_last_preprocessor_timing;

    double ElapsedMilliseconds(
        const PreprocessorClock::time_point &begin,
        const PreprocessorClock::time_point &end)
    {
        return std::chrono::duration<double, std::milli>(
                   end - begin)
            .count();
    }

    void ResetPreprocessorTiming()
    {
        g_last_preprocessor_timing =
            PreprocessorTiming();
    }
}

// ============================================================================
// Runtime switches used by the ROS node.
// ============================================================================

void PreProcessor::SetOutlierFiltersEnabled(
    bool enable_sor,
    bool enable_ror)
{
    config_.enable_SOR =
        enable_sor;

    config_.enable_ROR =
        enable_ror;
}

const PreprocessorTiming &PreProcessor::GetLastTiming() const
{
    return g_last_preprocessor_timing;
}

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
// IMPORTANT:
// Keep the currently validated registration order:
//
// Raw LiDAR
//     -> Deskew
//     -> Basic (NaN / Range / ROI / CropBox)
//     -> VoxelGrid
//     -> optional SOR
//     -> optional ROR
//
// Ground ICP V1.1 branches immediately after Basic:
//
// Basic dense cloud -------------------------------> Ground bridge
//       |
//       +-> Voxel -> SOR -> ROR -> sparse ICP cloud
// ============================================================================

LIDAR_FRAME PreProcessor::Process(
    const LIDAR_FRAME &lidar_frame,
    const std::vector<IMU_POSE> &imu_poses,
    bool use_translation)
{
    ResetPreprocessorTiming();

    // Never allow an older Basic cloud to survive a failed frame.
    fr_slam::ClearGroundIcpDenseInput();

    // ============================================================
    // 1. Deskew
    // ============================================================

    LIDAR_FRAME deskewed_frame;

    const PreprocessorClock::time_point deskew_begin =
        PreprocessorClock::now();

    const bool deskew_ok =
        Deskew(
            lidar_frame,
            imu_poses,
            deskewed_frame,
            use_translation);

    const PreprocessorClock::time_point deskew_end =
        PreprocessorClock::now();

    g_last_preprocessor_timing.deskew_ms =
        ElapsedMilliseconds(
            deskew_begin,
            deskew_end);

    if (!deskew_ok)
    {
        std::cerr
            << "PreProcessor::Process(): deskew failed."
            << std::endl;

        return LIDAR_FRAME();
    }

    // ============================================================
    // 2. Basic preprocessing
    //
    // preprocess() publishes the Basic dense cloud to the one-shot
    // Ground ICP bridge before the registration branch is downsampled.
    // ============================================================

    const PreprocessorClock::time_point basic_begin =
        PreprocessorClock::now();

    LIDAR_FRAME clean_frame =
        preprocess(
            deskewed_frame);

    const PreprocessorClock::time_point basic_end =
        PreprocessorClock::now();

    g_last_preprocessor_timing.basic_ms =
        ElapsedMilliseconds(
            basic_begin,
            basic_end);

    if (!clean_frame.cloud ||
        clean_frame.cloud->empty())
    {
        fr_slam::ClearGroundIcpDenseInput();

        std::cerr
            << "PreProcessor::Process(): "
            << "cloud is empty after Basic preprocessing."
            << std::endl;

        return LIDAR_FRAME();
    }

    g_last_preprocessor_timing.after_basic_points =
        clean_frame.cloud->size();

    // ============================================================
    // 3. Registration VoxelGrid
    // ============================================================

    const PreprocessorClock::time_point voxel_begin =
        PreprocessorClock::now();

    LIDAR_FRAME voxel_frame =
        VoxelGrid(
            clean_frame);

    const PreprocessorClock::time_point voxel_end =
        PreprocessorClock::now();

    g_last_preprocessor_timing.voxel_ms =
        ElapsedMilliseconds(
            voxel_begin,
            voxel_end);

    if (!voxel_frame.cloud ||
        voxel_frame.cloud->empty())
    {
        fr_slam::ClearGroundIcpDenseInput();

        std::cerr
            << "PreProcessor::Process(): "
            << "cloud is empty after VoxelGrid."
            << std::endl;

        return LIDAR_FRAME();
    }

    g_last_preprocessor_timing.after_voxel_points =
        voxel_frame.cloud->size();

    // ============================================================
    // 4. Statistical outlier removal
    // ============================================================

    const PreprocessorClock::time_point sor_begin =
        PreprocessorClock::now();

    LIDAR_FRAME sor_frame =
        SOR(
            voxel_frame);

    const PreprocessorClock::time_point sor_end =
        PreprocessorClock::now();

    g_last_preprocessor_timing.sor_ms =
        ElapsedMilliseconds(
            sor_begin,
            sor_end);

    if (!sor_frame.cloud ||
        sor_frame.cloud->empty())
    {
        fr_slam::ClearGroundIcpDenseInput();

        std::cerr
            << "PreProcessor::Process(): "
            << "cloud is empty after SOR."
            << std::endl;

        return LIDAR_FRAME();
    }

    // ============================================================
    // 5. Radius outlier removal
    // ============================================================

    const PreprocessorClock::time_point ror_begin =
        PreprocessorClock::now();

    LIDAR_FRAME ror_frame =
        ROR(
            sor_frame);

    const PreprocessorClock::time_point ror_end =
        PreprocessorClock::now();

    g_last_preprocessor_timing.ror_ms =
        ElapsedMilliseconds(
            ror_begin,
            ror_end);

    if (!ror_frame.cloud ||
        ror_frame.cloud->empty())
    {
        fr_slam::ClearGroundIcpDenseInput();

        std::cerr
            << "PreProcessor::Process(): "
            << "cloud is empty after ROR."
            << std::endl;

        return LIDAR_FRAME();
    }

    return ror_frame;
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
    // 2. Range filter
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
    // 5. Build Basic output frame
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

    // ============================================================
    // 6. Ground ICP V1.1 dense branch handoff
    //
    // This pointer is published BEFORE the registration Voxel/SOR/ROR
    // stages. Ground V4 later performs its own 0.15 m analysis voxel.
    // ============================================================

    fr_slam::PublishGroundIcpDenseInput(
        output_frame.cloud);

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
