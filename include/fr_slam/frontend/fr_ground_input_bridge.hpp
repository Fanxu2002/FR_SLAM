#pragma once

#include "fr_slam/common/fr_point_types.hpp"

#include <pcl/point_cloud.h>

namespace fr_slam
{

// ============================================================================
// Ground ICP V1.1 dense-input bridge
//
// Purpose:
//     The realtime frontend intentionally uses two point-cloud branches:
//
//       Basic/ROI cloud  -> Ground V4.0 analysis
//       Final sparse cloud -> ordinary Scan-to-LocalMap ICP
//
// The current ROS worker already executes PreProcessor::preprocess() and
// RegistrationScan2LocalMap::AddFrame() sequentially on the same worker thread.
// A thread-local one-shot bridge lets the Basic cloud reach AddFrame without
// changing the long-standing AddFrame public API or the ROS application file.
//
// Safety properties:
//   - thread_local: no cross-thread cloud mixing;
//   - one-shot consume: stale data cannot be reused after a successful AddFrame;
//   - PreProcessor clears the slot at the beginning of every Basic-filter pass;
//   - AddFrame falls back to its normal registration cloud if no dense cloud is
//     available, preserving the Ground ICP V1 fail-safe behavior.
// ============================================================================

void PublishGroundIcpDenseInput(
    const pcl::PointCloud<LIDAR_POINT>::ConstPtr &cloud);

pcl::PointCloud<LIDAR_POINT>::ConstPtr
ConsumeGroundIcpDenseInput();

void ClearGroundIcpDenseInput();

} // namespace fr_slam
