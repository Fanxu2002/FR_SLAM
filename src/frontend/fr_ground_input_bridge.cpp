#include "fr_slam/frontend/fr_ground_input_bridge.hpp"

#include <utility>

namespace fr_slam
{
namespace
{

thread_local pcl::PointCloud<LIDAR_POINT>::ConstPtr
    g_ground_icp_dense_input;

} // namespace

void PublishGroundIcpDenseInput(
    const pcl::PointCloud<LIDAR_POINT>::ConstPtr &cloud)
{
    g_ground_icp_dense_input =
        cloud;
}

pcl::PointCloud<LIDAR_POINT>::ConstPtr
ConsumeGroundIcpDenseInput()
{
    pcl::PointCloud<LIDAR_POINT>::ConstPtr cloud =
        g_ground_icp_dense_input;

    g_ground_icp_dense_input.reset();

    return cloud;
}

void ClearGroundIcpDenseInput()
{
    g_ground_icp_dense_input.reset();
}

} // namespace fr_slam
