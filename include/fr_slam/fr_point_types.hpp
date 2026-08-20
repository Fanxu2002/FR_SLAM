#pragma once
#ifndef PCL_NO_PRECOMPILE
#define PCL_NO_PRECOMPILE
#endif

#include <pcl/point_types.h>
#include <pcl/register_point_struct.h>

// THE original LIVOX MID 360 Point CONTENTS
struct EIGEN_ALIGN16 LIVOX_POINT
{
        PCL_ADD_POINT4D;
        float intensity;
        std::uint8_t tag;
        std::uint8_t line;
        double timestamp;
        PCL_MAKE_ALIGNED_OPERATOR_NEW
};

// THE original HESAI Point CONTENTS
struct EIGEN_ALIGN16 HESAI_POINT
{
        PCL_ADD_POINT4D;
        float intensity;
        std::uint16_t ring;
        double timestamp;
        PCL_MAKE_ALIGNED_OPERATOR_NEW
};

// THE Unified Point Contents
struct EIGEN_ALIGN16 LIDAR_POINT
{
        PCL_ADD_POINT4D;
        float intensity;
        std::uint16_t ring;
        double time_offset;
        // time relative to the start of the current scan(unified uint: second)
        PCL_MAKE_ALIGNED_OPERATOR_NEW
};
POINT_CLOUD_REGISTER_POINT_STRUCT(
    LIVOX_POINT,
    (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(std::uint8_t, tag, tag)(std::uint8_t, line, line)(double, timestamp, timestamp));
POINT_CLOUD_REGISTER_POINT_STRUCT(
    HESAI_POINT,
    (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(std::uint16_t, ring, ring)(double, timestamp, timestamp));
POINT_CLOUD_REGISTER_POINT_STRUCT(
    LIDAR_POINT,
    (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(std::uint16_t, ring, ring)(double, time_offset, time_offset));