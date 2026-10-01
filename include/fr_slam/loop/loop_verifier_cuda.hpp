#pragma once

#include <cstddef>

namespace fr_slam_cuda
{

struct BatchTransformStats
{
    bool ok = false;

    std::size_t point_count = 0U;
    std::size_t transform_count = 0U;

    double kernel_ms = 0.0;
    double total_ms = 0.0;

    int cuda_error = 0;
};


bool TransformPointsBatch(
    const float *source_xyz,
    std::size_t point_count,
    const float *transforms_4x4_row_major,
    std::size_t transform_count,
    float *output_xyz,
    BatchTransformStats &stats);


struct Knn5Stats
{
    bool ok = false;

    std::size_t query_count = 0U;
    std::size_t target_count = 0U;

    double kernel_ms = 0.0;
    double total_ms = 0.0;

    int cuda_error = 0;
};


bool BruteForceKnn5(
    const float *query_xyz,
    std::size_t query_count,
    const float *target_xyz,
    std::size_t target_count,
    int *output_indices,
    float *output_squared_distances,
    Knn5Stats &stats);


using PersistentKnn5Handle = void *;


struct PersistentKnn5Stats
{
    bool ok = false;

    std::size_t query_count = 0U;
    std::size_t target_count = 0U;
    std::size_t ambiguous_queries = 0U;

    double kernel_ms = 0.0;
    double total_ms = 0.0;

    int cuda_error = 0;
};


bool CreatePersistentKnn5(
    const float *target_xyz,
    std::size_t target_count,
    std::size_t max_query_count,
    PersistentKnn5Handle &handle,
    double &setup_ms,
    int &cuda_error);


void DestroyPersistentKnn5(
    PersistentKnn5Handle handle);


bool QueryPersistentKnn5(
    PersistentKnn5Handle handle,
    const float *query_xyz,
    std::size_t query_count,
    int *output_indices,
    float *output_squared_distances,
    unsigned char *output_ambiguous,
    float ambiguity_abs_epsilon,
    float ambiguity_relative_epsilon,
    PersistentKnn5Stats &stats);



struct PlaneHessianShadowStats
{
    bool ok = false;

    std::size_t query_count = 0U;
    std::size_t eligible_queries = 0U;
    std::size_t plane_valid = 0U;
    std::size_t plane_fit_failures = 0U;
    std::size_t correspondences = 0U;
    std::size_t downweighted = 0U;

    std::size_t ground_correspondences = 0U;
    std::size_t ground_downweighted = 0U;

    double kernel_ms = 0.0;
    double total_ms = 0.0;

    int cuda_error = 0;
};


bool ComputePlaneHessianShadowV1(
    PersistentKnn5Handle handle,
    const double *p_target_xyz,
    const int *neighbor_indices,
    const float *neighbor_squared_distances,
    const unsigned char *eligible,
    std::size_t query_count,
    const double *sensor_origin_xyz,
    double maximum_squared_distance,
    double maximum_plane_fit_error,
    double maximum_residual,
    double huber_delta,
    bool enable_ground_constraint,
    double ground_normal_cosine_threshold,
    double ground_min_below_sensor_m,
    double ground_max_residual,
    double ground_huber_delta,
    double ground_weight,
    double *output_H_6x6,
    double *output_b_6,
    double *output_H_ground_6x6,
    double *output_b_ground_6,
    double *output_ranges,
    unsigned char *output_range_valid,
    double &output_raw_sse,
    double &output_robust_sse,
    double &output_robust_weight_sum,
    PlaneHessianShadowStats &stats);



struct PersistentFusedStats
{
    bool ok = false;

    std::size_t query_count = 0U;

    std::size_t ambiguous_queries = 0U;
    std::size_t cpu_fallback_queries = 0U;

    std::size_t plane_fit_failures = 0U;
    std::size_t correspondences = 0U;
    std::size_t downweighted = 0U;

    std::size_t ground_correspondences = 0U;
    std::size_t ground_downweighted = 0U;

    double source_upload_ms = 0.0;

    double transform_kernel_ms = 0.0;
    double knn_kernel_ms = 0.0;
    double geometry_kernel_ms = 0.0;

    double total_ms = 0.0;

    int cuda_error = 0;
};


bool QueryPersistentFusedPlaneHessian(
    PersistentKnn5Handle handle,

    const float *source_xyz,
    std::size_t query_count,

    // Row-major 3x4 transform:
    //
    // [ r00 r01 r02 tx ]
    // [ r10 r11 r12 ty ]
    // [ r20 r21 r22 tz ]
    const double *T_target_source_3x4,

    float ambiguity_abs_epsilon,
    float ambiguity_relative_epsilon,

    double maximum_squared_distance,
    double maximum_plane_fit_error,
    double maximum_residual,
    double huber_delta,

    bool enable_ground_constraint,
    double ground_normal_cosine_threshold,
    double ground_min_below_sensor_m,
    double ground_max_residual,
    double ground_huber_delta,
    double ground_weight,

    double *output_H_6x6,
    double *output_b_6,

    double *output_H_ground_6x6,
    double *output_b_ground_6,

    // One entry per source point.
    // GPU-handled valid correspondence -> range_valid=1.
    // CPU fallback -> range_valid=0 here; CPU fills it later.
    double *output_ranges,
    unsigned char *output_range_valid,

    // 1 means this query must be recomputed by CPU KNN.
    unsigned char *output_cpu_fallback,

    double &output_raw_sse,
    double &output_robust_sse,
    double &output_robust_weight_sum,

    PersistentFusedStats &stats);


}  // namespace fr_slam_cuda
