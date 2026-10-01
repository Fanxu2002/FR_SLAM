#include "fr_slam/loop/loop_verifier_cuda.hpp"

namespace fr_slam_cuda
{
namespace
{
constexpr int kCudaUnavailable = -1;
}

bool TransformPointsBatch(
    const float *,
    std::size_t point_count,
    const float *,
    std::size_t transform_count,
    float *,
    BatchTransformStats &stats)
{
    stats = {};
    stats.point_count = point_count;
    stats.transform_count = transform_count;
    stats.cuda_error = kCudaUnavailable;
    return false;
}

bool BruteForceKnn5(
    const float *,
    std::size_t query_count,
    const float *,
    std::size_t target_count,
    int *,
    float *,
    Knn5Stats &stats)
{
    stats = {};
    stats.query_count = query_count;
    stats.target_count = target_count;
    stats.cuda_error = kCudaUnavailable;
    return false;
}

bool CreatePersistentKnn5(
    const float *,
    std::size_t,
    std::size_t,
    PersistentKnn5Handle &handle,
    double &setup_ms,
    int &cuda_error)
{
    handle = nullptr;
    setup_ms = 0.0;
    cuda_error = kCudaUnavailable;
    return false;
}

void DestroyPersistentKnn5(PersistentKnn5Handle)
{
}

bool QueryPersistentKnn5(
    PersistentKnn5Handle,
    const float *,
    std::size_t query_count,
    int *,
    float *,
    unsigned char *,
    float,
    float,
    PersistentKnn5Stats &stats)
{
    stats = {};
    stats.query_count = query_count;
    stats.cuda_error = kCudaUnavailable;
    return false;
}

bool ComputePlaneHessianShadowV1(
    PersistentKnn5Handle,
    const double *,
    const int *,
    const float *,
    const unsigned char *,
    std::size_t query_count,
    const double *,
    double,
    double,
    double,
    double,
    bool,
    double,
    double,
    double,
    double,
    double,
    double *,
    double *,
    double *,
    double *,
    double *,
    unsigned char *,
    double &output_raw_sse,
    double &output_robust_sse,
    double &output_robust_weight_sum,
    PlaneHessianShadowStats &stats)
{
    output_raw_sse = 0.0;
    output_robust_sse = 0.0;
    output_robust_weight_sum = 0.0;

    stats = {};
    stats.query_count = query_count;
    stats.cuda_error = kCudaUnavailable;
    return false;
}

bool QueryPersistentFusedPlaneHessian(
    PersistentKnn5Handle,
    const float *,
    std::size_t query_count,
    const double *,
    float,
    float,
    double,
    double,
    double,
    double,
    bool,
    double,
    double,
    double,
    double,
    double,
    double *,
    double *,
    double *,
    double *,
    double *,
    unsigned char *,
    unsigned char *,
    double &output_raw_sse,
    double &output_robust_sse,
    double &output_robust_weight_sum,
    PersistentFusedStats &stats)
{
    output_raw_sse = 0.0;
    output_robust_sse = 0.0;
    output_robust_weight_sum = 0.0;

    stats = {};
    stats.query_count = query_count;
    stats.cuda_error = kCudaUnavailable;
    return false;
}

}  // namespace fr_slam_cuda
