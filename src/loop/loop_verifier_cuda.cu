#include <iostream>
#include "fr_slam/loop/loop_verifier_cuda.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstddef>
#include <vector>

namespace
{

__global__ void TransformPointsBatchKernel(
    const float *source_xyz,
    const std::size_t point_count,
    const float *transforms,
    const std::size_t transform_count,
    float *output_xyz)
{
    const std::size_t linear_index =
        static_cast<std::size_t>(
            blockIdx.x) *
            static_cast<std::size_t>(
                blockDim.x) +
        static_cast<std::size_t>(
            threadIdx.x);

    const std::size_t total =
        point_count *
        transform_count;

    if (linear_index >= total)
    {
        return;
    }


    const std::size_t transform_index =
        linear_index /
        point_count;

    const std::size_t point_index =
        linear_index %
        point_count;


    const float *T =
        transforms +
        transform_index * 16U;

    const float *p =
        source_xyz +
        point_index * 3U;

    float *q =
        output_xyz +
        (
            transform_index *
                point_count +
            point_index
        ) *
            3U;


    const float x = p[0];
    const float y = p[1];
    const float z = p[2];


    q[0] =
        T[0] * x +
        T[1] * y +
        T[2] * z +
        T[3];

    q[1] =
        T[4] * x +
        T[5] * y +
        T[6] * z +
        T[7];

    q[2] =
        T[8] * x +
        T[9] * y +
        T[10] * z +
        T[11];
}

}  // namespace


namespace fr_slam_cuda
{

bool TransformPointsBatch(
    const float *source_xyz,
    const std::size_t point_count,
    const float *transforms_4x4_row_major,
    const std::size_t transform_count,
    float *output_xyz,
    BatchTransformStats &stats)
{
    stats = BatchTransformStats{};

    stats.point_count =
        point_count;

    stats.transform_count =
        transform_count;


    if (source_xyz == nullptr ||
        transforms_4x4_row_major == nullptr ||
        output_xyz == nullptr ||
        point_count == 0U ||
        transform_count == 0U)
    {
        return false;
    }


    const auto total_begin =
        std::chrono::steady_clock::now();


    const std::size_t source_bytes =
        point_count *
        3U *
        sizeof(float);

    const std::size_t transform_bytes =
        transform_count *
        16U *
        sizeof(float);

    const std::size_t output_bytes =
        point_count *
        transform_count *
        3U *
        sizeof(float);


    float *device_source = nullptr;
    float *device_transforms = nullptr;
    float *device_output = nullptr;


    auto fail =
        [&](const cudaError_t error)
        {
            stats.cuda_error =
                static_cast<int>(error);

            if (device_output != nullptr)
            {
                cudaFree(device_output);
            }

            if (device_transforms != nullptr)
            {
                cudaFree(device_transforms);
            }

            if (device_source != nullptr)
            {
                cudaFree(device_source);
            }

            stats.total_ms =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() -
                    total_begin)
                    .count();

            return false;
        };


    cudaError_t error =
        cudaMalloc(
            reinterpret_cast<void **>(
                &device_source),
            source_bytes);

    if (error != cudaSuccess)
    {
        return fail(error);
    }


    error =
        cudaMalloc(
            reinterpret_cast<void **>(
                &device_transforms),
            transform_bytes);

    if (error != cudaSuccess)
    {
        return fail(error);
    }


    error =
        cudaMalloc(
            reinterpret_cast<void **>(
                &device_output),
            output_bytes);

    if (error != cudaSuccess)
    {
        return fail(error);
    }


    error =
        cudaMemcpy(
            device_source,
            source_xyz,
            source_bytes,
            cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        return fail(error);
    }


    error =
        cudaMemcpy(
            device_transforms,
            transforms_4x4_row_major,
            transform_bytes,
            cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        return fail(error);
    }


    cudaEvent_t kernel_begin = nullptr;
    cudaEvent_t kernel_end = nullptr;

    error =
        cudaEventCreate(
            &kernel_begin);

    if (error != cudaSuccess)
    {
        return fail(error);
    }

    error =
        cudaEventCreate(
            &kernel_end);

    if (error != cudaSuccess)
    {
        cudaEventDestroy(
            kernel_begin);

        return fail(error);
    }


    constexpr int threads =
        256;

    const std::size_t total_items =
        point_count *
        transform_count;

    const int blocks =
        static_cast<int>(
            (
                total_items +
                static_cast<std::size_t>(
                    threads) -
                1U
            ) /
            static_cast<std::size_t>(
                threads));


    cudaEventRecord(
        kernel_begin);


    TransformPointsBatchKernel<<<
        blocks,
        threads>>>(
            device_source,
            point_count,
            device_transforms,
            transform_count,
            device_output);


    error =
        cudaGetLastError();

    if (error != cudaSuccess)
    {
        cudaEventDestroy(
            kernel_end);

        cudaEventDestroy(
            kernel_begin);

        return fail(error);
    }


    cudaEventRecord(
        kernel_end);

    error =
        cudaEventSynchronize(
            kernel_end);

    if (error != cudaSuccess)
    {
        cudaEventDestroy(
            kernel_end);

        cudaEventDestroy(
            kernel_begin);

        return fail(error);
    }


    float kernel_ms_float =
        0.0f;

    cudaEventElapsedTime(
        &kernel_ms_float,
        kernel_begin,
        kernel_end);

    stats.kernel_ms =
        static_cast<double>(
            kernel_ms_float);


    cudaEventDestroy(
        kernel_end);

    cudaEventDestroy(
        kernel_begin);


    error =
        cudaMemcpy(
            output_xyz,
            device_output,
            output_bytes,
            cudaMemcpyDeviceToHost);

    if (error != cudaSuccess)
    {
        return fail(error);
    }


    cudaFree(
        device_output);

    cudaFree(
        device_transforms);

    cudaFree(
        device_source);


    stats.total_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() -
            total_begin)
            .count();

    stats.ok = true;
    stats.cuda_error = 0;

    return true;
}

}  // namespace fr_slam_cuda


// ============================================================================
// GPU brute-force KNN K=5 shadow diagnostic.
// ============================================================================

namespace
{

constexpr int kCudaShadowKnnK = 5;


__global__ void BruteForceKnn5Kernel(
    const float *query_xyz,
    const std::size_t query_count,
    const float *target_xyz,
    const std::size_t target_count,
    int *output_indices,
    float *output_squared_distances)
{
    const std::size_t query_index =
        static_cast<std::size_t>(blockIdx.x) *
            static_cast<std::size_t>(blockDim.x) +
        static_cast<std::size_t>(threadIdx.x);

    if (query_index >= query_count)
    {
        return;
    }


    const float qx =
        query_xyz[query_index * 3U + 0U];

    const float qy =
        query_xyz[query_index * 3U + 1U];

    const float qz =
        query_xyz[query_index * 3U + 2U];


    float best_distance[kCudaShadowKnnK];
    int best_index[kCudaShadowKnnK];


#pragma unroll
    for (int k = 0;
         k < kCudaShadowKnnK;
         ++k)
    {
        best_distance[k] = 1.0e30f;
        best_index[k] = -1;
    }


    if (isfinite(qx) &&
        isfinite(qy) &&
        isfinite(qz))
    {
        for (std::size_t target_index = 0U;
             target_index < target_count;
             ++target_index)
        {
            const float tx =
                target_xyz[
                    target_index * 3U + 0U];

            const float ty =
                target_xyz[
                    target_index * 3U + 1U];

            const float tz =
                target_xyz[
                    target_index * 3U + 2U];


            if (!isfinite(tx) ||
                !isfinite(ty) ||
                !isfinite(tz))
            {
                continue;
            }


            const float dx =
                qx - tx;

            const float dy =
                qy - ty;

            const float dz =
                qz - tz;


            const float squared_distance =
                dx * dx +
                dy * dy +
                dz * dz;


            if (squared_distance >=
                best_distance[
                    kCudaShadowKnnK - 1])
            {
                continue;
            }


            int insert_position =
                kCudaShadowKnnK - 1;


            while (
                insert_position > 0 &&
                squared_distance <
                    best_distance[
                        insert_position - 1])
            {
                best_distance[
                    insert_position] =
                        best_distance[
                            insert_position - 1];

                best_index[
                    insert_position] =
                        best_index[
                            insert_position - 1];

                --insert_position;
            }


            best_distance[
                insert_position] =
                    squared_distance;

            best_index[
                insert_position] =
                    static_cast<int>(
                        target_index);
        }
    }


#pragma unroll
    for (int k = 0;
         k < kCudaShadowKnnK;
         ++k)
    {
        output_indices[
            query_index *
                static_cast<std::size_t>(
                    kCudaShadowKnnK) +
            static_cast<std::size_t>(k)] =
                best_index[k];

        output_squared_distances[
            query_index *
                static_cast<std::size_t>(
                    kCudaShadowKnnK) +
            static_cast<std::size_t>(k)] =
                best_distance[k];
    }
}

}  // namespace


namespace fr_slam_cuda
{

bool BruteForceKnn5(
    const float *query_xyz,
    const std::size_t query_count,
    const float *target_xyz,
    const std::size_t target_count,
    int *output_indices,
    float *output_squared_distances,
    Knn5Stats &stats)
{
    stats = Knn5Stats{};

    stats.query_count =
        query_count;

    stats.target_count =
        target_count;


    if (query_xyz == nullptr ||
        target_xyz == nullptr ||
        output_indices == nullptr ||
        output_squared_distances == nullptr ||
        query_count == 0U ||
        target_count < 5U)
    {
        return false;
    }


    const auto total_begin =
        std::chrono::steady_clock::now();


    const std::size_t query_bytes =
        query_count *
        3U *
        sizeof(float);

    const std::size_t target_bytes =
        target_count *
        3U *
        sizeof(float);

    const std::size_t result_count =
        query_count * 5U;

    const std::size_t index_bytes =
        result_count *
        sizeof(int);

    const std::size_t distance_bytes =
        result_count *
        sizeof(float);


    float *device_queries = nullptr;
    float *device_targets = nullptr;

    int *device_indices = nullptr;
    float *device_distances = nullptr;


    auto cleanup =
        [&]()
        {
            if (device_distances != nullptr)
            {
                cudaFree(device_distances);
            }

            if (device_indices != nullptr)
            {
                cudaFree(device_indices);
            }

            if (device_targets != nullptr)
            {
                cudaFree(device_targets);
            }

            if (device_queries != nullptr)
            {
                cudaFree(device_queries);
            }
        };


    auto fail =
        [&](const cudaError_t error)
        {
            stats.cuda_error =
                static_cast<int>(error);

            cleanup();

            stats.total_ms =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() -
                    total_begin)
                    .count();

            return false;
        };


    cudaError_t error =
        cudaMalloc(
            reinterpret_cast<void **>(
                &device_queries),
            query_bytes);

    if (error != cudaSuccess)
    {
        return fail(error);
    }


    error =
        cudaMalloc(
            reinterpret_cast<void **>(
                &device_targets),
            target_bytes);

    if (error != cudaSuccess)
    {
        return fail(error);
    }


    error =
        cudaMalloc(
            reinterpret_cast<void **>(
                &device_indices),
            index_bytes);

    if (error != cudaSuccess)
    {
        return fail(error);
    }


    error =
        cudaMalloc(
            reinterpret_cast<void **>(
                &device_distances),
            distance_bytes);

    if (error != cudaSuccess)
    {
        return fail(error);
    }


    error =
        cudaMemcpy(
            device_queries,
            query_xyz,
            query_bytes,
            cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        return fail(error);
    }


    error =
        cudaMemcpy(
            device_targets,
            target_xyz,
            target_bytes,
            cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        return fail(error);
    }


    cudaEvent_t kernel_begin = nullptr;
    cudaEvent_t kernel_end = nullptr;


    error =
        cudaEventCreate(
            &kernel_begin);

    if (error != cudaSuccess)
    {
        return fail(error);
    }


    error =
        cudaEventCreate(
            &kernel_end);

    if (error != cudaSuccess)
    {
        cudaEventDestroy(
            kernel_begin);

        return fail(error);
    }


    constexpr int threads =
        256;

    const int blocks =
        static_cast<int>(
            (
                query_count +
                static_cast<std::size_t>(
                    threads) -
                1U
            ) /
            static_cast<std::size_t>(
                threads));


    cudaEventRecord(
        kernel_begin);


    BruteForceKnn5Kernel<<<
        blocks,
        threads>>>(
            device_queries,
            query_count,
            device_targets,
            target_count,
            device_indices,
            device_distances);


    error =
        cudaGetLastError();

    if (error != cudaSuccess)
    {
        cudaEventDestroy(
            kernel_end);

        cudaEventDestroy(
            kernel_begin);

        return fail(error);
    }


    cudaEventRecord(
        kernel_end);

    error =
        cudaEventSynchronize(
            kernel_end);

    if (error != cudaSuccess)
    {
        cudaEventDestroy(
            kernel_end);

        cudaEventDestroy(
            kernel_begin);

        return fail(error);
    }


    float kernel_ms =
        0.0f;

    cudaEventElapsedTime(
        &kernel_ms,
        kernel_begin,
        kernel_end);


    stats.kernel_ms =
        static_cast<double>(
            kernel_ms);


    cudaEventDestroy(
        kernel_end);

    cudaEventDestroy(
        kernel_begin);


    error =
        cudaMemcpy(
            output_indices,
            device_indices,
            index_bytes,
            cudaMemcpyDeviceToHost);

    if (error != cudaSuccess)
    {
        return fail(error);
    }


    error =
        cudaMemcpy(
            output_squared_distances,
            device_distances,
            distance_bytes,
            cudaMemcpyDeviceToHost);

    if (error != cudaSuccess)
    {
        return fail(error);
    }


    cleanup();


    stats.total_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() -
            total_begin)
            .count();

    stats.ok = true;
    stats.cuda_error = 0;

    return true;
}

}  // namespace fr_slam_cuda


// ============================================================================
// GPU KNN PERSISTENT V2
//
// Target stays resident on GPU for the whole solver call.
// Kernel computes K=6:
//   first 5 -> formal candidate neighbours
//   6th     -> ambiguity detector for K5/K6 boundary.
//
// Diagnostic only.
// ============================================================================

namespace
{

struct PersistentKnn5ContextImpl
{
    std::size_t target_count = 0U;
    std::size_t max_query_count = 0U;

    float *device_target = nullptr;
    float *device_query = nullptr;

    int *device_indices = nullptr;
    float *device_distances = nullptr;
    unsigned char *device_ambiguous = nullptr;


    // ============================================================
    // GPU PERSISTENT FUSED V3
    // ============================================================

    float *device_fused_source = nullptr;

    // Double precision transformed point used by
    // plane/residual/Jacobian while device_query stays float
    // for the exact existing KNN path.
    double *device_fused_query_double = nullptr;

    unsigned char *device_fused_cpu_fallback = nullptr;

    double *device_fused_ranges = nullptr;
    unsigned char *device_fused_range_valid = nullptr;

    void *device_fused_accumulator = nullptr;

    bool fused_source_uploaded = false;
    std::size_t fused_source_count = 0U;

    cudaEvent_t fused_event_0 = nullptr;
    cudaEvent_t fused_event_1 = nullptr;
    cudaEvent_t fused_event_2 = nullptr;
    cudaEvent_t fused_event_3 = nullptr;

    // Dedicated non-blocking stream for this verifier context.
    // Different parallel verifier jobs therefore do not serialize
    // through the legacy default CUDA stream.
    cudaStream_t stream = nullptr;


    // ============================================================
    // V4 PRODUCTION DENSE GRID
    //
    // Target spatial index. Built once per persistent verifier
    // context. Production V3 remains unchanged.
    // ============================================================

    bool v4_grid_ready = false;

    float v4_voxel_size = 1.0f;

    float v4_min_x = 0.0f;
    float v4_min_y = 0.0f;
    float v4_min_z = 0.0f;

    int v4_nx = 0;
    int v4_ny = 0;
    int v4_nz = 0;

    std::size_t v4_cell_count = 0U;
    std::size_t v4_indexed_point_count = 0U;

    int *device_v4_cell_offsets = nullptr;
    int *device_v4_point_indices = nullptr;


};


constexpr int kPersistentOutputK = 5;
constexpr int kPersistentSearchK = 6;


// ================================================================
// V4 PRODUCTION DENSE GRID BUILDER
// Build a dense 1 m target voxel grid once per verifier context.
//
// Grid representation:
//
//   cell_offsets[cell_count + 1]
//   point_indices[indexed_point_count]
//
// Target indices are inserted in ascending original target index
// order. Therefore each cell also preserves ascending target index,
// which is useful for reproducing V3 equal-distance tie behaviour.
// ================================================================

bool BuildDenseGridV4(
    PersistentKnn5ContextImpl *context,
    const float *target_xyz)
{
    if (context == nullptr ||
        target_xyz == nullptr ||
        context->target_count < 6U)
    {
        return false;
    }


    constexpr float voxel_size = 1.0f;

    bool bbox_valid = false;

    float min_x = 0.0f;
    float min_y = 0.0f;
    float min_z = 0.0f;

    float max_x = 0.0f;
    float max_y = 0.0f;
    float max_z = 0.0f;

    std::size_t finite_count = 0U;


    // ------------------------------------------------------------
    // Pass 0: finite-point AABB
    // ------------------------------------------------------------

    for (std::size_t ti = 0U;
         ti < context->target_count;
         ++ti)
    {
        const float x =
            target_xyz[ti * 3U + 0U];

        const float y =
            target_xyz[ti * 3U + 1U];

        const float z =
            target_xyz[ti * 3U + 2U];


        if (!std::isfinite(
                static_cast<double>(x)) ||
            !std::isfinite(
                static_cast<double>(y)) ||
            !std::isfinite(
                static_cast<double>(z)))
        {
            continue;
        }


        ++finite_count;


        if (!bbox_valid)
        {
            min_x = max_x = x;
            min_y = max_y = y;
            min_z = max_z = z;

            bbox_valid = true;
        }
        else
        {
            min_x = std::min(min_x, x);
            min_y = std::min(min_y, y);
            min_z = std::min(min_z, z);

            max_x = std::max(max_x, x);
            max_y = std::max(max_y, y);
            max_z = std::max(max_z, z);
        }
    }


    if (!bbox_valid ||
        finite_count < 6U)
    {
        return false;
    }


    const int nx =
        static_cast<int>(
            std::floor(
                (max_x - min_x) /
                voxel_size)) +
        1;

    const int ny =
        static_cast<int>(
            std::floor(
                (max_y - min_y) /
                voxel_size)) +
        1;

    const int nz =
        static_cast<int>(
            std::floor(
                (max_z - min_z) /
                voxel_size)) +
        1;


    if (nx <= 0 ||
        ny <= 0 ||
        nz <= 0)
    {
        return false;
    }


    const std::size_t cell_count =
        static_cast<std::size_t>(nx) *
        static_cast<std::size_t>(ny) *
        static_cast<std::size_t>(nz);


    // Safety guard. Current dataset is ~30k cells at 1 m.
    constexpr std::size_t kMaxDenseCellsV4 =
        1000000U;


    if (cell_count == 0U ||
        cell_count >
            kMaxDenseCellsV4)
    {
        return false;
    }


    std::vector<int> cell_counts(
        cell_count,
        0);

    std::vector<int> target_cell_id(
        context->target_count,
        -1);


    auto coordinate_to_cell =
        [](
            const float value,
            const float origin,
            const int dimension)
        {
            int index =
                static_cast<int>(
                    std::floor(
                        value -
                        origin));

            if (index < 0)
            {
                index = 0;
            }

            if (index >= dimension)
            {
                index =
                    dimension - 1;
            }

            return index;
        };


    // ------------------------------------------------------------
    // Pass 1: cell counts
    // ------------------------------------------------------------

    for (std::size_t ti = 0U;
         ti < context->target_count;
         ++ti)
    {
        const float x =
            target_xyz[ti * 3U + 0U];

        const float y =
            target_xyz[ti * 3U + 1U];

        const float z =
            target_xyz[ti * 3U + 2U];


        if (!std::isfinite(
                static_cast<double>(x)) ||
            !std::isfinite(
                static_cast<double>(y)) ||
            !std::isfinite(
                static_cast<double>(z)))
        {
            continue;
        }


        const int ix =
            coordinate_to_cell(
                x,
                min_x,
                nx);

        const int iy =
            coordinate_to_cell(
                y,
                min_y,
                ny);

        const int iz =
            coordinate_to_cell(
                z,
                min_z,
                nz);


        const std::size_t cell =
            (
                static_cast<std::size_t>(iz) *
                    static_cast<std::size_t>(ny) +
                static_cast<std::size_t>(iy)
            ) *
                static_cast<std::size_t>(nx) +
            static_cast<std::size_t>(ix);


        target_cell_id[ti] =
            static_cast<int>(cell);

        ++cell_counts[cell];
    }


    // ------------------------------------------------------------
    // Prefix sum
    // ------------------------------------------------------------

    std::vector<int> cell_offsets(
        cell_count + 1U,
        0);


    for (std::size_t cell = 0U;
         cell < cell_count;
         ++cell)
    {
        cell_offsets[cell + 1U] =
            cell_offsets[cell] +
            cell_counts[cell];
    }


    if (static_cast<std::size_t>(
            cell_offsets[cell_count]) !=
        finite_count)
    {
        return false;
    }


    std::vector<int> point_indices(
        finite_count,
        -1);

    std::vector<int> cursors(
        cell_offsets.begin(),
        cell_offsets.end() - 1);


    // ------------------------------------------------------------
    // Pass 2: ascending original target index insertion.
    // ------------------------------------------------------------

    for (std::size_t ti = 0U;
         ti < context->target_count;
         ++ti)
    {
        const int cell =
            target_cell_id[ti];


        if (cell < 0)
        {
            continue;
        }


        const int dst =
            cursors[
                static_cast<std::size_t>(
                    cell)]++;


        point_indices[
            static_cast<std::size_t>(
                dst)] =
            static_cast<int>(ti);
    }


    // ------------------------------------------------------------
    // GPU storage.
    // Grid-build failure must NOT break production; V3 fallback remains available.
    // ------------------------------------------------------------

    int *device_offsets = nullptr;
    int *device_points = nullptr;


    cudaError_t error =
        cudaMalloc(
            reinterpret_cast<void **>(
                &device_offsets),
            (cell_count + 1U) *
                sizeof(int));


    if (error != cudaSuccess)
    {
        return false;
    }


    error =
        cudaMalloc(
            reinterpret_cast<void **>(
                &device_points),
            finite_count *
                sizeof(int));


    if (error != cudaSuccess)
    {
        cudaFree(device_offsets);
        return false;
    }


    error =
        cudaMemcpy(
            device_offsets,
            cell_offsets.data(),
            (cell_count + 1U) *
                sizeof(int),
            cudaMemcpyHostToDevice);


    if (error != cudaSuccess)
    {
        cudaFree(device_points);
        cudaFree(device_offsets);
        return false;
    }


    error =
        cudaMemcpy(
            device_points,
            point_indices.data(),
            finite_count *
                sizeof(int),
            cudaMemcpyHostToDevice);


    if (error != cudaSuccess)
    {
        cudaFree(device_points);
        cudaFree(device_offsets);
        return false;
    }


    context->v4_voxel_size =
        voxel_size;

    context->v4_min_x = min_x;
    context->v4_min_y = min_y;
    context->v4_min_z = min_z;

    context->v4_nx = nx;
    context->v4_ny = ny;
    context->v4_nz = nz;

    context->v4_cell_count =
        cell_count;

    context->v4_indexed_point_count =
        finite_count;

    context->device_v4_cell_offsets =
        device_offsets;

    context->device_v4_point_indices =
        device_points;

    context->v4_grid_ready = true;


    return true;
}


// ================================================================
// V4 PRODUCTION EXACT DENSE-GRID K=6
//
// Important:
//   * Production exact spatial KNN.
//   * Uses the same float squared-distance arithmetic as V3.
//   * Final ordering is lexicographic:
//
//         (squared_distance, target_index)
//
//     which reproduces V3's ascending-ti tie behaviour.
//   * Searches voxel shells until K6 is mathematically proven exact.
// ================================================================

__device__ __forceinline__
bool V4KnnPairLess(
    const float candidate_distance,
    const int candidate_index,
    const float reference_distance,
    const int reference_index)
{
    if (candidate_distance <
        reference_distance)
    {
        return true;
    }

    if (candidate_distance >
        reference_distance)
    {
        return false;
    }


    // V3 scans target indices in ascending order and uses strict
    // distance comparison. Therefore equal-distance ties retain
    // the smaller original target index.
    if (reference_index < 0)
    {
        return false;
    }


    return candidate_index <
        reference_index;
}


__device__ __forceinline__
void V4InsertExactK6(
    const float distance,
    const int target_index,
    float *best_d,
    int *best_i)
{
    if (target_index < 0 ||
        !isfinite(distance))
    {
        return;
    }


    if (!V4KnnPairLess(
            distance,
            target_index,
            best_d[
                kPersistentSearchK - 1],
            best_i[
                kPersistentSearchK - 1]))
    {
        return;
    }


    int insert_pos =
        kPersistentSearchK - 1;


    while (
        insert_pos > 0 &&
        V4KnnPairLess(
            distance,
            target_index,
            best_d[insert_pos - 1],
            best_i[insert_pos - 1]))
    {
        best_d[insert_pos] =
            best_d[insert_pos - 1];

        best_i[insert_pos] =
            best_i[insert_pos - 1];

        --insert_pos;
    }


    best_d[insert_pos] =
        distance;

    best_i[insert_pos] =
        target_index;
}


__device__ __forceinline__
float V4SquarePositiveDistance(
    const float value)
{
    const float positive =
        value > 0.0f
            ? value
            : 0.0f;

    return positive *
        positive;
}


__global__ void
PersistentKnn6KernelV4Exact(
    const float *query_xyz,
    const std::size_t query_count,

    const float *target_xyz,
    const std::size_t target_count,

    const int *cell_offsets,
    const int *point_indices,

    const float grid_min_x,
    const float grid_min_y,
    const float grid_min_z,

    const float voxel_size,

    const int nx,
    const int ny,
    const int nz,

    int *output_indices,
    float *output_distances,
    unsigned char *output_ambiguous,

    const float ambiguity_abs_epsilon,
    const float ambiguity_relative_epsilon)
{
    const std::size_t qi =
        static_cast<std::size_t>(
            blockIdx.x) *
            static_cast<std::size_t>(
                blockDim.x) +
        static_cast<std::size_t>(
            threadIdx.x);


    if (qi >= query_count)
    {
        return;
    }


    const float qx =
        query_xyz[
            qi * 3U + 0U];

    const float qy =
        query_xyz[
            qi * 3U + 1U];

    const float qz =
        query_xyz[
            qi * 3U + 2U];


    float best_d[
        kPersistentSearchK];

    int best_i[
        kPersistentSearchK];


#pragma unroll
    for (int k = 0;
         k < kPersistentSearchK;
         ++k)
    {
        best_d[k] =
            1.0e30f;

        best_i[k] =
            -1;
    }


    if (isfinite(qx) &&
        isfinite(qy) &&
        isfinite(qz) &&
        cell_offsets != nullptr &&
        point_indices != nullptr &&
        target_xyz != nullptr &&
        voxel_size > 0.0f &&
        nx > 0 &&
        ny > 0 &&
        nz > 0)
    {
        // --------------------------------------------------------
        // Query's logical voxel.
        //
        // Clamp only the starting cell to the finite target grid.
        // Query itself may lie outside the target AABB.
        // --------------------------------------------------------

        int cx =
            static_cast<int>(
                floorf(
                    (qx - grid_min_x) /
                    voxel_size));

        int cy =
            static_cast<int>(
                floorf(
                    (qy - grid_min_y) /
                    voxel_size));

        int cz =
            static_cast<int>(
                floorf(
                    (qz - grid_min_z) /
                    voxel_size));


        if (cx < 0)
        {
            cx = 0;
        }
        else if (cx >= nx)
        {
            cx = nx - 1;
        }


        if (cy < 0)
        {
            cy = 0;
        }
        else if (cy >= ny)
        {
            cy = ny - 1;
        }


        if (cz < 0)
        {
            cz = 0;
        }
        else if (cz >= nz)
        {
            cz = nz - 1;
        }


        const int rx =
            cx >
                (nx - 1 - cx)
                ? cx
                : (nx - 1 - cx);

        const int ry =
            cy >
                (ny - 1 - cy)
                ? cy
                : (ny - 1 - cy);

        const int rz =
            cz >
                (nz - 1 - cz)
                ? cz
                : (nz - 1 - cz);


        int max_shell = rx;

        if (ry > max_shell)
        {
            max_shell = ry;
        }

        if (rz > max_shell)
        {
            max_shell = rz;
        }


        // --------------------------------------------------------
        // Expand Chebyshev voxel shells:
        //
        // r=0: centre cell
        // r=1: outer surface of 3x3x3 cube
        // r=2: next outer shell
        // ...
        //
        // Every grid cell is visited exactly once.
        // --------------------------------------------------------

        for (int shell = 0;
             shell <= max_shell;
             ++shell)
        {
            int x0 = cx - shell;
            int x1 = cx + shell;

            int y0 = cy - shell;
            int y1 = cy + shell;

            int z0 = cz - shell;
            int z1 = cz + shell;


            if (x0 < 0)
            {
                x0 = 0;
            }

            if (x1 >= nx)
            {
                x1 = nx - 1;
            }


            if (y0 < 0)
            {
                y0 = 0;
            }

            if (y1 >= ny)
            {
                y1 = ny - 1;
            }


            if (z0 < 0)
            {
                z0 = 0;
            }

            if (z1 >= nz)
            {
                z1 = nz - 1;
            }


            for (int iz = z0;
                 iz <= z1;
                 ++iz)
            {
                for (int iy = y0;
                     iy <= y1;
                     ++iy)
                {
                    for (int ix = x0;
                         ix <= x1;
                         ++ix)
                    {
                        int dx_cell =
                            ix - cx;

                        if (dx_cell < 0)
                        {
                            dx_cell =
                                -dx_cell;
                        }


                        int dy_cell =
                            iy - cy;

                        if (dy_cell < 0)
                        {
                            dy_cell =
                                -dy_cell;
                        }


                        int dz_cell =
                            iz - cz;

                        if (dz_cell < 0)
                        {
                            dz_cell =
                                -dz_cell;
                        }


                        int cell_shell =
                            dx_cell;

                        if (dy_cell >
                            cell_shell)
                        {
                            cell_shell =
                                dy_cell;
                        }

                        if (dz_cell >
                            cell_shell)
                        {
                            cell_shell =
                                dz_cell;
                        }


                        // Inner cells were processed by earlier
                        // shells.
                        if (cell_shell !=
                            shell)
                        {
                            continue;
                        }


                        const std::size_t cell =
                            (
                                static_cast<
                                    std::size_t>(
                                        iz) *
                                    static_cast<
                                        std::size_t>(
                                            ny) +
                                static_cast<
                                    std::size_t>(
                                        iy)
                            ) *
                                static_cast<
                                    std::size_t>(
                                        nx) +
                            static_cast<
                                std::size_t>(
                                    ix);


                        const int begin =
                            cell_offsets[cell];

                        const int end =
                            cell_offsets[
                                cell + 1U];


                        for (int slot = begin;
                             slot < end;
                             ++slot)
                        {
                            const int ti =
                                point_indices[
                                    static_cast<
                                        std::size_t>(
                                            slot)];


                            if (ti < 0 ||
                                static_cast<
                                    std::size_t>(
                                        ti) >=
                                    target_count)
                            {
                                continue;
                            }


                            const float tx =
                                target_xyz[
                                    static_cast<
                                        std::size_t>(
                                            ti) *
                                        3U +
                                    0U];

                            const float ty =
                                target_xyz[
                                    static_cast<
                                        std::size_t>(
                                            ti) *
                                        3U +
                                    1U];

                            const float tz =
                                target_xyz[
                                    static_cast<
                                        std::size_t>(
                                            ti) *
                                        3U +
                                    2U];


                            if (!isfinite(tx) ||
                                !isfinite(ty) ||
                                !isfinite(tz))
                            {
                                continue;
                            }


                            // Same arithmetic and order as V3.
                            const float dx =
                                qx - tx;

                            const float dy =
                                qy - ty;

                            const float dz =
                                qz - tz;


                            const float d =
                                dx * dx +
                                dy * dy +
                                dz * dz;


                            V4InsertExactK6(
                                d,
                                ti,
                                best_d,
                                best_i);
                        }
                    }
                }
            }


            // ----------------------------------------------------
            // Exact early-stop proof.
            //
            // After this shell, [x0..x1] x [y0..y1] x
            // [z0..z1] has been exhaustively searched.
            //
            // Compute a CONSERVATIVE lower bound on the squared
            // distance to ANY still-unvisited grid cell.
            //
            // If d6 is strictly below this bound, no unvisited
            // target can replace K1...K6.
            //
            // Strict '<' is intentional: equality could contain
            // an equal-distance target with a smaller target index.
            // ----------------------------------------------------

            if (best_i[
                    kPersistentSearchK -
                    1] >= 0)
            {
                const bool all_x =
                    x0 == 0 &&
                    x1 == nx - 1;

                const bool all_y =
                    y0 == 0 &&
                    y1 == ny - 1;

                const bool all_z =
                    z0 == 0 &&
                    z1 == nz - 1;


                if (all_x &&
                    all_y &&
                    all_z)
                {
                    break;
                }


                float lower_bound_sq =
                    1.0e30f;


                // Unvisited cells left of searched box.
                if (x0 > 0)
                {
                    const float boundary =
                        grid_min_x +
                        static_cast<float>(
                            x0) *
                            voxel_size;

                    const float d2 =
                        V4SquarePositiveDistance(
                            qx -
                            boundary);

                    if (d2 <
                        lower_bound_sq)
                    {
                        lower_bound_sq =
                            d2;
                    }
                }


                // Unvisited cells right of searched box.
                if (x1 + 1 < nx)
                {
                    const float boundary =
                        grid_min_x +
                        static_cast<float>(
                            x1 + 1) *
                            voxel_size;

                    const float d2 =
                        V4SquarePositiveDistance(
                            boundary -
                            qx);

                    if (d2 <
                        lower_bound_sq)
                    {
                        lower_bound_sq =
                            d2;
                    }
                }


                if (y0 > 0)
                {
                    const float boundary =
                        grid_min_y +
                        static_cast<float>(
                            y0) *
                            voxel_size;

                    const float d2 =
                        V4SquarePositiveDistance(
                            qy -
                            boundary);

                    if (d2 <
                        lower_bound_sq)
                    {
                        lower_bound_sq =
                            d2;
                    }
                }


                if (y1 + 1 < ny)
                {
                    const float boundary =
                        grid_min_y +
                        static_cast<float>(
                            y1 + 1) *
                            voxel_size;

                    const float d2 =
                        V4SquarePositiveDistance(
                            boundary -
                            qy);

                    if (d2 <
                        lower_bound_sq)
                    {
                        lower_bound_sq =
                            d2;
                    }
                }


                if (z0 > 0)
                {
                    const float boundary =
                        grid_min_z +
                        static_cast<float>(
                            z0) *
                            voxel_size;

                    const float d2 =
                        V4SquarePositiveDistance(
                            qz -
                            boundary);

                    if (d2 <
                        lower_bound_sq)
                    {
                        lower_bound_sq =
                            d2;
                    }
                }


                if (z1 + 1 < nz)
                {
                    const float boundary =
                        grid_min_z +
                        static_cast<float>(
                            z1 + 1) *
                            voxel_size;

                    const float d2 =
                        V4SquarePositiveDistance(
                            boundary -
                            qz);

                    if (d2 <
                        lower_bound_sq)
                    {
                        lower_bound_sq =
                            d2;
                    }
                }


                if (best_d[
                        kPersistentSearchK -
                        1] <
                    lower_bound_sq)
                {
                    break;
                }
            }
        }
    }


    // ------------------------------------------------------------
    // Public V3-compatible output:
    // only K1...K5 are exported.
    // K6 remains local and is used for ambiguity.
    // ------------------------------------------------------------

#pragma unroll
    for (int k = 0;
         k < kPersistentOutputK;
         ++k)
    {
        const std::size_t offset =
            qi *
                static_cast<
                    std::size_t>(
                        kPersistentOutputK) +
            static_cast<
                std::size_t>(
                    k);


        output_indices[offset] =
            best_i[k];

        output_distances[offset] =
            best_d[k];
    }


    unsigned char ambiguous =
        0U;


    if (best_i[4] >= 0 &&
        best_i[5] >= 0 &&
        isfinite(best_d[4]) &&
        isfinite(best_d[5]))
    {
        const float gap =
            best_d[5] -
            best_d[4];


        const float scale =
            fmaxf(
                fabsf(
                    best_d[4]),
                1.0e-3f);


        const float threshold =
            ambiguity_abs_epsilon +
            ambiguity_relative_epsilon *
                scale;


        if (gap <= threshold)
        {
            ambiguous =
                1U;
        }
    }


    output_ambiguous[qi] =
        ambiguous;
}


// ================================================================
// Existing V3 production brute-force K=6.
// ================================================================

__global__ void PersistentKnn6Kernel(
    const float *query_xyz,
    const std::size_t query_count,
    const float *target_xyz,
    const std::size_t target_count,
    int *output_indices,
    float *output_distances,
    unsigned char *output_ambiguous,
    const float ambiguity_abs_epsilon,
    const float ambiguity_relative_epsilon)
{
    const std::size_t qi =
        static_cast<std::size_t>(blockIdx.x) *
            static_cast<std::size_t>(blockDim.x) +
        static_cast<std::size_t>(threadIdx.x);

    if (qi >= query_count)
    {
        return;
    }


    const float qx = query_xyz[qi * 3U + 0U];
    const float qy = query_xyz[qi * 3U + 1U];
    const float qz = query_xyz[qi * 3U + 2U];


    float best_d[kPersistentSearchK];
    int best_i[kPersistentSearchK];


#pragma unroll
    for (int k = 0;
         k < kPersistentSearchK;
         ++k)
    {
        best_d[k] = 1.0e30f;
        best_i[k] = -1;
    }


    if (isfinite(qx) &&
        isfinite(qy) &&
        isfinite(qz))
    {
        for (std::size_t ti = 0U;
             ti < target_count;
             ++ti)
        {
            const float tx =
                target_xyz[ti * 3U + 0U];

            const float ty =
                target_xyz[ti * 3U + 1U];

            const float tz =
                target_xyz[ti * 3U + 2U];


            if (!isfinite(tx) ||
                !isfinite(ty) ||
                !isfinite(tz))
            {
                continue;
            }


            const float dx = qx - tx;
            const float dy = qy - ty;
            const float dz = qz - tz;

            const float d =
                dx * dx +
                dy * dy +
                dz * dz;


            if (d >= best_d[kPersistentSearchK - 1])
            {
                continue;
            }


            int insert_pos =
                kPersistentSearchK - 1;


            while (
                insert_pos > 0 &&
                d < best_d[insert_pos - 1])
            {
                best_d[insert_pos] =
                    best_d[insert_pos - 1];

                best_i[insert_pos] =
                    best_i[insert_pos - 1];

                --insert_pos;
            }


            best_d[insert_pos] = d;
            best_i[insert_pos] =
                static_cast<int>(ti);
        }
    }


#pragma unroll
    for (int k = 0;
         k < kPersistentOutputK;
         ++k)
    {
        const std::size_t offset =
            qi *
                static_cast<std::size_t>(
                    kPersistentOutputK) +
            static_cast<std::size_t>(k);

        output_indices[offset] =
            best_i[k];

        output_distances[offset] =
            best_d[k];
    }


    unsigned char ambiguous = 0U;


    if (best_i[4] >= 0 &&
        best_i[5] >= 0 &&
        isfinite(best_d[4]) &&
        isfinite(best_d[5]))
    {
        const float gap =
            best_d[5] -
            best_d[4];

        const float scale =
            fmaxf(
                fabsf(best_d[4]),
                1.0e-3f);

        const float threshold =
            ambiguity_abs_epsilon +
            ambiguity_relative_epsilon *
                scale;


        if (gap <= threshold)
        {
            ambiguous = 1U;
        }
    }


    output_ambiguous[qi] =
        ambiguous;
}


void DestroyPersistentContextImpl(
    PersistentKnn5ContextImpl *context)
{
    if (context == nullptr)
    {
        return;
    }

    // ============================================================
    // FUSED V3 DESTROY
    // ============================================================

    // Ensure this verifier context has no outstanding GPU work
    // before destroying events and device buffers.
    if (context->stream != nullptr)
    {
        cudaStreamSynchronize(
            context->stream);
    }


    if (context->fused_event_3 != nullptr)
    {
        cudaEventDestroy(
            context->fused_event_3);
    }

    if (context->fused_event_2 != nullptr)
    {
        cudaEventDestroy(
            context->fused_event_2);
    }

    if (context->fused_event_1 != nullptr)
    {
        cudaEventDestroy(
            context->fused_event_1);
    }

    if (context->fused_event_0 != nullptr)
    {
        cudaEventDestroy(
            context->fused_event_0);
    }


    if (context->device_fused_accumulator != nullptr)
    {
        cudaFree(
            context->device_fused_accumulator);
    }

    if (context->device_fused_range_valid != nullptr)
    {
        cudaFree(
            context->device_fused_range_valid);
    }

    if (context->device_fused_ranges != nullptr)
    {
        cudaFree(
            context->device_fused_ranges);
    }

    if (context->device_fused_cpu_fallback != nullptr)
    {
        cudaFree(
            context->device_fused_cpu_fallback);
    }

    if (context->device_fused_query_double != nullptr)
    {
        cudaFree(
            context->device_fused_query_double);
    }

    if (context->device_fused_source != nullptr)
    {
        cudaFree(
            context->device_fused_source);
    }


    if (context->device_v4_point_indices != nullptr)
    {
        cudaFree(
            context->device_v4_point_indices);

        context->device_v4_point_indices =
            nullptr;
    }


    if (context->device_v4_cell_offsets != nullptr)
    {
        cudaFree(
            context->device_v4_cell_offsets);

        context->device_v4_cell_offsets =
            nullptr;
    }


    if (context->device_ambiguous != nullptr)
    {
        cudaFree(
            context->device_ambiguous);
    }

    if (context->device_distances != nullptr)
    {
        cudaFree(
            context->device_distances);
    }

    if (context->device_indices != nullptr)
    {
        cudaFree(
            context->device_indices);
    }

    if (context->device_query != nullptr)
    {
        cudaFree(
            context->device_query);
    }

    if (context->device_target != nullptr)
    {
        cudaFree(
            context->device_target);
    }

    if (context->stream != nullptr)
    {
        cudaStreamDestroy(
            context->stream);

        context->stream = nullptr;
    }


    delete context;
}

}  // namespace


namespace fr_slam_cuda
{

bool CreatePersistentKnn5(
    const float *target_xyz,
    const std::size_t target_count,
    const std::size_t max_query_count,
    PersistentKnn5Handle &handle,
    double &setup_ms,
    int &cuda_error)
{
    handle = nullptr;
    setup_ms = 0.0;
    cuda_error = 0;


    if (target_xyz == nullptr ||
        target_count < 6U ||
        max_query_count == 0U)
    {
        return false;
    }


    const auto begin =
        std::chrono::steady_clock::now();


    PersistentKnn5ContextImpl *context =
        new PersistentKnn5ContextImpl();


    const cudaError_t stream_error =
        cudaStreamCreateWithFlags(
            &context->stream,
            cudaStreamNonBlocking);


    if (stream_error != cudaSuccess)
    {
        cuda_error =
            static_cast<int>(
                stream_error);

        delete context;

        setup_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() -
                begin)
                .count();

        return false;
    }


    context->target_count =
        target_count;

    context->max_query_count =
        max_query_count;


    auto fail =
        [&](const cudaError_t error)
        {
            cuda_error =
                static_cast<int>(error);

            DestroyPersistentContextImpl(
                context);

            setup_ms =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() -
                    begin)
                    .count();

            return false;
        };


    cudaError_t error =
        cudaMalloc(
            reinterpret_cast<void **>(
                &context->device_target),
            target_count *
                3U *
                sizeof(float));

    if (error != cudaSuccess)
    {
        return fail(error);
    }


    error =
        cudaMalloc(
            reinterpret_cast<void **>(
                &context->device_query),
            max_query_count *
                3U *
                sizeof(float));

    if (error != cudaSuccess)
    {
        return fail(error);
    }


    error =
        cudaMalloc(
            reinterpret_cast<void **>(
                &context->device_indices),
            max_query_count *
                static_cast<std::size_t>(
                    kPersistentOutputK) *
                sizeof(int));

    if (error != cudaSuccess)
    {
        return fail(error);
    }


    error =
        cudaMalloc(
            reinterpret_cast<void **>(
                &context->device_distances),
            max_query_count *
                static_cast<std::size_t>(
                    kPersistentOutputK) *
                sizeof(float));

    if (error != cudaSuccess)
    {
        return fail(error);
    }


    error =
        cudaMalloc(
            reinterpret_cast<void **>(
                &context->device_ambiguous),
            max_query_count *
                sizeof(unsigned char));

    if (error != cudaSuccess)
    {
        return fail(error);
    }


    // Target upload ONCE.
    error =
        cudaMemcpy(
            context->device_target,
            target_xyz,
            target_count *
                3U *
                sizeof(float),
            cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        return fail(error);
    }


    const auto v4_grid_begin =
        std::chrono::steady_clock::now();


    const bool v4_grid_ok =
        BuildDenseGridV4(
            context,
            target_xyz);


    const double v4_grid_build_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() -
            v4_grid_begin)
            .count();


    std::cout
        << "V4_GRID_BUILD"
        << " | ready="
        << (v4_grid_ok ? 1 : 0)
        << " | target_points="
        << target_count
        << " | indexed_points="
        << context->v4_indexed_point_count
        << " | voxel_size="
        << context->v4_voxel_size
        << " | nx="
        << context->v4_nx
        << " | ny="
        << context->v4_ny
        << " | nz="
        << context->v4_nz
        << " | cells="
        << context->v4_cell_count
        << " | build_ms="
        << v4_grid_build_ms
        << std::endl;


    setup_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() -
            begin)
            .count();


    handle =
        reinterpret_cast<
            PersistentKnn5Handle>(
                context);

    return true;
}


void DestroyPersistentKnn5(
    PersistentKnn5Handle handle)
{
    DestroyPersistentContextImpl(
        reinterpret_cast<
            PersistentKnn5ContextImpl *>(
                handle));
}


bool QueryPersistentKnn5(
    PersistentKnn5Handle handle,
    const float *query_xyz,
    const std::size_t query_count,
    int *output_indices,
    float *output_squared_distances,
    unsigned char *output_ambiguous,
    const float ambiguity_abs_epsilon,
    const float ambiguity_relative_epsilon,
    PersistentKnn5Stats &stats)
{
    stats = PersistentKnn5Stats{};


    PersistentKnn5ContextImpl *context =
        reinterpret_cast<
            PersistentKnn5ContextImpl *>(
                handle);


    if (context == nullptr ||
        query_xyz == nullptr ||
        output_indices == nullptr ||
        output_squared_distances == nullptr ||
        output_ambiguous == nullptr ||
        query_count == 0U ||
        query_count >
            context->max_query_count)
    {
        return false;
    }


    stats.query_count =
        query_count;

    stats.target_count =
        context->target_count;


    const auto total_begin =
        std::chrono::steady_clock::now();


    cudaError_t error =
        cudaMemcpy(
            context->device_query,
            query_xyz,
            query_count *
                3U *
                sizeof(float),
            cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        stats.cuda_error =
            static_cast<int>(error);

        return false;
    }


    cudaEvent_t begin_event = nullptr;
    cudaEvent_t end_event = nullptr;


    error =
        cudaEventCreate(
            &begin_event);

    if (error != cudaSuccess)
    {
        stats.cuda_error =
            static_cast<int>(error);

        return false;
    }


    error =
        cudaEventCreate(
            &end_event);

    if (error != cudaSuccess)
    {
        stats.cuda_error =
            static_cast<int>(error);

        cudaEventDestroy(
            begin_event);

        return false;
    }


    constexpr int threads = 256;

    const int blocks =
        static_cast<int>(
            (
                query_count +
                static_cast<std::size_t>(
                    threads) -
                1U
            ) /
            static_cast<std::size_t>(
                threads));


    cudaEventRecord(
        begin_event);


    PersistentKnn6Kernel<<<
        blocks,
        threads>>>(
            context->device_query,
            query_count,
            context->device_target,
            context->target_count,
            context->device_indices,
            context->device_distances,
            context->device_ambiguous,
            ambiguity_abs_epsilon,
            ambiguity_relative_epsilon);


    error =
        cudaGetLastError();

    if (error != cudaSuccess)
    {
        stats.cuda_error =
            static_cast<int>(error);

        cudaEventDestroy(
            end_event);

        cudaEventDestroy(
            begin_event);

        return false;
    }


    cudaEventRecord(
        end_event);

    error =
        cudaEventSynchronize(
            end_event);

    if (error != cudaSuccess)
    {
        stats.cuda_error =
            static_cast<int>(error);

        cudaEventDestroy(
            end_event);

        cudaEventDestroy(
            begin_event);

        return false;
    }


    float kernel_ms = 0.0f;

    cudaEventElapsedTime(
        &kernel_ms,
        begin_event,
        end_event);

    stats.kernel_ms =
        static_cast<double>(
            kernel_ms);


    cudaEventDestroy(
        end_event);

    cudaEventDestroy(
        begin_event);


    const std::size_t output_count =
        query_count *
        static_cast<std::size_t>(
            kPersistentOutputK);


    error =
        cudaMemcpy(
            output_indices,
            context->device_indices,
            output_count *
                sizeof(int),
            cudaMemcpyDeviceToHost);

    if (error != cudaSuccess)
    {
        stats.cuda_error =
            static_cast<int>(error);

        return false;
    }


    error =
        cudaMemcpy(
            output_squared_distances,
            context->device_distances,
            output_count *
                sizeof(float),
            cudaMemcpyDeviceToHost);

    if (error != cudaSuccess)
    {
        stats.cuda_error =
            static_cast<int>(error);

        return false;
    }


    error =
        cudaMemcpy(
            output_ambiguous,
            context->device_ambiguous,
            query_count *
                sizeof(unsigned char),
            cudaMemcpyDeviceToHost);

    if (error != cudaSuccess)
    {
        stats.cuda_error =
            static_cast<int>(error);

        return false;
    }


    for (std::size_t i = 0U;
         i < query_count;
         ++i)
    {
        if (output_ambiguous[i] != 0U)
        {
            ++stats.ambiguous_queries;
        }
    }


    stats.total_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() -
            total_begin)
            .count();

    stats.cuda_error = 0;
    stats.ok = true;

    return true;
}

}  // namespace fr_slam_cuda


// ============================================================================
// GPU PLANE / HESSIAN SHADOW V1
//
// Input neighbour sets are exactly the non-ambiguous neighbour sets already
// used by GPU KNN Active V3.
//
// GPU:
//   5-point centroid
//   covariance
//   smallest-eigenvector plane normal
//   plane validity
//   point-to-plane residual
//   Huber
//   J = [lever x normal, normal]
//   per-correspondence H / b
//
// Host performs deterministic query-order reduction for V1 validation.
// Diagnostic only.
// ============================================================================

namespace
{

struct PlaneHessianContributionV1
{
    double h_upper[21];
    double b[6];

    double J[6];
    double residual;
    double range;
    double normal_z;
    double lever_z;

    double raw_sse;
    double robust_sse;
    double robust_weight;

    unsigned char status;
    unsigned char downweighted;
};


__device__ void JacobiRotate3x3V1(
    double A[3][3],
    double V[3][3],
    const int p,
    const int q)
{
    const double apq = A[p][q];

    if (fabs(apq) < 1.0e-18)
    {
        return;
    }

    const double app = A[p][p];
    const double aqq = A[q][q];

    const double tau =
        (aqq - app) /
        (2.0 * apq);

    const double t =
        copysign(
            1.0 /
                (
                    fabs(tau) +
                    sqrt(1.0 + tau * tau)
                ),
            tau);

    const double c =
        1.0 /
        sqrt(1.0 + t * t);

    const double s =
        t * c;


    A[p][p] =
        app -
        t * apq;

    A[q][q] =
        aqq +
        t * apq;

    A[p][q] = 0.0;
    A[q][p] = 0.0;


    for (int r = 0;
         r < 3;
         ++r)
    {
        if (r == p ||
            r == q)
        {
            continue;
        }

        const double arp = A[r][p];
        const double arq = A[r][q];

        const double new_rp =
            c * arp -
            s * arq;

        const double new_rq =
            s * arp +
            c * arq;

        A[r][p] = new_rp;
        A[p][r] = new_rp;

        A[r][q] = new_rq;
        A[q][r] = new_rq;
    }


    for (int r = 0;
         r < 3;
         ++r)
    {
        const double vrp = V[r][p];
        const double vrq = V[r][q];

        V[r][p] =
            c * vrp -
            s * vrq;

        V[r][q] =
            s * vrp +
            c * vrq;
    }
}


__device__ bool SmallestEigenvector3x3V1(
    const double covariance[3][3],
    double normal[3])
{
    double A[3][3];

    double V[3][3] =
    {
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0}
    };


    for (int r = 0;
         r < 3;
         ++r)
    {
        for (int c = 0;
             c < 3;
             ++c)
        {
            A[r][c] =
                covariance[r][c];
        }
    }


    // Small symmetric 3x3. Ten sweeps are deliberately conservative
    // for the first diagnostic implementation.
    for (int sweep = 0;
         sweep < 10;
         ++sweep)
    {
        JacobiRotate3x3V1(
            A,
            V,
            0,
            1);

        JacobiRotate3x3V1(
            A,
            V,
            0,
            2);

        JacobiRotate3x3V1(
            A,
            V,
            1,
            2);
    }


    int minimum_index = 0;

    if (A[1][1] <
        A[minimum_index][minimum_index])
    {
        minimum_index = 1;
    }

    if (A[2][2] <
        A[minimum_index][minimum_index])
    {
        minimum_index = 2;
    }


    normal[0] =
        V[0][minimum_index];

    normal[1] =
        V[1][minimum_index];

    normal[2] =
        V[2][minimum_index];


    const double norm =
        sqrt(
            normal[0] * normal[0] +
            normal[1] * normal[1] +
            normal[2] * normal[2]);


    if (!isfinite(norm) ||
        norm < 1.0e-12)
    {
        return false;
    }


    normal[0] /= norm;
    normal[1] /= norm;
    normal[2] /= norm;

    return true;
}


__device__ bool SmallestEigenvector3x3V2Fast(
    const double covariance[3][3],
    double normal[3],
    bool &used_v1_fallback)
{
    used_v1_fallback = false;


    const double a00 = covariance[0][0];
    const double a01 = covariance[0][1];
    const double a02 = covariance[0][2];

    const double a11 = covariance[1][1];
    const double a12 = covariance[1][2];

    const double a22 = covariance[2][2];


    if (!isfinite(a00) ||
        !isfinite(a01) ||
        !isfinite(a02) ||
        !isfinite(a11) ||
        !isfinite(a12) ||
        !isfinite(a22))
    {
        used_v1_fallback = true;

        return SmallestEigenvector3x3V1(
            covariance,
            normal);
    }


    // ------------------------------------------------------------
    // Analytic eigenvalues for a real symmetric 3x3 matrix.
    // ------------------------------------------------------------

    const double q =
        (a00 + a11 + a22) /
        3.0;


    const double b00 = a00 - q;
    const double b11 = a11 - q;
    const double b22 = a22 - q;


    const double p2 =
        b00 * b00 +
        b11 * b11 +
        b22 * b22 +
        2.0 *
            (
                a01 * a01 +
                a02 * a02 +
                a12 * a12
            );


    if (!isfinite(p2) ||
        p2 <= 1.0e-30)
    {
        used_v1_fallback = true;

        return SmallestEigenvector3x3V1(
            covariance,
            normal);
    }


    const double p =
        sqrt(
            p2 /
            6.0);


    if (!isfinite(p) ||
        p <= 1.0e-15)
    {
        used_v1_fallback = true;

        return SmallestEigenvector3x3V1(
            covariance,
            normal);
    }


    const double inv_p =
        1.0 /
        p;


    const double c00 =
        b00 * inv_p;

    const double c01 =
        a01 * inv_p;

    const double c02 =
        a02 * inv_p;

    const double c11 =
        b11 * inv_p;

    const double c12 =
        a12 * inv_p;

    const double c22 =
        b22 * inv_p;


    const double det_c =
        c00 * c11 * c22 +
        2.0 *
            c01 * c02 * c12 -
        c00 * c12 * c12 -
        c11 * c02 * c02 -
        c22 * c01 * c01;


    double r =
        0.5 *
        det_c;


    if (!isfinite(r))
    {
        used_v1_fallback = true;

        return SmallestEigenvector3x3V1(
            covariance,
            normal);
    }


    if (r < -1.0)
    {
        r = -1.0;
    }
    else if (r > 1.0)
    {
        r = 1.0;
    }


    constexpr double kTwoPiOverThree =
        2.094395102393195492308428922186335;


    const double phi =
        acos(r) /
        3.0;


    const double lambda_min =
        q +
        2.0 *
            p *
            cos(
                phi +
                kTwoPiOverThree);


    if (!isfinite(lambda_min))
    {
        used_v1_fallback = true;

        return SmallestEigenvector3x3V1(
            covariance,
            normal);
    }


    // ------------------------------------------------------------
    // Null space of A - lambda_min I.
    //
    // Take all three row cross products and choose the one with
    // largest norm for numerical stability.
    // ------------------------------------------------------------

    const double m00 =
        a00 -
        lambda_min;

    const double m01 = a01;
    const double m02 = a02;

    const double m10 = a01;

    const double m11 =
        a11 -
        lambda_min;

    const double m12 = a12;

    const double m20 = a02;
    const double m21 = a12;

    const double m22 =
        a22 -
        lambda_min;


    // row0 x row1
    const double c0x =
        m01 * m12 -
        m02 * m11;

    const double c0y =
        m02 * m10 -
        m00 * m12;

    const double c0z =
        m00 * m11 -
        m01 * m10;


    // row0 x row2
    const double c1x =
        m01 * m22 -
        m02 * m21;

    const double c1y =
        m02 * m20 -
        m00 * m22;

    const double c1z =
        m00 * m21 -
        m01 * m20;


    // row1 x row2
    const double c2x =
        m11 * m22 -
        m12 * m21;

    const double c2y =
        m12 * m20 -
        m10 * m22;

    const double c2z =
        m10 * m21 -
        m11 * m20;


    const double n0 =
        c0x * c0x +
        c0y * c0y +
        c0z * c0z;

    const double n1 =
        c1x * c1x +
        c1y * c1y +
        c1z * c1z;

    const double n2 =
        c2x * c2x +
        c2y * c2y +
        c2z * c2z;


    double nx = c0x;
    double ny = c0y;
    double nz = c0z;

    double best_norm2 = n0;


    if (n1 > best_norm2)
    {
        best_norm2 = n1;

        nx = c1x;
        ny = c1y;
        nz = c1z;
    }


    if (n2 > best_norm2)
    {
        best_norm2 = n2;

        nx = c2x;
        ny = c2y;
        nz = c2z;
    }


    if (!isfinite(best_norm2) ||
        best_norm2 <= 1.0e-30)
    {
        used_v1_fallback = true;

        return SmallestEigenvector3x3V1(
            covariance,
            normal);
    }


    const double norm =
        sqrt(best_norm2);


    if (!isfinite(norm) ||
        norm <= 1.0e-15)
    {
        used_v1_fallback = true;

        return SmallestEigenvector3x3V1(
            covariance,
            normal);
    }


    const double inv_norm =
        1.0 /
        norm;


    normal[0] =
        nx *
        inv_norm;

    normal[1] =
        ny *
        inv_norm;

    normal[2] =
        nz *
        inv_norm;


    return
        isfinite(normal[0]) &&
        isfinite(normal[1]) &&
        isfinite(normal[2]);
}


__global__ void PlaneHessianShadowKernelV1(
    const float *target_xyz,
    const std::size_t target_count,
    const double *p_target_xyz,
    const int *neighbor_indices,
    const float *neighbor_squared_distances,
    const unsigned char *eligible,
    const std::size_t query_count,
    const double sensor_origin_x,
    const double sensor_origin_y,
    const double sensor_origin_z,
    const double maximum_squared_distance,
    const double maximum_plane_fit_error,
    const double maximum_residual,
    const double huber_delta,
    PlaneHessianContributionV1 *contributions)
{
    const std::size_t qi =
        static_cast<std::size_t>(blockIdx.x) *
            static_cast<std::size_t>(blockDim.x) +
        static_cast<std::size_t>(threadIdx.x);


    if (qi >= query_count)
    {
        return;
    }


    PlaneHessianContributionV1 result{};

    result.status = 0U;
    result.downweighted = 0U;


    if (eligible[qi] == 0U)
    {
        contributions[qi] =
            result;

        return;
    }


    const std::size_t base =
        qi * 5U;


    if (!isfinite(
            static_cast<double>(
                neighbor_squared_distances[base])) ||
        static_cast<double>(
            neighbor_squared_distances[base]) >
            maximum_squared_distance)
    {
        contributions[qi] =
            result;

        return;
    }


    double points[5][3];


#pragma unroll
    for (int k = 0;
         k < 5;
         ++k)
    {
        const int index =
            neighbor_indices[
                base +
                static_cast<std::size_t>(k)];


        if (index < 0 ||
            static_cast<std::size_t>(index) >=
                target_count)
        {
            contributions[qi] =
                result;

            return;
        }


        points[k][0] =
            static_cast<double>(
                target_xyz[
                    static_cast<std::size_t>(index) *
                        3U +
                    0U]);

        points[k][1] =
            static_cast<double>(
                target_xyz[
                    static_cast<std::size_t>(index) *
                        3U +
                    1U]);

        points[k][2] =
            static_cast<double>(
                target_xyz[
                    static_cast<std::size_t>(index) *
                        3U +
                    2U]);
    }


    double centroid[3] =
    {
        0.0,
        0.0,
        0.0
    };


#pragma unroll
    for (int k = 0;
         k < 5;
         ++k)
    {
        centroid[0] += points[k][0];
        centroid[1] += points[k][1];
        centroid[2] += points[k][2];
    }


    centroid[0] /= 5.0;
    centroid[1] /= 5.0;
    centroid[2] /= 5.0;


    double covariance[3][3] =
    {
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0}
    };


#pragma unroll
    for (int k = 0;
         k < 5;
         ++k)
    {
        const double dx =
            points[k][0] -
            centroid[0];

        const double dy =
            points[k][1] -
            centroid[1];

        const double dz =
            points[k][2] -
            centroid[2];


        covariance[0][0] += dx * dx;
        covariance[0][1] += dx * dy;
        covariance[0][2] += dx * dz;

        covariance[1][0] += dy * dx;
        covariance[1][1] += dy * dy;
        covariance[1][2] += dy * dz;

        covariance[2][0] += dz * dx;
        covariance[2][1] += dz * dy;
        covariance[2][2] += dz * dz;
    }


#pragma unroll
    for (int r = 0;
         r < 3;
         ++r)
    {
#pragma unroll
        for (int c = 0;
             c < 3;
             ++c)
        {
            covariance[r][c] /=
                5.0;
        }
    }


    double normal[3];


    if (!SmallestEigenvector3x3V1(
            covariance,
            normal))
    {
        result.status = 1U;

        contributions[qi] =
            result;

        return;
    }


#pragma unroll
    for (int k = 0;
         k < 5;
         ++k)
    {
        const double dx =
            points[k][0] -
            centroid[0];

        const double dy =
            points[k][1] -
            centroid[1];

        const double dz =
            points[k][2] -
            centroid[2];


        const double distance =
            fabs(
                normal[0] * dx +
                normal[1] * dy +
                normal[2] * dz);


        if (!isfinite(distance) ||
            distance >
                maximum_plane_fit_error)
        {
            result.status = 1U;

            contributions[qi] =
                result;

            return;
        }
    }


    result.status = 2U;


    const double px =
        p_target_xyz[
            qi * 3U + 0U];

    const double py =
        p_target_xyz[
            qi * 3U + 1U];

    const double pz =
        p_target_xyz[
            qi * 3U + 2U];


    if (!isfinite(px) ||
        !isfinite(py) ||
        !isfinite(pz))
    {
        contributions[qi] =
            result;

        return;
    }


    const double residual =
        normal[0] *
            (px - centroid[0]) +
        normal[1] *
            (py - centroid[1]) +
        normal[2] *
            (pz - centroid[2]);


    const double absolute_residual =
        fabs(residual);


    if (!isfinite(residual) ||
        absolute_residual >
            maximum_residual)
    {
        contributions[qi] =
            result;

        return;
    }


    const double lx =
        px -
        sensor_origin_x;

    const double ly =
        py -
        sensor_origin_y;

    const double lz =
        pz -
        sensor_origin_z;


    if (!isfinite(lx) ||
        !isfinite(ly) ||
        !isfinite(lz))
    {
        contributions[qi] =
            result;

        return;
    }


    // lever x normal
    double J[6];

    J[0] =
        ly * normal[2] -
        lz * normal[1];

    J[1] =
        lz * normal[0] -
        lx * normal[2];

    J[2] =
        lx * normal[1] -
        ly * normal[0];

    J[3] = normal[0];
    J[4] = normal[1];
    J[5] = normal[2];


#pragma unroll
    for (int k = 0;
         k < 6;
         ++k)
    {
        result.J[k] = J[k];
    }

    result.residual = residual;
    result.normal_z = normal[2];
    result.lever_z = lz;

    result.range =
        sqrt(
            lx * lx +
            ly * ly +
            lz * lz);


    double huber_weight = 1.0;


    if (absolute_residual >
            huber_delta &&
        absolute_residual >
            1.0e-12)
    {
        huber_weight =
            huber_delta /
            absolute_residual;

        result.downweighted = 1U;
    }


    int upper_index = 0;


#pragma unroll
    for (int r = 0;
         r < 6;
         ++r)
    {
#pragma unroll
        for (int c = r;
             c < 6;
             ++c)
        {
            result.h_upper[
                upper_index++] =
                    huber_weight *
                    J[r] *
                    J[c];
        }


        result.b[r] =
            huber_weight *
            J[r] *
            residual;
    }


    result.raw_sse =
        residual *
        residual;

    result.robust_sse =
        huber_weight *
        residual *
        residual;

    result.robust_weight =
        huber_weight;

    result.status = 3U;


    contributions[qi] =
        result;
}

}  // namespace


namespace fr_slam_cuda
{

bool ComputePlaneHessianShadowV1(
    PersistentKnn5Handle handle,
    const double *p_target_xyz,
    const int *neighbor_indices,
    const float *neighbor_squared_distances,
    const unsigned char *eligible,
    const std::size_t query_count,
    const double *sensor_origin_xyz,
    const double maximum_squared_distance,
    const double maximum_plane_fit_error,
    const double maximum_residual,
    const double huber_delta,
    const bool enable_ground_constraint,
    const double ground_normal_cosine_threshold,
    const double ground_min_below_sensor_m,
    const double ground_max_residual,
    const double ground_huber_delta,
    const double ground_weight,
    double *output_H_6x6,
    double *output_b_6,
    double *output_H_ground_6x6,
    double *output_b_ground_6,
    double *output_ranges,
    unsigned char *output_range_valid,
    double &output_raw_sse,
    double &output_robust_sse,
    double &output_robust_weight_sum,
    PlaneHessianShadowStats &stats)
{
    stats =
        PlaneHessianShadowStats{};


    output_raw_sse = 0.0;
    output_robust_sse = 0.0;
    output_robust_weight_sum = 0.0;


    if (output_H_6x6 != nullptr)
    {
        for (int i = 0;
             i < 36;
             ++i)
        {
            output_H_6x6[i] = 0.0;
        }
    }


    if (output_b_6 != nullptr)
    {
        for (int i = 0;
             i < 6;
             ++i)
        {
            output_b_6[i] = 0.0;
        }
    }


    if (output_H_ground_6x6 != nullptr)
    {
        for (int i = 0;
             i < 36;
             ++i)
        {
            output_H_ground_6x6[i] = 0.0;
        }
    }


    if (output_b_ground_6 != nullptr)
    {
        for (int i = 0;
             i < 6;
             ++i)
        {
            output_b_ground_6[i] = 0.0;
        }
    }


    if (output_ranges != nullptr &&
        output_range_valid != nullptr)
    {
        for (std::size_t i = 0U;
             i < query_count;
             ++i)
        {
            output_ranges[i] =
                std::numeric_limits<double>::quiet_NaN();

            output_range_valid[i] = 0U;
        }
    }


    PersistentKnn5ContextImpl *context =
        reinterpret_cast<
            PersistentKnn5ContextImpl *>(
                handle);


    if (context == nullptr ||
        p_target_xyz == nullptr ||
        neighbor_indices == nullptr ||
        neighbor_squared_distances == nullptr ||
        eligible == nullptr ||
        sensor_origin_xyz == nullptr ||
        output_H_6x6 == nullptr ||
        output_b_6 == nullptr ||
        output_H_ground_6x6 == nullptr ||
        output_b_ground_6 == nullptr ||
        output_ranges == nullptr ||
        output_range_valid == nullptr ||
        query_count == 0U)
    {
        return false;
    }


    stats.query_count =
        query_count;


    const auto total_begin =
        std::chrono::steady_clock::now();


    double *device_p_target = nullptr;
    int *device_indices = nullptr;
    float *device_distances = nullptr;
    unsigned char *device_eligible = nullptr;

    PlaneHessianContributionV1
        *device_contributions = nullptr;


    auto fail =
        [&](const cudaError_t error)
        {
            stats.cuda_error =
                static_cast<int>(error);

            if (device_contributions != nullptr)
            {
                cudaFree(
                    device_contributions);
            }

            if (device_eligible != nullptr)
            {
                cudaFree(
                    device_eligible);
            }

            if (device_distances != nullptr)
            {
                cudaFree(
                    device_distances);
            }

            if (device_indices != nullptr)
            {
                cudaFree(
                    device_indices);
            }

            if (device_p_target != nullptr)
            {
                cudaFree(
                    device_p_target);
            }

            stats.total_ms =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() -
                    total_begin)
                    .count();

            return false;
        };


    cudaError_t error =
        cudaMalloc(
            reinterpret_cast<void **>(
                &device_p_target),
            query_count *
                3U *
                sizeof(double));

    if (error != cudaSuccess)
    {
        return fail(error);
    }


    error =
        cudaMalloc(
            reinterpret_cast<void **>(
                &device_indices),
            query_count *
                5U *
                sizeof(int));

    if (error != cudaSuccess)
    {
        return fail(error);
    }


    error =
        cudaMalloc(
            reinterpret_cast<void **>(
                &device_distances),
            query_count *
                5U *
                sizeof(float));

    if (error != cudaSuccess)
    {
        return fail(error);
    }


    error =
        cudaMalloc(
            reinterpret_cast<void **>(
                &device_eligible),
            query_count *
                sizeof(unsigned char));

    if (error != cudaSuccess)
    {
        return fail(error);
    }


    error =
        cudaMalloc(
            reinterpret_cast<void **>(
                &device_contributions),
            query_count *
                sizeof(
                    PlaneHessianContributionV1));

    if (error != cudaSuccess)
    {
        return fail(error);
    }


    error =
        cudaMemcpy(
            device_p_target,
            p_target_xyz,
            query_count *
                3U *
                sizeof(double),
            cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        return fail(error);
    }


    error =
        cudaMemcpy(
            device_indices,
            neighbor_indices,
            query_count *
                5U *
                sizeof(int),
            cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        return fail(error);
    }


    error =
        cudaMemcpy(
            device_distances,
            neighbor_squared_distances,
            query_count *
                5U *
                sizeof(float),
            cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        return fail(error);
    }


    error =
        cudaMemcpy(
            device_eligible,
            eligible,
            query_count *
                sizeof(unsigned char),
            cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        return fail(error);
    }


    for (std::size_t i = 0U;
         i < query_count;
         ++i)
    {
        if (eligible[i] != 0U)
        {
            ++stats.eligible_queries;
        }
    }


    cudaEvent_t begin_event = nullptr;
    cudaEvent_t end_event = nullptr;


    error =
        cudaEventCreate(
            &begin_event);

    if (error != cudaSuccess)
    {
        return fail(error);
    }


    error =
        cudaEventCreate(
            &end_event);

    if (error != cudaSuccess)
    {
        cudaEventDestroy(
            begin_event);

        return fail(error);
    }


    constexpr int threads = 256;

    const int blocks =
        static_cast<int>(
            (
                query_count +
                static_cast<std::size_t>(threads) -
                1U
            ) /
            static_cast<std::size_t>(threads));


    cudaEventRecord(
        begin_event);


    PlaneHessianShadowKernelV1<<<
        blocks,
        threads>>>(
            context->device_target,
            context->target_count,
            device_p_target,
            device_indices,
            device_distances,
            device_eligible,
            query_count,
            sensor_origin_xyz[0],
            sensor_origin_xyz[1],
            sensor_origin_xyz[2],
            maximum_squared_distance,
            maximum_plane_fit_error,
            maximum_residual,
            huber_delta,
            device_contributions);


    error =
        cudaGetLastError();

    if (error != cudaSuccess)
    {
        cudaEventDestroy(
            end_event);

        cudaEventDestroy(
            begin_event);

        return fail(error);
    }


    cudaEventRecord(
        end_event);

    error =
        cudaEventSynchronize(
            end_event);

    if (error != cudaSuccess)
    {
        cudaEventDestroy(
            end_event);

        cudaEventDestroy(
            begin_event);

        return fail(error);
    }


    float kernel_ms = 0.0f;

    cudaEventElapsedTime(
        &kernel_ms,
        begin_event,
        end_event);

    stats.kernel_ms =
        static_cast<double>(
            kernel_ms);


    cudaEventDestroy(
        end_event);

    cudaEventDestroy(
        begin_event);


    std::vector<
        PlaneHessianContributionV1>
            host_contributions(
                query_count);


    error =
        cudaMemcpy(
            host_contributions.data(),
            device_contributions,
            query_count *
                sizeof(
                    PlaneHessianContributionV1),
            cudaMemcpyDeviceToHost);

    if (error != cudaSuccess)
    {
        return fail(error);
    }


    // Deterministic query-order reduction.
    for (std::size_t qi = 0U;
         qi < query_count;
         ++qi)
    {
        const PlaneHessianContributionV1 &v =
            host_contributions[qi];


        if (v.status == 1U)
        {
            ++stats.plane_fit_failures;
        }


        if (v.status >= 2U)
        {
            ++stats.plane_valid;
        }


        if (v.status != 3U)
        {
            continue;
        }


        ++stats.correspondences;


        if (v.downweighted != 0U)
        {
            ++stats.downweighted;
        }


        int upper_index = 0;


        for (int r = 0;
             r < 6;
             ++r)
        {
            for (int c = r;
                 c < 6;
                 ++c)
            {
                const double value =
                    v.h_upper[
                        upper_index++];


                output_H_6x6[
                    r * 6 + c] +=
                        value;


                if (r != c)
                {
                    output_H_6x6[
                        c * 6 + r] +=
                            value;
                }
            }


            output_b_6[r] +=
                v.b[r];
        }


        output_raw_sse +=
            v.raw_sse;

        output_robust_sse +=
            v.robust_sse;

        output_robust_weight_sum +=
            v.robust_weight;


        if (std::isfinite(v.range) &&
            v.range > 1.0e-9)
        {
            output_ranges[qi] =
                v.range;

            output_range_valid[qi] =
                1U;
        }


        if (enable_ground_constraint &&
            std::abs(v.normal_z) >=
                ground_normal_cosine_threshold &&
            v.lever_z <=
                -ground_min_below_sensor_m &&
            std::abs(v.residual) <=
                ground_max_residual)
        {
            double J_ground[6];

            for (int k = 0;
                 k < 6;
                 ++k)
            {
                J_ground[k] =
                    v.J[k];
            }

            // [rx ry rz tx ty tz]
            J_ground[2] = 0.0;
            J_ground[3] = 0.0;
            J_ground[4] = 0.0;


            double ground_robust_weight =
                1.0;

            const double abs_residual =
                std::abs(v.residual);

            if (abs_residual >
                    ground_huber_delta &&
                abs_residual >
                    1.0e-12)
            {
                ground_robust_weight =
                    ground_huber_delta /
                    abs_residual;

                ++stats.ground_downweighted;
            }


            const double final_ground_weight =
                ground_weight *
                ground_robust_weight;


            for (int r = 0;
                 r < 6;
                 ++r)
            {
                for (int c = 0;
                     c < 6;
                     ++c)
                {
                    output_H_ground_6x6[
                        r * 6 + c] +=
                            final_ground_weight *
                            J_ground[r] *
                            J_ground[c];
                }

                output_b_ground_6[r] +=
                    final_ground_weight *
                    J_ground[r] *
                    v.residual;
            }


            ++stats.ground_correspondences;
        }
    }


    cudaFree(
        device_contributions);

    cudaFree(
        device_eligible);

    cudaFree(
        device_distances);

    cudaFree(
        device_indices);

    cudaFree(
        device_p_target);


    stats.total_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() -
            total_begin)
            .count();

    stats.cuda_error = 0;
    stats.ok = true;

    return true;
}

}  // namespace fr_slam_cuda


// ============================================================================
// GPU PERSISTENT FUSED PLANE / HESSIAN V3
//
// Source is uploaded once per solver context.
//
// Per GN iteration:
//
//      source resident
//           |
//           v
//      TransformKernel
//           |
//           +--> float query  -> existing PersistentKnn6Kernel
//           |
//           +--> double query
//                     |
//                     v
//              PlaneHessianKernel
//                     |
//                     v
//             GPU H/b reduction
//
// KNN indices / distances NEVER return to host.
//
// Host receives:
//   - CPU fallback mask
//   - correspondence ranges
//   - H / b
//   - counters
//
// Diagnostic/experimental V3 only.
// ============================================================================

namespace
{

struct FusedTransform3x4V3
{
    double m[12];
};


struct FusedAccumulatorV3
{
    // Symmetric upper triangle, order:
    // (0,0),(0,1)...(0,5),(1,1)...(5,5)
    double H_upper[21];
    double b[6];

    double H_ground_upper[21];
    double b_ground[6];

    double raw_sse;
    double robust_sse;
    double robust_weight_sum;

    unsigned long long ambiguous_queries;
    unsigned long long cpu_fallback_queries;

    unsigned long long plane_fit_failures;
    unsigned long long correspondences;
    unsigned long long downweighted;

    unsigned long long ground_correspondences;
    unsigned long long ground_downweighted;
};


__global__ void FusedTransformSourceKernelV3(
    const float *source_xyz,
    const std::size_t query_count,
    const FusedTransform3x4V3 T,
    float *query_float_xyz,
    double *query_double_xyz)
{
    const std::size_t qi =
        static_cast<std::size_t>(blockIdx.x) *
            static_cast<std::size_t>(blockDim.x) +
        static_cast<std::size_t>(threadIdx.x);


    if (qi >= query_count)
    {
        return;
    }


    const double x =
        static_cast<double>(
            source_xyz[qi * 3U + 0U]);

    const double y =
        static_cast<double>(
            source_xyz[qi * 3U + 1U]);

    const double z =
        static_cast<double>(
            source_xyz[qi * 3U + 2U]);


    const double qx =
        T.m[0] * x +
        T.m[1] * y +
        T.m[2] * z +
        T.m[3];

    const double qy =
        T.m[4] * x +
        T.m[5] * y +
        T.m[6] * z +
        T.m[7];

    const double qz =
        T.m[8] * x +
        T.m[9] * y +
        T.m[10] * z +
        T.m[11];


    query_double_xyz[
        qi * 3U + 0U] = qx;

    query_double_xyz[
        qi * 3U + 1U] = qy;

    query_double_xyz[
        qi * 3U + 2U] = qz;


    // Match existing CPU -> float KNN query conversion.
    query_float_xyz[
        qi * 3U + 0U] =
            static_cast<float>(qx);

    query_float_xyz[
        qi * 3U + 1U] =
            static_cast<float>(qy);

    query_float_xyz[
        qi * 3U + 2U] =
            static_cast<float>(qz);
}


__device__ inline int
UpperTriangleIndex6V3(
    const int row,
    const int col)
{
    // row <= col
    //
    // row 0 -> 0
    // row 1 -> 6
    // row 2 -> 11
    // row 3 -> 15
    // row 4 -> 18
    // row 5 -> 20
    const int start =
        row * 6 -
        (row * (row - 1)) / 2;

    return start +
           (col - row);
}


__global__ void
FusedPlaneHessianKernelV3(
    const float *target_xyz,
    const std::size_t target_count,

    const double *query_double_xyz,
    const std::size_t query_count,

    const int *neighbor_indices,
    const float *neighbor_squared_distances,
    const unsigned char *ambiguous,

    const double sensor_origin_x,
    const double sensor_origin_y,
    const double sensor_origin_z,

    const double maximum_squared_distance,
    const double maximum_plane_fit_error,
    const double maximum_residual,
    const double huber_delta,

    const bool enable_ground_constraint,
    const double ground_normal_cosine_threshold,
    const double ground_min_below_sensor_m,
    const double ground_max_residual,
    const double ground_huber_delta,
    const double ground_weight,

    unsigned char *cpu_fallback,
    double *ranges,
    unsigned char *range_valid,

    FusedAccumulatorV3 *accumulator)
{
    const std::size_t qi =
        static_cast<std::size_t>(blockIdx.x) *
            static_cast<std::size_t>(blockDim.x) +
        static_cast<std::size_t>(threadIdx.x);


    if (qi >= query_count)
    {
        return;
    }


    cpu_fallback[qi] = 0U;
    range_valid[qi] = 0U;


    const double px =
        query_double_xyz[
            qi * 3U + 0U];

    const double py =
        query_double_xyz[
            qi * 3U + 1U];

    const double pz =
        query_double_xyz[
            qi * 3U + 2U];


    // Formal CPU solver simply skips non-finite transformed points.
    if (!isfinite(px) ||
        !isfinite(py) ||
        !isfinite(pz))
    {
        return;
    }


    // K5/K6 boundary uncertainty -> exact CPU PCL fallback.
    if (ambiguous[qi] != 0U)
    {
        cpu_fallback[qi] = 1U;

        atomicAdd(
            &accumulator->ambiguous_queries,
            1ULL);

        atomicAdd(
            &accumulator->cpu_fallback_queries,
            1ULL);

        return;
    }


    constexpr int K = 5;

    const std::size_t base =
        qi *
        static_cast<std::size_t>(K);


    double points[K][3];


#pragma unroll
    for (int k = 0;
         k < K;
         ++k)
    {
        const std::size_t off =
            base +
            static_cast<std::size_t>(k);

        const int index =
            neighbor_indices[off];

        const float squared_distance =
            neighbor_squared_distances[off];


        // Match Active V3:
        // invalid GPU neighbour set -> CPU KNN fallback.
        if (index < 0 ||
            static_cast<std::size_t>(index) >=
                target_count ||
            !isfinite(
                static_cast<double>(
                    squared_distance)))
        {
            cpu_fallback[qi] = 1U;

            atomicAdd(
                &accumulator->cpu_fallback_queries,
                1ULL);

            return;
        }


        points[k][0] =
            static_cast<double>(
                target_xyz[
                    static_cast<std::size_t>(index) *
                        3U +
                    0U]);

        points[k][1] =
            static_cast<double>(
                target_xyz[
                    static_cast<std::size_t>(index) *
                        3U +
                    1U]);

        points[k][2] =
            static_cast<double>(
                target_xyz[
                    static_cast<std::size_t>(index) *
                        3U +
                    2U]);
    }


    const double nearest_squared_distance =
        static_cast<double>(
            neighbor_squared_distances[base]);


    if (!isfinite(nearest_squared_distance) ||
        nearest_squared_distance >
            maximum_squared_distance)
    {
        // GPU KNN itself is valid; formal solver rejects this
        // correspondence. No CPU fallback.
        return;
    }


    // ------------------------------------------------------------
    // 5-neighbour centroid
    // ------------------------------------------------------------

    double centroid[3] =
    {
        0.0,
        0.0,
        0.0
    };


#pragma unroll
    for (int k = 0;
         k < K;
         ++k)
    {
        centroid[0] += points[k][0];
        centroid[1] += points[k][1];
        centroid[2] += points[k][2];
    }


    centroid[0] /= 5.0;
    centroid[1] /= 5.0;
    centroid[2] /= 5.0;


    // ------------------------------------------------------------
    // Same population covariance as FitShadowPlane().
    // ------------------------------------------------------------

    double covariance[3][3] =
    {
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0}
    };


#pragma unroll
    for (int k = 0;
         k < K;
         ++k)
    {
        const double dx =
            points[k][0] -
            centroid[0];

        const double dy =
            points[k][1] -
            centroid[1];

        const double dz =
            points[k][2] -
            centroid[2];


        covariance[0][0] += dx * dx;
        covariance[0][1] += dx * dy;
        covariance[0][2] += dx * dz;

        covariance[1][0] += dy * dx;
        covariance[1][1] += dy * dy;
        covariance[1][2] += dy * dz;

        covariance[2][0] += dz * dx;
        covariance[2][1] += dz * dy;
        covariance[2][2] += dz * dz;
    }


#pragma unroll
    for (int r = 0;
         r < 3;
         ++r)
    {
#pragma unroll
        for (int c = 0;
             c < 3;
             ++c)
        {
            covariance[r][c] /=
                5.0;
        }
    }


    double normal[3];


    // Fast analytic smallest-eigenvector solver.
    //
    // Numerically degenerate covariance matrices fall back
    // internally to the validated Jacobi V1 implementation.
    bool fast_eigen_used_v1_fallback = false;

    if (!SmallestEigenvector3x3V2Fast(
            covariance,
            normal,
            fast_eigen_used_v1_fallback))
    {
        atomicAdd(
            &accumulator->plane_fit_failures,
            1ULL);

        return;
    }


    // ------------------------------------------------------------
    // Same all-K plane-fit validity test.
    // ------------------------------------------------------------

#pragma unroll
    for (int k = 0;
         k < K;
         ++k)
    {
        const double dx =
            points[k][0] -
            centroid[0];

        const double dy =
            points[k][1] -
            centroid[1];

        const double dz =
            points[k][2] -
            centroid[2];


        const double distance =
            fabs(
                normal[0] * dx +
                normal[1] * dy +
                normal[2] * dz);


        if (!isfinite(distance) ||
            distance >
                maximum_plane_fit_error)
        {
            atomicAdd(
                &accumulator->plane_fit_failures,
                1ULL);

            return;
        }
    }


    // ------------------------------------------------------------
    // Point-to-plane residual
    // ------------------------------------------------------------

    const double residual =
        normal[0] *
            (px - centroid[0]) +
        normal[1] *
            (py - centroid[1]) +
        normal[2] *
            (pz - centroid[2]);


    const double absolute_residual =
        fabs(residual);


    if (!isfinite(residual) ||
        absolute_residual >
            maximum_residual)
    {
        return;
    }


    // ------------------------------------------------------------
    // Sensor-centred perturbation.
    // J order = [rx ry rz tx ty tz]
    // ------------------------------------------------------------

    const double lx =
        px -
        sensor_origin_x;

    const double ly =
        py -
        sensor_origin_y;

    const double lz =
        pz -
        sensor_origin_z;


    if (!isfinite(lx) ||
        !isfinite(ly) ||
        !isfinite(lz))
    {
        return;
    }


    double J[6];

    J[0] =
        ly * normal[2] -
        lz * normal[1];

    J[1] =
        lz * normal[0] -
        lx * normal[2];

    J[2] =
        lx * normal[1] -
        ly * normal[0];

    J[3] = normal[0];
    J[4] = normal[1];
    J[5] = normal[2];


    // ------------------------------------------------------------
    // Huber
    // ------------------------------------------------------------

    double robust_weight = 1.0;


    if (absolute_residual >
            huber_delta &&
        absolute_residual >
            1.0e-12)
    {
        robust_weight =
            huber_delta /
            absolute_residual;

        atomicAdd(
            &accumulator->downweighted,
            1ULL);
    }


    // ------------------------------------------------------------
    // Geometry H / b
    // Only upper triangle is atomically accumulated.
    // ------------------------------------------------------------

#pragma unroll
    for (int r = 0;
         r < 6;
         ++r)
    {
#pragma unroll
        for (int c = r;
             c < 6;
             ++c)
        {
            const int index =
                UpperTriangleIndex6V3(
                    r,
                    c);

            atomicAdd(
                &accumulator->H_upper[index],
                robust_weight *
                    J[r] *
                    J[c]);
        }


        atomicAdd(
            &accumulator->b[r],
            robust_weight *
                J[r] *
                residual);
    }


    atomicAdd(
        &accumulator->raw_sse,
        residual *
            residual);

    atomicAdd(
        &accumulator->robust_sse,
        robust_weight *
            residual *
            residual);

    atomicAdd(
        &accumulator->robust_weight_sum,
        robust_weight);


    atomicAdd(
        &accumulator->correspondences,
        1ULL);


    // ------------------------------------------------------------
    // Correspondence range
    // ------------------------------------------------------------

    const double range =
        sqrt(
            lx * lx +
            ly * ly +
            lz * lz);


    if (isfinite(range) &&
        range > 1.0e-9)
    {
        ranges[qi] =
            range;

        range_valid[qi] =
            1U;
    }


    // ------------------------------------------------------------
    // Same loop-local Ground branch.
    // ------------------------------------------------------------

    if (enable_ground_constraint &&
        fabs(normal[2]) >=
            ground_normal_cosine_threshold &&
        lz <=
            -ground_min_below_sensor_m &&
        absolute_residual <=
            ground_max_residual)
    {
        double J_ground[6];


#pragma unroll
        for (int k = 0;
             k < 6;
             ++k)
        {
            J_ground[k] =
                J[k];
        }


        // Preserve [roll,pitch,z].
        J_ground[2] = 0.0; // yaw
        J_ground[3] = 0.0; // x
        J_ground[4] = 0.0; // y


        double ground_robust_weight =
            1.0;


        if (absolute_residual >
                ground_huber_delta &&
            absolute_residual >
                1.0e-12)
        {
            ground_robust_weight =
                ground_huber_delta /
                absolute_residual;

            atomicAdd(
                &accumulator->ground_downweighted,
                1ULL);
        }


        const double final_ground_weight =
            ground_weight *
            ground_robust_weight;


#pragma unroll
        for (int r = 0;
             r < 6;
             ++r)
        {
#pragma unroll
            for (int c = r;
                 c < 6;
                 ++c)
            {
                const int index =
                    UpperTriangleIndex6V3(
                        r,
                        c);

                atomicAdd(
                    &accumulator->
                        H_ground_upper[index],
                    final_ground_weight *
                        J_ground[r] *
                        J_ground[c]);
            }


            atomicAdd(
                &accumulator->b_ground[r],
                final_ground_weight *
                    J_ground[r] *
                    residual);
        }


        atomicAdd(
            &accumulator->ground_correspondences,
            1ULL);
    }
}


bool EnsurePersistentFusedBuffersV3(
    PersistentKnn5ContextImpl *context,
    int &cuda_error)
{
    cuda_error = 0;


    if (context == nullptr)
    {
        return false;
    }


    auto allocate =
        [&](void **pointer,
            const std::size_t bytes) -> bool
        {
            if (*pointer != nullptr)
            {
                return true;
            }


            const cudaError_t error =
                cudaMalloc(
                    pointer,
                    bytes);


            if (error != cudaSuccess)
            {
                cuda_error =
                    static_cast<int>(error);

                return false;
            }


            return true;
        };


    if (!allocate(
            reinterpret_cast<void **>(
                &context->device_fused_source),
            context->max_query_count *
                3U *
                sizeof(float)))
    {
        return false;
    }


    if (!allocate(
            reinterpret_cast<void **>(
                &context->
                    device_fused_query_double),
            context->max_query_count *
                3U *
                sizeof(double)))
    {
        return false;
    }


    if (!allocate(
            reinterpret_cast<void **>(
                &context->
                    device_fused_cpu_fallback),
            context->max_query_count *
                sizeof(unsigned char)))
    {
        return false;
    }


    if (!allocate(
            reinterpret_cast<void **>(
                &context->device_fused_ranges),
            context->max_query_count *
                sizeof(double)))
    {
        return false;
    }


    if (!allocate(
            reinterpret_cast<void **>(
                &context->
                    device_fused_range_valid),
            context->max_query_count *
                sizeof(unsigned char)))
    {
        return false;
    }


    if (!allocate(
            &context->device_fused_accumulator,
            sizeof(FusedAccumulatorV3)))
    {
        return false;
    }


    auto create_event =
        [&](cudaEvent_t &event) -> bool
        {
            if (event != nullptr)
            {
                return true;
            }


            const cudaError_t error =
                cudaEventCreate(
                    &event);


            if (error != cudaSuccess)
            {
                cuda_error =
                    static_cast<int>(error);

                return false;
            }


            return true;
        };


    if (!create_event(
            context->fused_event_0) ||
        !create_event(
            context->fused_event_1) ||
        !create_event(
            context->fused_event_2) ||
        !create_event(
            context->fused_event_3))
    {
        return false;
    }


    return true;
}

}  // namespace


namespace fr_slam_cuda
{

bool QueryPersistentFusedPlaneHessian(
    PersistentKnn5Handle handle,

    const float *source_xyz,
    const std::size_t query_count,

    const double *T_target_source_3x4,

    const float ambiguity_abs_epsilon,
    const float ambiguity_relative_epsilon,

    const double maximum_squared_distance,
    const double maximum_plane_fit_error,
    const double maximum_residual,
    const double huber_delta,

    const bool enable_ground_constraint,
    const double ground_normal_cosine_threshold,
    const double ground_min_below_sensor_m,
    const double ground_max_residual,
    const double ground_huber_delta,
    const double ground_weight,

    double *output_H_6x6,
    double *output_b_6,

    double *output_H_ground_6x6,
    double *output_b_ground_6,

    double *output_ranges,
    unsigned char *output_range_valid,

    unsigned char *output_cpu_fallback,

    double &output_raw_sse,
    double &output_robust_sse,
    double &output_robust_weight_sum,

    PersistentFusedStats &stats)
{
    stats =
        PersistentFusedStats{};


    output_raw_sse = 0.0;
    output_robust_sse = 0.0;
    output_robust_weight_sum = 0.0;


    PersistentKnn5ContextImpl *context =
        reinterpret_cast<
            PersistentKnn5ContextImpl *>(
                handle);


    if (context == nullptr ||
        source_xyz == nullptr ||
        T_target_source_3x4 == nullptr ||
        output_H_6x6 == nullptr ||
        output_b_6 == nullptr ||
        output_H_ground_6x6 == nullptr ||
        output_b_ground_6 == nullptr ||
        output_ranges == nullptr ||
        output_range_valid == nullptr ||
        output_cpu_fallback == nullptr ||
        query_count == 0U ||
        query_count >
            context->max_query_count)
    {
        return false;
    }


    stats.query_count =
        query_count;


    for (int i = 0;
         i < 36;
         ++i)
    {
        output_H_6x6[i] = 0.0;
        output_H_ground_6x6[i] = 0.0;
    }


    for (int i = 0;
         i < 6;
         ++i)
    {
        output_b_6[i] = 0.0;
        output_b_ground_6[i] = 0.0;
    }


    const auto total_begin =
        std::chrono::steady_clock::now();


    const auto setup_begin =
        std::chrono::steady_clock::now();


    int setup_cuda_error = 0;


    if (!EnsurePersistentFusedBuffersV3(
            context,
            setup_cuda_error))
    {
        stats.cuda_error =
            setup_cuda_error;

        return false;
    }


    stats.setup_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() -
            setup_begin)
            .count();


    // ------------------------------------------------------------
    // Upload source ONCE for this persistent solver context.
    // ------------------------------------------------------------

    if (!context->fused_source_uploaded)
    {
        const auto upload_begin =
            std::chrono::steady_clock::now();


        const cudaError_t error =
            cudaMemcpy(
                context->device_fused_source,
                source_xyz,
                query_count *
                    3U *
                    sizeof(float),
                cudaMemcpyHostToDevice);


        if (error != cudaSuccess)
        {
            stats.cuda_error =
                static_cast<int>(error);

            return false;
        }


        context->fused_source_uploaded = true;
        context->fused_source_count =
            query_count;


        stats.source_upload_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() -
                upload_begin)
                .count();
    }
    else if (
        context->fused_source_count !=
            query_count)
    {
        // Source cloud must be immutable for one solver context.
        return false;
    }


    // ------------------------------------------------------------
    // Reset only small GPU outputs.
    // ------------------------------------------------------------

    const auto memset_begin =
        std::chrono::steady_clock::now();


    cudaError_t error =
        cudaMemsetAsync(
            context->device_fused_accumulator,
            0,
            sizeof(FusedAccumulatorV3),
            context->stream);


    stats.memset_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() -
            memset_begin)
            .count();


    if (error != cudaSuccess)
    {
        stats.cuda_error =
            static_cast<int>(error);

        return false;
    }


    FusedTransform3x4V3 T;


    for (int i = 0;
         i < 12;
         ++i)
    {
        T.m[i] =
            T_target_source_3x4[i];
    }


    constexpr int threads = 256;


    const int blocks =
        static_cast<int>(
            (
                query_count +
                static_cast<std::size_t>(
                    threads) -
                1U
            ) /
            static_cast<std::size_t>(
                threads));


    // ------------------------------------------------------------
    // Transform
    // ------------------------------------------------------------

    cudaEventRecord(
        context->fused_event_0,
        context->stream);


    FusedTransformSourceKernelV3<<<
        blocks,
        threads,
        0,
        context->stream>>>(
            context->device_fused_source,
            query_count,
            T,
            context->device_query,
            context->device_fused_query_double);


    error =
        cudaGetLastError();


    if (error != cudaSuccess)
    {
        stats.cuda_error =
            static_cast<int>(error);

        return false;
    }


    cudaEventRecord(
        context->fused_event_1,
        context->stream);


    // ------------------------------------------------------------
    // Existing exact K=6 GPU retrieval.
    // ------------------------------------------------------------
    // V4 PRODUCTION exact dense-grid K=6 retrieval.
    //
    // Full-shadow validation:
    //   >319 million clean queries
    //   zero index mismatch
    //   zero distance mismatch
    //   zero ambiguity mismatch
    //
    // If the V4 grid is unavailable, preserve correctness by
    // falling back to the original V3 brute-force exact K=6.
    // ------------------------------------------------------------

    const bool use_v4_knn =
        context->v4_grid_ready &&
        context->device_v4_cell_offsets != nullptr &&
        context->device_v4_point_indices != nullptr;


    if (use_v4_knn)
    {
        PersistentKnn6KernelV4Exact<<<
            blocks,
            threads,
            0,
            context->stream>>>(
                context->device_query,
                query_count,

                context->device_target,
                context->target_count,

                context->device_v4_cell_offsets,
                context->device_v4_point_indices,

                context->v4_min_x,
                context->v4_min_y,
                context->v4_min_z,

                context->v4_voxel_size,

                context->v4_nx,
                context->v4_ny,
                context->v4_nz,

                context->device_indices,
                context->device_distances,
                context->device_ambiguous,

                ambiguity_abs_epsilon,
                ambiguity_relative_epsilon);
    }
    else
    {
        PersistentKnn6Kernel<<<
            blocks,
            threads,
            0,
            context->stream>>>(
                context->device_query,
                query_count,
                context->device_target,
                context->target_count,
                context->device_indices,
                context->device_distances,
                context->device_ambiguous,
                ambiguity_abs_epsilon,
                ambiguity_relative_epsilon);
    }


    error =
        cudaGetLastError();


    if (error != cudaSuccess)
    {
        stats.cuda_error =
            static_cast<int>(error);

        return false;
    }


    cudaEventRecord(
        context->fused_event_2,
        context->stream);


    // ------------------------------------------------------------
    // Plane / residual / H / b
    // ------------------------------------------------------------

    FusedPlaneHessianKernelV3<<<
        blocks,
        threads,
        0,
        context->stream>>>(
            context->device_target,
            context->target_count,

            context->device_fused_query_double,
            query_count,

            context->device_indices,
            context->device_distances,
            context->device_ambiguous,

            T.m[3],
            T.m[7],
            T.m[11],

            maximum_squared_distance,
            maximum_plane_fit_error,
            maximum_residual,
            huber_delta,

            enable_ground_constraint,
            ground_normal_cosine_threshold,
            ground_min_below_sensor_m,
            ground_max_residual,
            ground_huber_delta,
            ground_weight,

            context->device_fused_cpu_fallback,
            context->device_fused_ranges,
            context->device_fused_range_valid,

            reinterpret_cast<
                FusedAccumulatorV3 *>(
                    context->
                        device_fused_accumulator));


    error =
        cudaGetLastError();


    if (error != cudaSuccess)
    {
        stats.cuda_error =
            static_cast<int>(error);

        return false;
    }


    cudaEventRecord(
        context->fused_event_3,
        context->stream);


    const auto sync_begin =
        std::chrono::steady_clock::now();


    error =
        cudaEventSynchronize(
            context->fused_event_3);


    stats.sync_wait_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() -
            sync_begin)
            .count();


    if (error != cudaSuccess)
    {
        stats.cuda_error =
            static_cast<int>(error);

        return false;
    }


    float transform_ms = 0.0f;
    float knn_ms = 0.0f;
    float geometry_ms = 0.0f;


    cudaEventElapsedTime(
        &transform_ms,
        context->fused_event_0,
        context->fused_event_1);


    cudaEventElapsedTime(
        &knn_ms,
        context->fused_event_1,
        context->fused_event_2);


    cudaEventElapsedTime(
        &geometry_ms,
        context->fused_event_2,
        context->fused_event_3);


    stats.transform_kernel_ms =
        static_cast<double>(
            transform_ms);

    stats.knn_kernel_ms =
        static_cast<double>(
            knn_ms);

    stats.geometry_kernel_ms =
        static_cast<double>(
            geometry_ms);


    // V4 production KNN is executed above.


    // ------------------------------------------------------------
    // Tiny D2H outputs only.
    // ------------------------------------------------------------

    FusedAccumulatorV3 host_accumulator{};


    const auto d2h_accumulator_begin =
        std::chrono::steady_clock::now();


    error =
        cudaMemcpyAsync(
            &host_accumulator,
            context->device_fused_accumulator,
            sizeof(FusedAccumulatorV3),
            cudaMemcpyDeviceToHost,
            context->stream);


    stats.d2h_accumulator_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() -
            d2h_accumulator_begin)
            .count();


    if (error != cudaSuccess)
    {
        stats.cuda_error =
            static_cast<int>(error);

        return false;
    }


    const auto d2h_fallback_begin =
        std::chrono::steady_clock::now();


    error =
        cudaMemcpyAsync(
            output_cpu_fallback,
            context->device_fused_cpu_fallback,
            query_count *
                sizeof(unsigned char),
            cudaMemcpyDeviceToHost,
            context->stream);


    stats.d2h_fallback_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() -
            d2h_fallback_begin)
            .count();


    if (error != cudaSuccess)
    {
        stats.cuda_error =
            static_cast<int>(error);

        return false;
    }


    const auto d2h_ranges_begin =
        std::chrono::steady_clock::now();


    error =
        cudaMemcpyAsync(
            output_ranges,
            context->device_fused_ranges,
            query_count *
                sizeof(double),
            cudaMemcpyDeviceToHost,
            context->stream);


    stats.d2h_ranges_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() -
            d2h_ranges_begin)
            .count();


    if (error != cudaSuccess)
    {
        stats.cuda_error =
            static_cast<int>(error);

        return false;
    }


    const auto d2h_valid_begin =
        std::chrono::steady_clock::now();


    error =
        cudaMemcpyAsync(
            output_range_valid,
            context->device_fused_range_valid,
            query_count *
                sizeof(unsigned char),
            cudaMemcpyDeviceToHost,
            context->stream);


    stats.d2h_valid_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() -
            d2h_valid_begin)
            .count();


    if (error != cudaSuccess)
    {
        stats.cuda_error =
            static_cast<int>(error);

        return false;
    }


    const auto d2h_sync_begin =
        std::chrono::steady_clock::now();


    error =
        cudaStreamSynchronize(
            context->stream);


    stats.d2h_sync_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() -
            d2h_sync_begin)
            .count();


    if (error != cudaSuccess)
    {
        stats.cuda_error =
            static_cast<int>(error);

        return false;
    }


    const auto postprocess_begin =
        std::chrono::steady_clock::now();


    // ------------------------------------------------------------
    // Rebuild symmetric H from upper triangle.
    // ------------------------------------------------------------

    int upper_index = 0;


    for (int r = 0;
         r < 6;
         ++r)
    {
        for (int c = r;
             c < 6;
             ++c)
        {
            const double geometry_value =
                host_accumulator
                    .H_upper[
                        upper_index];

            const double ground_value =
                host_accumulator
                    .H_ground_upper[
                        upper_index];


            output_H_6x6[
                r * 6 + c] =
                    geometry_value;

            output_H_6x6[
                c * 6 + r] =
                    geometry_value;


            output_H_ground_6x6[
                r * 6 + c] =
                    ground_value;

            output_H_ground_6x6[
                c * 6 + r] =
                    ground_value;


            ++upper_index;
        }


        output_b_6[r] =
            host_accumulator.b[r];

        output_b_ground_6[r] =
            host_accumulator.b_ground[r];
    }


    output_raw_sse =
        host_accumulator.raw_sse;

    output_robust_sse =
        host_accumulator.robust_sse;

    output_robust_weight_sum =
        host_accumulator.robust_weight_sum;


    stats.ambiguous_queries =
        static_cast<std::size_t>(
            host_accumulator
                .ambiguous_queries);

    stats.cpu_fallback_queries =
        static_cast<std::size_t>(
            host_accumulator
                .cpu_fallback_queries);

    stats.plane_fit_failures =
        static_cast<std::size_t>(
            host_accumulator
                .plane_fit_failures);

    stats.correspondences =
        static_cast<std::size_t>(
            host_accumulator
                .correspondences);

    stats.downweighted =
        static_cast<std::size_t>(
            host_accumulator
                .downweighted);

    stats.ground_correspondences =
        static_cast<std::size_t>(
            host_accumulator
                .ground_correspondences);

    stats.ground_downweighted =
        static_cast<std::size_t>(
            host_accumulator
                .ground_downweighted);


    stats.postprocess_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() -
            postprocess_begin)
            .count();


    stats.total_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() -
            total_begin)
            .count();


    stats.cuda_error = 0;
    stats.ok = true;

    return true;
}

}  // namespace fr_slam_cuda
