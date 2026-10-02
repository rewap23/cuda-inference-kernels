#include <cuda_runtime.h>
#include "kernels.cuh"

// GPU workflow:
// 1) Allocate memory on the GPU.
// 2) Copy data from the host to the device.
// 3) Launch the kernel.
// 4) Copy data from the device back to the host.
// 5) Free the memory on the GPU.
// Tiled computation:
// 1) Load tile data.
// 2) __syncthreads() barrier.
// 3) Compute.

namespace {
constexpr int kBlockSize = 256;
constexpr float kEpsilon = 1.0e-5f;

__device__ float block_sum(float value, float* partial) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_down_sync(0xffffffff, value, offset);
    if (lane == 0) partial[warp] = value;
    __syncthreads();
    value = threadIdx.x < 8 ? partial[lane] : 0.0f;
    if (warp == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_down_sync(0xffffffff, value, offset);
    }
    if (threadIdx.x == 0) partial[0] = value;
    __syncthreads();
    return partial[0];
}

__global__ void layernorm_naive_kernel(const float* input, float* output, int rows, int cols) {
    const int row = blockIdx.x;
    if (row >= rows || threadIdx.x != 0) return;
    const float* row_input = input + row * cols;
    float mean = 0.0f;
    for (int col = 0; col < cols; ++col) mean += row_input[col];
    mean /= cols;
    float variance = 0.0f;
    for (int col = 0; col < cols; ++col) {
        const float delta = row_input[col] - mean;
        variance += delta * delta;
    }
    variance /= cols;
    const float inverse_stddev = rsqrtf(variance + kEpsilon);
    for (int col = 0; col < cols; ++col) output[row * cols + col] = (row_input[col] - mean) * inverse_stddev;
}

__global__ void layernorm_tiled_kernel(const float* input, float* output, int rows, int cols) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float partial[8];
    const float* row_input = input + row * cols;
    float local_sum = 0.0f;
    // 1) Load partial sums into shared memory for the row reduction.
    for (int col = threadIdx.x; col < cols; col += kBlockSize) local_sum += row_input[col];
    // 2) Sync barrier: wait for all threads to finish the reduction load.
    __syncthreads();
    const float mean = block_sum(local_sum, partial) / cols;
    float local_squared_delta = 0.0f;
    for (int col = threadIdx.x; col < cols; col += kBlockSize) {
        const float delta = row_input[col] - mean;
        local_squared_delta += delta * delta;
    }
    // 3) Compute the variance and normalized output using the tiled partial sums.
    const float variance = block_sum(local_squared_delta, partial) / cols;
    const float inverse_stddev = rsqrtf(variance + kEpsilon);
    for (int col = threadIdx.x; col < cols; col += kBlockSize) output[row * cols + col] = (row_input[col] - mean) * inverse_stddev;
}

__global__ void layernorm_vectorized_kernel(const float* input, float* output, int rows, int cols) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float partial[8];
    const float* row_input = input + row * cols;
    float local_sum = 0.0f;
    for (int base = threadIdx.x * 4; base < cols; base += kBlockSize * 4) {
        if (base + 3 < cols && cols % 4 == 0) {
            const float4 value = *reinterpret_cast<const float4*>(row_input + base);
            local_sum += value.x + value.y + value.z + value.w;
        } else {
            for (int lane = 0; lane < 4 && base + lane < cols; ++lane) local_sum += row_input[base + lane];
        }
    }
    const float mean = block_sum(local_sum, partial) / cols;
    float local_squared_delta = 0.0f;
    for (int base = threadIdx.x * 4; base < cols; base += kBlockSize * 4) {
        if (base + 3 < cols && cols % 4 == 0) {
            const float4 value = *reinterpret_cast<const float4*>(row_input + base);
            const float dx = value.x - mean;
            const float dy = value.y - mean;
            const float dz = value.z - mean;
            const float dw = value.w - mean;
            local_squared_delta += dx * dx + dy * dy + dz * dz + dw * dw;
        } else {
            for (int lane = 0; lane < 4 && base + lane < cols; ++lane) {
                const float delta = row_input[base + lane] - mean;
                local_squared_delta += delta * delta;
            }
        }
    }
    const float variance = block_sum(local_squared_delta, partial) / cols;
    const float inverse_stddev = rsqrtf(variance + kEpsilon);
    float* row_output = output + row * cols;
    for (int base = threadIdx.x * 4; base < cols; base += kBlockSize * 4) {
        if (base + 3 < cols && cols % 4 == 0) {
            const float4 value = *reinterpret_cast<const float4*>(row_input + base);
            const float4 result = make_float4((value.x - mean) * inverse_stddev, (value.y - mean) * inverse_stddev,
                                              (value.z - mean) * inverse_stddev, (value.w - mean) * inverse_stddev);
            *reinterpret_cast<float4*>(row_output + base) = result;
        } else {
            for (int lane = 0; lane < 4 && base + lane < cols; ++lane) row_output[base + lane] = (row_input[base + lane] - mean) * inverse_stddev;
        }
    }
}
}

void launch_layernorm(Variant variant, const float* input, float* output, int rows, int cols) {
    if (variant == Variant::Naive) {
        // 3) Launch the naive kernel.
        layernorm_naive_kernel<<<rows, 1>>>(input, output, rows, cols);
    } else if (variant == Variant::Tiled) {
        // 3) Launch the tiled kernel.
        layernorm_tiled_kernel<<<rows, kBlockSize>>>(input, output, rows, cols);
    } else {
        // 3) Launch the vectorized kernel.
        layernorm_vectorized_kernel<<<rows, kBlockSize>>>(input, output, rows, cols);
    }
}