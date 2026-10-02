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
constexpr int kMatmulTileSize = 16;
}  // namespace

// A: [M, K]
// B: [K, N]
// C: [M, N]

// MatMul Naive Kernel
// Each CUDA thread computes one output element C[row, col] - very slow
__global__ void matmul_naive_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int m, int k, int n) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= m || col >= n) {
        return;
    }

    float accumulator = 0.0f;

    for (int inner = 0; inner < k; ++inner) {
        accumulator += a[row * k + inner] * b[inner * n + col];
    }

    c[row * n + col] = accumulator;
}

// A: [M, K]
// B: [K, N]
// C: [M, N]

// Launch with:
// dim3 block(kMatmulTileSize, kMatmulTileSize);
// dim3 grid(
//     (n + kMatmulTileSize - 1) / kMatmulTileSize,
//     (m + kMatmulTileSize - 1) / kMatmulTileSize
// );

// Each block computes a kMatmulTileSize x kMatmulTileSize tile of C - very fast
__global__ void matmul_tiled_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int m, int k, int n) {
    __shared__ float tile_a[kMatmulTileSize][kMatmulTileSize];
    __shared__ float tile_b[kMatmulTileSize][kMatmulTileSize];

    const int local_row = threadIdx.y;
    const int local_col = threadIdx.x;

    const int row = blockIdx.y * kMatmulTileSize + local_row;
    const int col = blockIdx.x * kMatmulTileSize + local_col;

    float accumulator = 0.0f;

    const int num_k_tiles =
        (k + kMatmulTileSize - 1) / kMatmulTileSize;

    for (int tile = 0; tile < num_k_tiles; ++tile) {
        const int a_col = tile * kMatmulTileSize + local_col;
        const int b_row = tile * kMatmulTileSize + local_row;

        // 1) Load the tile values into shared memory.
        tile_a[local_row][local_col] =
            (row < m && a_col < k)
                ? a[row * k + a_col]
                : 0.0f;

        tile_b[local_row][local_col] =
            (b_row < k && col < n)
                ? b[b_row * n + col]
                : 0.0f;

        // 2) Sync barrier: wait for all threads to finish loading the tile.
        __syncthreads();

        // 3) Compute the partial dot product using the shared tile data.
        #pragma unroll
        for (int inner = 0; inner < kMatmulTileSize; ++inner) {
            accumulator +=
                tile_a[local_row][inner] *
                tile_b[inner][local_col];
        }

        // Sync barrier: ensure each thread finishes this tile before next tile loads.
        __syncthreads();
    }

    if (row < m && col < n) {
        c[row * n + col] = accumulator;
    }
}

__global__ void matmul_vectorized_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int m, int k, int n) {
    constexpr int kColumnTile = kMatmulTileSize * 4;
    __shared__ float tile_a[kMatmulTileSize][kMatmulTileSize];
    __shared__ float tile_b[kMatmulTileSize][kColumnTile];

    const int local_row = threadIdx.y;
    const int local_col = threadIdx.x * 4;
    const int row = blockIdx.y * kMatmulTileSize + local_row;
    const int col = blockIdx.x * kColumnTile + local_col;
    float4 accumulator = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    const int num_k_tiles = (k + kMatmulTileSize - 1) / kMatmulTileSize;

    for (int tile = 0; tile < num_k_tiles; ++tile) {
        const int a_col = tile * kMatmulTileSize + threadIdx.x;
        const int b_row = tile * kMatmulTileSize + local_row;
        tile_a[local_row][threadIdx.x] =
            (row < m && a_col < k) ? a[row * k + a_col] : 0.0f;

        if (b_row < k && col + 3 < n && n % 4 == 0) {
            const float4 values = *reinterpret_cast<const float4*>(b + b_row * n + col);
            tile_b[local_row][col - blockIdx.x * kColumnTile + 0] = values.x;
            tile_b[local_row][col - blockIdx.x * kColumnTile + 1] = values.y;
            tile_b[local_row][col - blockIdx.x * kColumnTile + 2] = values.z;
            tile_b[local_row][col - blockIdx.x * kColumnTile + 3] = values.w;
        } else {
            #pragma unroll
            for (int offset = 0; offset < 4; ++offset) {
                const int shared_col = threadIdx.x * 4 + offset;
                tile_b[local_row][shared_col] =
                    (b_row < k && col + offset < n) ? b[b_row * n + col + offset] : 0.0f;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int inner = 0; inner < kMatmulTileSize; ++inner) {
            const float lhs = tile_a[local_row][inner];
            accumulator.x += lhs * tile_b[inner][local_col + 0];
            accumulator.y += lhs * tile_b[inner][local_col + 1];
            accumulator.z += lhs * tile_b[inner][local_col + 2];
            accumulator.w += lhs * tile_b[inner][local_col + 3];
        }
        __syncthreads();
    }

    if (row < m && col + 3 < n) {
        *reinterpret_cast<float4*>(c + row * n + col) = accumulator;
    } else if (row < m) {
        if (col < n) c[row * n + col] = accumulator.x;
        if (col + 1 < n) c[row * n + col + 1] = accumulator.y;
        if (col + 2 < n) c[row * n + col + 2] = accumulator.z;
        if (col + 3 < n) c[row * n + col + 3] = accumulator.w;
    }
}

void launch_matmul(Variant variant, const float* a, const float* b, float* c, int m, int k, int n) {
    const dim3 block(kMatmulTileSize, kMatmulTileSize);
    if (variant == Variant::Naive) {
        const dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
        // 3) Launch the naive kernel.
        matmul_naive_kernel<<<grid, block>>>(a, b, c, m, k, n);
    } else if (variant == Variant::Tiled) {
        const dim3 grid((n + kMatmulTileSize - 1) / kMatmulTileSize,
                        (m + kMatmulTileSize - 1) / kMatmulTileSize);
        // 3) Launch the tiled kernel.
        matmul_tiled_kernel<<<grid, block>>>(a, b, c, m, k, n);
    } else {
        constexpr int column_tile = kMatmulTileSize * 4;
        const dim3 grid((n + column_tile - 1) / column_tile,
                        (m + kMatmulTileSize - 1) / kMatmulTileSize);
        // 3) Launch the vectorized kernel.
        matmul_vectorized_kernel<<<grid, block>>>(a, b, c, m, k, n);
    }
}