#include <cuda_runtime.h>

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

        // Every thread initializes its shared-memory locations
        tile_a[local_row][local_col] =
            (row < m && a_col < k)
                ? a[row * k + a_col]
                : 0.0f;

        tile_b[local_row][local_col] =
            (b_row < k && col < n)
                ? b[b_row * n + col]
                : 0.0f;

        // All threads must finish loading before any thread reads a tile
        __syncthreads();

        #pragma unroll
        for (int inner = 0; inner < kMatmulTileSize; ++inner) {
            accumulator +=
                tile_a[local_row][inner] *
                tile_b[inner][local_col];
        }

        // All threads must finish using the current tiles before another
        // iteration overwrites them
        __syncthreads();
    }

    if (row < m && col < n) {
        c[row * n + col] = accumulator;
    }
}