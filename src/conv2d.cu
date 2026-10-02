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
constexpr int kFilterSize = 3;
constexpr int kConvTile = 16;
}

// Convolution Naive Kernel
// Each CUDA thread computes one output element - very slow
__global__ void conv2d_naive_kernel(const float* input, const float* filter, float* output, int height, int width) {
	const int out_width = width - 2;
	const int out_height = height - 2;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= out_width || y >= out_height) return;

	float value = 0.0f;
	for (int fy = 0; fy < kFilterSize; ++fy) {
		for (int fx = 0; fx < kFilterSize; ++fx) {
			value += input[(y + fy) * width + x + fx] * filter[fy * kFilterSize + fx];
		}
	}
	output[y * out_width + x] = value;
}

// Convolution Tiled Kernel

// Each block computes a kConvTile x kConvTile tile of the output - very fast
__global__ void conv2d_tiled_kernel(const float* input, const float* filter, float* output, int height, int width) {
	__shared__ float tile[kConvTile + 2][kConvTile + 2];
	const int out_width = width - 2;
	const int out_height = height - 2;
	const int local_index = threadIdx.y * kConvTile + threadIdx.x;
	const int tile_elements = (kConvTile + 2) * (kConvTile + 2);
	const int tile_origin_x = blockIdx.x * kConvTile;
	const int tile_origin_y = blockIdx.y * kConvTile;

	// 1) Load the input tile into shared memory.
	for (int index = local_index; index < tile_elements; index += kConvTile * kConvTile) {
		const int local_y = index / (kConvTile + 2);
		const int local_x = index % (kConvTile + 2);
		const int input_y = tile_origin_y + local_y;
		const int input_x = tile_origin_x + local_x;
		tile[local_y][local_x] = (input_y < height && input_x < width)
			? input[input_y * width + input_x] : 0.0f;
	}

	// 2) Sync barrier: wait for all threads to finish loading the tile.
	__syncthreads();

	const int x = tile_origin_x + threadIdx.x;
	const int y = tile_origin_y + threadIdx.y;
	if (x < out_width && y < out_height) {
		float value = 0.0f;
		// 3) Compute the convolution using the shared tile data.
		#pragma unroll
		for (int fy = 0; fy < kFilterSize; ++fy) {
			#pragma unroll
			for (int fx = 0; fx < kFilterSize; ++fx) {
				value += tile[threadIdx.y + fy][threadIdx.x + fx] * filter[fy * kFilterSize + fx];
			}
		}
		output[y * out_width + x] = value;
	}
}

__global__ void conv2d_vectorized_kernel(const float* input, const float* filter, float* output, int height, int width) {
	const int out_width = width - 2;
	const int out_height = height - 2;
	const int x = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (y >= out_height || x >= out_width) return;

	float4 values = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	#pragma unroll
	for (int fy = 0; fy < kFilterSize; ++fy) {
		#pragma unroll
		for (int fx = 0; fx < kFilterSize; ++fx) {
			const float weight = filter[fy * kFilterSize + fx];
			const int base = (y + fy) * width + x + fx;
			if (x + 3 < out_width) {
				values.x += input[base] * weight;
				values.y += input[base + 1] * weight;
				values.z += input[base + 2] * weight;
				values.w += input[base + 3] * weight;
			} else {
				if (x < out_width) values.x += input[base] * weight;
				if (x + 1 < out_width) values.y += input[base + 1] * weight;
				if (x + 2 < out_width) values.z += input[base + 2] * weight;
				if (x + 3 < out_width) values.w += input[base + 3] * weight;
			}
		}
	}

	if (x + 3 < out_width) {
		*reinterpret_cast<float4*>(output + y * out_width + x) = values;
	} else {
		if (x < out_width) output[y * out_width + x] = values.x;
		if (x + 1 < out_width) output[y * out_width + x + 1] = values.y;
		if (x + 2 < out_width) output[y * out_width + x + 2] = values.z;
		if (x + 3 < out_width) output[y * out_width + x + 3] = values.w;
	}
}

void launch_conv2d(Variant variant, const float* input, const float* filter, float* output, int height, int width) {
	const dim3 block(kConvTile, kConvTile);
	if (variant == Variant::Naive) {
		const dim3 grid((width - 2 + block.x - 1) / block.x, (height - 2 + block.y - 1) / block.y);
		// 3) Launch the naive kernel.
		conv2d_naive_kernel<<<grid, block>>>(input, filter, output, height, width);
	} else if (variant == Variant::Tiled) {
		const dim3 grid((width - 2 + kConvTile - 1) / kConvTile, (height - 2 + kConvTile - 1) / kConvTile);
		// 3) Launch the tiled kernel.
		conv2d_tiled_kernel<<<grid, block>>>(input, filter, output, height, width);
	} else {
		const dim3 grid(((width - 2 + 3) / 4 + block.x - 1) / block.x,
						(height - 2 + block.y - 1) / block.y);
		// 3) Launch the vectorized kernel.
		conv2d_vectorized_kernel<<<grid, block>>>(input, filter, output, height, width);
	}
}
