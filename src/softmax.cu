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

// 
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

__device__ float block_max(float value, float* partial) {
	const int lane = threadIdx.x & 31; 
	const int warp = threadIdx.x >> 5;
	for (int offset = 16; offset > 0; offset >>= 1) value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
	if (lane == 0) partial[warp] = value;
	__syncthreads();
	value = threadIdx.x < 8 ? partial[lane] : -CUDART_INF_F;
	if (warp == 0) {
		for (int offset = 16; offset > 0; offset >>= 1) value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
	}
	if (threadIdx.x == 0) partial[0] = value;
	__syncthreads();
	return partial[0];
}
 // Softmax Naive Kernel
__global__ void softmax_naive_kernel(const float* input, float* output, int rows, int cols) {
	const int row = blockIdx.x;
	if (row >= rows || threadIdx.x != 0) return;
	const float* row_input = input + row * cols;
	float maximum = -CUDART_INF_F;
	for (int col = 0; col < cols; ++col) maximum = fmaxf(maximum, row_input[col]);
	float denominator = 0.0f;
	for (int col = 0; col < cols; ++col) denominator += expf(row_input[col] - maximum);
	for (int col = 0; col < cols; ++col) output[row * cols + col] = expf(row_input[col] - maximum) / denominator;
}

__global__ void softmax_tiled_kernel(const float* input, float* output, int rows, int cols) {
	const int row = blockIdx.x;
	if (row >= rows) return;
	__shared__ float partial[8];
	const float* row_input = input + row * cols;
	float local_max = -CUDART_INF_F;
	// 1) Load partial maxima into shared memory for the reduction.
	for (int col = threadIdx.x; col < cols; col += kBlockSize) local_max = fmaxf(local_max, row_input[col]);
	// 2) Sync barrier: regroup partial maxima before computing the global max.
	__syncthreads();
	const float maximum = block_max(local_max, partial);
	float local_sum = 0.0f;
	for (int col = threadIdx.x; col < cols; col += kBlockSize) local_sum += expf(row_input[col] - maximum);
	// 3) Compute the softmax denominator and normalize each value using the tiled reduction.
	const float denominator = block_sum(local_sum, partial);
	for (int col = threadIdx.x; col < cols; col += kBlockSize) output[row * cols + col] = expf(row_input[col] - maximum) / denominator;
}

__global__ void softmax_vectorized_kernel(const float* input, float* output, int rows, int cols) {
	const int row = blockIdx.x;
	if (row >= rows) return;
	__shared__ float partial[8];
	const float* row_input = input + row * cols;
	float local_max = -CUDART_INF_F;
	for (int base = threadIdx.x * 4; base < cols; base += kBlockSize * 4) {
		if (base + 3 < cols && cols % 4 == 0) {
			const float4 value = *reinterpret_cast<const float4*>(row_input + base);
			local_max = fmaxf(local_max, fmaxf(fmaxf(value.x, value.y), fmaxf(value.z, value.w)));
		} else {
			for (int lane = 0; lane < 4 && base + lane < cols; ++lane) local_max = fmaxf(local_max, row_input[base + lane]);
		}
	}
	const float maximum = block_max(local_max, partial);
	float local_sum = 0.0f;
	for (int base = threadIdx.x * 4; base < cols; base += kBlockSize * 4) {
		if (base + 3 < cols && cols % 4 == 0) {
			const float4 value = *reinterpret_cast<const float4*>(row_input + base);
			local_sum += expf(value.x - maximum) + expf(value.y - maximum) + expf(value.z - maximum) + expf(value.w - maximum);
		} else {
			for (int lane = 0; lane < 4 && base + lane < cols; ++lane) local_sum += expf(row_input[base + lane] - maximum);
		}
	}
	const float denominator = block_sum(local_sum, partial);
	float* row_output = output + row * cols;
	for (int base = threadIdx.x * 4; base < cols; base += kBlockSize * 4) {
		if (base + 3 < cols && cols % 4 == 0) {
			const float4 value = *reinterpret_cast<const float4*>(row_input + base);
			const float4 result = make_float4(expf(value.x - maximum) / denominator, expf(value.y - maximum) / denominator,
											  expf(value.z - maximum) / denominator, expf(value.w - maximum) / denominator);
			*reinterpret_cast<float4*>(row_output + base) = result;
		} else {
			for (int lane = 0; lane < 4 && base + lane < cols; ++lane) row_output[base + lane] = expf(row_input[base + lane] - maximum) / denominator;
		}
	}
}
}

void launch_softmax(Variant variant, const float* input, float* output, int rows, int cols) {
	if (variant == Variant::Naive) {
		// 3) Launch the naive kernel.
		softmax_naive_kernel<<<rows, 1>>>(input, output, rows, cols);
	} else if (variant == Variant::Tiled) {
		// 3) Launch the tiled kernel.
		softmax_tiled_kernel<<<rows, kBlockSize>>>(input, output, rows, cols);
	} else {
		// 3) Launch the vectorized kernel.
		softmax_vectorized_kernel<<<rows, kBlockSize>>>(input, output, rows, cols);
	}
}
