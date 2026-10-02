cmake_minimum_required(VERSION 3.24)
project(cuda_inference_kernels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(kernel_benchmark
	src/matmul.cu
	src/conv2d.cu
	src/softmax.cu
	src/layernorm.cu
	tests/benchmark.cu
)
target_include_directories(kernel_benchmark PRIVATE include)
target_compile_options(kernel_benchmark PRIVATE
	$<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)
