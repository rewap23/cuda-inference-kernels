#pragma once

#include <cuda_runtime.h>

enum class Variant { Naive, Tiled, Vectorized };

void launch_matmul(Variant variant, const float* a, const float* b, float* c, int m, int k, int n);
void launch_conv2d(Variant variant, const float* input, const float* filter, float* output, int height, int width);
void launch_softmax(Variant variant, const float* input, float* output, int rows, int cols);
void launch_layernorm(Variant variant, const float* input, float* output, int rows, int cols);