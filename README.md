# CUDA Inference Kernels

Four float32 operators are implemented in plain C++/CUDA, each with a deliberately simple baseline and two GPU-oriented variants. The goal is to make the exact math of each operator obvious while comparing naive, tiled, and vectorized implementations.

## What each kernel computes

- Matmul: multiplies two matrices A and B to form C = A @ B. In this project, A is `[M, K]`, B is `[K, N]`, and each output element is the dot product of one row of A with one column of B. This is the core GEMM pattern used by attention and MLP layers.
- Conv2D: performs a valid single-channel 3x3 convolution with no padding and stride 1. Each output pixel is the sum of the 3x3 input neighborhood multiplied by the 3x3 filter weights, so the output size is `(H - 2) x (W - 2)` for an input `H x W`.
- Softmax: applies the standard row-wise softmax across the last dimension. For each row, it subtracts the maximum value for numerical stability, exponentiates the shifted values, then normalizes by their sum so the row sums to 1.
- LayerNorm: normalizes each row independently along the last dimension. It computes the row mean and variance, then applies `(x - mean) / sqrt(variance + 1e-5)`. This keeps activations centered and scaled without using learned affine parameters.

## Implementation variants

| Operator | Naive | Tiled | Vectorized |
| --- | --- | --- | --- |
| Matmul | One thread per output element; each thread loops over the K dimension and rereads global-memory operands | Each block loads a tile of A and B into shared memory, then reuses those tiles across the dot-product accumulation | Computes four adjacent output columns per thread and uses aligned `float4` loads/stores when possible |
| Conv2D | One thread per valid output pixel; each thread walks the 3x3 window directly from global memory | Each block cooperatively loads an input tile plus halo into shared memory, then computes the convolution from that tile | Computes four neighboring output pixels per thread and stores a `float4` when aligned |
| Softmax | One thread processes a complete row by scanning all columns and computing max/denominator in sequence | One block per row uses shared-memory reductions to compute the row max and denominator in parallel | Processes four contiguous values per thread with vector loads and a shared-memory reduction for the row-wise normalization |
| LayerNorm | One thread processes a complete row by computing the mean and variance in a scalar loop | One block per row computes mean and variance with parallel reductions in shared memory | Four-value vector loads/stores and parallel reductions compute the same normalization more efficiently |

The conv operator is valid (no padding), stride 1, and single-channel. Softmax and layernorm operate on the last dimension; layernorm uses epsilon `1e-5` and no learned scale/bias. These are teaching kernels, not replacements for cuBLAS/cuDNN or production inference libraries.

## Run On Google Colab

1. In Colab, choose **Runtime > Change runtime type > T4 GPU** (or another GPU runtime).
2. Upload this repository as a ZIP, then run:

```bash
!unzip -q cuda-inference-kernels.zip
%cd cuda-inference-kernels
!nvidia-smi
!cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
!cmake --build build -j2
!./build/kernel_benchmark benchmark_results.csv
```

If you put the repository on GitHub instead, replace the unzip and `%cd` cells with `!git clone <your-repository-url>` and `%cd <repository-folder>`. Colab's standard GPU images include CUDA and CMake; the initial compilation can take a minute.

The executable first checks every result against a CPU reference. It then warms up each kernel and measures seven CUDA-event rounds of 30 launches each, reporting the median per-launch latency. It prints the GPU model and a table, and writes `benchmark_results.csv` with latency in microseconds and speedup relative to that operator's naive kernel. The CSV is the measured result; timings depend on the assigned GPU, clock state, and Colab load. Do not compare measurements from different GPU models as if they were the same test.

## Short Writeup

The progression tests three distinct sources of throughput. Tiling should help matmul and convolution by reusing data loaded from global memory. Row-wise softmax and layernorm benefit from distributing reductions across a block instead of assigning a whole row to one thread. The vectorized versions process adjacent values together and use 128-bit memory operations where alignment allows; they can reduce instruction overhead, but are not guaranteed to win when an operator is already bandwidth-bound or the workload is small.

After running in Colab, copy the printed per-operator speedups into your report and include the GPU model printed by the executable. Report the measured latency as well as speedup, since a large ratio on a very slow baseline can obscure the absolute runtime. There are intentionally no pre-filled speedup claims: no GPU benchmark has been run for this repository yet.


