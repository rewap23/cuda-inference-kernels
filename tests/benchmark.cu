#include <cuda_runtime.h>
#include "kernels.cuh"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check_cuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) : count_(count) {
        // 1) Allocate memory on the GPU.
        check_cuda(cudaMalloc(&data_, count * sizeof(T)), "cudaMalloc");
    }
    ~DeviceBuffer() {
        // 5) Free the memory on the GPU.
        if (data_) cudaFree(data_);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() { return data_; }
    const T* get() const { return data_; }
    size_t size() const { return count_; }
    void copy_from(const std::vector<T>& source) {
        // 2) Copy data from the host to the device.
        check_cuda(cudaMemcpy(data_, source.data(), count_ * sizeof(T), cudaMemcpyHostToDevice), "copy to device");
    }
    std::vector<T> copy_to_host() const {
        std::vector<T> result(count_);
        // 4) Copy data from the device back to the host.
        check_cuda(cudaMemcpy(result.data(), data_, count_ * sizeof(T), cudaMemcpyDeviceToHost), "copy to host");
        return result;
    }
private:
    T* data_ = nullptr;
    size_t count_;
};

void require_close(const std::vector<float>& actual, const std::vector<float>& expected, float tolerance, const std::string& label) {
    if (actual.size() != expected.size()) throw std::runtime_error(label + ": output size mismatch");
    for (size_t i = 0; i < actual.size(); ++i) {
        const float limit = tolerance * std::max(1.0f, std::fabs(expected[i]));
        if (!std::isfinite(actual[i]) || std::fabs(actual[i] - expected[i]) > limit) {
            throw std::runtime_error(label + ": correctness mismatch at element " + std::to_string(i));
        }
    }
}

template <typename Launch>
float measure_us(Launch launch) {
    for (int i = 0; i < 5; ++i) launch();
    check_cuda(cudaGetLastError(), "warmup launch");
    check_cuda(cudaDeviceSynchronize(), "warmup synchronize");
    constexpr int kIterations = 30;
    std::vector<float> samples;
    samples.reserve(7);
    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start), "create start event");
    check_cuda(cudaEventCreate(&stop), "create stop event");
    for (int round = 0; round < 7; ++round) {
        check_cuda(cudaEventRecord(start), "record start event");
        for (int i = 0; i < kIterations; ++i) launch();
        check_cuda(cudaEventRecord(stop), "record stop event");
        check_cuda(cudaEventSynchronize(stop), "synchronize stop event");
        float elapsed_ms = 0.0f;
        check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "measure events");
        samples.push_back(elapsed_ms * 1000.0f / kIterations);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

std::vector<float> make_data(size_t count, std::mt19937& generator) {
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    std::vector<float> values(count);
    for (float& value : values) value = distribution(generator);
    return values;
}

std::vector<float> matmul_reference(const std::vector<float>& a, const std::vector<float>& b, int m, int k, int n) {
    std::vector<float> output(static_cast<size_t>(m) * n, 0.0f);
    for (int row = 0; row < m; ++row) for (int col = 0; col < n; ++col)
        for (int inner = 0; inner < k; ++inner) output[row * n + col] += a[row * k + inner] * b[inner * n + col];
    return output;
}

std::vector<float> conv_reference(const std::vector<float>& input, const std::vector<float>& filter, int height, int width) {
    const int out_height = height - 2, out_width = width - 2;
    std::vector<float> output(static_cast<size_t>(out_height) * out_width, 0.0f);
    for (int y = 0; y < out_height; ++y) for (int x = 0; x < out_width; ++x)
        for (int fy = 0; fy < 3; ++fy) for (int fx = 0; fx < 3; ++fx)
            output[y * out_width + x] += input[(y + fy) * width + x + fx] * filter[fy * 3 + fx];
    return output;
}

std::vector<float> softmax_reference(const std::vector<float>& input, int rows, int cols) {
    std::vector<float> output(input.size());
    for (int row = 0; row < rows; ++row) {
        const float* source = input.data() + row * cols;
        const float maximum = *std::max_element(source, source + cols);
        float denominator = 0.0f;
        for (int col = 0; col < cols; ++col) denominator += std::exp(source[col] - maximum);
        for (int col = 0; col < cols; ++col) output[row * cols + col] = std::exp(source[col] - maximum) / denominator;
    }
    return output;
}

std::vector<float> layernorm_reference(const std::vector<float>& input, int rows, int cols) {
    std::vector<float> output(input.size());
    for (int row = 0; row < rows; ++row) {
        const float* source = input.data() + row * cols;
        float mean = 0.0f;
        for (int col = 0; col < cols; ++col) mean += source[col];
        mean /= cols;
        float variance = 0.0f;
        for (int col = 0; col < cols; ++col) variance += (source[col] - mean) * (source[col] - mean);
        variance /= cols;
        const float inverse_stddev = 1.0f / std::sqrt(variance + 1.0e-5f);
        for (int col = 0; col < cols; ++col) output[row * cols + col] = (source[col] - mean) * inverse_stddev;
    }
    return output;
}

void write_row(std::ofstream& csv, const std::string& op, const std::string& variant, float latency, float baseline) {
    csv << op << ',' << variant << ',' << std::fixed << std::setprecision(3) << latency << ',' << baseline / latency << '\n';
    std::cout << std::left << std::setw(12) << op << std::setw(12) << variant
              << std::right << std::fixed << std::setprecision(3) << std::setw(12) << latency
              << std::setw(12) << baseline / latency << "x\n";
}

void benchmark_matmul(std::ofstream& csv, std::mt19937& generator) {
    constexpr int m = 384, k = 384, n = 384;
    auto a = make_data(static_cast<size_t>(m) * k, generator);
    auto b = make_data(static_cast<size_t>(k) * n, generator);
    auto expected = matmul_reference(a, b, m, k, n);
    DeviceBuffer<float> da(a.size()), db(b.size()), dc(expected.size());
    da.copy_from(a); db.copy_from(b);
    const Variant variants[] = {Variant::Naive, Variant::Tiled, Variant::Vectorized};
    const char* names[] = {"naive", "tiled", "vectorized"};
    float baseline = 0.0f;
    for (int i = 0; i < 3; ++i) {
        auto launch = [&] { launch_matmul(variants[i], da.get(), db.get(), dc.get(), m, k, n); };
        launch(); check_cuda(cudaGetLastError(), "matmul launch"); check_cuda(cudaDeviceSynchronize(), "matmul synchronize");
        require_close(dc.copy_to_host(), expected, 2.0e-3f, std::string("matmul/") + names[i]);
        const float latency = measure_us(launch);
        if (i == 0) baseline = latency;
        write_row(csv, "matmul", names[i], latency, baseline);
    }
}

void benchmark_conv(std::ofstream& csv, std::mt19937& generator) {
    constexpr int height = 512, width = 512;
    auto input = make_data(static_cast<size_t>(height) * width, generator);
    auto filter = make_data(9, generator);
    auto expected = conv_reference(input, filter, height, width);
    DeviceBuffer<float> din(input.size()), df(9), dout(expected.size());
    din.copy_from(input); df.copy_from(filter);
    const Variant variants[] = {Variant::Naive, Variant::Tiled, Variant::Vectorized};
    const char* names[] = {"naive", "tiled", "vectorized"};
    float baseline = 0.0f;
    for (int i = 0; i < 3; ++i) {
        auto launch = [&] { launch_conv2d(variants[i], din.get(), df.get(), dout.get(), height, width); };
        launch(); check_cuda(cudaGetLastError(), "conv launch"); check_cuda(cudaDeviceSynchronize(), "conv synchronize");
        require_close(dout.copy_to_host(), expected, 2.0e-4f, std::string("conv2d/") + names[i]);
        const float latency = measure_us(launch);
        if (i == 0) baseline = latency;
        write_row(csv, "conv2d", names[i], latency, baseline);
    }
}

void benchmark_row_ops(std::ofstream& csv, std::mt19937& generator, bool is_softmax) {
    constexpr int rows = 2048, cols = 512;
    auto input = make_data(static_cast<size_t>(rows) * cols, generator);
    auto expected = is_softmax ? softmax_reference(input, rows, cols) : layernorm_reference(input, rows, cols);
    DeviceBuffer<float> din(input.size()), dout(expected.size());
    din.copy_from(input);
    const Variant variants[] = {Variant::Naive, Variant::Tiled, Variant::Vectorized};
    const char* names[] = {"naive", "tiled", "vectorized"};
    const char* op = is_softmax ? "softmax" : "layernorm";
    float baseline = 0.0f;
    for (int i = 0; i < 3; ++i) {
        auto launch = [&] {
            if (is_softmax) launch_softmax(variants[i], din.get(), dout.get(), rows, cols);
            else launch_layernorm(variants[i], din.get(), dout.get(), rows, cols);
        };
        launch(); check_cuda(cudaGetLastError(), "row operator launch"); check_cuda(cudaDeviceSynchronize(), "row operator synchronize");
        require_close(dout.copy_to_host(), expected, 3.0e-4f, std::string(op) + "/" + names[i]);
        const float latency = measure_us(launch);
        if (i == 0) baseline = latency;
        write_row(csv, op, names[i], latency, baseline);
    }
}
}

int main(int argc, char** argv) {
    try {
        int device = 0;
        check_cuda(cudaSetDevice(device), "select device");
        cudaDeviceProp properties{};
        check_cuda(cudaGetDeviceProperties(&properties, device), "get device properties");
        const std::string path = argc > 1 ? argv[1] : "benchmark_results.csv";
        std::ofstream csv(path);
        if (!csv) throw std::runtime_error("cannot open output CSV: " + path);
        csv << "operator,variant,latency_us,speedup_vs_naive\n";
        std::cout << "GPU: " << properties.name << "\nResults: " << path << "\n";
        std::cout << std::left << std::setw(12) << "operator" << std::setw(12) << "variant"
                  << std::right << std::setw(12) << "us" << std::setw(13) << "speedup" << "\n";
        std::mt19937 generator(2026);
        benchmark_matmul(csv, generator);
        benchmark_conv(csv, generator);
        benchmark_row_ops(csv, generator, true);
        benchmark_row_ops(csv, generator, false);
        check_cuda(cudaDeviceSynchronize(), "final synchronize");
        std::cout << "Correctness checks passed.\n";
    } catch (const std::exception& error) {
        std::cerr << "benchmark failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}