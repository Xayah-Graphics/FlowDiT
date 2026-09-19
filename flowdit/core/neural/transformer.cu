#include "transformer-kernels.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
namespace flowdit::neural::kernels {
    namespace {
        __device__ float reduce(float value, float* shared) {
            shared[threadIdx.x] = value;
            __syncthreads();
            for (unsigned n = blockDim.x / 2; n; n /= 2) {
                if (threadIdx.x < n) shared[threadIdx.x] += shared[threadIdx.x + n];
                __syncthreads();
            }
            return shared[0];
        }
        __global__ void norm_kernel(const __nv_bfloat16* x, const float* w, const float* b, __nv_bfloat16* y, float* mean, float* inverse, unsigned width) {
            __shared__ float shared[256];
            const unsigned row = blockIdx.x;
            float sum          = 0.f;
            for (unsigned c = threadIdx.x; c < width; c += blockDim.x) sum += __bfloat162float(x[row * width + c]);
            const float m = reduce(sum, shared) / width;
            __syncthreads();
            sum = 0.f;
            for (unsigned c = threadIdx.x; c < width; c += blockDim.x) {
                const float v = __bfloat162float(x[row * width + c]) - m;
                sum += v * v;
            }
            const float r = rsqrtf(reduce(sum, shared) / width + 1.e-5f);
            __syncthreads();
            if (threadIdx.x == 0) {
                mean[row]    = m;
                inverse[row] = r;
            }
            for (unsigned c = threadIdx.x; c < width; c += blockDim.x) y[row * width + c] = __float2bfloat16((__bfloat162float(x[row * width + c]) - m) * r * w[c] + b[c]);
        }
        __global__ void norm_dx_kernel(const __nv_bfloat16* x, const __nv_bfloat16* dy, const float* w, const float* m, const float* r, __nv_bfloat16* dx, const __nv_bfloat16* residual, unsigned width) {
            __shared__ float shared[256];
            const unsigned row = blockIdx.x;
            float sum = 0.f, product = 0.f;
            for (unsigned c = threadIdx.x; c < width; c += blockDim.x) {
                const auto i  = row * width + c;
                const float g = __bfloat162float(dy[i]) * w[c];
                sum += g;
                product += g * (__bfloat162float(x[i]) - m[row]) * r[row];
            }
            const float a = reduce(sum, shared) / width;
            __syncthreads();
            const float b = reduce(product, shared) / width;
            __syncthreads();
            for (unsigned c = threadIdx.x; c < width; c += blockDim.x) {
                const auto i       = row * width + c;
                const float result = r[row] * (__bfloat162float(dy[i]) * w[c] - a - (__bfloat162float(x[i]) - m[row]) * r[row] * b);
                dx[i]              = __float2bfloat16(result + (residual ? __bfloat162float(residual[i]) : 0.f));
            }
        }
        __global__ void norm_parameter_kernel(const __nv_bfloat16* x, const __nv_bfloat16* dy, const float* m, const float* r, float* dw, float* db, unsigned rows, unsigned width) {
            const unsigned c = blockIdx.x * blockDim.x + threadIdx.x;
            if (c >= width) return;
            float a = 0.f, b = 0.f;
            for (unsigned row = 0; row < rows; ++row) {
                const auto i  = row * width + c;
                const float g = __bfloat162float(dy[i]);
                a += g * (__bfloat162float(x[i]) - m[row]) * r[row];
                b += g;
            }
            dw[c] += a;
            db[c] += b;
        }
        __global__ void bias_kernel(__nv_bfloat16* x, const float* b, unsigned count, unsigned width) {
            const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < count) x[i] = __float2bfloat16(__bfloat162float(x[i]) + b[i % width]);
        }
        __global__ void bias_gradient_kernel(const __nv_bfloat16* dy, float* db, unsigned rows, unsigned width) {
            const unsigned c = blockIdx.x * blockDim.x + threadIdx.x;
            if (c >= width) return;
            float sum = 0.f;
            for (unsigned row = 0; row < rows; ++row) sum += __bfloat162float(dy[row * width + c]);
            db[c] += sum;
        }
        __global__ void gelu_kernel(const __nv_bfloat16* x, __nv_bfloat16* y, std::size_t count, bool backward) {
            const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= count) return;
            const float v = __bfloat162float(x[i]), cdf = .5f * (1.f + erff(v * .7071067811865475f));
            y[i] = __float2bfloat16(backward ? __bfloat162float(y[i]) * (cdf + v * .3989422804014327f * expf(-.5f * v * v)) : v * cdf);
        }
        __global__ void add_kernel(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* y, std::size_t count) {
            const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i < count) y[i] = __float2bfloat16(__bfloat162float(a[i]) + __bfloat162float(b[i]));
        }
        __global__ void concatenate_kernel(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* y, unsigned count, unsigned width) {
            const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= count) return;
            const unsigned row = i / width, c = i % width;
            y[row * width * 2 + c]         = a[i];
            y[row * width * 2 + width + c] = b[i];
        }
        __global__ void split_kernel(const __nv_bfloat16* x, __nv_bfloat16* a, __nv_bfloat16* b, unsigned count, unsigned width) {
            const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= count) return;
            const unsigned row = i / width, c = i % width;
            a[i] = x[row * width * 2 + c];
            b[i] = x[row * width * 2 + width + c];
        }
    } // namespace
    void normalize(::cuda::stream_ref s, const std::uint16_t* x, const float* w, const float* b, std::uint16_t* y, float* m, float* r, std::uint32_t rows, std::uint32_t width) {
        norm_kernel<<<rows, 256, 0, s.get()>>>(reinterpret_cast<const __nv_bfloat16*>(x), w, b, reinterpret_cast<__nv_bfloat16*>(y), m, r, width);
    }
    void normalize_backward(::cuda::stream_ref s, const std::uint16_t* x, const std::uint16_t* dy, const float* w, const float* m, const float* r, std::uint16_t* dx, float* dw, float* db, const std::uint16_t* residual, std::uint32_t rows, std::uint32_t width) {
        norm_parameter_kernel<<<(width + 255) / 256, 256, 0, s.get()>>>(reinterpret_cast<const __nv_bfloat16*>(x), reinterpret_cast<const __nv_bfloat16*>(dy), m, r, dw, db, rows, width);
        norm_dx_kernel<<<rows, 256, 0, s.get()>>>(reinterpret_cast<const __nv_bfloat16*>(x), reinterpret_cast<const __nv_bfloat16*>(dy), w, m, r, reinterpret_cast<__nv_bfloat16*>(dx), reinterpret_cast<const __nv_bfloat16*>(residual), width);
    }
    void bias(::cuda::stream_ref s, std::uint16_t* x, const float* b, std::uint32_t rows, std::uint32_t width) {
        bias_kernel<<<(rows * width + 255) / 256, 256, 0, s.get()>>>(reinterpret_cast<__nv_bfloat16*>(x), b, rows * width, width);
    }
    void bias_backward(::cuda::stream_ref s, const std::uint16_t* dy, float* db, std::uint32_t rows, std::uint32_t width) {
        bias_gradient_kernel<<<(width + 255) / 256, 256, 0, s.get()>>>(reinterpret_cast<const __nv_bfloat16*>(dy), db, rows, width);
    }
    void gelu(::cuda::stream_ref s, const std::uint16_t* x, std::uint16_t* y, std::size_t n) {
        gelu_kernel<<<(n + 255) / 256, 256, 0, s.get()>>>(reinterpret_cast<const __nv_bfloat16*>(x), reinterpret_cast<__nv_bfloat16*>(y), n, false);
    }
    void gelu_backward(::cuda::stream_ref s, const std::uint16_t* x, std::uint16_t* g, std::size_t n) {
        gelu_kernel<<<(n + 255) / 256, 256, 0, s.get()>>>(reinterpret_cast<const __nv_bfloat16*>(x), reinterpret_cast<__nv_bfloat16*>(g), n, true);
    }
    void add(::cuda::stream_ref s, const std::uint16_t* a, const std::uint16_t* b, std::uint16_t* y, std::size_t n) {
        add_kernel<<<(n + 255) / 256, 256, 0, s.get()>>>(reinterpret_cast<const __nv_bfloat16*>(a), reinterpret_cast<const __nv_bfloat16*>(b), reinterpret_cast<__nv_bfloat16*>(y), n);
    }
    void concatenate(::cuda::stream_ref s, const std::uint16_t* a, const std::uint16_t* b, std::uint16_t* y, std::uint32_t rows, std::uint32_t width) {
        concatenate_kernel<<<(rows * width + 255) / 256, 256, 0, s.get()>>>(reinterpret_cast<const __nv_bfloat16*>(a), reinterpret_cast<const __nv_bfloat16*>(b), reinterpret_cast<__nv_bfloat16*>(y), rows * width, width);
    }
    void split(::cuda::stream_ref s, const std::uint16_t* x, std::uint16_t* a, std::uint16_t* b, std::uint32_t rows, std::uint32_t width) {
        split_kernel<<<(rows * width + 255) / 256, 256, 0, s.get()>>>(reinterpret_cast<const __nv_bfloat16*>(x), reinterpret_cast<__nv_bfloat16*>(a), reinterpret_cast<__nv_bfloat16*>(b), rows * width, width);
    }
} // namespace flowdit::neural::kernels
