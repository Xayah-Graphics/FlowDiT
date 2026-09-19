#include "kernels.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
namespace flowdit::kernels {
    namespace {
        __global__ void tokens_kernel(const __nv_bfloat16* patches, const float* times, const std::uint32_t* labels, const float* classes, const float* position, __nv_bfloat16* tokens, unsigned batch, unsigned sequence, unsigned width) {
            const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= batch * sequence * width) return;
            const unsigned sample = i / (sequence * width), token = i / width % sequence, c = i % width;
            float v;
            if (token == 0) v = classes[labels[sample] * width + c];
            else if (token == 1) {
                const float phase = times[sample] * expf(-logf(10000.f) * (c % (width / 2)) / (width / 2));
                v                 = c < width / 2 ? cosf(phase) : sinf(phase);
            } else v = __bfloat162float(patches[(sample * (sequence - 2) + token - 2) * width + c]);
            tokens[i] = __float2bfloat16(v + position[token * width + c]);
        }
        __global__ void token_parameters_kernel(const __nv_bfloat16* g, const std::uint32_t* labels, float* classes, float* position, unsigned batch, unsigned sequence, unsigned width, unsigned class_count) {
            const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < sequence * width) {
                float sum = 0.f;
                for (unsigned b = 0; b < batch; ++b) sum += __bfloat162float(g[b * sequence * width + i]);
                position[i] += sum;
            }
            if (i < (class_count + 1) * width) {
                float sum            = 0.f;
                const unsigned label = i / width, c = i % width;
                for (unsigned b = 0; b < batch; ++b)
                    if (labels[b] == label) sum += __bfloat162float(g[b * sequence * width + c]);
                classes[i] += sum;
            }
        }
        __global__ void token_patch_gradient_kernel(const __nv_bfloat16* g, __nv_bfloat16* patches, unsigned count, unsigned sequence, unsigned width) {
            const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < count) patches[i] = g[(i / ((sequence - 2) * width) * sequence + i / width % (sequence - 2) + 2) * width + i % width];
        }
        __global__ void velocity_kernel(const __nv_bfloat16* x, float* y, unsigned count, unsigned sequence, unsigned width) {
            const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < count) y[i] = __bfloat162float(x[(i / ((sequence - 2) * width) * sequence + i / width % (sequence - 2) + 2) * width + i % width]);
        }
        __global__ void velocity_gradient_kernel(const float* x, __nv_bfloat16* y, unsigned count, unsigned sequence, unsigned width) {
            const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= count) return;
            const unsigned b = i / (sequence * width), t = i / width % sequence, c = i % width;
            y[i] = __float2bfloat16(t < 2 ? 0.f : x[(b * (sequence - 2) + t - 2) * width + c]);
        }
        __global__ void unpatchify_kernel(const float* patches, float* values, unsigned count, TensorLayout image) {
            const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= count) return;
            const unsigned area = image.width * image.height, c = i / area % image.channels, pixel = i % area, b = i / (area * image.channels), x = pixel % image.width, y = pixel / image.width;
            const unsigned token   = y / image.patch_size * (image.width / image.patch_size) + x / image.patch_size;
            const unsigned element = (y % image.patch_size * image.patch_size + x % image.patch_size) * image.channels + c;
            values[i]              = patches[(b * area / (image.patch_size * image.patch_size) + token) * image.patch_size * image.patch_size * image.channels + element];
        }
    } // namespace
    void make_tokens(::cuda::stream_ref s, const std::uint16_t* p, const float* times, const std::uint32_t* labels, const float* classes, const float* position, std::uint16_t* tokens, std::uint32_t b, std::uint32_t n, std::uint32_t d) {
        tokens_kernel<<<(b * n * d + 255) / 256, 256, 0, s.get()>>>(reinterpret_cast<const __nv_bfloat16*>(p), times, labels, classes, position, reinterpret_cast<__nv_bfloat16*>(tokens), b, n, d);
    }
    void tokens_backward(::cuda::stream_ref s, const std::uint16_t* g, const std::uint32_t* labels, float* classes, float* position, std::uint16_t* patches, std::uint32_t b, std::uint32_t n, std::uint32_t d, std::uint32_t class_count) {
        const unsigned count = (n > class_count + 1 ? n : class_count + 1) * d;
        token_parameters_kernel<<<(count + 255) / 256, 256, 0, s.get()>>>(reinterpret_cast<const __nv_bfloat16*>(g), labels, classes, position, b, n, d, class_count);
        token_patch_gradient_kernel<<<(b * (n - 2) * d + 255) / 256, 256, 0, s.get()>>>(reinterpret_cast<const __nv_bfloat16*>(g), reinterpret_cast<__nv_bfloat16*>(patches), b * (n - 2) * d, n, d);
    }
    void velocity(::cuda::stream_ref s, const std::uint16_t* x, float* y, std::uint32_t b, std::uint32_t n, std::uint32_t d) {
        velocity_kernel<<<(b * (n - 2) * d + 255) / 256, 256, 0, s.get()>>>(reinterpret_cast<const __nv_bfloat16*>(x), y, b * (n - 2) * d, n, d);
    }
    void velocity_backward(::cuda::stream_ref s, const float* x, std::uint16_t* y, std::uint32_t b, std::uint32_t n, std::uint32_t d) {
        velocity_gradient_kernel<<<(b * n * d + 255) / 256, 256, 0, s.get()>>>(x, reinterpret_cast<__nv_bfloat16*>(y), b * n * d, n, d);
    }
    void unpatchify(::cuda::stream_ref s, const float* p, float* v, std::uint32_t batch, TensorLayout image) {
        const auto count = batch * image.width * image.height * image.channels;
        unpatchify_kernel<<<(count + 255) / 256, 256, 0, s.get()>>>(p, v, count, image);
    }
} // namespace flowdit::kernels
