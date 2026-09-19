#include "kernels.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
namespace flowdit::tokenizer_kernels {
    namespace {
        __global__ void scale_kernel(const float* source, float* target, std::size_t count, float factor) {
            const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i < count) target[i] = source[i] * factor;
        }
        __global__ void pack_kernel(const float* source, __nv_bfloat16* target, std::size_t count) {
            const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i < count) target[i] = __float2bfloat16(source[i]);
        }
        __global__ void unpack_kernel(const __nv_bfloat16* source, float* target, std::size_t count, float scale) {
            const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i < count) target[i] = __bfloat162float(source[i]) * scale;
        }
        __global__ void pointwise_kernel(const __nv_bfloat16* source, const __nv_bfloat16* skip, const float* p, __nv_bfloat16* target, unsigned channels, unsigned area, int op) {
            const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= channels * area) return;
            const float x = __bfloat162float(source[i]);
            float y;
            if (op == 0) y = x + p[i / area];
            else if (op == 1) y = x / (1.f + expf(-x));
            else if (op == 2) y = fmaxf(x, 0.f);
            else if (op == 3) y = x + __bfloat162float(skip[i]);
            else {
                const float gate = __bfloat162float(source[i + channels * area]);
                // PyTorch evaluates SiLU before the BF16 product.
                y = x * __bfloat162float(__float2bfloat16(gate / (1.f + expf(-gate))));
            }
            target[i] = __float2bfloat16(y);
        }
        __global__ void rms_kernel(const __nv_bfloat16* source, const float* p, __nv_bfloat16* target, unsigned channels, unsigned area) {
            const unsigned pixel = blockIdx.x * blockDim.x + threadIdx.x;
            if (pixel >= area) return;
            float sum = 0.f;
            for (unsigned c = 0; c < channels; ++c) {
                const float x = __bfloat162float(source[c * area + pixel]);
                sum += x * x;
            }
            const float inverse = rsqrtf(sum / channels + 1.e-5f);
            for (unsigned c = 0; c < channels; ++c) {
                const float normalized   = __bfloat162float(__float2bfloat16(__bfloat162float(source[c * area + pixel]) * inverse));
                const float weighted     = __bfloat162float(__float2bfloat16(normalized * __bfloat162float(__float2bfloat16(p[c]))));
                target[c * area + pixel] = __float2bfloat16(weighted + __bfloat162float(__float2bfloat16(p[c + channels])));
            }
        }
        __global__ void rearrange_kernel(const __nv_bfloat16* source, __nv_bfloat16* target, unsigned w, unsigned h, unsigned channels, unsigned output_channels, int op) {
            const unsigned width  = op == 0 ? w / 2 : (op == 1 || op == 2 ? w * 2 : w);
            const unsigned height = op == 0 ? h / 2 : (op == 1 || op == 2 ? h * 2 : h);
            const unsigned i = blockIdx.x * blockDim.x + threadIdx.x, area = width * height;
            if (i >= output_channels * area) return;
            const unsigned c = i / area, y = (i % area) / width, x = i % width;
            float value = 0.f;
            if (op == 0) {
                const unsigned group = channels * 4 / output_channels;
                for (unsigned g = 0; g < group; ++g) {
                    const unsigned q = c * group + g;
                    value += __bfloat162float(source[(q / 4 * h + y * 2 + q / 2 % 2) * w + x * 2 + q % 2]);
                }
                value /= group;
            } else if (op == 1) value = __bfloat162float(source[(c * h + y / 2) * w + x / 2]);
            else if (op == 2) {
                const unsigned repeats = output_channels * 4 / channels;
                const unsigned q       = (c * 4 + y % 2 * 2 + x % 2) / repeats;
                value                  = __bfloat162float(source[(q * h + y / 2) * w + x / 2]);
            } else if (op == 3) {
                const unsigned group = channels / output_channels;
                for (unsigned g = 0; g < group; ++g) value += __bfloat162float(source[((c * group + g) * h + y) * w + x]);
                value /= group;
            } else value = __bfloat162float(source[(c / (output_channels / channels) * h + y) * w + x]);
            target[i] = __float2bfloat16(value);
        }
        __global__ void attention_pack_kernel(const __nv_bfloat16* a, const __nv_bfloat16* b, float* q, float* k, float* v, unsigned heads, unsigned area) {
            const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= heads * 33 * area) return;
            const unsigned pixel = i % area, d = i / area % 33, head = i / (33 * area);
            const auto* source    = head < heads / 2 ? a : b;
            const unsigned offset = (head % (heads / 2) * 96 + d) * area + pixel;
            v[i]                  = d == 32 ? 1.f : __bfloat162float(source[offset + 64 * area]);
            if (d < 32) {
                q[(head * 32 + d) * area + pixel] = fmaxf(__bfloat162float(source[offset]), 0.f);
                k[(head * 32 + d) * area + pixel] = fmaxf(__bfloat162float(source[offset + 32 * area]), 0.f);
            }
        }
        __global__ void attention_output_kernel(const float* source, __nv_bfloat16* target, unsigned heads, unsigned area) {
            const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= heads * 32 * area) return;
            const unsigned h = i / (32 * area), p = i % area, d = i / area % 32;
            target[i] = __float2bfloat16(source[(h * 33 + d) * area + p] / (source[(h * 33 + 32) * area + p] + 1.e-15f));
        }
    } // namespace
    void pack(::cuda::stream_ref s, const float* a, std::uint16_t* b, std::size_t n) {
        pack_kernel<<<(n + 255) / 256, 256, 0, s.get()>>>(a, reinterpret_cast<__nv_bfloat16*>(b), n);
    }
    void scale(::cuda::stream_ref s, const float* a, float* b, std::size_t n, float factor) {
        scale_kernel<<<(n + 255) / 256, 256, 0, s.get()>>>(a, b, n, factor);
    }
    void unpack(::cuda::stream_ref s, const std::uint16_t* a, float* b, std::size_t n, float scale) {
        unpack_kernel<<<(n + 255) / 256, 256, 0, s.get()>>>(reinterpret_cast<const __nv_bfloat16*>(a), b, n, scale);
    }
    void pointwise(::cuda::stream_ref s, const std::uint16_t* a, const std::uint16_t* skip, const float* p, std::uint16_t* b, std::uint32_t c, std::uint32_t area, int op) {
        pointwise_kernel<<<(c * area + 255) / 256, 256, 0, s.get()>>>(reinterpret_cast<const __nv_bfloat16*>(a), reinterpret_cast<const __nv_bfloat16*>(skip), p, reinterpret_cast<__nv_bfloat16*>(b), c, area, op);
    }
    void rms(::cuda::stream_ref s, const std::uint16_t* a, const float* p, std::uint16_t* b, std::uint32_t c, std::uint32_t area) {
        rms_kernel<<<(area + 255) / 256, 256, 0, s.get()>>>(reinterpret_cast<const __nv_bfloat16*>(a), p, reinterpret_cast<__nv_bfloat16*>(b), c, area);
    }
    void rearrange(::cuda::stream_ref s, const std::uint16_t* a, std::uint16_t* b, std::uint32_t w, std::uint32_t h, std::uint32_t c, std::uint32_t target_c, int op) {
        const auto n = target_c * w * h * (op == 1 || op == 2 ? 4 : 1) / (op == 0 ? 4 : 1);
        rearrange_kernel<<<(n + 255) / 256, 256, 0, s.get()>>>(reinterpret_cast<const __nv_bfloat16*>(a), reinterpret_cast<__nv_bfloat16*>(b), w, h, c, target_c, op);
    }
    void attention_pack(::cuda::stream_ref s, const std::uint16_t* a, const std::uint16_t* b, float* q, float* k, float* v, std::uint32_t heads, std::uint32_t area) {
        attention_pack_kernel<<<(heads * 33 * area + 255) / 256, 256, 0, s.get()>>>(reinterpret_cast<const __nv_bfloat16*>(a), reinterpret_cast<const __nv_bfloat16*>(b), q, k, v, heads, area);
    }
    void attention_output(::cuda::stream_ref s, const float* a, std::uint16_t* b, std::uint32_t heads, std::uint32_t area) {
        attention_output_kernel<<<(heads * 32 * area + 255) / 256, 256, 0, s.get()>>>(a, reinterpret_cast<__nv_bfloat16*>(b), heads, area);
    }
} // namespace flowdit::tokenizer_kernels
