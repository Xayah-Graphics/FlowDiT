#include "spatial-kernels.h"
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
namespace flowdit::kernels {
    namespace {
        __device__ float reduce(float v, float* shared) {
            shared[threadIdx.x] = v;
            __syncthreads();
            for (unsigned stride = 128; stride; stride /= 2) {
                if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
                __syncthreads();
            }
            return shared[0];
        }
        __global__ void pack_kernel(const float* x, __nv_bfloat16* y, std::size_t n) {
            const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i < n) y[i] = __float2bfloat16(x[i]);
        }
        __global__ void unpack_kernel(const __nv_bfloat16* x, float* y, std::size_t n, bool add) {
            const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i < n) y[i] = __bfloat162float(x[i]) + (add ? y[i] : 0);
        }
        __global__ void sum_kernel(const float* x, float* y, std::size_t n, float scale) {
            const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i < n) y[i] += x[i] * scale;
        }
        __global__ void bias_kernel(__nv_bfloat16* x, const float* b, std::size_t n, unsigned channels, unsigned area) {
            const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i < n) x[i] = __float2bfloat16(__bfloat162float(x[i]) + b[i / area % channels]);
        }
        __global__ void bias_backward_kernel(const float* dy, float* db, unsigned batch, unsigned channels, unsigned area) {
            __shared__ float shared[256];
            float v = 0;
            for (unsigned i = threadIdx.x; i < batch * area; i += blockDim.x) v += dy[(static_cast<std::size_t>(i / area) * channels + blockIdx.x) * area + i % area];
            v = reduce(v, shared);
            if (threadIdx.x == 0) db[blockIdx.x] += v;
        }
        __global__ void activation_kernel(const __nv_bfloat16* x, __nv_bfloat16* y, std::size_t n, int kind) {
            const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= n) return;
            const float v = __bfloat162float(x[i]);
            y[i]          = __float2bfloat16(kind == 0 ? v / (1 + expf(-v)) : v >= 0 ? v : kind == 1 ? 0 : v * .2f);
        }
        __global__ void activation_backward_kernel(const __nv_bfloat16* x, const float* dy, float* dx, std::size_t n, int kind) {
            const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= n) return;
            const float v = __bfloat162float(x[i]), sigmoid = 1 / (1 + expf(-v));
            dx[i] += dy[i] * (kind == 0 ? sigmoid * (1 + v * (1 - sigmoid)) : v > 0 ? 1 : kind == 1 ? 0 : .2f);
        }
        __global__ void residual_kernel(const __nv_bfloat16* x, const __nv_bfloat16* s, __nv_bfloat16* y, std::size_t n) {
            const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i < n) y[i] = __float2bfloat16(__bfloat162float(x[i]) + __bfloat162float(s[i]));
        }
        __global__ void norm_kernel(const __nv_bfloat16* x, const float* p, __nv_bfloat16* y, unsigned channels, unsigned area) {
            __shared__ float shared[256];
            __shared__ float mean, inv;
            const unsigned length = channels / 32 * area;
            const auto start      = static_cast<std::size_t>(blockIdx.x) * length;
            float v               = 0;
            for (unsigned i = threadIdx.x; i < length; i += blockDim.x) v += __bfloat162float(x[start + i]);
            v = reduce(v, shared);
            if (threadIdx.x == 0) mean = v / length;
            __syncthreads();
            v = 0;
            for (unsigned i = threadIdx.x; i < length; i += blockDim.x) {
                const float d = __bfloat162float(x[start + i]) - mean;
                v += d * d;
            }
            v = reduce(v, shared);
            if (threadIdx.x == 0) inv = rsqrtf(v / length + 1e-6f);
            __syncthreads();
            for (unsigned i = threadIdx.x; i < length; i += blockDim.x) {
                const auto c = (start + i) / area % channels;
                y[start + i] = __float2bfloat16((__bfloat162float(x[start + i]) - mean) * inv * p[c] + p[channels + c]);
            }
        }
        __global__ void norm_backward_kernel(const __nv_bfloat16* x, const float* p, const float* dy, float* dx, float* dp, unsigned channels, unsigned area) {
            __shared__ float shared[256];
            __shared__ float mean, inv, a, b;
            const unsigned length = channels / 32 * area;
            const auto start      = static_cast<std::size_t>(blockIdx.x) * length;
            float v               = 0;
            for (unsigned i = threadIdx.x; i < length; i += blockDim.x) v += __bfloat162float(x[start + i]);
            v = reduce(v, shared);
            if (threadIdx.x == 0) mean = v / length;
            __syncthreads();
            v = 0;
            for (unsigned i = threadIdx.x; i < length; i += blockDim.x) {
                const float d = __bfloat162float(x[start + i]) - mean;
                v += d * d;
            }
            v = reduce(v, shared);
            if (threadIdx.x == 0) inv = rsqrtf(v / length + 1e-6f);
            __syncthreads();
            v = 0;
            for (unsigned i = threadIdx.x; i < length; i += blockDim.x) v += dy[start + i] * p[(start + i) / area % channels];
            v = reduce(v, shared);
            if (threadIdx.x == 0) a = v / length;
            __syncthreads();
            v = 0;
            for (unsigned i = threadIdx.x; i < length; i += blockDim.x) v += dy[start + i] * p[(start + i) / area % channels] * (__bfloat162float(x[start + i]) - mean) * inv;
            v = reduce(v, shared);
            if (threadIdx.x == 0) b = v / length;
            __syncthreads();
            for (unsigned i = threadIdx.x; i < length; i += blockDim.x) {
                const auto c           = (start + i) / area % channels;
                const float normalized = (__bfloat162float(x[start + i]) - mean) * inv;
                dx[start + i] += inv * (dy[start + i] * p[c] - a - b * normalized);
            }
            if (dp)
                for (unsigned c = 0; c < channels / 32; ++c) {
                    float gamma = 0, beta = 0;
                    for (unsigned i = threadIdx.x; i < area; i += blockDim.x) {
                        const auto j = start + c * area + i;
                        gamma += dy[j] * (__bfloat162float(x[j]) - mean) * inv;
                        beta += dy[j];
                    }
                    gamma = reduce(gamma, shared);
                    if (threadIdx.x == 0) atomicAdd(dp + blockIdx.x % 32 * (channels / 32) + c, gamma);
                    __syncthreads();
                    beta = reduce(beta, shared);
                    if (threadIdx.x == 0) atomicAdd(dp + channels + blockIdx.x % 32 * (channels / 32) + c, beta);
                    __syncthreads();
                }
        }
        __global__ void resize_kernel(const __nv_bfloat16* x, __nv_bfloat16* y, unsigned planes, unsigned w, unsigned h, bool pool) {
            const unsigned ow = pool ? w / 2 : w * 2, oh = pool ? h / 2 : h * 2;
            const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= static_cast<std::size_t>(planes) * ow * oh) return;
            const auto p = i / (ow * oh), px = i % ow, py = i / ow % oh;
            if (!pool) y[i] = x[(p * h + py / 2) * w + px / 2];
            else {
                float v = -INFINITY;
                for (unsigned dy = 0; dy < 2; ++dy)
                    for (unsigned dx = 0; dx < 2; ++dx) v = fmaxf(v, __bfloat162float(x[(p * h + py * 2 + dy) * w + px * 2 + dx]));
                y[i] = __float2bfloat16(v);
            }
        }
        __global__ void resize_backward_kernel(const __nv_bfloat16* x, const float* dy, float* dx, unsigned planes, unsigned w, unsigned h, bool pool) {
            const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= static_cast<std::size_t>(planes) * w * h) return;
            const auto p = i / (w * h), px = i % w, py = i / w % h;
            if (!pool) {
                float v = 0;
                for (unsigned yy = 0; yy < 2; ++yy)
                    for (unsigned xx = 0; xx < 2; ++xx) v += dy[(p * h * 2 + py * 2 + yy) * w * 2 + px * 2 + xx];
                dx[i] += v;
            } else {
                if (px >= w / 2 * 2 || py >= h / 2 * 2) return;
                const unsigned ox = px / 2, oy = py / 2;
                float v          = -INFINITY;
                std::size_t best = 0;
                for (unsigned yy = 0; yy < 2; ++yy)
                    for (unsigned xx = 0; xx < 2; ++xx) {
                        const auto j = (p * h + oy * 2 + yy) * w + ox * 2 + xx;
                        if (__bfloat162float(x[j]) > v) {
                            v    = __bfloat162float(x[j]);
                            best = j;
                        }
                    }
                if (i == best) dx[i] += dy[(p * (h / 2) + oy) * (w / 2) + ox];
            }
        }
        __global__ void softmax_kernel(__nv_bfloat16* scores, unsigned n) {
            __shared__ float shared[256];
            __shared__ float maximum, denominator;
            const auto start = static_cast<std::size_t>(blockIdx.x / n) * n * n + blockIdx.x % n;
            float v          = -INFINITY;
            for (unsigned k = threadIdx.x; k < n; k += blockDim.x) v = fmaxf(v, __bfloat162float(scores[start + k * n]));
            shared[threadIdx.x] = v;
            __syncthreads();
            for (unsigned stride = 128; stride; stride /= 2) {
                if (threadIdx.x < stride) shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
                __syncthreads();
            }
            if (threadIdx.x == 0) maximum = shared[0];
            __syncthreads();
            v = 0;
            for (unsigned k = threadIdx.x; k < n; k += blockDim.x) v += expf(__bfloat162float(scores[start + k * n]) - maximum);
            v = reduce(v, shared);
            if (threadIdx.x == 0) denominator = v;
            __syncthreads();
            for (unsigned k = threadIdx.x; k < n; k += blockDim.x) scores[start + k * n] = __float2bfloat16(expf(__bfloat162float(scores[start + k * n]) - maximum) / denominator);
        }
        __global__ void softmax_backward_kernel(const __nv_bfloat16* p, __nv_bfloat16* g, unsigned n, float scale) {
            __shared__ float shared[256];
            const auto start = static_cast<std::size_t>(blockIdx.x / n) * n * n + blockIdx.x % n;
            float v          = 0;
            for (unsigned k = threadIdx.x; k < n; k += blockDim.x) v += __bfloat162float(p[start + k * n]) * __bfloat162float(g[start + k * n]);
            v = reduce(v, shared);
            for (unsigned k = threadIdx.x; k < n; k += blockDim.x) g[start + k * n] = __float2bfloat16((__bfloat162float(g[start + k * n]) - v) * __bfloat162float(p[start + k * n]) * scale);
        }
        __global__ void posterior_kernel(const float* moments, float* z, unsigned batch, unsigned elements, std::uint64_t seed, std::uint64_t step, bool sample) {
            const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= static_cast<std::size_t>(batch) * elements) return;
            const auto source = (i / elements * 2) * elements + i % elements;
            curandStatePhilox4_32_10_t rng;
            curand_init(seed, step, i * 4, &rng);
            z[i] = moments[source] + (sample ? expf(.5f * fminf(fmaxf(moments[source + elements], -30.f), 20.f)) * curand_normal(&rng) : 0);
        }
        __global__ void posterior_backward_kernel(const float* m, const float* z, const float* dz, float* dm, float* loss, unsigned batch, unsigned elements, float weight, float scale) {
            __shared__ float shared[256];
            const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            float kl     = 0;
            if (i < static_cast<std::size_t>(batch) * elements) {
                const auto j     = i / elements * elements * 2 + i % elements;
                const float mean = m[j], raw = m[j + elements], logvar = fminf(fmaxf(raw, -30.f), 20.f), variance = expf(logvar);
                const float factor = weight * scale / (batch * elements);
                dm[j]              = dz[i] + mean * factor;
                dm[j + elements]   = raw >= -30 && raw <= 20 ? .5f * (dz[i] * (z[i] - mean) + (variance - 1) * factor) : 0;
                kl                 = .5f * (mean * mean + variance - 1 - logvar) / (batch * elements);
            }
            kl = reduce(kl, shared);
            if (threadIdx.x == 0) atomicAdd(loss, kl);
        }
        __global__ void reconstruction_kernel(const float* x, const float* y, float* dy, float* loss, std::size_t n, float scale) {
            __shared__ float shared[256];
            const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            float l      = 0;
            if (i < n) {
                const float d = y[i] - x[i];
                l             = fabsf(d) / n;
                dy[i]         = (d > 0 ? 1.f : d < 0 ? -1.f : 0.f) * scale / n;
            }
            l = reduce(l, shared);
            if (threadIdx.x == 0) atomicAdd(loss, l);
        }
        __global__ void adversarial_kernel(const float* x, float* dx, float* loss, std::size_t n, int kind, float scale) {
            __shared__ float shared[256];
            const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            float l      = 0;
            if (i < n) {
                l     = kind == 0 ? -x[i] : fmaxf(0, 1 + (kind == 1 ? -x[i] : x[i]));
                dx[i] = (kind == 0 ? -1.f : l > 0 ? kind == 1 ? -1.f : 1.f : 0.f) * scale / n;
            }
            l = reduce(l / n, shared);
            if (threadIdx.x == 0) atomicAdd(loss, l);
        }
        __global__ void perceptual_input_kernel(const float* x, float* y, unsigned batch, unsigned channels, unsigned area) {
            const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= static_cast<std::size_t>(batch) * 3 * area) return;
            const float shifts[3]{-.030f, -.088f, -.188f}, scales[3]{.458f, .448f, .450f};
            const unsigned c = i / area % 3;
            y[i]             = (x[(i / (3 * area) * channels + (channels == 1 ? 0 : c)) * area + i % area] - shifts[c]) / scales[c];
        }
        __global__ void perceptual_backward_kernel(const float* dy, float* dx, unsigned batch, unsigned channels, unsigned area) {
            const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= static_cast<std::size_t>(batch) * channels * area) return;
            const float scales[3]{.458f, .448f, .450f};
            const auto b = i / (channels * area), c = i / area % channels, p = i % area;
            if (channels == 1)
                for (unsigned k = 0; k < 3; ++k) dx[i] += dy[(b * 3 + k) * area + p] / scales[k];
            else dx[i] += dy[(b * 3 + c) * area + p] / scales[c];
        }
        __global__ void perceptual_loss_kernel(const __nv_bfloat16* a, const __nv_bfloat16* b, const float* weights, float* db, float* loss, unsigned batch, unsigned channels, unsigned area, float scale) {
            const auto p = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (p >= static_cast<std::size_t>(batch) * area) return;
            const auto start = p / area * channels * area + p % area;
            float an = 0, bn = 0;
            for (unsigned c = 0; c < channels; ++c) {
                const float x = __bfloat162float(a[start + c * area]), y = __bfloat162float(b[start + c * area]);
                an += x * x;
                bn += y * y;
            }
            an      = 1 / (sqrtf(an) + 1e-10f);
            bn      = 1 / (sqrtf(bn) + 1e-10f);
            float l = 0, dot = 0;
            for (unsigned c = 0; c < channels; ++c) {
                const float x = __bfloat162float(a[start + c * area]) * an, y = __bfloat162float(b[start + c * area]) * bn, d = y - x;
                l += weights[c] * d * d;
                dot += weights[c] * d * y;
            }
            atomicAdd(loss, l / (batch * area));
            for (unsigned c = 0; c < channels; ++c) {
                const float x = __bfloat162float(a[start + c * area]) * an, y = __bfloat162float(b[start + c * area]) * bn;
                db[start + c * area] += 2 * bn * (weights[c] * (y - x) - y * dot) * scale / (batch * area);
            }
        }
        __global__ void standardize_kernel(const float* x, float* y, const float* mean, const float* deviation, std::size_t n, unsigned channels, unsigned area, bool inverse) {
            const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i < n) {
                const auto c = i / area % channels;
                y[i]         = inverse ? x[i] * deviation[c] + mean[c] : (x[i] - mean[c]) / deviation[c];
            }
        }
    } // namespace
    void pack(::cuda::stream_ref s, const float* x, std::uint16_t* y, std::size_t n) {
        pack_kernel<<<(n + 255) / 256, 256, 0, s.get()>>>(x, reinterpret_cast<__nv_bfloat16*>(y), n);
    }
    void unpack(::cuda::stream_ref s, const std::uint16_t* x, float* y, std::size_t n, bool a) {
        unpack_kernel<<<(n + 255) / 256, 256, 0, s.get()>>>(reinterpret_cast<const __nv_bfloat16*>(x), y, n, a);
    }
    void sum(::cuda::stream_ref s, const float* x, float* y, std::size_t n, float v) {
        sum_kernel<<<(n + 255) / 256, 256, 0, s.get()>>>(x, y, n, v);
    }
    void bias(::cuda::stream_ref s, std::uint16_t* x, const float* b, unsigned n, unsigned c, unsigned a) {
        bias_kernel<<<(static_cast<std::size_t>(n) * c * a + 255) / 256, 256, 0, s.get()>>>(reinterpret_cast<__nv_bfloat16*>(x), b, static_cast<std::size_t>(n) * c * a, c, a);
    }
    void bias_gradient(::cuda::stream_ref s, const float* g, float* b, unsigned n, unsigned c, unsigned a) {
        bias_backward_kernel<<<c, 256, 0, s.get()>>>(g, b, n, c, a);
    }
    void activation(::cuda::stream_ref s, const std::uint16_t* x, std::uint16_t* y, std::size_t n, int k) {
        activation_kernel<<<(n + 255) / 256, 256, 0, s.get()>>>(reinterpret_cast<const __nv_bfloat16*>(x), reinterpret_cast<__nv_bfloat16*>(y), n, k);
    }
    void activation_backward(::cuda::stream_ref s, const std::uint16_t* x, const float* g, float* y, std::size_t n, int k) {
        activation_backward_kernel<<<(n + 255) / 256, 256, 0, s.get()>>>(reinterpret_cast<const __nv_bfloat16*>(x), g, y, n, k);
    }
    void residual(::cuda::stream_ref s, const std::uint16_t* x, const std::uint16_t* a, std::uint16_t* y, std::size_t n) {
        residual_kernel<<<(n + 255) / 256, 256, 0, s.get()>>>(reinterpret_cast<const __nv_bfloat16*>(x), reinterpret_cast<const __nv_bfloat16*>(a), reinterpret_cast<__nv_bfloat16*>(y), n);
    }
    void group_norm(::cuda::stream_ref s, const std::uint16_t* x, const float* p, std::uint16_t* y, unsigned n, unsigned c, unsigned a) {
        norm_kernel<<<n * 32, 256, 0, s.get()>>>(reinterpret_cast<const __nv_bfloat16*>(x), p, reinterpret_cast<__nv_bfloat16*>(y), c, a);
    }
    void group_norm_backward(::cuda::stream_ref s, const std::uint16_t* x, const float* p, const float* g, float* y, float* dp, unsigned n, unsigned c, unsigned a) {
        norm_backward_kernel<<<n * 32, 256, 0, s.get()>>>(reinterpret_cast<const __nv_bfloat16*>(x), p, g, y, dp, c, a);
    }
    void resize(::cuda::stream_ref s, const std::uint16_t* x, std::uint16_t* y, unsigned p, unsigned w, unsigned h, bool pool) {
        const auto count = static_cast<std::size_t>(p) * (pool ? w / 2 : w * 2) * (pool ? h / 2 : h * 2);
        resize_kernel<<<(count + 255) / 256, 256, 0, s.get()>>>(reinterpret_cast<const __nv_bfloat16*>(x), reinterpret_cast<__nv_bfloat16*>(y), p, w, h, pool);
    }
    void resize_backward(::cuda::stream_ref s, const std::uint16_t* x, const float* g, float* y, unsigned p, unsigned w, unsigned h, bool pool) {
        const auto count = static_cast<std::size_t>(p) * w * h;
        resize_backward_kernel<<<(count + 255) / 256, 256, 0, s.get()>>>(reinterpret_cast<const __nv_bfloat16*>(x), g, y, p, w, h, pool);
    }
    void attention_softmax(::cuda::stream_ref s, std::uint16_t* x, unsigned n, unsigned q) {
        softmax_kernel<<<n * q, 256, 0, s.get()>>>(reinterpret_cast<__nv_bfloat16*>(x), q);
    }
    void attention_softmax_backward(::cuda::stream_ref s, const std::uint16_t* p, std::uint16_t* g, unsigned n, unsigned q, float scale) {
        softmax_backward_kernel<<<n * q, 256, 0, s.get()>>>(reinterpret_cast<const __nv_bfloat16*>(p), reinterpret_cast<__nv_bfloat16*>(g), q, scale);
    }
    void posterior(::cuda::stream_ref s, const float* m, float* z, unsigned b, unsigned e, std::uint64_t seed, std::uint64_t step, bool sample) {
        posterior_kernel<<<(static_cast<std::size_t>(b) * e + 255) / 256, 256, 0, s.get()>>>(m, z, b, e, seed, step, sample);
    }
    void posterior_backward(::cuda::stream_ref s, const float* m, const float* z, const float* g, float* out, float* l, unsigned b, unsigned e, float weight, float scale) {
        posterior_backward_kernel<<<(static_cast<std::size_t>(b) * e + 255) / 256, 256, 0, s.get()>>>(m, z, g, out, l, b, e, weight, scale);
    }
    void reconstruction_loss(::cuda::stream_ref s, const float* x, const float* y, float* g, float* l, std::size_t n, float scale) {
        reconstruction_kernel<<<(n + 255) / 256, 256, 0, s.get()>>>(x, y, g, l, n, scale);
    }
    void adversarial_loss(::cuda::stream_ref s, const float* x, float* g, float* l, std::size_t n, int kind, float scale) {
        adversarial_kernel<<<(n + 255) / 256, 256, 0, s.get()>>>(x, g, l, n, kind, scale);
    }
    void perceptual_input(::cuda::stream_ref s, const float* x, float* y, unsigned n, unsigned c, unsigned a) {
        perceptual_input_kernel<<<(static_cast<std::size_t>(n) * 3 * a + 255) / 256, 256, 0, s.get()>>>(x, y, n, c, a);
    }
    void perceptual_backward(::cuda::stream_ref s, const float* g, float* y, unsigned n, unsigned c, unsigned a) {
        perceptual_backward_kernel<<<(static_cast<std::size_t>(n) * c * a + 255) / 256, 256, 0, s.get()>>>(g, y, n, c, a);
    }
    void perceptual_loss(::cuda::stream_ref s, const std::uint16_t* x, const std::uint16_t* y, const float* w, float* g, float* l, unsigned n, unsigned c, unsigned a, float scale) {
        perceptual_loss_kernel<<<(static_cast<std::size_t>(n) * a + 255) / 256, 256, 0, s.get()>>>(reinterpret_cast<const __nv_bfloat16*>(x), reinterpret_cast<const __nv_bfloat16*>(y), w, g, l, n, c, a, scale);
    }
    void standardize(::cuda::stream_ref s, const float* x, float* y, const float* m, const float* d, unsigned n, unsigned c, unsigned a, bool inverse) {
        const auto count = static_cast<std::size_t>(n) * c * a;
        standardize_kernel<<<(count + 255) / 256, 256, 0, s.get()>>>(x, y, m, d, count, c, a, inverse);
    }
} // namespace flowdit::kernels
