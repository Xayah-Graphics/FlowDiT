#include "kernels.h"
#include <cmath>
#include <flowdit/cuda.h>
#include <cuda/launch>
#include <curand_kernel.h>
namespace flowdit::kernels {
    namespace {
        constexpr std::uint32_t thread_count = 256u;
        __global__ void make_sampling_noise_kernel(float* const state, const std::uint64_t seed, const std::uint32_t sample_elements) {
            const std::uint32_t sample = blockIdx.x;
            for (std::uint32_t group = threadIdx.x; group < (sample_elements + 3u) / 4u; group += blockDim.x) {
                curandStatePhilox4_32_10_t random{};
                curand_init(seed, static_cast<std::uint64_t>(sample) * 8u + 4u, static_cast<unsigned long long>(group) * 4ull, &random);
                const float4 noise      = curand_normal4(&random);
                const std::size_t index = static_cast<std::size_t>(sample) * sample_elements + group * 4u;
                const float gaussian[4]{noise.x, noise.y, noise.z, noise.w};
                for (std::uint32_t lane = 0; lane < 4u && group * 4u + lane < sample_elements; ++lane) state[index + lane] = gaussian[lane];
            }
        }
        __global__ void make_sampling_time_kernel(float* const times, const float time, const std::uint32_t batch) {
            const std::uint32_t sample = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (sample < batch) times[sample] = time;
        }
        __global__ void make_labels_kernel(std::uint32_t* const labels, const std::uint32_t batch, const std::uint32_t class_index, const std::uint32_t class_count) {
            const std::uint32_t sample = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (sample < batch) labels[sample] = class_index == UINT32_MAX ? sample % class_count : class_index;
        }
        __global__ void combine_guidance_kernel(const float* const conditional, const float* const unconditional, float* const output, const float guidance, const std::size_t count) {
            const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index < count) output[index] = fmaf(guidance, conditional[index] - unconditional[index], unconditional[index]);
        }
        __global__ void euler_step_kernel(float* const state, const float* const velocity, const float step_size, const std::size_t count) {
            const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index < count) state[index] = fmaf(step_size, velocity[index], state[index]);
        }
        __global__ void heun_predict_kernel(const float* const state, const float* const velocity, float* const prediction, const float step_size, const std::size_t count) {
            const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index < count) prediction[index] = fmaf(step_size, velocity[index], state[index]);
        }
        __global__ void heun_step_kernel(float* const state, const float* const first_velocity, const float* const second_velocity, const float step_size, const std::size_t count) {
            const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index < count) state[index] = fmaf(0.5F * step_size, first_velocity[index] + second_velocity[index], state[index]);
        }
        __global__ void rk4_intermediate_kernel(const float* const state, const float* const velocity, float* const intermediate, const float step_size, const std::size_t count) {
            const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index < count) intermediate[index] = fmaf(step_size, velocity[index], state[index]);
        }
        __global__ void rk4_step_kernel(float* const state, const float* const first, const float* const second, const float* const third, const float* const fourth, const float step_size, const std::size_t count) {
            const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index < count) state[index] += step_size * (first[index] + 2.0F * second[index] + 2.0F * third[index] + fourth[index]) / 6.0F;
        }
    }
    void make_sampling_noise(const ::cuda::stream_ref stream, float* const state, const std::uint64_t seed, const std::uint32_t batch, const std::uint32_t sample_elements) {
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims(batch), ::cuda::block_dims(thread_count))), make_sampling_noise_kernel, state, seed, sample_elements);
    }
    void make_sampling_time(const ::cuda::stream_ref stream, float* const times, const float time, const std::uint32_t batch) {
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims(::cuda::ceil_div(batch, thread_count)), ::cuda::block_dims(thread_count))), make_sampling_time_kernel, times, time, batch);
    }
    void make_labels(const ::cuda::stream_ref stream, std::uint32_t* const labels, const std::uint32_t batch, const std::uint32_t class_index, const std::uint32_t class_count) {
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims(::cuda::ceil_div(batch, thread_count)), ::cuda::block_dims(thread_count))), make_labels_kernel, labels, batch, class_index, class_count);
    }
    void combine_guidance(const ::cuda::stream_ref stream, const float* const conditional, const float* const unconditional, float* const output, const float guidance, const std::size_t count) {
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims(::cuda::ceil_div(count, static_cast<std::size_t>(thread_count))), ::cuda::block_dims(thread_count))), combine_guidance_kernel, conditional, unconditional, output, guidance, count);
    }
    void euler_step(const ::cuda::stream_ref stream, float* const state, const float* const velocity, const float step_size, const std::size_t count) {
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims(::cuda::ceil_div(count, static_cast<std::size_t>(thread_count))), ::cuda::block_dims(thread_count))), euler_step_kernel, state, velocity, step_size, count);
    }
    void heun_predict(const ::cuda::stream_ref stream, const float* const state, const float* const velocity, float* const prediction, const float step_size, const std::size_t count) {
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims(::cuda::ceil_div(count, static_cast<std::size_t>(thread_count))), ::cuda::block_dims(thread_count))), heun_predict_kernel, state, velocity, prediction, step_size, count);
    }
    void heun_step(const ::cuda::stream_ref stream, float* const state, const float* const first_velocity, const float* const second_velocity, const float step_size, const std::size_t count) {
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims(::cuda::ceil_div(count, static_cast<std::size_t>(thread_count))), ::cuda::block_dims(thread_count))), heun_step_kernel, state, first_velocity, second_velocity, step_size, count);
    }
    void rk4_intermediate(const ::cuda::stream_ref stream, const float* const state, const float* const velocity, float* const intermediate, const float step_size, const std::size_t count) {
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims(::cuda::ceil_div(count, static_cast<std::size_t>(thread_count))), ::cuda::block_dims(thread_count))), rk4_intermediate_kernel, state, velocity, intermediate, step_size, count);
    }
    void rk4_step(const ::cuda::stream_ref stream, float* const state, const float* const first, const float* const second, const float* const third, const float* const fourth, const float step_size, const std::size_t count) {
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims(::cuda::ceil_div(count, static_cast<std::size_t>(thread_count))), ::cuda::block_dims(thread_count))), rk4_step_kernel, state, first, second, third, fourth, step_size, count);
    }
} // namespace flowdit::kernels
