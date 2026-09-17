#include "kernels.h"
#include <cmath>
#include <cuda/launch>
#include <curand_kernel.h>
namespace flowdit::kernels {
    namespace {
        constexpr std::uint32_t thread_count = 256u;
        __device__ unsigned long long subsequence(const std::uint64_t step, const std::uint32_t batch, const std::uint32_t sample, const std::uint32_t domain) {
            return (step * batch + sample) * 8ull + domain;
        }
        __global__ void make_training_batch_kernel(const float* const data, const std::uint32_t* const input_labels, float* const path, float* const target, float* const times, std::uint32_t* const labels, const std::uint64_t* const step, const std::uint64_t* const seed, const std::uint32_t batch, const TensorLayout image, const std::uint32_t class_count) {
            const std::uint32_t sample_elements = image.width * image.height * image.channels;
            const std::uint32_t patch_width         = image.patch_size * image.patch_size * image.channels;
            __shared__ float time;
            const std::uint32_t sample = blockIdx.x;
            if (threadIdx.x == 0u) {
                curandStatePhilox4_32_10_t state{};
                curand_init(*seed, subsequence(*step, batch, sample, 1u), 0ull, &state);
                time = static_cast<float>(curand(&state) >> 8u) * 0x1p-24F;
                curand_init(*seed, subsequence(*step, batch, sample, 2u), 0ull, &state);
                labels[sample] = static_cast<float>(curand(&state) >> 8u) * 0x1p-24F < 0.1F ? class_count : input_labels[sample];
                times[sample]  = time;
            }
            __syncthreads();
            for (std::uint32_t group = threadIdx.x; group < (sample_elements + 3u) / 4u; group += blockDim.x) {
                curandStatePhilox4_32_10_t state{};
                curand_init(*seed, subsequence(*step, batch, sample, 3u), static_cast<unsigned long long>(group) * 4ull, &state);
                const float4 noise = curand_normal4(&state);
                const float gaussian[4]{noise.x, noise.y, noise.z, noise.w};
                for (std::uint32_t lane = 0u; lane < 4u; ++lane) {
                    const std::uint32_t patch_index = group * 4u + lane;
                    if (patch_index >= sample_elements) continue;
                    const std::uint32_t token         = patch_index / patch_width;
                    const std::uint32_t patch_element = patch_index % patch_width;
                    const std::uint32_t patch_y       = token / (image.width / image.patch_size);
                    const std::uint32_t patch_x       = token % (image.width / image.patch_size);
                    const std::uint32_t pixel         = patch_element / image.channels;
                    const std::uint32_t channel       = patch_element % image.channels;
                    const std::uint32_t y             = patch_y * image.patch_size + pixel / image.patch_size;
                    const std::uint32_t x   = patch_x * image.patch_size + pixel % image.patch_size;
                    const std::size_t source          = static_cast<std::size_t>(sample) * sample_elements + static_cast<std::size_t>(channel) * image.width * image.height + y * image.width + x;
                    const float value                 = data[source];
                    const std::size_t destination     = static_cast<std::size_t>(sample) * sample_elements + patch_index;
                    path[destination]                 = fmaf(time, value - gaussian[lane], gaussian[lane]);
                    target[destination]               = value - gaussian[lane];
                }
            }
        }
        __global__ void flow_matching_sample_loss_kernel(const float* const prediction, const float* const target, float* const prediction_gradient, float* const sample_loss, const std::uint32_t sample_elements) {
            __shared__ float reduction[thread_count];
            const std::uint32_t sample = blockIdx.x;
            float loss{};
            for (std::uint32_t element = threadIdx.x; element < sample_elements; element += blockDim.x) {
                const std::size_t index    = static_cast<std::size_t>(sample) * sample_elements + element;
                const float difference     = prediction[index] - target[index];
                loss                       = fmaf(difference, difference, loss);
                prediction_gradient[index] = 2.0F * difference / static_cast<float>(gridDim.x * sample_elements);
            }
            reduction[threadIdx.x] = loss;
            __syncthreads();
            for (std::uint32_t stride = thread_count / 2u; stride != 0u; stride /= 2u) {
                if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
                __syncthreads();
            }
            if (threadIdx.x == 0u) sample_loss[sample] = reduction[0];
        }
        __global__ void flow_matching_loss_kernel(const float* const sample_loss, float* const loss, const std::uint32_t batch, const std::uint32_t sample_elements) {
            __shared__ float reduction[thread_count];
            float sum{};
            for (std::uint32_t sample = threadIdx.x; sample < batch; sample += blockDim.x) sum += sample_loss[sample];
            reduction[threadIdx.x] = sum;
            __syncthreads();
            for (std::uint32_t stride = thread_count / 2u; stride != 0u; stride /= 2u) {
                if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
                __syncthreads();
            }
            if (threadIdx.x == 0u) *loss = reduction[0] / static_cast<float>(batch * sample_elements);
        }
        __global__ void advance_training_state_kernel(std::uint64_t* const step, std::uint64_t* const processed_samples, const std::uint32_t samples_per_step) {
            ++*step;
            *processed_samples += samples_per_step;
        }
    }
    void make_training_batch(const ::cuda::stream_ref stream, const float* const data, const std::uint32_t* const input_labels, float* const path, float* const target, float* const times, std::uint32_t* const labels, const std::uint64_t* const step, const std::uint64_t* const seed, const std::uint32_t batch, const TensorLayout image, const std::uint32_t class_count) {
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims(batch), ::cuda::block_dims(thread_count))), make_training_batch_kernel, data, input_labels, path, target, times, labels, step, seed, batch, image, class_count);
    }
    void flow_matching_loss(const ::cuda::stream_ref stream, const float* const prediction, const float* const target, float* const prediction_gradient, float* const sample_loss, float* const loss, const std::uint32_t batch, const std::uint32_t sample_elements) {
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims(batch), ::cuda::block_dims(thread_count))), flow_matching_sample_loss_kernel, prediction, target, prediction_gradient, sample_loss, sample_elements);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims(1u), ::cuda::block_dims(thread_count))), flow_matching_loss_kernel, sample_loss, loss, batch, sample_elements);
    }
    void advance_training_state(const ::cuda::stream_ref stream, std::uint64_t* const step, std::uint64_t* const processed_samples, const std::uint32_t samples_per_step) {
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims(1u), ::cuda::block_dims(1u))), advance_training_state_kernel, step, processed_samples, samples_per_step);
    }
} // namespace flowdit::kernels
