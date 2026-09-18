#include "kernels.h"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda/launch>
#include <flowdit/cuda.h>
namespace flowdit::kernels {
    namespace {
        constexpr std::uint32_t thread_count = 256u;
        __global__ void time_embedding_kernel(const float* const times, float* const embedding, const std::uint32_t width, const std::size_t count) {
            const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= count) return;
            const std::uint32_t feature         = static_cast<std::uint32_t>(index % width);
            const std::uint32_t frequency_index = feature % (width / 2u);
            const float frequency               = expf(-logf(10'000.0F) * static_cast<float>(frequency_index) / static_cast<float>(width / 2u));
            const float angle                   = times[index / width] * frequency;
            embedding[index]                    = feature < width / 2u ? cosf(angle) : sinf(angle);
        }
        __global__ void silu_forward_kernel(const float* const input, float* const output, const std::size_t count) {
            const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= count) return;
            const float value = input[index];
            output[index]     = value / (1.0F + expf(-value));
        }
        __global__ void silu_backward_kernel(const float* const input, const float* const output_gradient, float* const input_gradient, const std::size_t count) {
            const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= count) return;
            const float value     = input[index];
            const float sigmoid   = 1.0F / (1.0F + expf(-value));
            input_gradient[index] = output_gradient[index] * sigmoid * (1.0F + value * (1.0F - sigmoid));
        }
        __global__ void add_position_kernel(float* const tokens, const float* const position, const std::size_t count, const std::size_t position_count) {
            const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index < count) tokens[index] += position[index % position_count];
        }
        __global__ void make_condition_kernel(float* const time_condition, const float* const class_embedding, const std::uint32_t* const labels, const std::uint32_t width, const std::size_t count) {
            const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= count) return;
            const std::uint32_t feature = static_cast<std::uint32_t>(index % width);
            time_condition[index] += class_embedding[static_cast<std::size_t>(labels[index / width]) * width + feature];
        }
        __global__ void class_embedding_backward_kernel(const float* const condition_gradient, const std::uint32_t* const labels, float* const class_embedding_gradient, const std::uint32_t batch, const std::uint32_t width, const std::uint32_t class_count) {
            const std::uint32_t index = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= (class_count + 1u) * width) return;
            const std::uint32_t label   = index / width;
            const std::uint32_t feature = index % width;
            float gradient{};
            for (std::uint32_t sample = 0u; sample < batch; ++sample)
                if (labels[sample] == label) gradient += condition_gradient[static_cast<std::size_t>(sample) * width + feature];
            class_embedding_gradient[index] = gradient;
        }
        __global__ void final_adaln_forward_kernel(const float* const input, const float* const modulation, float* const output, float* const means, float* const inverse_standard_deviations, const std::uint32_t sequence_size, const std::uint32_t width) {
            __shared__ float reduction[thread_count];
            const std::uint32_t row     = blockIdx.x;
            const std::uint32_t feature = threadIdx.x;
            const std::uint32_t sample  = row / sequence_size;
            const float value           = feature < width ? input[static_cast<std::size_t>(row) * width + feature] : 0.0F;
            reduction[feature]          = value;
            __syncthreads();
            for (std::uint32_t stride = thread_count / 2u; stride != 0u; stride /= 2u) {
                if (feature < stride) reduction[feature] += reduction[feature + stride];
                __syncthreads();
            }
            const float mean = reduction[0] / static_cast<float>(width);
            __syncthreads();
            reduction[feature] = feature < width ? (value - mean) * (value - mean) : 0.0F;
            __syncthreads();
            for (std::uint32_t stride = thread_count / 2u; stride != 0u; stride /= 2u) {
                if (feature < stride) reduction[feature] += reduction[feature + stride];
                __syncthreads();
            }
            const float inverse_standard_deviation = rsqrtf(reduction[0] / static_cast<float>(width) + 1.0e-6F);
            if (feature == 0u) {
                means[row]                       = mean;
                inverse_standard_deviations[row] = inverse_standard_deviation;
            }
            if (feature < width) {
                const std::size_t modulation_offset                     = static_cast<std::size_t>(sample) * 2u * width;
                output[static_cast<std::size_t>(row) * width + feature] = (value - mean) * inverse_standard_deviation * (1.0F + modulation[modulation_offset + width + feature]) + modulation[modulation_offset + feature];
            }
        }
        __global__ void final_adaln_input_backward_kernel(const float* const input, const float* const modulation, const float* const output_gradient, const float* const means, const float* const inverse_standard_deviations, float* const input_gradient, const std::uint32_t sequence_size, const std::uint32_t width) {
            __shared__ float gradient_sum[thread_count];
            __shared__ float normalized_gradient_sum[thread_count];
            const std::uint32_t row     = blockIdx.x;
            const std::uint32_t feature = threadIdx.x;
            const std::uint32_t sample  = row / sequence_size;
            float normalized{};
            float gradient{};
            if (feature < width) {
                normalized = (input[static_cast<std::size_t>(row) * width + feature] - means[row]) * inverse_standard_deviations[row];
                gradient   = output_gradient[static_cast<std::size_t>(row) * width + feature] * (1.0F + modulation[static_cast<std::size_t>(sample) * 2u * width + width + feature]);
            }
            gradient_sum[feature]            = gradient;
            normalized_gradient_sum[feature] = gradient * normalized;
            __syncthreads();
            for (std::uint32_t stride = thread_count / 2u; stride != 0u; stride /= 2u) {
                if (feature < stride) {
                    gradient_sum[feature] += gradient_sum[feature + stride];
                    normalized_gradient_sum[feature] += normalized_gradient_sum[feature + stride];
                }
                __syncthreads();
            }
            if (feature < width) input_gradient[static_cast<std::size_t>(row) * width + feature] = inverse_standard_deviations[row] * (gradient - (gradient_sum[0] + normalized * normalized_gradient_sum[0]) / static_cast<float>(width));
        }
        __global__ void final_adaln_modulation_backward_kernel(const float* const input, const float* const output_gradient, const float* const means, const float* const inverse_standard_deviations, float* const modulation_gradient, const std::uint32_t sequence_size, const std::uint32_t width) {
            const std::uint32_t sample  = blockIdx.x;
            const std::uint32_t feature = threadIdx.x;
            if (feature >= width) return;
            float shift_gradient{};
            float scale_gradient{};
            for (std::uint32_t token = 0u; token < sequence_size; ++token) {
                const std::size_t row = static_cast<std::size_t>(sample) * sequence_size + token;
                const float gradient  = output_gradient[row * width + feature];
                shift_gradient += gradient;
                scale_gradient = fmaf(gradient, (input[row * width + feature] - means[row]) * inverse_standard_deviations[row], scale_gradient);
            }
            const std::size_t offset                      = static_cast<std::size_t>(sample) * 2u * width;
            modulation_gradient[offset + feature]         = shift_gradient;
            modulation_gradient[offset + width + feature] = scale_gradient;
        }
        __global__ void unpatchify_kernel(const float* const patches, float* const values, const std::size_t count, const TensorLayout image) {
            const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= count) return;
            const std::uint32_t image_pixels = image.width * image.height;
            const std::uint32_t patch_width  = image.patch_size * image.patch_size * image.channels;
            const std::size_t sample         = index / image_pixels;
            const std::uint32_t pixel        = static_cast<std::uint32_t>(index % image_pixels);
            const std::uint32_t y            = pixel / image.width;
            const std::uint32_t x            = pixel % image.width;
            const std::uint32_t token        = (y / image.patch_size) * (image.width / image.patch_size) + x / image.patch_size;
            const std::uint32_t patch_pixel  = (y % image.patch_size) * image.patch_size + x % image.patch_size;
            for (std::uint32_t channel = 0u; channel < image.channels; ++channel) values[(sample * image.channels + channel) * image_pixels + pixel] = patches[sample * image_pixels * image.channels + token * patch_width + patch_pixel * image.channels + channel];
        }
    } // namespace
    void make_time_embedding(const ::cuda::stream_ref stream, const float* const times, float* const embedding, const std::uint32_t batch, const std::uint32_t width) {
        const std::size_t count = static_cast<std::size_t>(batch) * width;
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims(::cuda::ceil_div(count, static_cast<std::size_t>(thread_count))), ::cuda::block_dims(thread_count))), time_embedding_kernel, times, embedding, width, count);
    }
    void silu_forward(const ::cuda::stream_ref stream, const float* const input, float* const output, const std::size_t count) {
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims(::cuda::ceil_div(count, static_cast<std::size_t>(thread_count))), ::cuda::block_dims(thread_count))), silu_forward_kernel, input, output, count);
    }
    void silu_backward(const ::cuda::stream_ref stream, const float* const input, const float* const output_gradient, float* const input_gradient, const std::size_t count) {
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims(::cuda::ceil_div(count, static_cast<std::size_t>(thread_count))), ::cuda::block_dims(thread_count))), silu_backward_kernel, input, output_gradient, input_gradient, count);
    }
    void add_position(const ::cuda::stream_ref stream, float* const tokens, const float* const position, const std::size_t count, const std::size_t position_count) {
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims(::cuda::ceil_div(count, static_cast<std::size_t>(thread_count))), ::cuda::block_dims(thread_count))), add_position_kernel, tokens, position, count, position_count);
    }
    void make_condition(const ::cuda::stream_ref stream, float* const time_condition, const float* const class_embedding, const std::uint32_t* const labels, const std::uint32_t batch, const std::uint32_t width) {
        const std::size_t count = static_cast<std::size_t>(batch) * width;
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims(::cuda::ceil_div(count, static_cast<std::size_t>(thread_count))), ::cuda::block_dims(thread_count))), make_condition_kernel, time_condition, class_embedding, labels, width, count);
    }
    void class_embedding_backward(const ::cuda::stream_ref stream, const float* const condition_gradient, const std::uint32_t* const labels, float* const class_embedding_gradient, const std::uint32_t batch, const std::uint32_t width, const std::uint32_t class_count) {
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims(::cuda::ceil_div((class_count + 1u) * width, thread_count)), ::cuda::block_dims(thread_count))), class_embedding_backward_kernel, condition_gradient, labels, class_embedding_gradient, batch, width, class_count);
    }
    void final_adaln_forward(const ::cuda::stream_ref stream, const float* const input, const float* const modulation, float* const output, float* const means, float* const inverse_standard_deviations, const std::uint32_t batch, const std::uint32_t sequence_size, const std::uint32_t width) {
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims(batch * sequence_size), ::cuda::block_dims(thread_count))), final_adaln_forward_kernel, input, modulation, output, means, inverse_standard_deviations, sequence_size, width);
    }
    void final_adaln_backward(const ::cuda::stream_ref stream, const float* const input, const float* const modulation, const float* const output_gradient, const float* const means, const float* const inverse_standard_deviations, float* const input_gradient, float* const modulation_gradient, const std::uint32_t batch, const std::uint32_t sequence_size, const std::uint32_t width) {
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims(batch * sequence_size), ::cuda::block_dims(thread_count))), final_adaln_input_backward_kernel, input, modulation, output_gradient, means, inverse_standard_deviations, input_gradient, sequence_size, width);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims(batch), ::cuda::block_dims(thread_count))), final_adaln_modulation_backward_kernel, input, output_gradient, means, inverse_standard_deviations, modulation_gradient, sequence_size, width);
    }
    void unpatchify(const ::cuda::stream_ref stream, const float* const patches, float* const values, const std::uint32_t batch, const TensorLayout image) {
        const std::size_t count = static_cast<std::size_t>(batch) * image.width * image.height;
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims(::cuda::ceil_div(count, static_cast<std::size_t>(thread_count))), ::cuda::block_dims(thread_count))), unpatchify_kernel, patches, values, count, image);
    }
} // namespace flowdit::kernels
