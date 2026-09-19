#include "kernels.h"
#include <cuda/launch>
#include <flowdit/cuda.h>
namespace flowdit::kernels {
    namespace {
        constexpr std::uint32_t thread_count = 256u;
        __global__ void encode_pixels_kernel(const std::uint8_t* images, float* values, const std::uint32_t width, const std::uint32_t height, const std::uint32_t channels) {
            const std::uint32_t sample = blockIdx.x;
            const std::uint32_t pixels = width * height;
            for (std::uint32_t i = threadIdx.x; i < pixels * channels; i += blockDim.x) {
                const std::size_t index = static_cast<std::size_t>(sample) * pixels * channels + i;
                values[index]           = static_cast<float>(images[index]) * (2.0F / 255.0F) - 1.0F;
            }
        }
        __global__ void decode_pixels_kernel(const float* values, std::uint8_t* rgba, const std::uint32_t pixels, const std::uint32_t channels, const std::size_t count) {
            const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= count) return;
            const std::size_t sample  = index / pixels;
            const std::uint32_t pixel = static_cast<std::uint32_t>(index % pixels);
            for (std::uint32_t channel = 0; channel < 3; ++channel) {
                const std::uint32_t source_channel = channels == 1 ? 0 : channel;
                const float value                  = fminf(fmaxf(values[(sample * channels + source_channel) * pixels + pixel], -1.0F), 1.0F);
                rgba[index * 4 + channel]          = static_cast<std::uint8_t>(rintf((value + 1.0F) * 127.5F));
            }
            rgba[index * 4 + 3] = 255;
        }
    } // namespace
    void encode_pixels(const ::cuda::stream_ref stream, const std::uint8_t* images, float* values, const std::uint32_t batch, const std::uint32_t width, const std::uint32_t height, const std::uint32_t channels) {
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims(batch), ::cuda::block_dims(thread_count))), encode_pixels_kernel, images, values, width, height, channels);
    }
    void decode_pixels(const ::cuda::stream_ref stream, const float* values, std::uint8_t* rgba, const std::uint32_t batch, const std::uint32_t width, const std::uint32_t height, const std::uint32_t channels) {
        const std::size_t count = static_cast<std::size_t>(batch) * width * height;
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims(::cuda::ceil_div(count, static_cast<std::size_t>(thread_count))), ::cuda::block_dims(thread_count))), decode_pixels_kernel, values, rgba, width * height, channels, count);
    }
} // namespace flowdit::kernels
