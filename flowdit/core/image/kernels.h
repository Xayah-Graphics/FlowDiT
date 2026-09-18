#ifndef FLOWDIT_IMAGE_KERNELS_H
#define FLOWDIT_IMAGE_KERNELS_H
#include <cstdint>
#include <flowdit/cuda_stream.h>
namespace flowdit::kernels {
    void encode_pixels(::cuda::stream_ref stream, const std::uint8_t* images, float* values, std::uint32_t batch, std::uint32_t width, std::uint32_t height, std::uint32_t channels, bool horizontal_flip, std::uint64_t seed, std::uint64_t step);
    void decode_pixels(::cuda::stream_ref stream, const float* values, std::uint8_t* rgba, std::uint32_t batch, std::uint32_t width, std::uint32_t height, std::uint32_t channels);
} // namespace flowdit::kernels
#endif
