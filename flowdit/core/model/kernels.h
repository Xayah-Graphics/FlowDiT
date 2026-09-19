#ifndef FLOWDIT_KERNELS_H
#define FLOWDIT_KERNELS_H
#include <cstddef>
#include <cstdint>
#include <flowdit/cuda_stream.h>
namespace flowdit::kernels {
    struct TensorLayout final {
        std::uint32_t width, height, channels, patch_size;
    };
    void make_tokens(::cuda::stream_ref s, const std::uint16_t* patches, const float* times, const std::uint32_t* labels, const float* classes, const float* position, std::uint16_t* tokens, std::uint32_t batch, std::uint32_t sequence, std::uint32_t width);
    void tokens_backward(::cuda::stream_ref s, const std::uint16_t* gradient, const std::uint32_t* labels, float* classes, float* position, std::uint16_t* patches, std::uint32_t batch, std::uint32_t sequence, std::uint32_t width, std::uint32_t class_count);
    void velocity(::cuda::stream_ref s, const std::uint16_t* full, float* output, std::uint32_t batch, std::uint32_t sequence, std::uint32_t width);
    void velocity_backward(::cuda::stream_ref s, const float* gradient, std::uint16_t* full, std::uint32_t batch, std::uint32_t sequence, std::uint32_t width);
    void unpatchify(::cuda::stream_ref stream, const float* patches, float* values, std::uint32_t batch, TensorLayout image);
} // namespace flowdit::kernels
#endif
