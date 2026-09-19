#ifndef FLOWDIT_TOKENIZER_KERNELS_H
#define FLOWDIT_TOKENIZER_KERNELS_H
#include <cstddef>
#include <cstdint>
#include <flowdit/cuda_stream.h>
namespace flowdit::tokenizer_kernels {
    void pack(::cuda::stream_ref stream, const float* source, std::uint16_t* target, std::size_t count);
    void scale(::cuda::stream_ref stream, const float* source, float* target, std::size_t count, float factor);
    void unpack(::cuda::stream_ref stream, const std::uint16_t* source, float* target, std::size_t count, float scale = 1.f);
    void pointwise(::cuda::stream_ref stream, const std::uint16_t* source, const std::uint16_t* skip, const float* parameters, std::uint16_t* target, std::uint32_t channels, std::uint32_t area, int operation);
    void rms(::cuda::stream_ref stream, const std::uint16_t* source, const float* parameters, std::uint16_t* target, std::uint32_t channels, std::uint32_t area);
    void rearrange(::cuda::stream_ref stream, const std::uint16_t* source, std::uint16_t* target, std::uint32_t width, std::uint32_t height, std::uint32_t channels, std::uint32_t target_channels, int operation);
    void attention_pack(::cuda::stream_ref stream, const std::uint16_t* source, const std::uint16_t* multiscale, float* q, float* k, float* v, std::uint32_t heads, std::uint32_t area);
    void attention_output(::cuda::stream_ref stream, const float* source, std::uint16_t* target, std::uint32_t heads, std::uint32_t area);
} // namespace flowdit::tokenizer_kernels
#endif
