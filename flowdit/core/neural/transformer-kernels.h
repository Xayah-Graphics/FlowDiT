#ifndef FLOWDIT_TRANSFORMER_KERNELS_H
#define FLOWDIT_TRANSFORMER_KERNELS_H
#include <cstddef>
#include <cstdint>
#include <flowdit/cuda_stream.h>
namespace flowdit::neural::kernels {
    void normalize(::cuda::stream_ref s, const std::uint16_t* x, const float* weight, const float* bias, std::uint16_t* y, float* mean, float* inverse, std::uint32_t rows, std::uint32_t width);
    void normalize_backward(::cuda::stream_ref s, const std::uint16_t* x, const std::uint16_t* dy, const float* weight, const float* mean, const float* inverse, std::uint16_t* dx, float* dw, float* db, const std::uint16_t* residual, std::uint32_t rows, std::uint32_t width);
    void bias(::cuda::stream_ref s, std::uint16_t* x, const float* bias, std::uint32_t rows, std::uint32_t width);
    void bias_backward(::cuda::stream_ref s, const std::uint16_t* dy, float* db, std::uint32_t rows, std::uint32_t width);
    void gelu(::cuda::stream_ref s, const std::uint16_t* x, std::uint16_t* y, std::size_t count);
    void gelu_backward(::cuda::stream_ref s, const std::uint16_t* x, std::uint16_t* gradient, std::size_t count);
    void add(::cuda::stream_ref s, const std::uint16_t* a, const std::uint16_t* b, std::uint16_t* output, std::size_t count);
    void concatenate(::cuda::stream_ref s, const std::uint16_t* a, const std::uint16_t* b, std::uint16_t* output, std::uint32_t rows, std::uint32_t width);
    void split(::cuda::stream_ref s, const std::uint16_t* input, std::uint16_t* a, std::uint16_t* b, std::uint32_t rows, std::uint32_t width);
} // namespace flowdit::neural::kernels
#endif
