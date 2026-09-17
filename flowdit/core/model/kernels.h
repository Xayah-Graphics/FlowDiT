#ifndef FLOWDIT_KERNELS_H
#define FLOWDIT_KERNELS_H
#include <cstddef>
#include <cstdint>
#include <flowdit/cuda_stream.h>
namespace flowdit::kernels {
    struct TensorLayout final {
        std::uint32_t width, height, channels, patch_size;
    };
    void make_time_embedding(::cuda::stream_ref stream, const float* times, float* embedding, std::uint32_t batch, std::uint32_t width);
    void silu_forward(::cuda::stream_ref stream, const float* input, float* output, std::size_t count);
    void silu_backward(::cuda::stream_ref stream, const float* input, const float* output_gradient, float* input_gradient, std::size_t count);
    void add_position(::cuda::stream_ref stream, float* tokens, const float* position, std::size_t count, std::size_t position_count);
    void make_condition(::cuda::stream_ref stream, float* time_condition, const float* class_embedding, const std::uint32_t* labels, std::uint32_t batch, std::uint32_t width);
    void class_embedding_backward(::cuda::stream_ref stream, const float* condition_gradient, const std::uint32_t* labels, float* class_embedding_gradient, std::uint32_t batch, std::uint32_t width, std::uint32_t class_count);
    void final_adaln_forward(::cuda::stream_ref stream, const float* input, const float* modulation, float* output, float* means, float* inverse_standard_deviations, std::uint32_t batch, std::uint32_t sequence, std::uint32_t width);
    void final_adaln_backward(::cuda::stream_ref stream, const float* input, const float* modulation, const float* output_gradient, const float* means, const float* inverse_standard_deviations, float* input_gradient, float* modulation_gradient, std::uint32_t batch, std::uint32_t sequence, std::uint32_t width);
    void unpatchify(::cuda::stream_ref stream, const float* patches, float* values, std::uint32_t batch, TensorLayout image);
} // namespace flowdit::kernels
#endif
