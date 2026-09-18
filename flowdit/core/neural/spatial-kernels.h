#ifndef FLOWDIT_SPATIAL_KERNELS_H
#define FLOWDIT_SPATIAL_KERNELS_H
#include <cstddef>
#include <cstdint>
#include <flowdit/cuda_stream.h>
namespace flowdit::kernels {
    void pack(::cuda::stream_ref stream, const float* input, std::uint16_t* output, std::size_t count);
    void unpack(::cuda::stream_ref stream, const std::uint16_t* input, float* output, std::size_t count, bool add = false);
    void sum(::cuda::stream_ref stream, const float* input, float* output, std::size_t count, float scale = 1);
    void bias(::cuda::stream_ref stream, std::uint16_t* values, const float* bias, std::uint32_t batch, std::uint32_t channels, std::uint32_t area);
    void bias_gradient(::cuda::stream_ref stream, const float* gradient, float* output, std::uint32_t batch, std::uint32_t channels, std::uint32_t area);
    void activation(::cuda::stream_ref stream, const std::uint16_t* input, std::uint16_t* output, std::size_t count, int kind);
    void activation_backward(::cuda::stream_ref stream, const std::uint16_t* input, const float* gradient, float* output, std::size_t count, int kind);
    void residual(::cuda::stream_ref stream, const std::uint16_t* input, const std::uint16_t* skip, std::uint16_t* output, std::size_t count);
    void group_norm(::cuda::stream_ref stream, const std::uint16_t* input, const float* parameters, std::uint16_t* output, std::uint32_t batch, std::uint32_t channels, std::uint32_t area);
    void group_norm_backward(::cuda::stream_ref stream, const std::uint16_t* input, const float* parameters, const float* gradient, float* output, float* parameter_gradient, std::uint32_t batch, std::uint32_t channels, std::uint32_t area);
    void resize(::cuda::stream_ref stream, const std::uint16_t* input, std::uint16_t* output, std::uint32_t planes, std::uint32_t width, std::uint32_t height, bool pool);
    void resize_backward(::cuda::stream_ref stream, const std::uint16_t* input, const float* gradient, float* output, std::uint32_t planes, std::uint32_t width, std::uint32_t height, bool pool);
    void attention_softmax(::cuda::stream_ref stream, std::uint16_t* scores, std::uint32_t batch, std::uint32_t sequence);
    void attention_softmax_backward(::cuda::stream_ref stream, const std::uint16_t* probabilities, std::uint16_t* gradient, std::uint32_t batch, std::uint32_t sequence, float scale);
    void posterior(::cuda::stream_ref stream, const float* moments, float* latent, std::uint32_t batch, std::uint32_t elements, std::uint64_t seed, std::uint64_t step, bool sample);
    void posterior_backward(::cuda::stream_ref stream, const float* moments, const float* latent, const float* gradient, float* output, float* loss, std::uint32_t batch, std::uint32_t elements, float weight, float scale);
    void reconstruction_loss(::cuda::stream_ref stream, const float* reference, const float* reconstructed, float* gradient, float* loss, std::size_t count, float scale);
    void adversarial_loss(::cuda::stream_ref stream, const float* logits, float* gradient, float* loss, std::size_t count, int kind, float scale);
    void perceptual_input(::cuda::stream_ref stream, const float* input, float* output, std::uint32_t batch, std::uint32_t channels, std::uint32_t area);
    void perceptual_backward(::cuda::stream_ref stream, const float* gradient, float* output, std::uint32_t batch, std::uint32_t channels, std::uint32_t area);
    void perceptual_loss(::cuda::stream_ref stream, const std::uint16_t* reference, const std::uint16_t* reconstructed, const float* weights, float* gradient, float* loss, std::uint32_t batch, std::uint32_t channels, std::uint32_t area, float scale);
    void standardize(::cuda::stream_ref stream, const float* input, float* output, const float* mean, const float* deviation, std::uint32_t batch, std::uint32_t channels, std::uint32_t area, bool inverse);
} // namespace flowdit::kernels
#endif
