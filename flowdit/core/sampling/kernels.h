#ifndef FLOWDIT_SAMPLING_KERNELS_H
#define FLOWDIT_SAMPLING_KERNELS_H
#include "../model/kernels.h"
namespace flowdit::kernels {
    void make_sampling_noise(::cuda::stream_ref stream, float* state, std::uint64_t seed, std::uint32_t batch, std::uint32_t sample_elements, std::uint32_t first_sample = 0);
    void make_sampling_time(::cuda::stream_ref stream, float* times, float time, std::uint32_t batch);
    void make_labels(::cuda::stream_ref stream, std::uint32_t* labels, std::uint32_t batch, std::uint32_t class_index, std::uint32_t class_count, std::uint32_t first_sample = 0);
    void combine_guidance(::cuda::stream_ref stream, const float* conditional, const float* unconditional, float* output, float guidance, std::size_t count);
    void euler_step(::cuda::stream_ref stream, float* state, const float* velocity, float step_size, std::size_t count);
    void heun_predict(::cuda::stream_ref stream, const float* state, const float* velocity, float* prediction, float step_size, std::size_t count);
    void heun_step(::cuda::stream_ref stream, float* state, const float* first_velocity, const float* second_velocity, float step_size, std::size_t count);
    void rk4_intermediate(::cuda::stream_ref stream, const float* state, const float* velocity, float* intermediate, float step_size, std::size_t count);
    void rk4_step(::cuda::stream_ref stream, float* state, const float* first, const float* second, const float* third, const float* fourth, float step_size, std::size_t count);
} // namespace flowdit::kernels
#endif
