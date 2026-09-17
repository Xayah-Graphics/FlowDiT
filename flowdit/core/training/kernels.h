#ifndef FLOWDIT_TRAINING_KERNELS_H
#define FLOWDIT_TRAINING_KERNELS_H
#include "../model/kernels.h"
namespace flowdit::kernels {
    void make_training_batch(::cuda::stream_ref stream, const float* data, const std::uint32_t* input_labels, float* path, float* target, float* times, std::uint32_t* labels, const std::uint64_t* step, const std::uint64_t* seed, std::uint32_t batch, TensorLayout image, std::uint32_t class_count);
    void flow_matching_loss(::cuda::stream_ref stream, const float* prediction, const float* target, float* prediction_gradient, float* sample_loss, float* loss, std::uint32_t batch, std::uint32_t sample_elements);
    void advance_training_state(::cuda::stream_ref stream, std::uint64_t* step, std::uint64_t* processed_samples, std::uint32_t samples_per_step);
} // namespace flowdit::kernels
#endif
