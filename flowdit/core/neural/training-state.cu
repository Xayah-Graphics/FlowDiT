#include "training-state-kernels.h"
#include <cuda/launch>
#include <cuda/stream>
namespace flowdit::neural::kernels {
    namespace {
        __global__ void step_scalars_kernel(const std::uint64_t* const step, const std::uint64_t* const processed_samples, float* const step_scalars, const float first_decay, const float second_decay, const std::uint32_t samples_per_step, const std::uint64_t half_life_samples, const float ramp_up_ratio, const std::uint32_t warmup_steps) {
            step_scalars[3]       = warmup_steps ? fminf(1.f, static_cast<float>(*step) / warmup_steps) : 1.f;
            step_scalars[0]       = 1.0F - powf(first_decay, static_cast<float>(*step));
            step_scalars[1]       = 1.0F - powf(second_decay, static_cast<float>(*step));
            const float half_life = fminf(static_cast<float>(half_life_samples), static_cast<float>(*processed_samples) * ramp_up_ratio);
            step_scalars[2]       = *processed_samples == 0u ? 0.0F : exp2f(-static_cast<float>(samples_per_step) / half_life);
        }
        __global__ void optimize_device_kernel(float* const parameters, float* const gradients, float* const first_moments, float* const second_moments, float* const ema, const std::size_t count, const float learning_rate, const float first_decay, const float second_decay, const float* const step_scalars, const float epsilon, const float weight_decay) {
            const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= count) return;
            const float gradient = gradients[index];
            const float first = first_moments[index] = fmaf(1.0F - first_decay, gradient, first_decay * first_moments[index]);
            const float second = second_moments[index] = fmaf(1.0F - second_decay, gradient * gradient, second_decay * second_moments[index]);
            const float corrected                      = first / step_scalars[0] / (sqrtf(second / step_scalars[1]) + epsilon);
            const float parameter                      = parameters[index] - learning_rate * step_scalars[3] * fmaf(weight_decay, parameters[index], corrected);
            parameters[index]                          = parameter;
            gradients[index]                           = 0.0F;
            ema[index]                                 = fmaf(1.0F - step_scalars[2], parameter, step_scalars[2] * ema[index]);
        }
    } // namespace
    void optimize(const ::cuda::stream_ref stream, float* const parameters, float* const gradients, float* const first_moments, float* const second_moments, float* const ema, const std::size_t count, const float learning_rate, const float first_decay, const float second_decay, const std::uint64_t* const step, const std::uint64_t* const processed_samples, float* const step_scalars, const float epsilon, const float weight_decay, const std::uint32_t samples_per_step, const std::uint64_t half_life_samples, const float ramp_up_ratio, const std::uint32_t warmup_steps) {
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims(1u), ::cuda::block_dims(1u))), step_scalars_kernel, step, processed_samples, step_scalars, first_decay, second_decay, samples_per_step, half_life_samples, ramp_up_ratio, warmup_steps);
        ::cuda::launch(stream, ::cuda::distribute<256u>(count), optimize_device_kernel, parameters, gradients, first_moments, second_moments, ema, count, learning_rate, first_decay, second_decay, step_scalars, epsilon, weight_decay);
    }
} // namespace flowdit::neural::kernels
