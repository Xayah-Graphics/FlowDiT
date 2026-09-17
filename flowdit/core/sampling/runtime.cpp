module;
#include "../model/kernels.h"
#include <flowdit/cuda.h>
module flowdit.sampling.runtime;
import std;
import flowdit.model;
namespace flowdit {
    SamplingRuntime::SamplingRuntime(const ::cuda::stream_ref source_stream, FlowDiT& source_model)
        : value_count{static_cast<std::size_t>(batch) * source_model.configuration.image.width * source_model.configuration.image.height * source_model.configuration.image.channels}, stream{source_stream}, model{source_model}, model_workspace_layout{batch, model.configuration}, model_workspace{stream, ::cuda::device_default_memory_pool(stream.device()), model_workspace_layout.byte_count, ::cuda::no_init}, state{stream, ::cuda::device_default_memory_pool(stream.device()), value_count, ::cuda::no_init}, times{stream, ::cuda::device_default_memory_pool(stream.device()), batch, ::cuda::no_init}, labels{stream, ::cuda::device_default_memory_pool(stream.device()), batch, ::cuda::no_init}, null_labels{stream, ::cuda::device_default_memory_pool(stream.device()), batch, ::cuda::no_init}, conditional_velocity{stream, ::cuda::device_default_memory_pool(stream.device()), value_count, ::cuda::no_init}, unconditional_velocity{stream, ::cuda::device_default_memory_pool(stream.device()), value_count, ::cuda::no_init},
          velocity{stream, ::cuda::device_default_memory_pool(stream.device()), value_count, ::cuda::no_init}, intermediate{stream, ::cuda::device_default_memory_pool(stream.device()), value_count, ::cuda::no_init}, first{stream, ::cuda::device_default_memory_pool(stream.device()), value_count, ::cuda::no_init}, second{stream, ::cuda::device_default_memory_pool(stream.device()), value_count, ::cuda::no_init}, third{stream, ::cuda::device_default_memory_pool(stream.device()), value_count, ::cuda::no_init}, fourth{stream, ::cuda::device_default_memory_pool(stream.device()), value_count, ::cuda::no_init}, image_rgba{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(batch) * model.configuration.image.width * model.configuration.image.height * 4uz, ::cuda::no_init} {
        kernels::make_labels(stream, null_labels.data(), batch, static_cast<std::uint32_t>(model.configuration.image.classes.size()), static_cast<std::uint32_t>(model.configuration.image.classes.size()));
    }
    void SamplingRuntime::initialize(const SamplingRequest& source) {
        request  = source;
        progress = {.step_count = request.step_count};
        kernels::make_sampling_noise(stream, state.data(), request.seed, batch, static_cast<std::uint32_t>(value_count / batch));
        kernels::make_labels(stream, labels.data(), batch, request.class_index.value_or(UINT32_MAX), static_cast<std::uint32_t>(model.configuration.image.classes.size()));
    }
    void SamplingRuntime::advance(const float* const parameters) {
        const float step_size = 1.0F / static_cast<float>(request.step_count);
        const float time      = static_cast<float>(progress.step) * step_size;
        if (request.solver == SamplingSolver::euler) {
            evaluate(parameters, state.data(), time, request.guidance, velocity.data());
            kernels::euler_step(stream, state.data(), velocity.data(), step_size, value_count);
            ++progress.nfe;
        } else if (request.solver == SamplingSolver::heun) {
            evaluate(parameters, state.data(), time, request.guidance, first.data());
            kernels::heun_predict(stream, state.data(), first.data(), intermediate.data(), step_size, value_count);
            evaluate(parameters, intermediate.data(), time + step_size, request.guidance, second.data());
            kernels::heun_step(stream, state.data(), first.data(), second.data(), step_size, value_count);
            progress.nfe += 2u;
        } else {
            evaluate(parameters, state.data(), time, request.guidance, first.data());
            kernels::rk4_intermediate(stream, state.data(), first.data(), intermediate.data(), 0.5F * step_size, value_count);
            evaluate(parameters, intermediate.data(), time + 0.5F * step_size, request.guidance, second.data());
            kernels::rk4_intermediate(stream, state.data(), second.data(), intermediate.data(), 0.5F * step_size, value_count);
            evaluate(parameters, intermediate.data(), time + 0.5F * step_size, request.guidance, third.data());
            kernels::rk4_intermediate(stream, state.data(), third.data(), intermediate.data(), step_size, value_count);
            evaluate(parameters, intermediate.data(), time + step_size, request.guidance, fourth.data());
            kernels::rk4_step(stream, state.data(), first.data(), second.data(), third.data(), fourth.data(), step_size, value_count);
            progress.nfe += 4u;
        }
        ++progress.step;
        progress.time = static_cast<float>(progress.step) * step_size;
    }
    void SamplingRuntime::publish(const SamplingObserver& observer) {
        if (observer.image) {
            kernels::unpatchify(stream, state.data(), image_rgba.data(), batch, {model.configuration.image.width, model.configuration.image.height, model.configuration.image.channels, model.configuration.patch_size});
            observer.image(progress, image_rgba.data(), model.configuration.image.width, batch * model.configuration.image.height, stream);
        }
        stream.sync();
        if (observer.progress) observer.progress(progress);
    }
    SamplingResult SamplingRuntime::finish() {
        kernels::unpatchify(stream, state.data(), image_rgba.data(), batch, {model.configuration.image.width, model.configuration.image.height, model.configuration.image.channels, model.configuration.patch_size});
        SamplingResult result{.model = model.configuration, .nfe = progress.nfe, .labels = std::vector<std::uint32_t>(batch), .rgba = std::vector<std::uint8_t>(image_rgba.size())};
        ::cuda::copy_bytes(stream, image_rgba, ::cuda::std::span<std::uint8_t>{result.rgba.data(), result.rgba.size()});
        ::cuda::copy_bytes(stream, labels, ::cuda::std::span<std::uint32_t>{result.labels.data(), result.labels.size()});
        stream.sync();
        return result;
    }
    std::optional<SamplingResult> SamplingRuntime::sample(const float* const parameters, const SamplingRequest& source, const SamplingObserver& observer) {
        initialize(source);
        publish(observer);
        while (progress.step < progress.step_count) {
            if (observer.stop.stop_requested()) return std::nullopt;
            advance(parameters);
            publish(observer);
        }
        return finish();
    }
    void SamplingRuntime::evaluate(const float* const parameters, const float* const input, const float time, const float guidance, float* const output) {
        kernels::make_sampling_time(stream, times.data(), time, batch);
        model.forward(parameters, input, times.data(), labels.data(), model_workspace.data(), model_workspace_layout);
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{reinterpret_cast<const float*>(model_workspace.data() + model_workspace_layout.velocity), value_count}, conditional_velocity);
        model.forward(parameters, input, times.data(), null_labels.data(), model_workspace.data(), model_workspace_layout);
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{reinterpret_cast<const float*>(model_workspace.data() + model_workspace_layout.velocity), value_count}, unconditional_velocity);
        kernels::combine_guidance(stream, conditional_velocity.data(), unconditional_velocity.data(), output, guidance, value_count);
    }
} // namespace flowdit
