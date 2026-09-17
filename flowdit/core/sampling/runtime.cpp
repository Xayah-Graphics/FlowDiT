module;
#include "../model/kernels.h"
#include <flowdit/cuda.h>
module flowdit.sampling.runtime;
import std;
import flowdit.model;
namespace flowdit {
    SamplingRuntime::SamplingRuntime(const ::cuda::stream_ref source_stream, FlowDiT& source_model)
        : value_count{static_cast<std::size_t>(batch) * source_model.configuration.image.width * source_model.configuration.image.height * source_model.configuration.image.channels}, stream{source_stream}, model{source_model}, model_workspace_layout{batch, model.configuration}, model_workspace{stream, ::cuda::device_default_memory_pool(stream.device()), model_workspace_layout.byte_count, ::cuda::no_init}, state{stream, ::cuda::device_default_memory_pool(stream.device()), value_count, ::cuda::no_init}, times{stream, ::cuda::device_default_memory_pool(stream.device()), batch, ::cuda::no_init}, labels{stream, ::cuda::device_default_memory_pool(stream.device()), batch, ::cuda::no_init}, null_labels{stream, ::cuda::device_default_memory_pool(stream.device()), batch, ::cuda::no_init}, conditional_velocity{stream, ::cuda::device_default_memory_pool(stream.device()), value_count, ::cuda::no_init}, unconditional_velocity{stream, ::cuda::device_default_memory_pool(stream.device()), value_count, ::cuda::no_init},
          image_rgba{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(batch) * model.configuration.image.width * model.configuration.image.height * 4uz, ::cuda::no_init} {
        kernels::make_labels(stream, null_labels.data(), batch, static_cast<std::uint32_t>(model.configuration.image.classes.size()), static_cast<std::uint32_t>(model.configuration.image.classes.size()));
    }
    std::optional<SamplingResult> SamplingRuntime::sample(const float* const parameters, const SamplingRequest& source, const SamplingObserver& observer) {
        initialize(source);
        publish(observer);
        while (progress.step < progress.step_count) {
            if (observer.stop.stop_requested()) return std::nullopt;
            advance(parameters);
            if (progress.step < progress.step_count) publish(observer);
        }
        return finish(observer);
    }
    void SamplingRuntime::initialize(const SamplingRequest& source) {
        request  = source;
        progress = {.step_count = request.step_count};
        const std::size_t count = value_count * (request.solver == SamplingSolver::euler ? 1uz : request.solver == SamplingSolver::heun ? 3uz : 5uz);
        if (!stages || stages->size() != count) stages.emplace(stream, ::cuda::device_default_memory_pool(stream.device()), count, ::cuda::no_init);
        kernels::make_sampling_noise(stream, state.data(), request.seed, batch, static_cast<std::uint32_t>(value_count / batch));
        kernels::make_labels(stream, labels.data(), batch, request.class_index.value_or(UINT32_MAX), static_cast<std::uint32_t>(model.configuration.image.classes.size()));
    }
    void SamplingRuntime::advance(const float* const parameters) {
        const float step_size = 1.0F / static_cast<float>(request.step_count);
        const float time      = static_cast<float>(progress.step) * step_size;
        float* const first = stages->data();
        if (request.solver == SamplingSolver::euler) {
            evaluate(parameters, state.data(), time, request.guidance, first);
            kernels::euler_step(stream, state.data(), first, step_size, value_count);
            ++progress.nfe;
        } else if (request.solver == SamplingSolver::heun) {
            float* const second = first + value_count;
            float* const intermediate = second + value_count;
            evaluate(parameters, state.data(), time, request.guidance, first);
            kernels::heun_predict(stream, state.data(), first, intermediate, step_size, value_count);
            evaluate(parameters, intermediate, time + step_size, request.guidance, second);
            kernels::heun_step(stream, state.data(), first, second, step_size, value_count);
            progress.nfe += 2u;
        } else {
            float* const second = first + value_count;
            float* const third = second + value_count;
            float* const fourth = third + value_count;
            float* const intermediate = fourth + value_count;
            evaluate(parameters, state.data(), time, request.guidance, first);
            kernels::rk4_intermediate(stream, state.data(), first, intermediate, 0.5F * step_size, value_count);
            evaluate(parameters, intermediate, time + 0.5F * step_size, request.guidance, second);
            kernels::rk4_intermediate(stream, state.data(), second, intermediate, 0.5F * step_size, value_count);
            evaluate(parameters, intermediate, time + 0.5F * step_size, request.guidance, third);
            kernels::rk4_intermediate(stream, state.data(), third, intermediate, step_size, value_count);
            evaluate(parameters, intermediate, time + step_size, request.guidance, fourth);
            kernels::rk4_step(stream, state.data(), first, second, third, fourth, step_size, value_count);
            progress.nfe += 4u;
        }
        ++progress.step;
    }
    void SamplingRuntime::publish(const SamplingObserver& observer) {
        if (observer.image) {
            kernels::unpatchify(stream, state.data(), image_rgba.data(), batch, {model.configuration.image.width, model.configuration.image.height, model.configuration.image.channels, model.configuration.patch_size});
            observer.image(image_rgba.data(), model.configuration.image.width, batch * model.configuration.image.height, stream);
        }
        stream.sync();
        if (observer.progress) observer.progress(progress);
    }
    SamplingResult SamplingRuntime::finish(const SamplingObserver& observer) {
        kernels::unpatchify(stream, state.data(), image_rgba.data(), batch, {model.configuration.image.width, model.configuration.image.height, model.configuration.image.channels, model.configuration.patch_size});
        if (observer.image) observer.image(image_rgba.data(), model.configuration.image.width, batch * model.configuration.image.height, stream);
        SamplingResult result{.model = model.configuration, .nfe = progress.nfe, .labels = std::vector<std::uint32_t>(batch), .rgba = std::vector<std::uint8_t>(image_rgba.size())};
        ::cuda::copy_bytes(stream, image_rgba, ::cuda::std::span<std::uint8_t>{result.rgba.data(), result.rgba.size()});
        ::cuda::copy_bytes(stream, labels, ::cuda::std::span<std::uint32_t>{result.labels.data(), result.labels.size()});
        stream.sync();
        if (observer.progress) observer.progress(progress);
        return result;
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
