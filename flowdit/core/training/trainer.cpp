module;
#include "../model/kernels.h"
#include <cuda_runtime_api.h>
#include <flowdit/cuda.h>
module flowdit.training.trainer;
import std;
import flowdit.dataset.types;
import flowdit.model;
import flowdit.sampling.runtime;
import flowdit.neural.matmul;
import flowdit.neural.training_state;
import flowdit.serialization.safetensors;
namespace flowdit {
    Trainer::Trainer(const Dataset& training_set, const int device_ordinal, const std::uint64_t seed, const neural::TrainingConfiguration& configuration, const std::uint32_t patch_size)
        : state{.seed = seed}, value_count{static_cast<std::size_t>(batch) * training_set.specification.width * training_set.specification.height * training_set.specification.channels}, image_count{static_cast<std::uint32_t>(training_set.labels.size())}, horizontal_flip{training_set.horizontal_flip}, stream{::cuda::devices[device_ordinal]}, dataset_images{stream, ::cuda::device_default_memory_pool(stream.device()), training_set.images.size(), ::cuda::no_init}, dataset_labels{stream, ::cuda::device_default_memory_pool(stream.device()), training_set.labels.size(), ::cuda::no_init}, matmul{stream, flow_matmul_runtime_configuration}, model{stream, matmul, {training_set.specification, patch_size}}, parameter_buffer{stream, model.parameters.parameter_count}, training_configuration{configuration}, model_workspace_layout{batch, model.configuration}, model_workspace{stream, ::cuda::device_default_memory_pool(stream.device()), model_workspace_layout.byte_count, ::cuda::no_init},
          path{stream, ::cuda::device_default_memory_pool(stream.device()), value_count, ::cuda::no_init}, target{stream, ::cuda::device_default_memory_pool(stream.device()), value_count, ::cuda::no_init}, times{stream, ::cuda::device_default_memory_pool(stream.device()), batch, ::cuda::no_init}, labels{stream, ::cuda::device_default_memory_pool(stream.device()), batch, ::cuda::no_init}, patch_gradient{stream, ::cuda::device_default_memory_pool(stream.device()), value_count, ::cuda::no_init}, loss{stream, ::cuda::device_default_memory_pool(stream.device()), 1uz, ::cuda::no_init}, loss_sum{stream, ::cuda::device_default_memory_pool(stream.device()), 1uz, ::cuda::no_init}, device_step{stream, ::cuda::device_default_memory_pool(stream.device()), 1uz, ::cuda::no_init}, device_processed_samples{stream, ::cuda::device_default_memory_pool(stream.device()), 1uz, ::cuda::no_init}, device_seed{stream, ::cuda::device_default_memory_pool(stream.device()), 1uz, ::cuda::no_init} {
        ::cuda::copy_bytes(stream, ::cuda::std::span<const std::uint8_t>{training_set.images.data(), training_set.images.size()}, dataset_images);
        ::cuda::copy_bytes(stream, ::cuda::std::span<const std::uint32_t>{training_set.labels.data(), training_set.labels.size()}, dataset_labels);
        parameter_buffer.initialize(model.initialize_parameters(seed));
        const std::uint64_t initial_step = 1u;
        ::cuda::copy_bytes(stream, ::cuda::std::span<const std::uint64_t>{&initial_step, 1uz}, device_step);
        ::cuda::copy_bytes(stream, ::cuda::std::span<const std::uint64_t>{&state.processed_samples, 1uz}, device_processed_samples);
        ::cuda::copy_bytes(stream, ::cuda::std::span<const std::uint64_t>{&state.seed, 1uz}, device_seed);
        kernels::make_training_batch(stream, dataset_images.data(), dataset_labels.data(), path.data(), target.data(), times.data(), labels.data(), device_step.data(), device_seed.data(), batch, {model.configuration.image.width, model.configuration.image.height, model.configuration.image.channels, model.configuration.patch_size}, image_count, static_cast<std::uint32_t>(model.configuration.image.classes.size()), horizontal_flip);
        model.forward(parameter_buffer.parameters.data(), path.data(), times.data(), labels.data(), model_workspace.data(), model_workspace_layout);
        model.loss(target.data(), loss.data(), model_workspace.data(), model_workspace_layout);
        model.backward(parameter_buffer.parameters.data(), parameter_buffer.gradients.data(), path.data(), times.data(), labels.data(), patch_gradient.data(), model_workspace.data(), model_workspace_layout);
        parameter_buffer.clear_gradients();
        stream.sync();
        if (const cudaError_t status = cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeThreadLocal); status != cudaSuccess) throw std::runtime_error{std::format("CUDA graph capture: {}", cudaGetErrorString(status))};
        training_step();
        if (const cudaError_t status = cudaStreamEndCapture(stream.get(), &graph); status != cudaSuccess) throw std::runtime_error{std::format("CUDA graph completion: {}", cudaGetErrorString(status))};
        if (const cudaError_t status = cudaGraphInstantiate(&graph_execution, graph, 0u); status != cudaSuccess) throw std::runtime_error{std::format("CUDA graph instantiation: {}", cudaGetErrorString(status))};
    }
    Trainer::~Trainer() noexcept {
        if (graph_execution != nullptr) cudaGraphExecDestroy(graph_execution);
        if (graph != nullptr) cudaGraphDestroy(graph);
    }
    TrainingStatistics Trainer::optimize(const std::uint64_t iterations) {
        ::cuda::fill_bytes(stream, loss_sum, 0u);
        const auto start = std::chrono::steady_clock::now();
        for (std::uint64_t iteration = 0u; iteration < iterations; ++iteration)
            if (const cudaError_t status = cudaGraphLaunch(graph_execution, stream.get()); status != cudaSuccess) throw std::runtime_error{std::format("CUDA graph launch: {}", cudaGetErrorString(status))};
        float accumulated_loss{};
        ::cuda::copy_bytes(stream, loss_sum, ::cuda::std::span<float>{&accumulated_loss, 1uz});
        stream.sync();
        const double elapsed = std::chrono::duration<double>{std::chrono::steady_clock::now() - start}.count();
        state.step += iterations;
        state.processed_samples += iterations * batch;
        state.elapsed_seconds += elapsed;
        return {
            .average_loss       = accumulated_loss / static_cast<float>(iterations),
            .elapsed_seconds    = elapsed,
        };
    }
    std::optional<SamplingResult> Trainer::sample(const SamplingRequest& request, const ParameterSource source, const SamplingObserver& observer) {
        SamplingRuntime runtime{stream, model};
        if (source == ParameterSource::parameters) return runtime.sample(parameter_buffer.parameters.data(), request, observer);
        return runtime.sample(parameter_buffer.ema.data(), request, observer);
    }
    void Trainer::save(const std::filesystem::path& path) const {
        const neural::ParameterState parameters = parameter_buffer.download();
        const std::array<std::uint64_t, 4u> training_state{state.step, state.processed_samples, state.seed, std::bit_cast<std::uint64_t>(state.elapsed_seconds)};
        const std::array<serialization::safetensors::TensorView, 5u> tensors{
            serialization::safetensors::TensorView{"model.parameters", "F32", {parameters.parameters.size()}, parameters.parameters.data(), parameters.parameters.size() * sizeof(float)},
            serialization::safetensors::TensorView{"optimizer.first_moments", "F32", {parameters.first_moments.size()}, parameters.first_moments.data(), parameters.first_moments.size() * sizeof(float)},
            serialization::safetensors::TensorView{"optimizer.second_moments", "F32", {parameters.second_moments.size()}, parameters.second_moments.data(), parameters.second_moments.size() * sizeof(float)},
            serialization::safetensors::TensorView{"model.ema", "F32", {parameters.ema.size()}, parameters.ema.data(), parameters.ema.size() * sizeof(float)},
            serialization::safetensors::TensorView{"training.state", "U64", {training_state.size()}, training_state.data(), training_state.size() * sizeof(std::uint64_t)},
        };
        serialization::safetensors::write(path, "flow-matching", tensors, {{"flowdit.model", serialize_model(model.configuration)}});
    }
    void Trainer::load(const std::filesystem::path& path) {
        const serialization::safetensors::File file = serialization::safetensors::read(path);
        neural::ParameterState parameters;
        const auto master = std::ranges::find(file.tensors, std::string{"model.parameters"}, &serialization::safetensors::Tensor::name);
        const auto first  = std::ranges::find(file.tensors, std::string{"optimizer.first_moments"}, &serialization::safetensors::Tensor::name);
        const auto second = std::ranges::find(file.tensors, std::string{"optimizer.second_moments"}, &serialization::safetensors::Tensor::name);
        const auto ema    = std::ranges::find(file.tensors, std::string{"model.ema"}, &serialization::safetensors::Tensor::name);
        parameters.parameters.resize(master->data.size() / sizeof(float));
        parameters.first_moments.resize(first->data.size() / sizeof(float));
        parameters.second_moments.resize(second->data.size() / sizeof(float));
        parameters.ema.resize(ema->data.size() / sizeof(float));
        std::memcpy(parameters.parameters.data(), master->data.data(), master->data.size());
        std::memcpy(parameters.first_moments.data(), first->data.data(), first->data.size());
        std::memcpy(parameters.second_moments.data(), second->data.data(), second->data.size());
        std::memcpy(parameters.ema.data(), ema->data.data(), ema->data.size());
        parameter_buffer.upload(parameters);
        const auto training = std::ranges::find(file.tensors, std::string{"training.state"}, &serialization::safetensors::Tensor::name);
        std::array<std::uint64_t, 4u> training_state{};
        std::memcpy(training_state.data(), training->data.data(), training->data.size());
        state                         = {.step = training_state[0], .processed_samples = training_state[1], .seed = training_state[2], .elapsed_seconds = std::bit_cast<double>(training_state[3])};
        const std::uint64_t next_step = state.step + 1u;
        ::cuda::copy_bytes(stream, ::cuda::std::span<const std::uint64_t>{&next_step, 1uz}, device_step);
        ::cuda::copy_bytes(stream, ::cuda::std::span<const std::uint64_t>{&state.processed_samples, 1uz}, device_processed_samples);
        ::cuda::copy_bytes(stream, ::cuda::std::span<const std::uint64_t>{&state.seed, 1uz}, device_seed);
        stream.sync();
    }
    void Trainer::training_step() {
        kernels::make_training_batch(stream, dataset_images.data(), dataset_labels.data(), path.data(), target.data(), times.data(), labels.data(), device_step.data(), device_seed.data(), batch, {model.configuration.image.width, model.configuration.image.height, model.configuration.image.channels, model.configuration.patch_size}, image_count, static_cast<std::uint32_t>(model.configuration.image.classes.size()), horizontal_flip);
        model.forward(parameter_buffer.parameters.data(), path.data(), times.data(), labels.data(), model_workspace.data(), model_workspace_layout);
        model.loss(target.data(), loss.data(), model_workspace.data(), model_workspace_layout);
        model.backward(parameter_buffer.parameters.data(), parameter_buffer.gradients.data(), path.data(), times.data(), labels.data(), patch_gradient.data(), model_workspace.data(), model_workspace_layout);
        parameter_buffer.step(training_configuration, device_step.data(), device_processed_samples.data(), batch);
        kernels::add_loss(stream, loss.data(), loss_sum.data());
        kernels::advance_training_state(stream, device_step.data(), device_processed_samples.data(), batch);
    }
} // namespace flowdit
