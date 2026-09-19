module;
#include "kernels.h"
#include <cuda_runtime_api.h>
#include <flowdit/cuda.h>
#include <nlohmann/json.hpp>
module flowdit.training.trainer;
import std;
import flowdit.model;
import flowdit.neural.matmul;
import flowdit.neural.training_state;
import flowdit.serialization.safetensors;
namespace flowdit {
    Trainer::Trainer(const ::cuda::stream_ref source_stream, const ModelConfiguration& configuration, const std::uint32_t source_batch, const std::uint64_t seed, const neural::TrainingConfiguration& optimizer, const std::uint32_t source_accumulation)
        : batch{source_batch}, accumulation{source_accumulation}, state{.seed = seed}, value_count{static_cast<std::size_t>(batch) * configuration.shape.width * configuration.shape.height * configuration.shape.channels}, stream{source_stream}, input_values{stream, ::cuda::device_default_memory_pool(stream.device()), value_count, ::cuda::no_init}, input_labels{stream, ::cuda::device_default_memory_pool(stream.device()), batch, ::cuda::no_init}, matmul{stream, flow_matmul_runtime_configuration}, model{stream, matmul, configuration}, parameter_buffer{stream, model.parameters.parameter_count}, training_configuration{optimizer}, model_workspace_layout{batch, model.configuration}, model_workspace{stream, ::cuda::device_default_memory_pool(stream.device()), model_workspace_layout.byte_count, ::cuda::no_init}, path{stream, ::cuda::device_default_memory_pool(stream.device()), value_count, ::cuda::no_init}, target{stream, ::cuda::device_default_memory_pool(stream.device()), value_count, ::cuda::no_init},
          times{stream, ::cuda::device_default_memory_pool(stream.device()), batch, ::cuda::no_init}, labels{stream, ::cuda::device_default_memory_pool(stream.device()), batch, ::cuda::no_init}, loss{stream, ::cuda::device_default_memory_pool(stream.device()), 1uz, ::cuda::no_init}, sample_loss{stream, ::cuda::device_default_memory_pool(stream.device()), batch, ::cuda::no_init}, device_step{stream, ::cuda::device_default_memory_pool(stream.device()), 1uz, ::cuda::no_init}, device_processed_samples{stream, ::cuda::device_default_memory_pool(stream.device()), 1uz, ::cuda::no_init}, device_seed{stream, ::cuda::device_default_memory_pool(stream.device()), 1uz, ::cuda::no_init}, device_microstep{stream, ::cuda::device_default_memory_pool(stream.device()), 1uz, ::cuda::no_init} {
        training_identity = nlohmann::json{{"microbatch", batch}, {"accumulation", accumulation}, {"learning_rate", optimizer.learning_rate}, {"betas", {optimizer.first_decay, optimizer.second_decay}}, {"epsilon", optimizer.epsilon}, {"weight_decay", optimizer.weight_decay}, {"warmup_steps", optimizer.warmup_steps}, {"ema_half_life", optimizer.exponential_average.half_life_samples}, {"ema_ramp", optimizer.exponential_average.ramp_up_ratio}, {"rng", "philox-sample-v1"}, {"class_dropout", .1f}}.dump();
        ::cuda::fill_bytes(stream, input_values, 0);
        ::cuda::fill_bytes(stream, input_labels, 0);
        parameter_buffer.initialize(model.initialize_parameters(seed));
        const std::uint64_t initial_step = 1u;
        ::cuda::fill_bytes(stream, device_microstep, 0);
        ::cuda::copy_bytes(stream, ::cuda::std::span<const std::uint64_t>{&initial_step, 1uz}, device_step);
        ::cuda::copy_bytes(stream, ::cuda::std::span<const std::uint64_t>{&state.processed_samples, 1uz}, device_processed_samples);
        ::cuda::copy_bytes(stream, ::cuda::std::span<const std::uint64_t>{&state.seed, 1uz}, device_seed);
        kernels::make_training_batch(stream, input_values.data(), input_labels.data(), path.data(), target.data(), times.data(), labels.data(), device_microstep.data(), device_seed.data(), batch, {model.configuration.shape.width, model.configuration.shape.height, model.configuration.shape.channels, model.configuration.patch_size}, model.configuration.class_count);
        model.forward(parameter_buffer.parameters.data(), path.data(), times.data(), labels.data(), model_workspace.data(), model_workspace_layout);
        kernels::flow_matching_loss(stream, reinterpret_cast<const float*>(model_workspace.data() + model_workspace_layout.velocity), target.data(), reinterpret_cast<float*>(model_workspace.data() + model_workspace_layout.velocity_gradient), sample_loss.data(), loss.data(), batch, static_cast<std::uint32_t>(value_count / batch), 1.f / accumulation);
        model.backward(parameter_buffer.parameters.data(), parameter_buffer.gradients.data(), labels.data(), model_workspace.data(), model_workspace_layout);
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
    float Trainer::optimize(const TensorBatch& input) {
        float total{};
        for (std::uint32_t micro = 0; micro < accumulation; ++micro) {
            ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{input.values + micro * value_count, value_count}, input_values);
            ::cuda::copy_bytes(stream, ::cuda::std::span<const std::uint32_t>{input.labels + micro * batch, batch}, input_labels);
            if (const cudaError_t status = cudaGraphLaunch(graph_execution, stream.get()); status != cudaSuccess) throw std::runtime_error{std::format("CUDA graph launch: {}", cudaGetErrorString(status))};
            float value{};
            ::cuda::copy_bytes(stream, loss, ::cuda::std::span<float>{&value, 1});
            stream.sync();
            total += value;
        }
        parameter_buffer.step(training_configuration, device_step.data(), device_processed_samples.data(), batch * accumulation);
        kernels::advance_training_state(stream, device_step.data(), device_processed_samples.data(), batch * accumulation);
        stream.sync();
        ++state.step;
        state.processed_samples += batch * accumulation;
        return total / accumulation;
    }
    void Trainer::save(const std::filesystem::path& path, std::map<std::string, std::string> metadata) const {
        const neural::ParameterState parameters = parameter_buffer.download();
        const std::array<std::uint64_t, 4u> training_state{state.step, state.processed_samples, state.seed, std::bit_cast<std::uint64_t>(state.elapsed_seconds)};
        const std::array<serialization::safetensors::TensorView, 5u> tensors{
            serialization::safetensors::TensorView{"model.parameters", "F32", {parameters.parameters.size()}, parameters.parameters.data(), parameters.parameters.size() * sizeof(float)},
            serialization::safetensors::TensorView{"optimizer.first_moments", "F32", {parameters.first_moments.size()}, parameters.first_moments.data(), parameters.first_moments.size() * sizeof(float)},
            serialization::safetensors::TensorView{"optimizer.second_moments", "F32", {parameters.second_moments.size()}, parameters.second_moments.data(), parameters.second_moments.size() * sizeof(float)},
            serialization::safetensors::TensorView{"model.ema", "F32", {parameters.ema.size()}, parameters.ema.data(), parameters.ema.size() * sizeof(float)},
            serialization::safetensors::TensorView{"training.state", "U64", {training_state.size()}, training_state.data(), training_state.size() * sizeof(std::uint64_t)},
        };
        metadata["flowdit.model"]    = serialize_model(model.configuration);
        metadata["flowdit.training"] = training_identity;
        serialization::safetensors::write(path, "usit-dcae-v1", tensors, metadata);
    }
    void Trainer::load(const std::filesystem::path& path) {
        const serialization::safetensors::File file = serialization::safetensors::read(path);
        if (file.metadata.at("flowdit.system") != "usit-dcae-v1" || file.metadata.at("flowdit.model") != serialize_model(model.configuration) || file.metadata.at("flowdit.training") != training_identity) throw std::runtime_error{"Checkpoint model or training configuration does not match this run"};
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
        const std::uint64_t next_step = state.step + 1u, next_microstep = state.step * accumulation;
        ::cuda::copy_bytes(stream, ::cuda::std::span<const std::uint64_t>{&next_microstep, 1uz}, device_microstep);
        ::cuda::copy_bytes(stream, ::cuda::std::span<const std::uint64_t>{&next_step, 1uz}, device_step);
        ::cuda::copy_bytes(stream, ::cuda::std::span<const std::uint64_t>{&state.processed_samples, 1uz}, device_processed_samples);
        ::cuda::copy_bytes(stream, ::cuda::std::span<const std::uint64_t>{&state.seed, 1uz}, device_seed);
        stream.sync();
    }
    void Trainer::training_step() {
        kernels::make_training_batch(stream, input_values.data(), input_labels.data(), path.data(), target.data(), times.data(), labels.data(), device_microstep.data(), device_seed.data(), batch, {model.configuration.shape.width, model.configuration.shape.height, model.configuration.shape.channels, model.configuration.patch_size}, model.configuration.class_count);
        model.forward(parameter_buffer.parameters.data(), path.data(), times.data(), labels.data(), model_workspace.data(), model_workspace_layout);
        kernels::flow_matching_loss(stream, reinterpret_cast<const float*>(model_workspace.data() + model_workspace_layout.velocity), target.data(), reinterpret_cast<float*>(model_workspace.data() + model_workspace_layout.velocity_gradient), sample_loss.data(), loss.data(), batch, static_cast<std::uint32_t>(value_count / batch), 1.f / accumulation);
        model.backward(parameter_buffer.parameters.data(), parameter_buffer.gradients.data(), labels.data(), model_workspace.data(), model_workspace_layout);
        kernels::advance_training_state(stream, device_microstep.data(), device_processed_samples.data(), 0);
    }
} // namespace flowdit
