module;
#include "kernels.h"
#include <flowdit/cuda.h>
module flowdit.model;
import std;
import flowdit.neural.matmul;
import flowdit.neural.transformer;
namespace flowdit {
    namespace {
        neural::TransformerConfiguration make_transformer_configuration(const ModelConfiguration& configuration) {
            return {.sequence = configuration.shape.width / configuration.patch_size * (configuration.shape.height / configuration.patch_size), .width = 256u, .block_count = 8u, .head_count = 8u, .mlp_width = 1024u};
        }
        std::size_t align_workspace(const std::size_t value) {
            return (value + 255uz) & ~255uz;
        }
        template <class Type>
        Type* workspace_pointer(std::uint8_t* const workspace, const std::size_t offset) {
            return reinterpret_cast<Type*>(workspace + offset);
        }
        void initialize_xavier(std::vector<float>& values, const std::size_t offset, const std::size_t count, const std::uint32_t input_width, const std::uint32_t output_width, std::mt19937_64& generator) {
            const float extent = std::sqrt(6.0F / static_cast<float>(input_width + output_width));
            std::uniform_real_distribution<float> distribution{-extent, extent};
            for (std::size_t index = 0uz; index < count; ++index) values[offset + index] = distribution(generator);
        }
    } // namespace
    FlowDiTParameterLayout::FlowDiTParameterLayout(const ModelConfiguration& configuration) {
        const auto shape              = make_transformer_configuration(configuration);
        const std::size_t patch_width = configuration.patch_size * configuration.patch_size * configuration.shape.channels;
        const std::size_t width       = shape.width;
        std::size_t offset{};
        patch_weight = offset;
        offset += patch_width * width;
        patch_bias = offset;
        offset += width;
        class_embedding = offset;
        offset += (configuration.class_count + 1uz) * width;
        time_input_weight = offset;
        offset += width * width;
        time_input_bias = offset;
        offset += width;
        time_output_weight = offset;
        offset += width * width;
        time_output_bias = offset;
        offset += width;
        transformer = offset;
        offset += neural::TransformerParameterLayout{shape}.parameter_count;
        final_modulation_weight = offset;
        offset += width * 2uz * width;
        final_modulation_bias = offset;
        offset += 2uz * width;
        velocity_weight = offset;
        offset += width * patch_width;
        velocity_bias   = offset;
        parameter_count = offset + patch_width;
    }
    FlowDiTWorkspaceLayout::FlowDiTWorkspaceLayout(const std::uint32_t source_batch, const ModelConfiguration& configuration) : batch{source_batch}, transformer_layout{make_transformer_configuration(configuration), batch} {
        const auto shape              = make_transformer_configuration(configuration);
        const std::size_t patch_width = configuration.patch_size * configuration.patch_size * configuration.shape.channels;
        const std::size_t sequence    = shape.sequence;
        const std::size_t width       = shape.width;
        const std::size_t token_count = static_cast<std::size_t>(batch) * sequence;
        std::size_t offset{};
        tokens                            = offset;
        offset                            = align_workspace(offset + token_count * width * sizeof(float));
        time_embedding                    = offset;
        offset                            = align_workspace(offset + static_cast<std::size_t>(batch) * width * sizeof(float));
        time_preactivation                = offset;
        offset                            = align_workspace(offset + static_cast<std::size_t>(batch) * width * sizeof(float));
        time_hidden                       = offset;
        offset                            = align_workspace(offset + static_cast<std::size_t>(batch) * width * sizeof(float));
        condition                         = offset;
        offset                            = align_workspace(offset + static_cast<std::size_t>(batch) * width * sizeof(float));
        condition_activated               = offset;
        offset                            = align_workspace(offset + static_cast<std::size_t>(batch) * width * sizeof(float));
        transformed                       = offset;
        offset                            = align_workspace(offset + token_count * width * sizeof(float));
        final_modulation                  = offset;
        offset                            = align_workspace(offset + static_cast<std::size_t>(batch) * 2uz * width * sizeof(float));
        final_normalized                  = offset;
        offset                            = align_workspace(offset + token_count * width * sizeof(float));
        final_means                       = offset;
        offset                            = align_workspace(offset + token_count * sizeof(float));
        final_inverse_standard_deviations = offset;
        offset                            = align_workspace(offset + token_count * sizeof(float));
        velocity                          = offset;
        offset                            = align_workspace(offset + token_count * patch_width * sizeof(float));
        velocity_gradient                 = offset;
        offset                            = align_workspace(offset + token_count * patch_width * sizeof(float));
        normalized_gradient               = offset;
        offset                            = align_workspace(offset + token_count * width * sizeof(float));
        transformed_gradient              = offset;
        offset                            = align_workspace(offset + token_count * width * sizeof(float));
        final_modulation_gradient         = offset;
        offset                            = align_workspace(offset + static_cast<std::size_t>(batch) * 2uz * width * sizeof(float));
        condition_gradient                = offset;
        offset                            = align_workspace(offset + static_cast<std::size_t>(batch) * width * sizeof(float));
        time_hidden_gradient              = offset;
        offset                            = align_workspace(offset + static_cast<std::size_t>(batch) * width * sizeof(float));
        time_preactivation_gradient       = offset;
        offset                            = align_workspace(offset + static_cast<std::size_t>(batch) * width * sizeof(float));
        transformer_workspace             = offset;
        byte_count                        = align_workspace(offset + transformer_layout.byte_count);
    }
    FlowDiT::FlowDiT(const ::cuda::stream_ref source_stream, neural::MatmulRuntime& source_matmul, const ModelConfiguration& source_configuration) : configuration{source_configuration}, shape{make_transformer_configuration(configuration)}, patch_width{configuration.patch_size * configuration.patch_size * configuration.shape.channels}, stream{source_stream}, matmul{source_matmul}, parameters{configuration}, transformer{matmul, shape}, position{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(shape.sequence) * shape.width, ::cuda::no_init} {
        std::vector<float> host_position(position.size());
        const std::uint32_t frequency_count = shape.width / 4u;
        for (std::uint32_t y = 0u; y < configuration.shape.height / configuration.patch_size; ++y)
            for (std::uint32_t x = 0u; x < configuration.shape.width / configuration.patch_size; ++x)
                for (std::uint32_t frequency = 0u; frequency < frequency_count; ++frequency) {
                    const float scale                                         = std::exp(-std::log(10'000.0F) * static_cast<float>(frequency) / static_cast<float>(frequency_count));
                    const std::size_t offset                                  = static_cast<std::size_t>(y * (configuration.shape.width / configuration.patch_size) + x) * shape.width;
                    host_position[offset + frequency]                         = std::sin(static_cast<float>(y) * scale);
                    host_position[offset + frequency_count + frequency]       = std::cos(static_cast<float>(y) * scale);
                    host_position[offset + 2uz * frequency_count + frequency] = std::sin(static_cast<float>(x) * scale);
                    host_position[offset + 3uz * frequency_count + frequency] = std::cos(static_cast<float>(x) * scale);
                }
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{host_position.data(), host_position.size()}, position);
        stream.sync();
    }
    std::vector<float> FlowDiT::initialize_parameters(const std::uint64_t seed) const {
        std::vector<float> values(parameters.parameter_count);
        std::mt19937_64 generator{seed};
        initialize_xavier(values, parameters.patch_weight, static_cast<std::size_t>(patch_width) * shape.width, patch_width, shape.width, generator);
        std::normal_distribution<float> small_normal{0.0F, 0.02F};
        for (std::size_t index = parameters.class_embedding; index < parameters.class_embedding + (configuration.class_count + 1uz) * shape.width; ++index) values[index] = small_normal(generator);
        for (std::size_t index = parameters.time_input_weight; index < parameters.time_input_weight + static_cast<std::size_t>(shape.width) * shape.width; ++index) values[index] = small_normal(generator);
        for (std::size_t index = parameters.time_output_weight; index < parameters.time_output_weight + static_cast<std::size_t>(shape.width) * shape.width; ++index) values[index] = small_normal(generator);
        const std::vector<float> transformer_values = transformer.initialize_parameters(seed + 1u);
        std::ranges::copy(transformer_values, values.begin() + static_cast<std::ptrdiff_t>(parameters.transformer));
        return values;
    }
    void FlowDiT::forward(const float* const parameter_values, const float* const patches, const float* const times, const std::uint32_t* const labels, std::uint8_t* const workspace, const FlowDiTWorkspaceLayout& layout) {
        const std::uint32_t token_count = layout.batch * shape.sequence;
        float* tokens                   = workspace_pointer<float>(workspace, layout.tokens);
        float* time_embedding           = workspace_pointer<float>(workspace, layout.time_embedding);
        float* time_preactivation       = workspace_pointer<float>(workspace, layout.time_preactivation);
        float* time_hidden              = workspace_pointer<float>(workspace, layout.time_hidden);
        float* condition                = workspace_pointer<float>(workspace, layout.condition);
        float* condition_activated      = workspace_pointer<float>(workspace, layout.condition_activated);
        float* transformed              = workspace_pointer<float>(workspace, layout.transformed);
        float* final_modulation         = workspace_pointer<float>(workspace, layout.final_modulation);
        float* final_normalized         = workspace_pointer<float>(workspace, layout.final_normalized);
        float* velocity                 = workspace_pointer<float>(workspace, layout.velocity);
        matmul.execute({patches, parameter_values + parameters.patch_weight, tokens, token_count, shape.width, patch_width, false, false, neural::MatmulEpilogue::bias, parameter_values + parameters.patch_bias});
        kernels::add_position(stream, tokens, position.data(), static_cast<std::size_t>(token_count) * shape.width, position.size());
        kernels::make_time_embedding(stream, times, time_embedding, layout.batch, shape.width);
        matmul.execute({time_embedding, parameter_values + parameters.time_input_weight, time_preactivation, layout.batch, shape.width, shape.width, false, false, neural::MatmulEpilogue::bias, parameter_values + parameters.time_input_bias});
        kernels::silu_forward(stream, time_preactivation, time_hidden, static_cast<std::size_t>(layout.batch) * shape.width);
        matmul.execute({time_hidden, parameter_values + parameters.time_output_weight, condition, layout.batch, shape.width, shape.width, false, false, neural::MatmulEpilogue::bias, parameter_values + parameters.time_output_bias});
        kernels::make_condition(stream, condition, parameter_values + parameters.class_embedding, labels, layout.batch, shape.width);
        kernels::silu_forward(stream, condition, condition_activated, static_cast<std::size_t>(layout.batch) * shape.width);
        transformer.forward(parameter_values + parameters.transformer, tokens, condition_activated, transformed, workspace + layout.transformer_workspace, layout.transformer_layout);
        matmul.execute({condition_activated, parameter_values + parameters.final_modulation_weight, final_modulation, layout.batch, 2u * shape.width, shape.width, false, false, neural::MatmulEpilogue::bias, parameter_values + parameters.final_modulation_bias});
        kernels::final_adaln_forward(stream, transformed, final_modulation, final_normalized, workspace_pointer<float>(workspace, layout.final_means), workspace_pointer<float>(workspace, layout.final_inverse_standard_deviations), layout.batch, shape.sequence, shape.width);
        matmul.execute({final_normalized, parameter_values + parameters.velocity_weight, velocity, token_count, patch_width, shape.width, false, false, neural::MatmulEpilogue::bias, parameter_values + parameters.velocity_bias});
    }
    void FlowDiT::backward(const float* const parameter_values, float* const parameter_gradients, const float* const patches, const float*, const std::uint32_t* const labels, float* const input_patch_gradient, std::uint8_t* const workspace, const FlowDiTWorkspaceLayout& layout) {
        const std::uint32_t token_count    = layout.batch * shape.sequence;
        const float* tokens                = workspace_pointer<float>(workspace, layout.tokens);
        const float* time_embedding        = workspace_pointer<float>(workspace, layout.time_embedding);
        const float* time_preactivation    = workspace_pointer<float>(workspace, layout.time_preactivation);
        const float* time_hidden           = workspace_pointer<float>(workspace, layout.time_hidden);
        const float* condition             = workspace_pointer<float>(workspace, layout.condition);
        const float* condition_activated   = workspace_pointer<float>(workspace, layout.condition_activated);
        const float* transformed           = workspace_pointer<float>(workspace, layout.transformed);
        const float* final_modulation      = workspace_pointer<float>(workspace, layout.final_modulation);
        const float* final_normalized      = workspace_pointer<float>(workspace, layout.final_normalized);
        const float* velocity_gradient     = workspace_pointer<float>(workspace, layout.velocity_gradient);
        float* normalized_gradient         = workspace_pointer<float>(workspace, layout.normalized_gradient);
        float* transformed_gradient        = workspace_pointer<float>(workspace, layout.transformed_gradient);
        float* final_modulation_gradient   = workspace_pointer<float>(workspace, layout.final_modulation_gradient);
        float* condition_gradient          = workspace_pointer<float>(workspace, layout.condition_gradient);
        float* time_hidden_gradient        = workspace_pointer<float>(workspace, layout.time_hidden_gradient);
        float* time_preactivation_gradient = workspace_pointer<float>(workspace, layout.time_preactivation_gradient);
        ::cuda::fill_bytes(stream, ::cuda::std::span<float>{condition_gradient, static_cast<std::size_t>(layout.batch) * shape.width}, 0u);
        matmul.execute({final_normalized, velocity_gradient, parameter_gradients + parameters.velocity_weight, shape.width, patch_width, token_count, true, false, neural::MatmulEpilogue::bias_gradient, parameter_gradients + parameters.velocity_bias});
        matmul.execute({velocity_gradient, parameter_values + parameters.velocity_weight, normalized_gradient, token_count, shape.width, patch_width, false, true});
        kernels::final_adaln_backward(stream, transformed, final_modulation, normalized_gradient, workspace_pointer<float>(workspace, layout.final_means), workspace_pointer<float>(workspace, layout.final_inverse_standard_deviations), transformed_gradient, final_modulation_gradient, layout.batch, shape.sequence, shape.width);
        matmul.execute({condition_activated, final_modulation_gradient, parameter_gradients + parameters.final_modulation_weight, shape.width, 2u * shape.width, layout.batch, true, false, neural::MatmulEpilogue::bias_gradient, parameter_gradients + parameters.final_modulation_bias});
        matmul.execute({final_modulation_gradient, parameter_values + parameters.final_modulation_weight, condition_gradient, layout.batch, shape.width, 2u * shape.width, false, true});
        transformer.backward(parameter_values + parameters.transformer, parameter_gradients + parameters.transformer, tokens, condition_activated, transformed_gradient, normalized_gradient, condition_gradient, workspace + layout.transformer_workspace, layout.transformer_layout);
        kernels::silu_backward(stream, condition, condition_gradient, time_preactivation_gradient, static_cast<std::size_t>(layout.batch) * shape.width);
        kernels::class_embedding_backward(stream, time_preactivation_gradient, labels, parameter_gradients + parameters.class_embedding, layout.batch, shape.width, configuration.class_count);
        matmul.execute({time_hidden, time_preactivation_gradient, parameter_gradients + parameters.time_output_weight, shape.width, shape.width, layout.batch, true, false, neural::MatmulEpilogue::bias_gradient, parameter_gradients + parameters.time_output_bias});
        matmul.execute({time_preactivation_gradient, parameter_values + parameters.time_output_weight, time_hidden_gradient, layout.batch, shape.width, shape.width, false, true});
        kernels::silu_backward(stream, time_preactivation, time_hidden_gradient, condition_gradient, static_cast<std::size_t>(layout.batch) * shape.width);
        matmul.execute({time_embedding, condition_gradient, parameter_gradients + parameters.time_input_weight, shape.width, shape.width, layout.batch, true, false, neural::MatmulEpilogue::bias_gradient, parameter_gradients + parameters.time_input_bias});
        matmul.execute({patches, normalized_gradient, parameter_gradients + parameters.patch_weight, patch_width, shape.width, token_count, true, false, neural::MatmulEpilogue::bias_gradient, parameter_gradients + parameters.patch_bias});
        matmul.execute({normalized_gradient, parameter_values + parameters.patch_weight, input_patch_gradient, token_count, patch_width, shape.width, false, true});
    }
} // namespace flowdit
