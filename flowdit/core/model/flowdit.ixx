module;
#include <flowdit/cuda.h>
export module flowdit.model;
import std;
export import flowdit.model.configuration;
import flowdit.neural.matmul;
import flowdit.neural.transformer;
export namespace flowdit {
    inline constexpr neural::MatmulRuntimeConfiguration flow_matmul_runtime_configuration{
        .workspace_byte_count   = 64uz * 1024uz * 1024uz,
        .tuning_byte_count      = 256uz * 1024uz * 1024uz,
        .tuning_bias_byte_count = 64uz * 1024uz,
    };
    struct FlowDiTParameterLayout final {
        std::size_t patch_weight;
        std::size_t patch_bias;
        std::size_t class_embedding;
        std::size_t time_input_weight;
        std::size_t time_input_bias;
        std::size_t time_output_weight;
        std::size_t time_output_bias;
        std::size_t transformer;
        std::size_t final_modulation_weight;
        std::size_t final_modulation_bias;
        std::size_t velocity_weight;
        std::size_t velocity_bias;
        std::size_t parameter_count;
        explicit FlowDiTParameterLayout(const ModelConfiguration& configuration);
    };
    struct FlowDiTWorkspaceLayout final {
        std::uint32_t batch;
        std::size_t tokens;
        std::size_t time_embedding;
        std::size_t time_preactivation;
        std::size_t time_hidden;
        std::size_t condition;
        std::size_t condition_activated;
        std::size_t transformed;
        std::size_t final_modulation;
        std::size_t final_normalized;
        std::size_t final_means;
        std::size_t final_inverse_standard_deviations;
        std::size_t velocity;
        std::size_t velocity_gradient;
        std::size_t normalized_gradient;
        std::size_t transformed_gradient;
        std::size_t final_modulation_gradient;
        std::size_t condition_gradient;
        std::size_t time_hidden_gradient;
        std::size_t time_preactivation_gradient;
        std::size_t transformer_workspace;
        neural::TransformerWorkspaceLayout transformer_layout;
        std::size_t byte_count;
        FlowDiTWorkspaceLayout(std::uint32_t batch, const ModelConfiguration& configuration);
    };
    struct FlowDiT final {
        ModelConfiguration configuration;
        neural::TransformerConfiguration shape;
        std::uint32_t patch_width;
        ::cuda::stream_ref stream;
        neural::MatmulRuntime& matmul;
        FlowDiTParameterLayout parameters;
        neural::Transformer transformer;
        ::cuda::device_buffer<float> position;
        FlowDiT(::cuda::stream_ref stream, neural::MatmulRuntime& matmul, const ModelConfiguration& configuration);
        std::vector<float> initialize_parameters(std::uint64_t seed) const;
        void forward(const float* parameter_values, const float* patches, const float* times, const std::uint32_t* labels, std::uint8_t* workspace, const FlowDiTWorkspaceLayout& workspace_layout);
        void backward(const float* parameter_values, float* parameter_gradients, const float* patches, const float* times, const std::uint32_t* labels, float* patch_gradient, std::uint8_t* workspace, const FlowDiTWorkspaceLayout& workspace_layout);
    };
} // namespace flowdit
