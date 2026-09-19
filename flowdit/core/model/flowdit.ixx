module;
#include <flowdit/cuda.h>
export module flowdit.model;
import std;
export import flowdit.model.configuration;
import flowdit.neural.matmul;
import flowdit.neural.transformer;
export namespace flowdit {
    inline constexpr neural::MatmulRuntimeConfiguration flow_matmul_runtime_configuration{64uz * 1024 * 1024};
    struct FlowDiTParameterLayout final {
        std::size_t patch_weight, patch_bias, class_embedding, position, transformer, final_weight, final_bias, velocity_weight, velocity_bias, parameter_count;
        explicit FlowDiTParameterLayout(const ModelConfiguration& configuration);
    };
    struct FlowDiTWorkspaceLayout final {
        std::uint32_t batch;
        bool training;
        std::size_t weights, patches, embedded, normalized, means, inverse, full_velocity, velocity, velocity_gradient, full_gradient, norm_gradient, patch_gradient, transformer_workspace, byte_count{};
        neural::TransformerWorkspaceLayout transformer_layout;
        FlowDiTWorkspaceLayout(std::uint32_t batch, const ModelConfiguration& configuration, bool training = true);
    };
    struct FlowDiT final {
        ModelConfiguration configuration;
        neural::TransformerConfiguration shape;
        std::uint32_t patch_width;
        ::cuda::stream_ref stream;
        neural::MatmulRuntime& matmul;
        FlowDiTParameterLayout parameters;
        neural::Transformer transformer;
        FlowDiT(::cuda::stream_ref stream, neural::MatmulRuntime& matmul, const ModelConfiguration& configuration);
        std::vector<float> initialize_parameters(std::uint64_t seed) const;
        void forward(const float* parameters, const float* patches, const float* times, const std::uint32_t* labels, std::uint8_t* workspace, const FlowDiTWorkspaceLayout& layout);
        void backward(const float* parameters, float* gradient, const std::uint32_t* labels, std::uint8_t* workspace, const FlowDiTWorkspaceLayout& layout);
    };
} // namespace flowdit
