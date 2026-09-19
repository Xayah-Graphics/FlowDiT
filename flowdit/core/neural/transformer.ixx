module;
#include <flowdit/cuda.h>
export module flowdit.neural.transformer;
import std;
import flowdit.neural.matmul;
import flowdit.neural.attention;
export namespace flowdit::neural {
    struct TransformerConfiguration final {
        std::uint32_t sequence, width, heads, side_blocks, mlp_width;
    };
    struct TransformerBlockLayout final {
        std::size_t norm1_weight, norm1_bias, qkv, projection, projection_bias, norm2_weight, norm2_bias, mlp1, mlp1_bias, mlp2, mlp2_bias, skip, skip_bias;
    };
    struct TransformerParameterLayout final {
        std::vector<TransformerBlockLayout> blocks;
        std::size_t count{};
        explicit TransformerParameterLayout(TransformerConfiguration configuration);
    };
    struct TransformerWorkspaceLayout final {
        std::uint32_t batch;
        bool training;
        std::vector<std::size_t> states;
        std::size_t concatenated, projected, norm1, qkv, attended, attention_residual, norm2, mlp, activated, temporary;
        std::size_t mean1, inverse1, mean2, inverse2, statistics;
        std::size_t gradient, scratch_gradient, skip_gradients, d1, d2, dqkv, dmlp;
        std::size_t byte_count{};
        TransformerWorkspaceLayout(std::uint32_t batch, TransformerConfiguration configuration, bool training);
    };
    struct Transformer final {
        ::cuda::stream_ref stream;
        MatmulRuntime& matmul;
        TransformerConfiguration configuration;
        TransformerParameterLayout parameters;
        std::list<Attention> attention;
        Transformer(::cuda::stream_ref stream, MatmulRuntime& matmul, TransformerConfiguration configuration);
        void initialize(std::span<float> values, std::mt19937_64& random) const;
        void forward(const float* master, const std::uint16_t* weights, std::uint8_t* workspace, const TransformerWorkspaceLayout& layout);
        void backward(const float* master, const std::uint16_t* weights, float* gradient, std::uint8_t* workspace, const TransformerWorkspaceLayout& layout);

    private:
        const std::uint16_t* block_forward(std::uint32_t index, const float* master, const std::uint16_t* weights, std::uint16_t* output, std::uint8_t* workspace, const TransformerWorkspaceLayout& layout, Attention& attention);
        void linear_backward(const std::uint16_t* input, const std::uint16_t* gradient, std::uint16_t* input_gradient, const std::uint16_t* weight, float* weight_gradient, float* bias_gradient, std::uint32_t rows, std::uint32_t in, std::uint32_t out);
    };
} // namespace flowdit::neural
