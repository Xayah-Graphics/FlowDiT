module;
#include "transformer-kernels.h"
#include <flowdit/cuda.h>
module flowdit.neural.transformer;
import std;
namespace flowdit::neural {
    TransformerParameterLayout::TransformerParameterLayout(TransformerConfiguration c) {
        const std::size_t d = c.width, m = c.mlp_width;
        const auto take = [&](std::size_t n) {
            const auto offset = count;
            count += n;
            return offset;
        };
        for (std::uint32_t i = 0; i < 2 * c.side_blocks + 1; ++i) {
            TransformerBlockLayout b{};
            b.norm1_weight    = take(d);
            b.norm1_bias      = take(d);
            b.qkv             = take(3 * d * d);
            b.projection      = take(d * d);
            b.projection_bias = take(d);
            b.norm2_weight    = take(d);
            b.norm2_bias      = take(d);
            b.mlp1            = take(m * d);
            b.mlp1_bias       = take(m);
            b.mlp2            = take(d * m);
            b.mlp2_bias       = take(d);
            if (i > c.side_blocks) {
                b.skip      = take(2 * d * d);
                b.skip_bias = take(d);
            }
            blocks.push_back(b);
        }
    }
    TransformerWorkspaceLayout::TransformerWorkspaceLayout(std::uint32_t b, TransformerConfiguration c, bool train) : batch{b}, training{train} {
        const std::size_t rows = batch * c.sequence, n = rows * c.width;
        const auto take = [&](std::size_t bytes) {
            const auto offset = byte_count;
            byte_count += (bytes + 255) / 256 * 256;
            return offset;
        };
        for (std::uint32_t i = 0; i <= 2 * c.side_blocks + 1; ++i) {
            if (training || i <= c.side_blocks + 2) states.push_back(take(n * 2));
            else states.push_back(states[c.side_blocks + 1 + (i - c.side_blocks - 1) % 2]);
        }
        concatenated       = take(n * 4);
        projected          = take(n * 2);
        norm1              = take(n * 2);
        qkv                = take(n * 6);
        attended           = take(n * 2);
        attention_residual = take(n * 2);
        norm2              = take(n * 2);
        mlp                = take(rows * c.mlp_width * 2);
        activated          = take(rows * c.mlp_width * 2);
        temporary          = take(n * 2);
        mean1              = take(rows * 4);
        inverse1           = take(rows * 4);
        mean2              = take(rows * 4);
        inverse2           = take(rows * 4);
        statistics         = take(training ? rows * c.heads * 4 : 0);
        if (training) {
            gradient         = take(n * 2);
            scratch_gradient = take(n * 2);
            skip_gradients   = take(c.side_blocks * n * 2);
            d1               = take(n * 2);
            d2               = take(n * 2);
            dqkv             = take(n * 6);
            dmlp             = take(rows * c.mlp_width * 2);
        }
    }
    Transformer::Transformer(::cuda::stream_ref s, MatmulRuntime& m, TransformerConfiguration c) : stream{s}, matmul{m}, configuration{c}, parameters{c} {}
    void Transformer::initialize(std::span<float> values, std::mt19937_64& random) const {
        std::normal_distribution<float> normal{0.f, .02f};
        const auto matrix = [&](std::size_t at, std::size_t n) {
            for (std::size_t i = 0; i < n; ++i) {
                float x;
                do {
                    x = normal(random);
                } while (std::abs(x) > 2.f);
                values[at + i] = x;
            }
        };
        const auto d = configuration.width, m = configuration.mlp_width;
        for (const auto& b : parameters.blocks) {
            std::fill_n(values.data() + b.norm1_weight, d, 1.f);
            std::fill_n(values.data() + b.norm2_weight, d, 1.f);
            matrix(b.qkv, 3uz * d * d);
            matrix(b.projection, static_cast<std::size_t>(d) * d);
            matrix(b.mlp1, static_cast<std::size_t>(d) * m);
            matrix(b.mlp2, static_cast<std::size_t>(d) * m);
            if (b.skip) matrix(b.skip, 2uz * d * d);
        }
    }
    void Transformer::forward(const float* master, const std::uint16_t* weights, std::uint8_t* workspace, const TransformerWorkspaceLayout& l) {
        auto plan = std::ranges::find_if(attention, [&](const auto& p) { return p.batch == l.batch && p.training == l.training; });
        if (plan == attention.end()) plan = attention.emplace(attention.end(), stream, l.batch, configuration.sequence, configuration.width, configuration.heads, l.training);
        for (std::uint32_t i = 0; i < parameters.blocks.size(); ++i) block_forward(i, master, weights, reinterpret_cast<std::uint16_t*>(workspace + l.states[i + 1]), workspace, l, *plan);
    }
    void Transformer::backward(const float* master, const std::uint16_t* weights, float* gradient, std::uint8_t* workspace, const TransformerWorkspaceLayout& l) {
        auto& plan         = *std::ranges::find_if(attention, [&](const auto& p) { return p.batch == l.batch && p.training; });
        const auto pointer = [&](std::size_t offset) { return reinterpret_cast<std::uint16_t*>(workspace + offset); };
        const auto scalar  = [&](std::size_t offset) { return reinterpret_cast<float*>(workspace + offset); };
        const auto rows = l.batch * configuration.sequence, d = configuration.width, m = configuration.mlp_width;
        const std::size_t n = static_cast<std::size_t>(rows) * d;
        ::cuda::fill_bytes(stream, ::cuda::std::span<std::uint16_t>{pointer(l.skip_gradients), n * configuration.side_blocks}, 0);
        auto* dy = pointer(l.gradient);
        auto* dx = pointer(l.scratch_gradient);
        for (int index = static_cast<int>(parameters.blocks.size()) - 1; index >= 0; --index) {
            const auto& b = parameters.blocks[index];
            if (index < static_cast<int>(configuration.side_blocks)) kernels::add(stream, dy, pointer(l.skip_gradients) + index * n, dy, n);
            const auto* input = block_forward(index, master, weights, pointer(l.temporary), workspace, l, plan);
            linear_backward(pointer(l.activated), dy, pointer(l.dmlp), weights + b.mlp2, gradient + b.mlp2, gradient + b.mlp2_bias, rows, m, d);
            kernels::gelu_backward(stream, pointer(l.mlp), pointer(l.dmlp), static_cast<std::size_t>(rows) * m);
            linear_backward(pointer(l.norm2), pointer(l.dmlp), pointer(l.d1), weights + b.mlp1, gradient + b.mlp1, gradient + b.mlp1_bias, rows, d, m);
            kernels::normalize_backward(stream, pointer(l.attention_residual), pointer(l.d1), master + b.norm2_weight, scalar(l.mean2), scalar(l.inverse2), pointer(l.d2), gradient + b.norm2_weight, gradient + b.norm2_bias, dy, rows, d);
            linear_backward(pointer(l.attended), pointer(l.d2), pointer(l.d1), weights + b.projection, gradient + b.projection, gradient + b.projection_bias, rows, d, d);
            plan.backward(pointer(l.qkv), pointer(l.attended), scalar(l.statistics), pointer(l.d1), pointer(l.dqkv));
            linear_backward(pointer(l.norm1), pointer(l.dqkv), pointer(l.d1), weights + b.qkv, gradient + b.qkv, nullptr, rows, d, 3 * d);
            kernels::normalize_backward(stream, input, pointer(l.d1), master + b.norm1_weight, scalar(l.mean1), scalar(l.inverse1), dx, gradient + b.norm1_weight, gradient + b.norm1_bias, pointer(l.d2), rows, d);
            if (b.skip) {
                linear_backward(pointer(l.concatenated), dx, pointer(l.dqkv), weights + b.skip, gradient + b.skip, gradient + b.skip_bias, rows, 2 * d, d);
                kernels::split(stream, pointer(l.dqkv), dx, pointer(l.skip_gradients) + (2 * configuration.side_blocks - index) * n, rows, d);
            }
            std::swap(dy, dx);
        }
    }
    const std::uint16_t* Transformer::block_forward(std::uint32_t index, const float* master, const std::uint16_t* weights, std::uint16_t* output, std::uint8_t* workspace, const TransformerWorkspaceLayout& l, Attention& plan) {
        const auto p    = [&](std::size_t offset) { return reinterpret_cast<std::uint16_t*>(workspace + offset); };
        const auto f    = [&](std::size_t offset) { return reinterpret_cast<float*>(workspace + offset); };
        const auto rows = l.batch * configuration.sequence, d = configuration.width, m = configuration.mlp_width;
        const std::size_t n = static_cast<std::size_t>(rows) * d;
        const auto& b       = parameters.blocks[index];
        const auto* input   = p(l.states[index]);
        if (b.skip) {
            const auto skip_index = 2 * configuration.side_blocks + 1 - index;
            kernels::concatenate(stream, input, p(l.states[skip_index]), p(l.concatenated), rows, d);
            matmul.execute({p(l.concatenated), weights + b.skip, p(l.projected), rows, d, 2 * d, false, true});
            kernels::bias(stream, p(l.projected), master + b.skip_bias, rows, d);
            input = p(l.projected);
        }
        kernels::normalize(stream, input, master + b.norm1_weight, master + b.norm1_bias, p(l.norm1), f(l.mean1), f(l.inverse1), rows, d);
        matmul.execute({p(l.norm1), weights + b.qkv, p(l.qkv), rows, 3 * d, d, false, true});
        plan.forward(p(l.qkv), p(l.attended), f(l.statistics));
        matmul.execute({p(l.attended), weights + b.projection, p(l.temporary), rows, d, d, false, true});
        kernels::bias(stream, p(l.temporary), master + b.projection_bias, rows, d);
        kernels::add(stream, input, p(l.temporary), p(l.attention_residual), n);
        kernels::normalize(stream, p(l.attention_residual), master + b.norm2_weight, master + b.norm2_bias, p(l.norm2), f(l.mean2), f(l.inverse2), rows, d);
        matmul.execute({p(l.norm2), weights + b.mlp1, p(l.mlp), rows, m, d, false, true});
        kernels::bias(stream, p(l.mlp), master + b.mlp1_bias, rows, m);
        kernels::gelu(stream, p(l.mlp), p(l.activated), static_cast<std::size_t>(rows) * m);
        matmul.execute({p(l.activated), weights + b.mlp2, p(l.temporary), rows, d, m, false, true});
        kernels::bias(stream, p(l.temporary), master + b.mlp2_bias, rows, d);
        kernels::add(stream, p(l.attention_residual), p(l.temporary), output, n);
        return input;
    }
    void Transformer::linear_backward(const std::uint16_t* input, const std::uint16_t* gradient, std::uint16_t* input_gradient, const std::uint16_t* weight, float* weight_gradient, float* bias_gradient, std::uint32_t rows, std::uint32_t in, std::uint32_t out) {
        matmul.execute({gradient, input, weight_gradient, out, in, rows, true, false, true, 1.f});
        if (bias_gradient) kernels::bias_backward(stream, gradient, bias_gradient, rows, out);
        matmul.execute({gradient, weight, input_gradient, rows, in, out});
    }
} // namespace flowdit::neural
