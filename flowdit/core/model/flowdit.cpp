module;
#include "../neural/transformer-kernels.h"
#include "../tokenizer/kernels.h"
#include "kernels.h"
#include <flowdit/cuda.h>
module flowdit.model;
import std;
namespace flowdit {
    namespace {
        neural::TransformerConfiguration transformer_configuration(const ModelConfiguration& c) {
            return {c.shape.width * c.shape.height / (c.patch_size * c.patch_size) + 2, c.width, c.heads, c.side_blocks, c.mlp_width};
        }
    } // namespace
    FlowDiTParameterLayout::FlowDiTParameterLayout(const ModelConfiguration& c) {
        const auto d = c.width, p = c.patch_size * c.patch_size * c.shape.channels;
        const auto shape = transformer_configuration(c);
        patch_weight     = 0;
        patch_bias       = static_cast<std::size_t>(d) * p;
        class_embedding  = patch_bias + d;
        position         = class_embedding + (c.class_count + 1) * d;
        transformer      = position + shape.sequence * d;
        final_weight     = transformer + neural::TransformerParameterLayout{shape}.count;
        final_bias       = final_weight + d;
        velocity_weight  = final_bias + d;
        velocity_bias    = velocity_weight + static_cast<std::size_t>(p) * d;
        parameter_count  = velocity_bias + p;
    }
    FlowDiTWorkspaceLayout::FlowDiTWorkspaceLayout(std::uint32_t b, const ModelConfiguration& c, bool train) : batch{b}, training{train}, transformer_layout{b, transformer_configuration(c), train} {
        const auto shape = transformer_configuration(c);
        const auto take  = [&](std::size_t bytes) {
            const auto offset = byte_count;
            byte_count += (bytes + 255) / 256 * 256;
            return offset;
        };
        const std::size_t rows = b * shape.sequence, images = b * (shape.sequence - 2), p = c.patch_size * c.patch_size * c.shape.channels;
        weights       = take(FlowDiTParameterLayout{c}.parameter_count * 2);
        patches       = take(images * p * 2);
        embedded      = take(images * c.width * 2);
        normalized    = take(rows * c.width * 2);
        means         = take(rows * 4);
        inverse       = take(rows * 4);
        full_velocity = take(rows * p * 2);
        velocity      = take(images * p * 4);
        if (training) {
            velocity_gradient = take(images * p * 4);
            full_gradient     = take(rows * p * 2);
            norm_gradient     = take(rows * c.width * 2);
            patch_gradient    = take(images * c.width * 2);
        }
        transformer_workspace = take(transformer_layout.byte_count);
    }
    FlowDiT::FlowDiT(::cuda::stream_ref s, neural::MatmulRuntime& m, const ModelConfiguration& c) : configuration{c}, shape{transformer_configuration(c)}, patch_width{c.patch_size * c.patch_size * c.shape.channels}, stream{s}, matmul{m}, parameters{c}, transformer{s, m, shape} {}
    std::vector<float> FlowDiT::initialize_parameters(std::uint64_t seed) const {
        std::vector<float> result(parameters.parameter_count);
        std::mt19937_64 random{seed};
        std::normal_distribution<float> normal{0.f, .02f};
        std::uniform_real_distribution<float> patch{-1.f / std::sqrt(static_cast<float>(patch_width)), 1.f / std::sqrt(static_cast<float>(patch_width))};
        for (std::size_t i = parameters.patch_weight; i < parameters.class_embedding; ++i) result[i] = patch(random);
        std::normal_distribution<float> embedding{0.f, 1.f};
        for (std::size_t i = parameters.class_embedding; i < parameters.position; ++i) result[i] = embedding(random);
        for (std::size_t i = parameters.position; i < parameters.transformer; ++i) result[i] = normal(random);
        transformer.initialize(std::span{result}.subspan(parameters.transformer, transformer.parameters.count), random);
        std::fill_n(result.data() + parameters.final_weight, configuration.width, 1.f);
        for (std::size_t i = parameters.velocity_weight; i < parameters.velocity_bias; ++i) result[i] = normal(random);
        return result;
    }
    void FlowDiT::forward(const float* master, const float* input, const float* times, const std::uint32_t* labels, std::uint8_t* workspace, const FlowDiTWorkspaceLayout& l) {
        const auto p    = [&](std::size_t offset) { return reinterpret_cast<std::uint16_t*>(workspace + offset); };
        const auto f    = [&](std::size_t offset) { return reinterpret_cast<float*>(workspace + offset); };
        const auto rows = l.batch * shape.sequence, images = l.batch * (shape.sequence - 2), d = configuration.width;
        auto* tw       = workspace + l.transformer_workspace;
        const auto& tl = l.transformer_layout;
        tokenizer_kernels::pack(stream, master, p(l.weights), parameters.parameter_count);
        tokenizer_kernels::pack(stream, input, p(l.patches), static_cast<std::size_t>(images) * patch_width);
        matmul.execute({p(l.patches), p(l.weights) + parameters.patch_weight, p(l.embedded), images, d, patch_width, false, true});
        neural::kernels::bias(stream, p(l.embedded), master + parameters.patch_bias, images, d);
        kernels::make_tokens(stream, p(l.embedded), times, labels, master + parameters.class_embedding, master + parameters.position, reinterpret_cast<std::uint16_t*>(tw + tl.states.front()), l.batch, shape.sequence, d);
        transformer.forward(master + parameters.transformer, p(l.weights) + parameters.transformer, tw, tl);
        neural::kernels::normalize(stream, reinterpret_cast<const std::uint16_t*>(tw + tl.states.back()), master + parameters.final_weight, master + parameters.final_bias, p(l.normalized), f(l.means), f(l.inverse), rows, d);
        matmul.execute({p(l.normalized), p(l.weights) + parameters.velocity_weight, p(l.full_velocity), rows, patch_width, d, false, true});
        neural::kernels::bias(stream, p(l.full_velocity), master + parameters.velocity_bias, rows, patch_width);
        kernels::velocity(stream, p(l.full_velocity), f(l.velocity), l.batch, shape.sequence, patch_width);
    }
    void FlowDiT::backward(const float* master, float* gradient, const std::uint32_t* labels, std::uint8_t* workspace, const FlowDiTWorkspaceLayout& l) {
        const auto p    = [&](std::size_t offset) { return reinterpret_cast<std::uint16_t*>(workspace + offset); };
        const auto f    = [&](std::size_t offset) { return reinterpret_cast<float*>(workspace + offset); };
        const auto rows = l.batch * shape.sequence, images = l.batch * (shape.sequence - 2), d = configuration.width;
        auto* tw       = workspace + l.transformer_workspace;
        const auto& tl = l.transformer_layout;
        kernels::velocity_backward(stream, f(l.velocity_gradient), p(l.full_gradient), l.batch, shape.sequence, patch_width);
        matmul.execute({p(l.full_gradient), p(l.normalized), gradient + parameters.velocity_weight, patch_width, d, rows, true, false, true, 1.f});
        neural::kernels::bias_backward(stream, p(l.full_gradient), gradient + parameters.velocity_bias, rows, patch_width);
        matmul.execute({p(l.full_gradient), p(l.weights) + parameters.velocity_weight, p(l.norm_gradient), rows, d, patch_width});
        neural::kernels::normalize_backward(stream, reinterpret_cast<const std::uint16_t*>(tw + tl.states.back()), p(l.norm_gradient), master + parameters.final_weight, f(l.means), f(l.inverse), reinterpret_cast<std::uint16_t*>(tw + tl.gradient), gradient + parameters.final_weight, gradient + parameters.final_bias, nullptr, rows, d);
        transformer.backward(master + parameters.transformer, p(l.weights) + parameters.transformer, gradient + parameters.transformer, tw, tl);
        kernels::tokens_backward(stream, reinterpret_cast<const std::uint16_t*>(tw + tl.scratch_gradient), labels, gradient + parameters.class_embedding, gradient + parameters.position, p(l.patch_gradient), l.batch, shape.sequence, d, configuration.class_count);
        matmul.execute({p(l.patch_gradient), p(l.patches), gradient + parameters.patch_weight, d, patch_width, images, true, false, true, 1.f});
        neural::kernels::bias_backward(stream, p(l.patch_gradient), gradient + parameters.patch_bias, images, d);
    }
} // namespace flowdit
