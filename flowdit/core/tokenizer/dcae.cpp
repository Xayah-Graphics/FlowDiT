module;
#include "kernels.h"
#include <cublas_v2.h>
#include <cudnn.h>
#include <flowdit/cuda.h>
module flowdit.tokenizer.dcae;
import flowdit.serialization.safetensors;
import flowdit.serialization.digest;
import std;
namespace flowdit {
    namespace {
        void check(cudnnStatus_t result) {
            if (result != CUDNN_STATUS_SUCCESS) throw std::runtime_error{cudnnGetErrorString(result)};
        }
        void check(cublasStatus_t result) {
            if (result != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::format("DC-AE cuBLAS: {}", static_cast<int>(result))};
        }
    } // namespace
    TokenizerConvolution::TokenizerConvolution(cudnnHandle_t handle, TensorShape source, TensorShape target, std::uint32_t kernel, std::uint32_t stride, std::uint32_t groups) {
        check(cudnnCreateTensorDescriptor(&input));
        check(cudnnCreateTensorDescriptor(&output));
        check(cudnnCreateFilterDescriptor(&filter));
        check(cudnnCreateConvolutionDescriptor(&operation));
        check(cudnnSetTensor4dDescriptor(input, CUDNN_TENSOR_NCHW, CUDNN_DATA_BFLOAT16, 1, source.channels, source.height, source.width));
        check(cudnnSetTensor4dDescriptor(output, CUDNN_TENSOR_NCHW, CUDNN_DATA_BFLOAT16, 1, target.channels, target.height, target.width));
        check(cudnnSetFilter4dDescriptor(filter, CUDNN_DATA_BFLOAT16, CUDNN_TENSOR_NCHW, target.channels, source.channels / groups, kernel, kernel));
        check(cudnnSetConvolution2dDescriptor(operation, kernel / 2, kernel / 2, stride, stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
        check(cudnnSetConvolutionGroupCount(operation, groups));
        check(cudnnSetConvolutionMathType(operation, CUDNN_TENSOR_OP_MATH));
        cudnnConvolutionFwdAlgoPerf_t performance{};
        int count{};
        check(cudnnGetConvolutionForwardAlgorithm_v7(handle, input, filter, operation, output, 1, &count, &performance));
        check(performance.status);
        algorithm = performance.algo;
        workspace = performance.memory;
    }
    TokenizerConvolution::~TokenizerConvolution() {
        cudnnDestroyConvolutionDescriptor(operation);
        cudnnDestroyFilterDescriptor(filter);
        cudnnDestroyTensorDescriptor(output);
        cudnnDestroyTensorDescriptor(input);
    }
    DCAE::DCAE(::cuda::stream_ref s, TokenizerSpecification spec, TensorShape image, TokenizerDirection mode) : stream{s}, specification{std::move(spec)}, direction{mode} {
        if (specification.model != "dc-ae-f32c32-sana-1.1") throw std::runtime_error{"Unsupported tokenizer architecture"};
        const auto path = tokenizer_directory(specification) / "weights.safetensors";
        if (serialization::digest_file(path) != specification.weights_digest) throw std::runtime_error{"Tokenizer weights do not match their identity"};
        check(cudnnCreate(&cudnn));
        check(cudnnSetStream(cudnn, stream.get()));
        check(cublasCreate(&blas));
        check(cublasSetStream(blas, stream.get()));
        check(cublasSetMathMode(blas, CUBLAS_PEDANTIC_MATH));
        const TensorShape latent{image.width / specification.spatial_factor, image.height / specification.spatial_factor, specification.latent_channels};
        layers.push_back({.operation = TokenizerOperation::input, .shape = direction == TokenizerDirection::encode ? image : latent});
        const std::array<std::uint32_t, 6> widths{128, 256, 512, 512, 1024, 1024};
        std::size_t x{};
        if (direction == TokenizerDirection::encode) {
            x = append(TokenizerOperation::convolution, x, 128, 0, "encoder.conv_in");
            for (std::uint32_t stage = 0; stage < 6; ++stage) {
                const auto depth = stage < 3 ? 2u : 3u;
                for (std::uint32_t j = 0; j < depth; ++j) x = block(x, std::format("encoder.down_blocks.{}.{}", stage, j), stage >= 3);
                if (stage == 5) continue;
                const auto shortcut = append(TokenizerOperation::down_average, x, widths[stage + 1]);
                x                   = append(TokenizerOperation::convolution, x, widths[stage + 1], 0, std::format("encoder.down_blocks.{}.{}.conv", stage, depth), 2);
                x                   = append(TokenizerOperation::add, x, 0, shortcut);
            }
            const auto shortcut = append(TokenizerOperation::average, x, latent.channels);
            x                   = append(TokenizerOperation::convolution, x, latent.channels, 0, "encoder.conv_out");
            append(TokenizerOperation::add, x, 0, shortcut);
        } else {
            const auto shortcut = append(TokenizerOperation::duplicate, x, 1024);
            x                   = append(TokenizerOperation::convolution, x, 1024, 0, "decoder.conv_in");
            x                   = append(TokenizerOperation::add, x, 0, shortcut);
            for (int stage = 5; stage >= 0; --stage) {
                if (stage < 5) {
                    const auto skip = append(TokenizerOperation::up_duplicate, x, widths[stage]);
                    x               = append(TokenizerOperation::nearest, x);
                    x               = append(TokenizerOperation::convolution, x, widths[stage], 0, std::format("decoder.up_blocks.{}.0.conv", stage));
                    x               = append(TokenizerOperation::add, x, 0, skip);
                }
                for (std::uint32_t j = 0; j < 3; ++j) x = block(x, std::format("decoder.up_blocks.{}.{}", stage, j + (stage < 5 ? 1 : 0)), stage >= 3);
            }
            x = append(TokenizerOperation::rms, x, 0, 0, "decoder.norm_out");
            x = append(TokenizerOperation::relu, x);
            append(TokenizerOperation::convolution, x, image.channels, 0, "decoder.conv_out");
        }
        std::vector<std::string> names;
        for (const auto& layer : layers) {
            if (layer.operation != TokenizerOperation::convolution && layer.operation != TokenizerOperation::rms) continue;
            names.push_back(layer.name + ".weight");
            if (layer.has_bias || layer.operation == TokenizerOperation::rms) names.push_back(layer.name + ".bias");
        }
        std::vector<std::string_view> views{names.begin(), names.end()};
        const auto file = serialization::safetensors::read(path, views);
        std::map<std::string, const serialization::safetensors::Tensor*> tensors;
        for (const auto& tensor : file.tensors) tensors.emplace(tensor.name, &tensor);
        std::vector<float> host, convolution_weights;
        std::size_t scratch{}, attention_count{};
        for (auto& layer : layers) {
            if (layer.operation == TokenizerOperation::convolution || layer.operation == TokenizerOperation::rms) {
                auto& destination  = layer.operation == TokenizerOperation::convolution ? convolution_weights : host;
                layer.weights      = destination.size();
                const auto& weight = *tensors.at(layer.name + ".weight");
                destination.resize(destination.size() + weight.data.size() / sizeof(float));
                std::memcpy(destination.data() + layer.weights, weight.data.data(), weight.data.size());
                layer.bias = host.size();
                if (layer.has_bias || layer.operation == TokenizerOperation::rms) {
                    const auto& bias = *tensors.at(layer.name + ".bias");
                    host.resize(host.size() + bias.data.size() / sizeof(float));
                    std::memcpy(host.data() + layer.bias, bias.data.data(), bias.data.size());
                }
                if (layer.convolution) scratch = std::max(scratch, layer.convolution->workspace);
            }
            if (layer.operation == TokenizerOperation::attention) attention_count = std::max(attention_count, static_cast<std::size_t>(layer.shape.channels / 32) * (130 * layer.shape.width * layer.shape.height + 33 * 32));
            layer.elements = static_cast<std::size_t>(layer.shape.width) * layer.shape.height * layer.shape.channels;
        }
        // Allocate only simultaneously live activations, preserving residuals.
        std::vector<std::size_t> last(layers.size());
        for (std::size_t i = 1; i < layers.size(); ++i) {
            last[layers[i].source] = i;
            if (layers[i].operation == TokenizerOperation::add || layers[i].operation == TokenizerOperation::attention) last[layers[i].skip] = i;
        }
        struct Slot {
            std::size_t offset, size, until;
        };
        std::vector<Slot> slots;
        std::size_t size{};
        for (std::size_t i = 0; i < layers.size(); ++i) {
            const auto count = (layers[i].elements + 127) / 128 * 128;
            auto slot        = std::ranges::find_if(slots, [&](const auto& s) { return s.until < i && s.size >= count; });
            if (slot == slots.end()) {
                slots.push_back({size, count, last[i]});
                layers[i].offset = size;
                size += count;
            } else {
                slot->until      = last[i];
                layers[i].offset = slot->offset;
            }
        }
        const auto pool = ::cuda::device_default_memory_pool(stream.device());
        parameters.emplace(stream, pool, host.size(), ::cuda::no_init);
        weights.emplace(stream, pool, convolution_weights.size(), ::cuda::no_init);
        arena.emplace(stream, pool, size, ::cuda::no_init);
        workspace.emplace(stream, pool, scratch, ::cuda::no_init);
        attention_workspace.emplace(stream, pool, attention_count, ::cuda::no_init);
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{host.data(), host.size()}, *parameters);
        ::cuda::device_buffer<float> staging{stream, pool, convolution_weights.size(), ::cuda::no_init};
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{convolution_weights.data(), convolution_weights.size()}, staging);
        tokenizer_kernels::pack(stream, staging.data(), weights->data(), convolution_weights.size());
        stream.sync();
    }
    DCAE::~DCAE() {
        cublasDestroy(blas);
        cudnnDestroy(cudnn);
    }
    const std::uint16_t* DCAE::forward(const float* input) {
        tokenizer_kernels::pack(stream, input, arena->data() + layers.front().offset, layers.front().elements);
        const float one = 1.f, zero = 0.f;
        for (std::size_t i = 1; i < layers.size(); ++i) {
            const auto& l      = layers[i];
            const auto& parent = layers[l.source];
            const auto* source = arena->data() + parent.offset;
            auto* target       = arena->data() + l.offset;
            const auto area    = l.shape.width * l.shape.height;
            switch (l.operation) {
            case TokenizerOperation::convolution:
                {
                    const auto& conv = *l.convolution;
                    check(cudnnConvolutionForward(cudnn, &one, conv.input, source, conv.filter, weights->data() + l.weights, conv.operation, conv.algorithm, workspace->data(), workspace->size(), &zero, conv.output, target));
                    if (l.has_bias) tokenizer_kernels::pointwise(stream, target, nullptr, parameters->data() + l.bias, target, l.shape.channels, area, 0);
                    break;
                }
            case TokenizerOperation::rms: tokenizer_kernels::rms(stream, source, parameters->data() + l.weights, target, l.shape.channels, area); break;
            case TokenizerOperation::silu: tokenizer_kernels::pointwise(stream, source, nullptr, nullptr, target, l.shape.channels, area, 1); break;
            case TokenizerOperation::relu: tokenizer_kernels::pointwise(stream, source, nullptr, nullptr, target, l.shape.channels, area, 2); break;
            case TokenizerOperation::add: tokenizer_kernels::pointwise(stream, source, arena->data() + layers[l.skip].offset, nullptr, target, l.shape.channels, area, 3); break;
            case TokenizerOperation::glu: tokenizer_kernels::pointwise(stream, source, nullptr, nullptr, target, l.shape.channels, area, 4); break;
            case TokenizerOperation::down_average:
            case TokenizerOperation::nearest:
            case TokenizerOperation::up_duplicate:
            case TokenizerOperation::average:
            case TokenizerOperation::duplicate: tokenizer_kernels::rearrange(stream, source, target, parent.shape.width, parent.shape.height, parent.shape.channels, l.shape.channels, static_cast<int>(l.operation) - static_cast<int>(TokenizerOperation::down_average)); break;
            case TokenizerOperation::attention:
                {
                    const auto heads = l.shape.channels / 32;
                    auto* q          = attention_workspace->data();
                    auto* k          = q + heads * 32 * area;
                    auto* v          = k + heads * 32 * area;
                    auto* kv         = v + heads * 33 * area;
                    auto* out        = kv + heads * 33 * 32;
                    tokenizer_kernels::attention_pack(stream, source, arena->data() + layers[l.skip].offset, q, k, v, heads, area);
                    check(cublasSgemmStridedBatched(blas, CUBLAS_OP_T, CUBLAS_OP_N, 32, 33, area, &one, k, area, 32 * area, v, area, 33 * area, &zero, kv, 32, 33 * 32, heads));
                    check(cublasSgemmStridedBatched(blas, CUBLAS_OP_N, CUBLAS_OP_N, area, 33, 32, &one, q, area, 32 * area, kv, 32, 33 * 32, &zero, out, area, 33 * area, heads));
                    tokenizer_kernels::attention_output(stream, out, target, heads, area);
                    break;
                }
            default: break;
            }
        }
        return arena->data() + layers.back().offset;
    }
    std::size_t DCAE::append(TokenizerOperation operation, std::size_t source, std::uint32_t channels, std::size_t skip, std::string name, std::uint32_t stride, std::uint32_t groups) {
        const auto shape = layers[source].shape;
        TokenizerLayer l{.operation = operation, .shape = shape, .source = source, .skip = skip, .name = std::move(name)};
        if (channels) l.shape.channels = channels;
        if (operation == TokenizerOperation::convolution) {
            l.shape.width /= stride;
            l.shape.height /= stride;
            const bool point  = l.name.ends_with("qkv") || l.name.ends_with("proj_out") || l.name.ends_with("to_out") || l.name.ends_with("conv_point") || l.name.ends_with("conv_inverted");
            const auto kernel = point ? 1u : l.name.ends_with("proj_in") ? 5u : 3u;
            l.has_bias        = !(l.name.ends_with("conv2") || l.name.ends_with("qkv") || l.name.ends_with("proj_in") || l.name.ends_with("proj_out") || l.name.ends_with("to_out") || l.name.ends_with("conv_point"));
            l.convolution     = std::make_unique<TokenizerConvolution>(cudnn, shape, l.shape, kernel, stride, groups);
        } else if (operation == TokenizerOperation::down_average) {
            l.shape.width /= 2;
            l.shape.height /= 2;
        } else if (operation == TokenizerOperation::nearest || operation == TokenizerOperation::up_duplicate) {
            l.shape.width *= 2;
            l.shape.height *= 2;
        }
        layers.push_back(std::move(l));
        return layers.size() - 1;
    }
    std::size_t DCAE::block(std::size_t source, const std::string& name, bool attention) {
        const auto channels = layers[source].shape.channels;
        auto x              = source;
        if (attention) {
            const auto qkv = append(TokenizerOperation::convolution, x, channels * 3, 0, name + ".attn.qkv");
            x              = append(TokenizerOperation::convolution, qkv, channels * 3, 0, name + ".attn.to_qkv_multiscale.0.proj_in", 1, channels * 3);
            x              = append(TokenizerOperation::convolution, x, channels * 3, 0, name + ".attn.to_qkv_multiscale.0.proj_out", 1, channels * 3 / 32);
            x              = append(TokenizerOperation::attention, qkv, channels * 2, x);
            x              = append(TokenizerOperation::convolution, x, channels, 0, name + ".attn.to_out");
            x              = append(TokenizerOperation::rms, x, 0, 0, name + ".attn.norm_out");
            source         = append(TokenizerOperation::add, x, 0, source);
            x              = append(TokenizerOperation::convolution, source, channels * 8, 0, name + ".conv_out.conv_inverted");
            x              = append(TokenizerOperation::silu, x);
            x              = append(TokenizerOperation::convolution, x, channels * 8, 0, name + ".conv_out.conv_depth", 1, channels * 8);
            x              = append(TokenizerOperation::glu, x, channels * 4);
            x              = append(TokenizerOperation::convolution, x, channels, 0, name + ".conv_out.conv_point");
            x              = append(TokenizerOperation::rms, x, 0, 0, name + ".conv_out.norm");
        } else {
            x = append(TokenizerOperation::convolution, x, channels, 0, name + ".conv1");
            x = append(TokenizerOperation::silu, x);
            x = append(TokenizerOperation::convolution, x, channels, 0, name + ".conv2");
            x = append(TokenizerOperation::rms, x, 0, 0, name + ".norm");
        }
        return append(TokenizerOperation::add, x, 0, source);
    }
} // namespace flowdit
