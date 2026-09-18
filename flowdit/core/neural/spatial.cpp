module;
#include "spatial-kernels.h"
#include <cublas_v2.h>
#include <cudnn.h>
#include <flowdit/cuda.h>
module flowdit.neural.spatial;
import std;
namespace flowdit::neural {
    namespace {
        void check(const cudnnStatus_t status) {
            if (status != CUDNN_STATUS_SUCCESS) throw std::runtime_error{cudnnGetErrorString(status)};
        }
        void check_blas(const cublasStatus_t status) {
            if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::format("Spatial cuBLAS: {}", static_cast<int>(status))};
        }
    } // namespace
    Convolution::Convolution(cudnnHandle_t handle, std::uint32_t batch, TensorShape source, TensorShape target, std::uint32_t kernel, std::uint32_t stride, std::uint32_t padding) {
        check(cudnnCreateTensorDescriptor(&input));
        check(cudnnCreateTensorDescriptor(&output));
        check(cudnnCreateFilterDescriptor(&filter));
        check(cudnnCreateConvolutionDescriptor(&operation));
        check(cudnnSetTensor4dDescriptor(input, CUDNN_TENSOR_NCHW, CUDNN_DATA_BFLOAT16, batch, source.channels, source.height, source.width));
        check(cudnnSetTensor4dDescriptor(output, CUDNN_TENSOR_NCHW, CUDNN_DATA_BFLOAT16, batch, target.channels, target.height, target.width));
        check(cudnnSetFilter4dDescriptor(filter, CUDNN_DATA_BFLOAT16, CUDNN_TENSOR_NCHW, target.channels, source.channels, kernel, kernel));
        check(cudnnSetConvolution2dDescriptor(operation, padding, padding, stride, stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
        check(cudnnSetConvolutionMathType(operation, CUDNN_TENSOR_OP_MATH));
        int count{};
        cudnnConvolutionFwdAlgoPerf_t fwd{};
        cudnnConvolutionBwdDataAlgoPerf_t data{};
        cudnnConvolutionBwdFilterAlgoPerf_t weight{};
        check(cudnnGetConvolutionForwardAlgorithm_v7(handle, input, filter, operation, output, 1, &count, &fwd));
        check(fwd.status);
        check(cudnnGetConvolutionBackwardDataAlgorithm_v7(handle, filter, output, operation, input, 1, &count, &data));
        check(data.status);
        check(cudnnGetConvolutionBackwardFilterAlgorithm_v7(handle, input, output, operation, filter, 1, &count, &weight));
        check(weight.status);
        forward_algorithm = fwd.algo;
        data_algorithm    = data.algo;
        weight_algorithm  = weight.algo;
        workspace         = std::max({fwd.memory, data.memory, weight.memory});
    }
    Convolution::~Convolution() {
        cudnnDestroyConvolutionDescriptor(operation);
        cudnnDestroyFilterDescriptor(filter);
        cudnnDestroyTensorDescriptor(output);
        cudnnDestroyTensorDescriptor(input);
    }
    SpatialNetwork::SpatialNetwork(::cuda::stream_ref source, std::uint32_t count, TensorShape shape) : stream{source}, batch{count} {
        check(cudnnCreate(&cudnn));
        check(cudnnSetStream(cudnn, stream.get()));
        check_blas(cublasCreate(&blas));
        check_blas(cublasSetStream(blas, stream.get()));
        layers.push_back({.operation = SpatialOperation::input, .shape = shape});
    }
    SpatialNetwork::~SpatialNetwork() {
        cublasDestroy(blas);
        cudnnDestroy(cudnn);
    }
    std::size_t SpatialNetwork::append(SpatialOperation operation, std::uint32_t channels, std::uint32_t kernel, std::uint32_t stride, std::uint32_t padding, std::optional<std::size_t> source, std::size_t skip) {
        const auto parent = source.value_or(layers.size() - 1);
        const auto input  = layers[parent].shape;
        SpatialLayer layer{.operation = operation, .shape = input, .source = parent, .skip = skip, .parameters = initial.size()};
        if (operation == SpatialOperation::convolution) {
            layer.shape        = {(input.width + 2 * padding - kernel) / stride + 1, (input.height + 2 * padding - kernel) / stride + 1, channels};
            layer.weight_count = static_cast<std::size_t>(channels) * input.channels * kernel * kernel;
            initial.resize(initial.size() + layer.weight_count + channels);
            layer.convolution = std::make_unique<Convolution>(cudnn, batch, input, layer.shape, kernel, stride, padding);
            workspace_bytes   = std::max(workspace_bytes, layer.convolution->workspace);
        } else if (operation == SpatialOperation::normalization) {
            initial.resize(initial.size() + input.channels, 1);
            initial.resize(initial.size() + input.channels, 0);
        } else if (operation == SpatialOperation::upsample) {
            layer.shape.width *= 2;
            layer.shape.height *= 2;
        } else if (operation == SpatialOperation::pool) {
            layer.shape.width /= 2;
            layer.shape.height /= 2;
        } else if (operation == SpatialOperation::attention) layer.shape.channels /= 3;
        const auto elements = static_cast<std::size_t>(batch) * layer.shape.width * layer.shape.height * layer.shape.channels;
        temporary_count     = std::max({temporary_count, elements, static_cast<std::size_t>(batch) * input.width * input.height * input.channels, layer.weight_count});
        layers.push_back(std::move(layer));
        return layers.size() - 1;
    }
    void SpatialNetwork::residual(std::uint32_t channels) {
        const auto source = layers.size() - 1;
        append(SpatialOperation::normalization);
        append(SpatialOperation::silu);
        append(SpatialOperation::convolution, channels);
        append(SpatialOperation::normalization);
        append(SpatialOperation::silu);
        const auto branch   = append(SpatialOperation::convolution, channels);
        const auto shortcut = layers[source].shape.channels == channels ? source : append(SpatialOperation::convolution, channels, 1, 1, 0, source);
        append(SpatialOperation::add, 0, 3, 1, 1, branch, shortcut);
    }
    void SpatialNetwork::attention() {
        const auto source   = layers.size() - 1;
        const auto channels = layers[source].shape.channels;
        append(SpatialOperation::normalization);
        append(SpatialOperation::convolution, channels * 3, 1, 1, 0);
        append(SpatialOperation::attention);
        append(SpatialOperation::convolution, channels, 1, 1, 0);
        append(SpatialOperation::add, 0, 3, 1, 1, {}, source);
    }
    std::vector<float> SpatialNetwork::initialize(std::uint64_t seed) const {
        auto result = initial;
        std::mt19937_64 random{seed};
        for (const auto& layer : layers) {
            if (layer.operation != SpatialOperation::convolution) continue;
            const float deviation = std::sqrt(2.f / static_cast<float>(layer.weight_count / layer.shape.channels));
            std::normal_distribution<float> normal{0, deviation};
            for (std::size_t i = 0; i < layer.weight_count; ++i) result[layer.parameters + i] = normal(random);
        }
        return result;
    }
    const std::uint16_t* SpatialNetwork::forward(const float* parameters, const float* input) {
        if (!weights) weights.emplace(stream, ::cuda::device_default_memory_pool(stream.device()), initial.size(), ::cuda::no_init);
        if (!workspace) workspace.emplace(stream, ::cuda::device_default_memory_pool(stream.device()), std::max(workspace_bytes, 1uz), ::cuda::no_init);
        if (!temporary) temporary.emplace(stream, ::cuda::device_default_memory_pool(stream.device()), temporary_count, ::cuda::no_init);
        if (!convolution_gradient) convolution_gradient.emplace(stream, ::cuda::device_default_memory_pool(stream.device()), temporary_count, ::cuda::no_init);
        kernels::pack(stream, parameters, weights->data(), initial.size());
        const float one = 1, zero = 0;
        for (std::size_t index = 0; index < layers.size(); ++index) {
            auto& layer = layers[index];
            layer.gradient.reset();
            const auto elements = static_cast<std::size_t>(batch) * layer.shape.width * layer.shape.height * layer.shape.channels;
            if (!layer.values) layer.values.emplace(stream, ::cuda::device_default_memory_pool(stream.device()), elements, ::cuda::no_init);
            auto* output = layer.values->data();
            if (index == 0) {
                kernels::pack(stream, input, output, elements);
                continue;
            }
            const auto& source = layers[layer.source];
            const auto* x      = source.values->data();
            switch (layer.operation) {
            case SpatialOperation::convolution:
                {
                    const auto& conv = *layer.convolution;
                    check(cudnnConvolutionForward(cudnn, &one, conv.input, x, conv.filter, weights->data() + layer.parameters, conv.operation, conv.forward_algorithm, workspace->data(), workspace->size(), &zero, conv.output, output));
                    kernels::bias(stream, output, parameters + layer.parameters + layer.weight_count, batch, layer.shape.channels, layer.shape.width * layer.shape.height);
                    break;
                }
            case SpatialOperation::normalization: kernels::group_norm(stream, x, parameters + layer.parameters, output, batch, layer.shape.channels, layer.shape.width * layer.shape.height); break;
            case SpatialOperation::silu:
            case SpatialOperation::relu:
            case SpatialOperation::leaky_relu: kernels::activation(stream, x, output, elements, static_cast<int>(layer.operation) - static_cast<int>(SpatialOperation::silu)); break;
            case SpatialOperation::add: kernels::residual(stream, x, layers[layer.skip].values->data(), output, elements); break;
            case SpatialOperation::upsample:
            case SpatialOperation::pool: kernels::resize(stream, x, output, batch * source.shape.channels, source.shape.width, source.shape.height, layer.operation == SpatialOperation::pool); break;
            case SpatialOperation::attention: attend(layer, x, nullptr); break;
            default: break;
            }
        }
        return layers.back().values->data();
    }
    float* SpatialNetwork::accumulate(std::size_t index) {
        auto& layer = layers[index];
        if (!layer.gradient) {
            layer.gradient.emplace(stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(batch) * layer.shape.width * layer.shape.height * layer.shape.channels, ::cuda::no_init);
            ::cuda::fill_bytes(stream, *layer.gradient, 0);
        }
        return layer.gradient->data();
    }
    const float* SpatialNetwork::backward(const float* parameters, float* parameter_gradient, const float* gradient) {
        if (gradient) kernels::sum(stream, gradient, accumulate(layers.size() - 1), layers.back().values->size());
        const float one = 1, zero = 0;
        for (std::size_t index = layers.size() - 1; index > 0; --index) {
            auto& layer = layers[index];
            if (!layer.gradient) continue;
            auto& source     = layers[layer.source];
            const auto* x    = source.values->data();
            const auto* dy   = layer.gradient->data();
            auto* dx         = accumulate(layer.source);
            const auto count = layer.values->size();
            switch (layer.operation) {
            case SpatialOperation::convolution:
                {
                    const auto& conv = *layer.convolution;
                    kernels::pack(stream, dy, temporary->data(), count);
                    check(cudnnConvolutionBackwardData(cudnn, &one, conv.filter, weights->data() + layer.parameters, conv.output, temporary->data(), conv.operation, conv.data_algorithm, workspace->data(), workspace->size(), &zero, conv.input, convolution_gradient->data()));
                    kernels::unpack(stream, convolution_gradient->data(), dx, source.values->size(), true);
                    if (parameter_gradient) {
                        check(cudnnConvolutionBackwardFilter(cudnn, &one, conv.input, x, conv.output, temporary->data(), conv.operation, conv.weight_algorithm, workspace->data(), workspace->size(), &zero, conv.filter, convolution_gradient->data()));
                        kernels::unpack(stream, convolution_gradient->data(), parameter_gradient + layer.parameters, layer.weight_count, true);
                        kernels::bias_gradient(stream, dy, parameter_gradient + layer.parameters + layer.weight_count, batch, layer.shape.channels, layer.shape.width * layer.shape.height);
                    }
                    break;
                }
            case SpatialOperation::normalization: kernels::group_norm_backward(stream, x, parameters + layer.parameters, dy, dx, parameter_gradient ? parameter_gradient + layer.parameters : nullptr, batch, layer.shape.channels, layer.shape.width * layer.shape.height); break;
            case SpatialOperation::silu:
            case SpatialOperation::relu:
            case SpatialOperation::leaky_relu: kernels::activation_backward(stream, x, dy, dx, count, static_cast<int>(layer.operation) - static_cast<int>(SpatialOperation::silu)); break;
            case SpatialOperation::add:
                kernels::sum(stream, dy, dx, count);
                kernels::sum(stream, dy, accumulate(layer.skip), count);
                break;
            case SpatialOperation::upsample:
            case SpatialOperation::pool: kernels::resize_backward(stream, x, dy, dx, batch * source.shape.channels, source.shape.width, source.shape.height, layer.operation == SpatialOperation::pool); break;
            case SpatialOperation::attention: attend(layer, x, dy); break;
            default: break;
            }
            layer.gradient.reset();
        }
        return layers.front().gradient->data();
    }
    void SpatialNetwork::attend(SpatialLayer& layer, const std::uint16_t* input, const float* gradient) {
        const auto n = layer.shape.width * layer.shape.height, c = layer.shape.channels;
        const auto stride = static_cast<long long>(n) * c;
        ::cuda::device_buffer<std::uint16_t> probabilities{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(batch) * n * n, ::cuda::no_init};
        const float one = 1, zero = 0, scale = 1.f / std::sqrt(static_cast<float>(c));
        check_blas(cublasGemmStridedBatchedEx(blas, CUBLAS_OP_N, CUBLAS_OP_T, n, n, c, &scale, input, CUDA_R_16BF, n, stride * 3, input + stride, CUDA_R_16BF, n, stride * 3, &zero, probabilities.data(), CUDA_R_16BF, n, static_cast<long long>(n) * n, batch, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        kernels::attention_softmax(stream, probabilities.data(), batch, n);
        if (!gradient) {
            check_blas(cublasGemmStridedBatchedEx(blas, CUBLAS_OP_N, CUBLAS_OP_N, n, c, n, &one, probabilities.data(), CUDA_R_16BF, n, static_cast<long long>(n) * n, input + stride * 2, CUDA_R_16BF, n, stride * 3, &zero, layer.values->data(), CUDA_R_16BF, n, stride, batch, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
            return;
        }
        ::cuda::device_buffer<std::uint16_t> scores_gradient{stream, ::cuda::device_default_memory_pool(stream.device()), probabilities.size(), ::cuda::no_init};
        kernels::pack(stream, gradient, temporary->data(), static_cast<std::size_t>(batch) * stride);
        check_blas(cublasGemmStridedBatchedEx(blas, CUBLAS_OP_N, CUBLAS_OP_T, n, n, c, &one, temporary->data(), CUDA_R_16BF, n, stride, input + stride * 2, CUDA_R_16BF, n, stride * 3, &zero, scores_gradient.data(), CUDA_R_16BF, n, static_cast<long long>(n) * n, batch, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        check_blas(cublasGemmStridedBatchedEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, c, n, &one, probabilities.data(), CUDA_R_16BF, n, static_cast<long long>(n) * n, temporary->data(), CUDA_R_16BF, n, stride, &zero, convolution_gradient->data() + stride * 2, CUDA_R_16BF, n, stride * 3, batch, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        kernels::attention_softmax_backward(stream, probabilities.data(), scores_gradient.data(), batch, n, scale);
        check_blas(cublasGemmStridedBatchedEx(blas, CUBLAS_OP_N, CUBLAS_OP_N, n, c, n, &one, scores_gradient.data(), CUDA_R_16BF, n, static_cast<long long>(n) * n, input + stride, CUDA_R_16BF, n, stride * 3, &zero, convolution_gradient->data(), CUDA_R_16BF, n, stride * 3, batch, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        check_blas(cublasGemmStridedBatchedEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, c, n, &one, scores_gradient.data(), CUDA_R_16BF, n, static_cast<long long>(n) * n, input, CUDA_R_16BF, n, stride * 3, &zero, convolution_gradient->data() + stride, CUDA_R_16BF, n, stride * 3, batch, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        kernels::unpack(stream, convolution_gradient->data(), accumulate(layer.source), static_cast<std::size_t>(batch) * stride * 3, true);
    }
} // namespace flowdit::neural
