module;
#include <cublas_v2.h>
#include <cudnn.h>
#include <flowdit/cuda.h>
export module flowdit.neural.spatial;
export import flowdit.tensor.types;
import std;
export namespace flowdit::neural {
    enum class SpatialOperation { input, convolution, normalization, silu, relu, leaky_relu, add, upsample, pool, attention };
    struct Convolution final {
        cudnnTensorDescriptor_t input{}, output{};
        cudnnFilterDescriptor_t filter{};
        cudnnConvolutionDescriptor_t operation{};
        cudnnConvolutionFwdAlgo_t forward_algorithm{};
        cudnnConvolutionBwdDataAlgo_t data_algorithm{};
        cudnnConvolutionBwdFilterAlgo_t weight_algorithm{};
        std::size_t workspace{};
        Convolution(cudnnHandle_t handle, std::uint32_t batch, TensorShape source, TensorShape target, std::uint32_t kernel, std::uint32_t stride, std::uint32_t padding);
        ~Convolution();
        Convolution(const Convolution&)            = delete;
        Convolution& operator=(const Convolution&) = delete;
    };
    struct SpatialLayer final {
        SpatialOperation operation;
        TensorShape shape;
        std::size_t source{}, skip{}, parameters{}, weight_count{};
        std::unique_ptr<Convolution> convolution;
        std::optional<::cuda::device_buffer<std::uint16_t>> values;
        std::optional<::cuda::device_buffer<float>> gradient;
    };
    // Explicit spatial layers shared by the autoencoder, PatchGAN and frozen VGG.
    // Activations are BF16; parameter masters, accumulated gradients and reductions are FP32.
    struct SpatialNetwork {
        ::cuda::stream_ref stream;
        std::uint32_t batch;
        std::vector<SpatialLayer> layers;
        std::vector<float> initial;
        cudnnHandle_t cudnn{};
        cublasHandle_t blas{};
        SpatialNetwork(::cuda::stream_ref stream, std::uint32_t batch, TensorShape input);
        ~SpatialNetwork();
        SpatialNetwork(const SpatialNetwork&)            = delete;
        SpatialNetwork& operator=(const SpatialNetwork&) = delete;
        std::size_t append(SpatialOperation operation, std::uint32_t channels = 0, std::uint32_t kernel = 3, std::uint32_t stride = 1, std::uint32_t padding = 1, std::optional<std::size_t> source = {}, std::size_t skip = 0);
        void residual(std::uint32_t channels);
        void attention();
        std::vector<float> initialize(std::uint64_t seed) const;
        const std::uint16_t* forward(const float* parameters, const float* input);
        float* accumulate(std::size_t layer);
        const float* backward(const float* parameters, float* parameter_gradient, const float* gradient = nullptr);

    private:
        std::optional<::cuda::device_buffer<std::uint16_t>> weights, temporary, convolution_gradient;
        std::optional<::cuda::device_buffer<std::uint8_t>> workspace;
        std::size_t workspace_bytes{}, temporary_count{};
        void attend(SpatialLayer& layer, const std::uint16_t* input, const float* gradient);
    };
} // namespace flowdit::neural
