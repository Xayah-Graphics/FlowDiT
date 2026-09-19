module;
#include <cublas_v2.h>
#include <cudnn.h>
#include <flowdit/cuda.h>
export module flowdit.tokenizer.dcae;
export import flowdit.tokenizer.configuration;
import std;
export namespace flowdit {
    enum class TokenizerDirection { encode, decode };
    enum class TokenizerOperation { input, convolution, rms, silu, relu, add, glu, down_average, nearest, up_duplicate, average, duplicate, attention };
    struct TokenizerConvolution final {
        cudnnTensorDescriptor_t input{}, output{};
        cudnnFilterDescriptor_t filter{};
        cudnnConvolutionDescriptor_t operation{};
        cudnnConvolutionFwdAlgo_t algorithm{};
        std::size_t workspace{};
        TokenizerConvolution(cudnnHandle_t handle, TensorShape source, TensorShape target, std::uint32_t kernel, std::uint32_t stride, std::uint32_t groups);
        ~TokenizerConvolution();
        TokenizerConvolution(const TokenizerConvolution&)            = delete;
        TokenizerConvolution& operator=(const TokenizerConvolution&) = delete;
    };
    struct TokenizerLayer final {
        TokenizerOperation operation;
        TensorShape shape;
        std::size_t source{}, skip{}, weights{}, bias{}, offset{}, elements{};
        bool has_bias{};
        std::string name;
        std::unique_ptr<TokenizerConvolution> convolution;
    };
    // A fixed pretrained network, evaluated one image at a time. The arena reuses
    // storage after the last consumer; it never allocates training state.
    struct DCAE final {
        ::cuda::stream_ref stream;
        TokenizerSpecification specification;
        TokenizerDirection direction;
        std::vector<TokenizerLayer> layers;
        cudnnHandle_t cudnn{};
        cublasHandle_t blas{};
        DCAE(::cuda::stream_ref stream, TokenizerSpecification specification, TensorShape image, TokenizerDirection direction);
        ~DCAE();
        DCAE(const DCAE&)            = delete;
        DCAE& operator=(const DCAE&) = delete;
        const std::uint16_t* forward(const float* input);

    private:
        std::optional<::cuda::device_buffer<float>> parameters, attention_workspace;
        std::optional<::cuda::device_buffer<std::uint16_t>> weights, arena;
        std::optional<::cuda::device_buffer<std::uint8_t>> workspace;
        std::size_t append(TokenizerOperation operation, std::size_t source, std::uint32_t channels = 0, std::size_t skip = 0, std::string name = {}, std::uint32_t stride = 1, std::uint32_t groups = 1);
        std::size_t block(std::size_t source, const std::string& name, bool attention);
    };
} // namespace flowdit
