module;
#include <cudnn_frontend.h>
#include <flowdit/cuda.h>
export module flowdit.neural.attention;
import std;
export namespace flowdit::neural {
    struct Attention final {
        std::uint32_t batch, width;
        bool training;
        cudnnHandle_t handle{};
        cudnn_frontend::graph::Graph forward_graph, backward_graph;
        std::optional<::cuda::device_buffer<std::uint8_t>> workspace;
        Attention(::cuda::stream_ref stream, std::uint32_t batch, std::uint32_t sequence, std::uint32_t width, std::uint32_t heads, bool training);
        ~Attention();
        Attention(const Attention&)            = delete;
        Attention& operator=(const Attention&) = delete;
        void forward(const std::uint16_t* qkv, std::uint16_t* output, float* statistics);
        void backward(const std::uint16_t* qkv, const std::uint16_t* output, const float* statistics, const std::uint16_t* gradient, std::uint16_t* qkv_gradient);
    };
} // namespace flowdit::neural
