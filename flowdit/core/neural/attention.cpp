module;
#include <cudnn_frontend.h>
#include <flowdit/cuda.h>
module flowdit.neural.attention;
import std;
namespace flowdit::neural {
    namespace {
        void check(cudnn_frontend::error_t result) {
            if (!result.is_good()) throw std::runtime_error{"cuDNN attention: " + result.get_message()};
        }
        void check(cudnnStatus_t result) {
            if (result != CUDNN_STATUS_SUCCESS) throw std::runtime_error{cudnnGetErrorString(result)};
        }
    } // namespace
    Attention::Attention(::cuda::stream_ref stream, std::uint32_t b, std::uint32_t n, std::uint32_t w, std::uint32_t h, bool train) : batch{b}, width{w}, training{train} {
        check(cudnnCreate(&handle));
        check(cudnnSetStream(handle, stream.get()));
        const std::int64_t B = b, S = n, D = w / h, H = h, W = w;
        const std::vector<std::int64_t> dimensions{B, H, S, D}, qkv_stride{S * W * 3, D, W * 3, 1}, output_stride{S * W, D, W, 1};
        for (int pass = 0; pass < (training ? 2 : 1); ++pass) {
            auto& graph = pass == 0 ? forward_graph : backward_graph;
            graph.set_io_data_type(cudnn_frontend::DataType_t::BFLOAT16).set_intermediate_data_type(cudnn_frontend::DataType_t::FLOAT).set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);
            const auto q = graph.tensor(cudnn_frontend::graph::Tensor_attributes{}.set_name("Q").set_uid(1).set_dim(dimensions).set_stride(qkv_stride));
            const auto k = graph.tensor(cudnn_frontend::graph::Tensor_attributes{}.set_name("K").set_uid(2).set_dim(dimensions).set_stride(qkv_stride));
            const auto v = graph.tensor(cudnn_frontend::graph::Tensor_attributes{}.set_name("V").set_uid(3).set_dim(dimensions).set_stride(qkv_stride));
            if (pass == 0) {
                const auto [o, stats] = graph.sdpa(q, k, v, cudnn_frontend::graph::SDPA_attributes{}.set_name("full_attention").set_generate_stats(training).set_attn_scale(1.f / std::sqrt(static_cast<float>(D))));
                o->set_output(true).set_uid(4).set_dim(dimensions).set_stride(output_stride);
                if (training) stats->set_output(true).set_uid(5).set_data_type(cudnn_frontend::DataType_t::FLOAT);
            } else {
                const auto o            = graph.tensor(cudnn_frontend::graph::Tensor_attributes{}.set_name("O").set_uid(4).set_dim(dimensions).set_stride(output_stride));
                const auto stats        = graph.tensor(cudnn_frontend::graph::Tensor_attributes{}.set_name("stats").set_uid(5).set_dim({B, H, S, 1}).set_stride({H * S, S, 1, 1}).set_data_type(cudnn_frontend::DataType_t::FLOAT));
                const auto gradient     = graph.tensor(cudnn_frontend::graph::Tensor_attributes{}.set_name("dO").set_uid(6).set_dim(dimensions).set_stride(output_stride));
                const auto [dq, dk, dv] = graph.sdpa_backward(q, k, v, o, gradient, stats, cudnn_frontend::graph::SDPA_backward_attributes{}.set_name("full_attention_backward").set_attn_scale(1.f / std::sqrt(static_cast<float>(D))).set_deterministic_algorithm(true));
                dq->set_output(true).set_uid(7).set_dim(dimensions).set_stride(qkv_stride);
                dk->set_output(true).set_uid(8).set_dim(dimensions).set_stride(qkv_stride);
                dv->set_output(true).set_uid(9).set_dim(dimensions).set_stride(qkv_stride);
            }
            check(graph.build(handle, {cudnn_frontend::HeurMode_t::A}));
        }
        const auto bytes = std::max(forward_graph.get_workspace_size(), training ? backward_graph.get_workspace_size() : 0);
        workspace.emplace(stream, ::cuda::device_default_memory_pool(stream.device()), bytes, ::cuda::no_init);
    }
    Attention::~Attention() {
        cudnnDestroy(handle);
    }
    void Attention::forward(const std::uint16_t* qkv, std::uint16_t* output, float* statistics) {
        std::unordered_map<std::int64_t, void*> pointers{{1, const_cast<std::uint16_t*>(qkv)}, {2, const_cast<std::uint16_t*>(qkv + width)}, {3, const_cast<std::uint16_t*>(qkv + 2 * width)}, {4, output}};
        if (training) pointers.emplace(5, statistics);
        check(forward_graph.execute(handle, pointers, workspace->data()));
    }
    void Attention::backward(const std::uint16_t* qkv, const std::uint16_t* output, const float* statistics, const std::uint16_t* gradient, std::uint16_t* qkv_gradient) {
        std::unordered_map<std::int64_t, void*> pointers{{1, const_cast<std::uint16_t*>(qkv)}, {2, const_cast<std::uint16_t*>(qkv + width)}, {3, const_cast<std::uint16_t*>(qkv + 2 * width)}, {4, const_cast<std::uint16_t*>(output)}, {5, const_cast<float*>(statistics)}, {6, const_cast<std::uint16_t*>(gradient)}, {7, qkv_gradient}, {8, qkv_gradient + width}, {9, qkv_gradient + 2 * width}};
        check(backward_graph.execute(handle, pointers, workspace->data()));
    }
} // namespace flowdit::neural
