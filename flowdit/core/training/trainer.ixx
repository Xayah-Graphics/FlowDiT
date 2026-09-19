module;
#include <cuda_runtime_api.h>
#include <flowdit/cuda.h>
export module flowdit.training.trainer;
import std;
import flowdit.model;
import flowdit.neural.matmul;
export import flowdit.neural.training_state;
export import flowdit.training.state;
export namespace flowdit {
    struct Trainer final {
        std::uint32_t batch, accumulation;
        TrainingState state;
        Trainer(::cuda::stream_ref stream, const ModelConfiguration& model, std::uint32_t batch, std::uint64_t seed, const neural::TrainingConfiguration& configuration, std::uint32_t accumulation = 1);
        ~Trainer() noexcept;
        Trainer(const Trainer&)            = delete;
        Trainer& operator=(const Trainer&) = delete;
        Trainer(Trainer&&)                 = delete;
        Trainer& operator=(Trainer&&)      = delete;
        float optimize(const TensorBatch& input);
        void save(const std::filesystem::path& path, std::map<std::string, std::string> metadata) const;
        void load(const std::filesystem::path& path);

    private:
        std::size_t value_count;
        ::cuda::stream_ref stream;
        ::cuda::device_buffer<float> input_values;
        ::cuda::device_buffer<std::uint32_t> input_labels;
        neural::MatmulRuntime matmul;

    public:
        FlowDiT model;
        neural::ParameterBuffer parameter_buffer;

    private:
        neural::TrainingConfiguration training_configuration;
        std::string training_identity;
        FlowDiTWorkspaceLayout model_workspace_layout;
        ::cuda::device_buffer<std::uint8_t> model_workspace;
        ::cuda::device_buffer<float> path;
        ::cuda::device_buffer<float> target;
        ::cuda::device_buffer<float> times;
        ::cuda::device_buffer<std::uint32_t> labels;
        ::cuda::device_buffer<float> loss;
        ::cuda::device_buffer<float> sample_loss;
        ::cuda::device_buffer<std::uint64_t> device_step;
        ::cuda::device_buffer<std::uint64_t> device_processed_samples;
        ::cuda::device_buffer<std::uint64_t> device_seed;
        ::cuda::device_buffer<std::uint64_t> device_microstep;
        cudaGraph_t graph{};
        cudaGraphExec_t graph_execution{};
        void training_step();
    };
} // namespace flowdit
