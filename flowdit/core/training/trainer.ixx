module;
#include <cuda_runtime_api.h>
#include <flowdit/cuda.h>
export module flowdit.training.trainer;
import std;
import flowdit.dataset.types;
import flowdit.model;
export import flowdit.sampling.runtime;
import flowdit.neural.matmul;
export import flowdit.neural.training_state;
export namespace flowdit {
    struct TrainingState final {
        std::uint64_t step{};
        std::uint64_t processed_samples{};
        std::uint64_t seed{};
        double elapsed_seconds{};
    };
    struct TrainingStatistics final {
        float average_loss;
        double elapsed_seconds;
    };
    struct Trainer final {
        inline static constexpr std::uint32_t batch = 256u;
        TrainingState state;
        Trainer(const Dataset& training_set, int device_ordinal, std::uint64_t seed, const neural::TrainingConfiguration& configuration = {}, std::uint32_t patch_size = 2u);
        ~Trainer() noexcept;
        Trainer(const Trainer&)            = delete;
        Trainer& operator=(const Trainer&) = delete;
        Trainer(Trainer&&)                 = delete;
        Trainer& operator=(Trainer&&)      = delete;
        TrainingStatistics optimize(std::uint64_t iterations);
        std::optional<SamplingResult> sample(const SamplingRequest& request, ParameterSource source = ParameterSource::exponential_average, const SamplingObserver& observer = {});
        void save(const std::filesystem::path& path) const;
        void load(const std::filesystem::path& path);

    private:
        std::size_t value_count;
        std::uint32_t image_count;
        bool horizontal_flip;
        ::cuda::stream stream;
        ::cuda::device_buffer<std::uint8_t> dataset_images;
        ::cuda::device_buffer<std::uint32_t> dataset_labels;
        neural::MatmulRuntime matmul;
        FlowDiT model;
        neural::ParameterBuffer parameter_buffer;
        neural::TrainingConfiguration training_configuration;
        FlowDiTWorkspaceLayout model_workspace_layout;
        ::cuda::device_buffer<std::uint8_t> model_workspace;
        ::cuda::device_buffer<float> path;
        ::cuda::device_buffer<float> target;
        ::cuda::device_buffer<float> times;
        ::cuda::device_buffer<std::uint32_t> labels;
        ::cuda::device_buffer<float> patch_gradient;
        ::cuda::device_buffer<float> loss;
        ::cuda::device_buffer<float> loss_sum;
        ::cuda::device_buffer<std::uint64_t> device_step;
        ::cuda::device_buffer<std::uint64_t> device_processed_samples;
        ::cuda::device_buffer<std::uint64_t> device_seed;
        cudaGraph_t graph{};
        cudaGraphExec_t graph_execution{};
        void training_step();
    };
} // namespace flowdit
