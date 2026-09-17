module;
#include <flowdit/cuda.h>
export module flowdit.sampling.runtime;
export import flowdit.sampling.types;
import flowdit.model;
import std;
export namespace flowdit {
    struct SamplingObserver final {
        std::stop_token stop;
        std::function<void(const SamplingProgress&)> progress;
        std::function<void(const std::uint8_t*, std::uint32_t, std::uint32_t, ::cuda::stream_ref)> image;
    };
    struct SamplingRuntime final {
        SamplingRuntime(::cuda::stream_ref stream, FlowDiT& model);
        std::optional<SamplingResult> sample(const float* parameters, const SamplingRequest& request, const SamplingObserver& observer = {});

    private:
        SamplingRequest request;
        SamplingProgress progress;
        inline static constexpr std::uint32_t batch = 100u;
        std::size_t value_count;
        ::cuda::stream_ref stream;
        FlowDiT& model;
        FlowDiTWorkspaceLayout model_workspace_layout;
        ::cuda::device_buffer<std::uint8_t> model_workspace;
        ::cuda::device_buffer<float> state;
        ::cuda::device_buffer<float> times;
        ::cuda::device_buffer<std::uint32_t> labels;
        ::cuda::device_buffer<std::uint32_t> null_labels;
        ::cuda::device_buffer<float> conditional_velocity;
        ::cuda::device_buffer<float> unconditional_velocity;
        std::optional<::cuda::device_buffer<float>> stages;
        ::cuda::device_buffer<std::uint8_t> image_rgba;
        void initialize(const SamplingRequest& request);
        void advance(const float* parameters);
        void publish(const SamplingObserver& observer);
        SamplingResult finish(const SamplingObserver& observer);
        void evaluate(const float* parameters, const float* input, float time, float guidance, float* output);
    };
} // namespace flowdit
