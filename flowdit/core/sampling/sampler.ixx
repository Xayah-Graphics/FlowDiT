module;
#include <flowdit/cuda.h>
export module flowdit.sampling.sampler;
export import flowdit.sampling.runtime;
import flowdit.model;
import flowdit.neural.matmul;
import flowdit.neural.training_state;
import std;
export namespace flowdit {
    struct Sampler final {
        ModelConfiguration configuration;
        Sampler(const std::filesystem::path& checkpoint, int device_ordinal);
        std::optional<SamplingResult> sample(const SamplingRequest& request, const SamplingObserver& observer = {});

    private:
        ::cuda::stream stream;
        neural::MatmulRuntime matmul;
        FlowDiT model;
        neural::InferenceParameterBuffer parameters;
        SamplingRuntime runtime;
    };
} // namespace flowdit
