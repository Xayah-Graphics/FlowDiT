module;
#include <flowdit/cuda.h>
module flowdit.sampling.sampler;
import flowdit.serialization.safetensors;
import std;
namespace flowdit {
    namespace {
        std::vector<float> load_parameters(const std::filesystem::path& path) {
            const serialization::safetensors::File file = serialization::safetensors::read(path, std::array<std::string_view, 1>{"model.ema"});
            const auto& tensor                          = file.tensors.front();
            std::vector<float> result(tensor.data.size() / sizeof(float));
            std::memcpy(result.data(), tensor.data.data(), tensor.data.size());
            return result;
        }
    } // namespace
    Sampler::Sampler(const ModelConfiguration& source_configuration, const std::filesystem::path& checkpoint, const int device_ordinal, const std::uint32_t count) : configuration{source_configuration}, stream{::cuda::devices[device_ordinal]}, batch{count}, matmul{stream, flow_matmul_runtime_configuration}, model{stream, matmul, configuration}, parameters{stream, load_parameters(checkpoint)}, runtime{stream, model, batch} {}
    std::optional<SamplingResult> Sampler::sample(const SamplingRequest& request, const SamplingObserver& observer) {
        return runtime.sample(parameters.parameters.data(), request, observer);
    }
} // namespace flowdit
