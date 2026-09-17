module;
#include <flowdit/cuda.h>
module flowdit.sampling.sampler;
import flowdit.serialization.safetensors;
import std;
namespace flowdit {
    namespace {
        std::vector<float> load_parameters(const std::filesystem::path& path, const ParameterSource source) {
            const serialization::safetensors::File file = serialization::safetensors::read(path);
            const auto tensor                           = std::ranges::find(file.tensors, std::string{source == ParameterSource::parameters ? "model.parameters" : "model.ema"}, &serialization::safetensors::Tensor::name);
            std::vector<float> result(tensor->data.size() / sizeof(float));
            std::memcpy(result.data(), tensor->data.data(), tensor->data.size());
            return result;
        }
    } // namespace
    Sampler::Sampler(const std::filesystem::path& checkpoint, const int device_ordinal, const ParameterSource source) : configuration{read_model_configuration(checkpoint)}, stream{::cuda::devices[device_ordinal]}, matmul{stream, flow_matmul_runtime_configuration}, model{stream, matmul, configuration}, checkpoint_values{load_parameters(checkpoint, source)}, parameters{stream, checkpoint_values}, runtime{stream, model} {}
    std::optional<SamplingResult> Sampler::sample(const SamplingRequest& request, const SamplingObserver& observer) {
        return runtime.sample(parameters.parameters.data(), request, observer);
    }
} // namespace flowdit
