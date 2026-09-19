export module flowdit.sampling.types;
export import flowdit.model.configuration;
import std;
export namespace flowdit {
    enum class ParameterSource : std::uint8_t {
        parameters,
        exponential_average,
    };
    enum class SamplingSolver : std::uint8_t {
        euler,
        heun,
        rk4,
    };
    struct SamplingRequest final {
        SamplingSolver solver{SamplingSolver::heun};
        std::uint32_t step_count{50u}, count{100u};
        std::uint32_t first_sample{};
        float guidance{2.0F};
        std::uint64_t seed{42u};
        std::optional<std::uint32_t> class_index;
    };
    struct SamplingResult final {
        TensorBatch tensor;
        std::uint32_t nfe{};
    };
    struct SamplingProgress final {
        std::uint32_t step{}, step_count{}, nfe{};
    };
} // namespace flowdit
