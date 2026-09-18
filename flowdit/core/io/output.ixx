export module flowdit.io.output;
export import flowdit.sampling.types;
export import flowdit.training.configuration;
export import flowdit.image.transfer;
export import flowdit.representation.latent;
import std;
export namespace flowdit {
    struct TrainingRecord final {
        std::uint64_t step{};
        float loss{};
        double samples_per_second{}, training_seconds{};
        std::array<float, 4> components{};
    };
    struct SampleInfo final {
        std::filesystem::path path, checkpoint;
        SamplingRequest request;
        bool reconstruction{};
        ParameterSource source{ParameterSource::exponential_average};
        std::uint64_t training_step{};
        std::uint32_t nfe{};
        ModelConfiguration model;
        ImageSpecification image;
        std::vector<std::uint32_t> labels;
    };
    struct SampleOutput final {
        SampleInfo info;
        std::vector<std::uint8_t> rgba;
    };
    struct RunHistory final {
        std::vector<TrainingRecord> metrics;
        std::filesystem::path preview;
    };
} // namespace flowdit
export namespace flowdit::output {
    void write_configuration(const RunConfiguration& configuration);
    RunConfiguration read_configuration(const std::filesystem::path& directory);
    void write_sample(const SampleOutput& sample);
    SampleOutput read_sample(const std::filesystem::path& path);
    RunHistory read_history(const std::filesystem::path& directory);
} // namespace flowdit::output
