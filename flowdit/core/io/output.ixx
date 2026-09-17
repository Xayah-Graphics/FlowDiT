export module flowdit.io.output;
export import flowdit.sampling.types;
export import flowdit.neural.training_state;
import std;
export namespace flowdit {
    struct RunConfiguration final {
        std::filesystem::path dataset, output;
        DatasetKind dataset_type{DatasetKind::cifar10};
        std::uint32_t patch_size{2u};
        std::uint64_t end_step{400'000u}, seed{42u};
        int device{};
        std::uint32_t execution_steps{10u}, log_interval{100u}, preview_interval{1'000u}, save_interval{50'000u};
        neural::TrainingConfiguration optimizer;
        SamplingRequest preview;
    };
    struct TrainingRecord final {
        std::uint64_t step{};
        float loss{};
        double samples_per_second{}, training_seconds{};
    };
    struct SampleInfo final {
        std::filesystem::path path, checkpoint;
        SamplingRequest request;
        ParameterSource source{ParameterSource::exponential_average};
        std::uint64_t training_step{};
        std::uint32_t nfe{};
        ModelConfiguration model;
        std::vector<std::uint32_t> labels;
    };
    struct SampleOutput final {
        SampleInfo info;
        SamplingResult images;
    };
    struct RunHistory final {
        std::vector<TrainingRecord> metrics;
        std::vector<SampleInfo> samples;
    };
} // namespace flowdit
export namespace flowdit::output {
    void write_configuration(const RunConfiguration& configuration);
    RunConfiguration read_configuration(const std::filesystem::path& directory);
    void write_png(const std::filesystem::path& path, const SamplingResult& images, std::optional<std::size_t> index = {});
    void write_sample(const SampleOutput& sample);
    SampleOutput read_sample(const std::filesystem::path& path);
    RunHistory read_history(const std::filesystem::path& directory);
    std::string_view solver_name(SamplingSolver solver);
} // namespace flowdit::output
