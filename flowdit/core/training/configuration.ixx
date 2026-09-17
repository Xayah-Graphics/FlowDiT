export module flowdit.training.configuration;
export import flowdit.sampling.types;
export import flowdit.image.types;
export import flowdit.neural.training_state;
import std;
export namespace flowdit {
    struct RunConfiguration final {
        std::filesystem::path dataset, output;
        ModelConfiguration model;
        ImageSpecification image;
        std::uint32_t batch{256u};
        bool horizontal_flip{};
        std::uint64_t end_step{400'000u}, seed{42u};
        inline static constexpr std::uint32_t execution_steps{10u}, log_interval{100u}, preview_interval{1'000u}, save_interval{50'000u};
        neural::TrainingConfiguration optimizer;
        SamplingRequest preview;
    };
    RunConfiguration training_configuration(const ImageSpecification& image);
} // namespace flowdit
