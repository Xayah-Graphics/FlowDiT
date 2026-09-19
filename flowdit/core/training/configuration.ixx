export module flowdit.training.configuration;
export import flowdit.sampling.types;
export import flowdit.image.types;
export import flowdit.tokenizer.configuration;
export import flowdit.neural.training_state;
import std;
export namespace flowdit {
    struct RunConfiguration final {
        TokenizerSpecification tokenizer;
        std::filesystem::path dataset, output;
        ModelConfiguration model;
        ImageSpecification image;
        std::uint32_t batch{16}, accumulation{4};
        bool horizontal_flip{true};
        std::uint64_t end_step{20'000}, seed{42};
        std::uint32_t log_interval{10}, preview_interval{1'000}, save_interval{1'000};
        neural::TrainingConfiguration optimizer;
        SamplingRequest preview;
    };
    RunConfiguration training_configuration(const ImageSpecification& image);
} // namespace flowdit
