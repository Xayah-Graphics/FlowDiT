export module flowdit.training.configuration;
export import flowdit.sampling.types;
export import flowdit.image.types;
export import flowdit.autoencoder.configuration;
export import flowdit.neural.training_state;
import std;
export namespace flowdit {
    enum class TrainingStage { autoencoder, flowdit };
    struct RunConfiguration final {
        TrainingStage stage{TrainingStage::autoencoder};
        AutoencoderConfiguration autoencoder;
        std::filesystem::path autoencoder_checkpoint;
        std::filesystem::path dataset, output;
        ModelConfiguration model;
        ImageSpecification image;
        std::uint32_t batch{1u};
        bool horizontal_flip{};
        std::uint64_t end_step{400'000u}, seed{42u};
        std::uint32_t log_interval{10u}, preview_interval{100u}, save_interval{1'000u};
        neural::TrainingConfiguration optimizer;
        SamplingRequest preview;
    };
    RunConfiguration training_configuration(const ImageSpecification& image, TrainingStage stage = TrainingStage::autoencoder);
} // namespace flowdit
