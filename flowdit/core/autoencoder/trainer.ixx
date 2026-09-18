module;
#include <flowdit/cuda.h>
export module flowdit.autoencoder.trainer;
export import flowdit.autoencoder.model;
export import flowdit.training.state;
import flowdit.neural.training_state;
import flowdit.autoencoder.perceptual;
import std;
export namespace flowdit {
    struct AutoencoderMetrics final {
        float reconstruction{}, kl{}, perceptual{}, generator{}, discriminator{};
    };
    struct AutoencoderTrainer final {
        ::cuda::stream_ref stream;
        std::uint32_t batch;
        TrainingState state;
        Autoencoder model;
        neural::ParameterBuffer parameters;
        AutoencoderTrainer(::cuda::stream_ref stream, const AutoencoderConfiguration& configuration, std::uint32_t batch, std::uint64_t seed, const neural::TrainingConfiguration& optimizer, const std::filesystem::path& perceptual_weights);
        AutoencoderMetrics optimize(const TensorBatch& input);
        const float* reconstruct(const TensorBatch& input);
        void save(const std::filesystem::path& path, std::map<std::string, std::string> metadata) const;
        void load(const std::filesystem::path& path);

    private:
        neural::TrainingConfiguration optimizer;
        neural::SpatialNetwork discriminator;
        std::optional<neural::ParameterBuffer> discriminator_parameters;
        PerceptualLoss perceptual;
        std::uint32_t microstep{};
        ::cuda::device_buffer<float> moments, latent, reconstructed, reconstruction_gradient, moments_gradient, losses, logits, logits_gradient;
        ::cuda::device_buffer<std::uint64_t> device_state;
    };
} // namespace flowdit
