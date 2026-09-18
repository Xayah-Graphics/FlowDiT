module flowdit.training.configuration;
import std;
namespace flowdit {
    RunConfiguration training_configuration(const ImageSpecification& image, const TrainingStage stage) {
        RunConfiguration result;
        result.stage         = stage;
        result.image         = image;
        auto& c              = result.autoencoder;
        c.image              = {image.width, image.height, image.channels};
        const bool large     = std::min(image.width, image.height) >= 256;
        c.widths             = large ? std::vector<std::uint32_t>{128, 256, 512, 512} : image.channels == 1 ? std::vector<std::uint32_t>{32, 64} : std::vector<std::uint32_t>{64, 128};
        c.latent_channels    = image.channels == 1 ? 2 : 4;
        c.perceptual_weight  = image.channels == 1 ? .01f : .1f;
        c.adversarial_weight = image.channels == 1 ? .001f : .01f;
        result.model         = {c.latent_shape(), static_cast<std::uint32_t>(image.classes.size()), 2};
        // RTX 5090 defaults measured with training previews included in peak memory.
        if (stage == TrainingStage::autoencoder) result.batch = large ? 3 : image.channels == 1 ? 512 : 256;
        else result.batch = large ? 16 : 256;
        result.end_step                = stage == TrainingStage::autoencoder ? 20'000 : image.channels == 1 ? 20'000 : 400'000;
        result.horizontal_flip         = image.channels == 3;
        result.preview.count           = large ? 6 : 20;
        result.optimizer.learning_rate = stage == TrainingStage::autoencoder ? 5e-5f : 1e-4f;
        if (stage == TrainingStage::flowdit) {
            result.log_interval     = 100;
            result.preview_interval = 1'000;
            result.save_interval    = 50'000;
        }
        return result;
    }
} // namespace flowdit
