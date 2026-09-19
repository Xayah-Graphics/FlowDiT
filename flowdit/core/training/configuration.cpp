module flowdit.training.configuration;
import std;
namespace flowdit {
    RunConfiguration training_configuration(const ImageSpecification& image) {
        RunConfiguration c;
        c.image              = image;
        c.tokenizer          = installed_tokenizer();
        c.model              = {{image.width / c.tokenizer.spatial_factor, image.height / c.tokenizer.spatial_factor, c.tokenizer.latent_channels}, static_cast<std::uint32_t>(image.classes.size())};
        c.preview.count      = 6;
        c.preview.step_count = 50;
        return c;
    }
} // namespace flowdit
