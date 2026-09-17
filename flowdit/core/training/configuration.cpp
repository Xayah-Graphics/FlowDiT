module flowdit.training.configuration;
import std;
namespace flowdit {
    RunConfiguration training_configuration(const ImageSpecification& image) {
        RunConfiguration result;
        result.image = image;
        result.model = {{image.width, image.height, image.channels}, static_cast<std::uint32_t>(image.classes.size()), 2u};
        // Existing experiment presets; file-format detection belongs to dataset.
        if (image.name == "MNIST") result.end_step = 20'000;
        if (image.name == "CIFAR-10") result.horizontal_flip = true;
        return result;
    }
} // namespace flowdit
