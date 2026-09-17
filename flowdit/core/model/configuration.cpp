module;
#include <nlohmann/json.hpp>
module flowdit.model.configuration;
import flowdit.serialization.safetensors;
import std;
namespace flowdit {
    std::string serialize_model(const ModelConfiguration& configuration) {
        const auto& image = configuration.image;
        return nlohmann::json{{"dataset", image.name}, {"width", image.width}, {"height", image.height}, {"channels", image.channels}, {"classes", image.classes}, {"patch_size", configuration.patch_size}}.dump();
    }
    ModelConfiguration deserialize_model(const std::string_view text) {
        const auto json = nlohmann::json::parse(text);
        return {{json.at("dataset").get<std::string>(), json.at("width"), json.at("height"), json.at("channels"), json.at("classes").get<std::vector<std::string>>()}, json.at("patch_size")};
    }
    ModelConfiguration read_model_configuration(const std::filesystem::path& checkpoint) {
        return deserialize_model(serialization::safetensors::read_metadata(checkpoint).at("flowdit.model"));
    }
} // namespace flowdit
