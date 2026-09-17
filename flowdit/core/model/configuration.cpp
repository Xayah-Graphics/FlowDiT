module;
#include <nlohmann/json.hpp>
module flowdit.model.configuration;
import std;
namespace flowdit {
    std::string serialize_model(const ModelConfiguration& configuration) {
        const auto& shape = configuration.shape;
        return nlohmann::json{{"width", shape.width}, {"height", shape.height}, {"channels", shape.channels}, {"class_count", configuration.class_count}, {"patch_size", configuration.patch_size}, {"architecture", "flowdit-256-8-8-1024"}}.dump();
    }
    ModelConfiguration deserialize_model(const std::string_view text) {
        const auto json = nlohmann::json::parse(text);
        if (json.at("architecture") != "flowdit-256-8-8-1024") throw std::runtime_error{"Unsupported model architecture"};
        return {{json.at("width"), json.at("height"), json.at("channels")}, json.at("class_count"), json.at("patch_size")};
    }
} // namespace flowdit
