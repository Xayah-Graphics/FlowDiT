module;
#include <nlohmann/json.hpp>
module flowdit.model.configuration;
import std;
namespace flowdit {
    std::string serialize_model(const ModelConfiguration& c) {
        return nlohmann::json{{"architecture", "usit-v1"}, {"shape", {c.shape.width, c.shape.height, c.shape.channels}}, {"class_count", c.class_count}, {"patch_size", c.patch_size}, {"width", c.width}, {"heads", c.heads}, {"side_blocks", c.side_blocks}, {"mlp_width", c.mlp_width}}.dump();
    }
    ModelConfiguration deserialize_model(std::string_view text) {
        const auto j = nlohmann::json::parse(text);
        if (j.at("architecture") != "usit-v1") throw std::runtime_error{"Unsupported model architecture"};
        return {{j.at("shape").at(0), j.at("shape").at(1), j.at("shape").at(2)}, j.at("class_count"), j.at("patch_size"), j.at("width"), j.at("heads"), j.at("side_blocks"), j.at("mlp_width")};
    }
} // namespace flowdit
