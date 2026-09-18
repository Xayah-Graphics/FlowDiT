module;
#include <nlohmann/json.hpp>
module flowdit.autoencoder.configuration;
import std;
namespace flowdit {
    TensorShape AutoencoderConfiguration::latent_shape() const {
        const auto scale = 1u << (widths.size() - 1);
        return {image.width / scale, image.height / scale, latent_channels};
    }
    std::string serialize_autoencoder(const AutoencoderConfiguration& c) {
        return nlohmann::json{{"architecture", "residual-kl"}, {"image", {c.image.width, c.image.height, c.image.channels}}, {"widths", c.widths}, {"latent_channels", c.latent_channels}, {"residual_blocks", c.residual_blocks}, {"accumulation", c.accumulation}, {"perceptual_weight", c.perceptual_weight}, {"kl_weight", c.kl_weight}, {"adversarial_weight", c.adversarial_weight}, {"adversarial_start", c.adversarial_start}}.dump();
    }
    AutoencoderConfiguration deserialize_autoencoder(std::string_view text) {
        const auto j = nlohmann::json::parse(text);
        if (j.at("architecture") != "residual-kl") throw std::runtime_error{"Unsupported autoencoder architecture"};
        return {{j.at("image").at(0), j.at("image").at(1), j.at("image").at(2)}, j.at("widths").get<std::vector<std::uint32_t>>(), j.at("latent_channels"), j.at("residual_blocks"), j.at("accumulation"), j.at("perceptual_weight"), j.at("kl_weight"), j.at("adversarial_weight"), j.at("adversarial_start")};
    }
} // namespace flowdit
