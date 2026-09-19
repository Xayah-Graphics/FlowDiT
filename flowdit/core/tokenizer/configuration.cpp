module;
#include <nlohmann/json.hpp>
module flowdit.tokenizer.configuration;
import std;
namespace flowdit {
    std::string serialize_tokenizer(const TokenizerSpecification& s) {
        return nlohmann::json{{"model", s.model}, {"repository", s.repository}, {"revision", s.revision}, {"source_digest", s.source_digest}, {"weights_digest", s.weights_digest}, {"input_channels", s.input_channels}, {"spatial_factor", s.spatial_factor}, {"latent_channels", s.latent_channels}, {"scaling_factor", s.scaling_factor}, {"channel_order", s.channel_order}, {"input_convention", s.input_convention}, {"prefixes", s.prefixes}}.dump();
    }
    TokenizerSpecification deserialize_tokenizer(std::string_view text) {
        const auto j = nlohmann::json::parse(text);
        return {j.at("model"), j.at("repository"), j.at("revision"), j.at("source_digest"), j.at("weights_digest"), j.at("input_channels"), j.at("spatial_factor"), j.at("latent_channels"), j.at("scaling_factor"), j.at("channel_order"), j.at("input_convention"), j.at("prefixes").get<std::vector<std::uint32_t>>()};
    }
    TokenizerSpecification installed_tokenizer() {
        const auto root = std::filesystem::path{FLOWDIT_ASSET_DIRECTORY} / "tokenizers/dc-ae-f32c32-sana-1.1";
        if (std::filesystem::exists(root))
            for (const auto& entry : std::filesystem::directory_iterator{root}) {
                if (!std::filesystem::exists(entry.path() / "tokenizer.json")) continue;
                std::ifstream file{entry.path() / "tokenizer.json"};
                const auto s = deserialize_tokenizer(nlohmann::json::parse(file).dump());
                if (s.revision == "6f7b3f3b289a439a11ef4fb1034989fd4b9a4766") return s;
            }
        throw std::runtime_error{"DC-AE assets are missing from " + root.string()};
    }
    std::filesystem::path tokenizer_directory(const TokenizerSpecification& s) {
        return std::filesystem::path{FLOWDIT_ASSET_DIRECTORY} / "tokenizers" / s.model / s.source_digest;
    }
} // namespace flowdit
