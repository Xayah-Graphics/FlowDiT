export module flowdit.tokenizer.configuration;
export import flowdit.tensor.types;
import std;
export namespace flowdit {
    struct TokenizerSpecification final {
        std::string model, repository, revision, source_digest, weights_digest;
        std::uint32_t input_channels{}, spatial_factor{}, latent_channels{};
        float scaling_factor{};
        std::string channel_order, input_convention;
        std::vector<std::uint32_t> prefixes;
    };
    std::string serialize_tokenizer(const TokenizerSpecification& specification);
    TokenizerSpecification deserialize_tokenizer(std::string_view text);
    TokenizerSpecification installed_tokenizer();
    std::filesystem::path tokenizer_directory(const TokenizerSpecification& specification);
} // namespace flowdit
