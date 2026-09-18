export module flowdit.autoencoder.configuration;
export import flowdit.image.types;
import std;
export namespace flowdit {
    struct AutoencoderConfiguration final {
        TensorShape image;
        std::vector<std::uint32_t> widths;
        std::uint32_t latent_channels{4}, residual_blocks{2};
        std::uint32_t accumulation{1};
        float perceptual_weight{0.1F}, kl_weight{1e-4F}, adversarial_weight{0.01F};
        std::uint64_t adversarial_start{1000};
        TensorShape latent_shape() const;
    };
    std::string serialize_autoencoder(const AutoencoderConfiguration& configuration);
    AutoencoderConfiguration deserialize_autoencoder(std::string_view text);
} // namespace flowdit
