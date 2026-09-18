module;
#include <flowdit/cuda.h>
export module flowdit.representation.latent;
export import flowdit.autoencoder.model;
import flowdit.image.transfer;
import std;
export namespace flowdit {
    struct LatentConfiguration final {
        std::filesystem::path checkpoint;
        std::string identity, cache_identity;
        AutoencoderConfiguration autoencoder;
        std::vector<float> mean, deviation;
    };
    std::string serialize_latent(const LatentConfiguration& configuration);
    LatentConfiguration deserialize_latent(std::string_view text);
    struct LatentDecoder final {
        ::cuda::stream_ref stream;
        LatentConfiguration configuration;
        std::uint32_t batch;
        LatentDecoder(::cuda::stream_ref stream, LatentConfiguration configuration, std::uint32_t batch, const std::filesystem::path& dataset);
        const std::uint8_t* decode(const TensorBatch& tensor);

    private:
        Decoder model;
        ImageTransfer image;
        ::cuda::device_buffer<float> parameters, values, pixels, mean, deviation;
        ::cuda::device_buffer<std::uint8_t> rgba;
    };
} // namespace flowdit
