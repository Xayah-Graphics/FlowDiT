module;
#include <flowdit/cuda.h>
export module flowdit.representation.latent;
export import flowdit.tokenizer.configuration;
import flowdit.tokenizer.dcae;
import flowdit.image.transfer;
import std;
export namespace flowdit {
    struct LatentConfiguration final {
        TokenizerSpecification tokenizer;
        TensorShape image, shape;
        std::string cache_identity;
    };
    std::string serialize_latent(const LatentConfiguration& configuration);
    LatentConfiguration deserialize_latent(std::string_view text);
    struct LatentDecoder final {
        ::cuda::stream_ref stream;
        LatentConfiguration configuration;
        LatentDecoder(::cuda::stream_ref stream, LatentConfiguration configuration, std::uint32_t batch);
        const std::uint8_t* decode(const TensorBatch& tensor);

    private:
        DCAE model;
        ImageTransfer image;
        ::cuda::device_buffer<float> values, pixels;
        ::cuda::device_buffer<std::uint8_t> rgba;
    };
} // namespace flowdit
