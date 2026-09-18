module;
#include <flowdit/cuda.h>
export module flowdit.autoencoder.model;
export import flowdit.autoencoder.configuration;
export import flowdit.neural.spatial;
import std;
export namespace flowdit {
    struct Encoder final : neural::SpatialNetwork {
        Encoder(::cuda::stream_ref stream, const AutoencoderConfiguration& configuration, std::uint32_t batch);
        void encode(const float* parameters, const float* images, float* moments);
    };
    struct Decoder final : neural::SpatialNetwork {
        Decoder(::cuda::stream_ref stream, const AutoencoderConfiguration& configuration, std::uint32_t batch);
        void decode(const float* parameters, const float* latent, float* images);
    };
    struct Autoencoder final {
        AutoencoderConfiguration configuration;
        Encoder encoder;
        Decoder decoder;
        std::size_t encoder_parameters;
        Autoencoder(::cuda::stream_ref stream, const AutoencoderConfiguration& configuration, std::uint32_t batch);
        std::vector<float> initialize(std::uint64_t seed) const;
    };
} // namespace flowdit
