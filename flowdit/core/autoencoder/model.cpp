module;
#include "../neural/spatial-kernels.h"
#include <cublas_v2.h>
#include <cudnn.h>
#include <flowdit/cuda.h>
module flowdit.autoencoder.model;
import std;
namespace flowdit {
    Encoder::Encoder(::cuda::stream_ref stream, const AutoencoderConfiguration& config, std::uint32_t batch) : SpatialNetwork{stream, batch, config.image} {
        append(neural::SpatialOperation::convolution, config.widths.front());
        for (std::size_t level = 0; level < config.widths.size(); ++level) {
            for (std::uint32_t block = 0; block < config.residual_blocks; ++block) residual(config.widths[level]);
            if (level + 1 < config.widths.size()) append(neural::SpatialOperation::convolution, config.widths[level], 3, 2, 1);
        }
        residual(config.widths.back());
        attention();
        residual(config.widths.back());
        append(neural::SpatialOperation::normalization);
        append(neural::SpatialOperation::silu);
        append(neural::SpatialOperation::convolution, config.latent_channels * 2);
    }
    void Encoder::encode(const float* parameters, const float* images, float* moments) {
        const auto* values = forward(parameters, images);
        kernels::unpack(stream, values, moments, layers.back().values->size());
    }
    Decoder::Decoder(::cuda::stream_ref stream, const AutoencoderConfiguration& config, std::uint32_t batch) : SpatialNetwork{stream, batch, config.latent_shape()} {
        append(neural::SpatialOperation::convolution, config.widths.back());
        residual(config.widths.back());
        attention();
        residual(config.widths.back());
        for (std::size_t level = config.widths.size(); level-- > 0;) {
            for (std::uint32_t block = 0; block <= config.residual_blocks; ++block) residual(config.widths[level]);
            if (level > 0) {
                append(neural::SpatialOperation::upsample);
                append(neural::SpatialOperation::convolution, config.widths[level]);
            }
        }
        append(neural::SpatialOperation::normalization);
        append(neural::SpatialOperation::silu);
        append(neural::SpatialOperation::convolution, config.image.channels);
    }
    void Decoder::decode(const float* parameters, const float* latent, float* images) {
        const auto* values = forward(parameters, latent);
        kernels::unpack(stream, values, images, layers.back().values->size());
    }
    Autoencoder::Autoencoder(::cuda::stream_ref stream, const AutoencoderConfiguration& config, std::uint32_t batch) : configuration{config}, encoder{stream, config, batch}, decoder{stream, config, batch}, encoder_parameters{encoder.initial.size()} {}
    std::vector<float> Autoencoder::initialize(std::uint64_t seed) const {
        auto result     = encoder.initialize(seed);
        const auto tail = decoder.initialize(seed + 1);
        result.insert(result.end(), tail.begin(), tail.end());
        return result;
    }
} // namespace flowdit
