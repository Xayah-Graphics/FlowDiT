module;
#include "../neural/spatial-kernels.h"
#include <cublas_v2.h>
#include <cudnn.h>
#include <flowdit/cuda.h>
#include <nlohmann/json.hpp>
module flowdit.representation.latent;
import flowdit.serialization.safetensors;
import flowdit.serialization.digest;
import std;
namespace flowdit {
    std::string serialize_latent(const LatentConfiguration& c) {
        return nlohmann::json{{"type", "kl"}, {"checkpoint", c.checkpoint.generic_string()}, {"identity", c.identity}, {"cache_identity", c.cache_identity}, {"autoencoder", nlohmann::json::parse(serialize_autoencoder(c.autoencoder))}, {"mean", c.mean}, {"deviation", c.deviation}}.dump();
    }
    LatentConfiguration deserialize_latent(std::string_view text) {
        const auto j = nlohmann::json::parse(text);
        if (j.at("type") != "kl") throw std::runtime_error{"Unsupported latent representation"};
        return {j.at("checkpoint").get<std::string>(), j.at("identity").get<std::string>(), j.at("cache_identity").get<std::string>(), deserialize_autoencoder(j.at("autoencoder").dump()), j.at("mean").get<std::vector<float>>(), j.at("deviation").get<std::vector<float>>()};
    }
    LatentDecoder::LatentDecoder(::cuda::stream_ref source, LatentConfiguration c, std::uint32_t count, const std::filesystem::path& dataset)
        : stream{source}, configuration{std::move(c)}, batch{count}, model{stream, configuration.autoencoder, 1}, image{stream, configuration.autoencoder.image, 1}, parameters{stream, ::cuda::device_default_memory_pool(stream.device()), model.initial.size(), ::cuda::no_init}, values{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(1) * configuration.autoencoder.latent_shape().width * configuration.autoencoder.latent_shape().height * configuration.autoencoder.latent_channels, ::cuda::no_init}, pixels{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(configuration.autoencoder.image.width) * configuration.autoencoder.image.height * configuration.autoencoder.image.channels, ::cuda::no_init}, mean{stream, ::cuda::device_default_memory_pool(stream.device()), configuration.mean.size(), ::cuda::no_init}, deviation{stream, ::cuda::device_default_memory_pool(stream.device()), configuration.deviation.size(), ::cuda::no_init},
          rgba{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(batch) * configuration.autoencoder.image.width * configuration.autoencoder.image.height * 4, ::cuda::no_init} {
        const auto path = dataset / configuration.checkpoint;
        if (serialization::digest_file(path) != configuration.identity) throw std::runtime_error{"The bound autoencoder checkpoint has changed"};
        const auto file = serialization::safetensors::read(path, std::array<std::string_view, 1>{"decoder.parameters"});
        ::cuda::copy_bytes(stream, ::cuda::std::span<const std::byte>{file.tensors.front().data.data(), file.tensors.front().data.size()}, parameters);
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{configuration.mean.data(), configuration.mean.size()}, mean);
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{configuration.deviation.data(), configuration.deviation.size()}, deviation);
        stream.sync();
    }
    const std::uint8_t* LatentDecoder::decode(const TensorBatch& tensor) {
        const std::size_t bytes = static_cast<std::size_t>(configuration.autoencoder.image.width) * configuration.autoencoder.image.height * 4;
        for (std::uint32_t i = 0; i < tensor.count; ++i) {
            kernels::standardize(stream, tensor.values + i * values.size(), values.data(), mean.data(), deviation.data(), 1, tensor.shape.channels, tensor.shape.width * tensor.shape.height, true);
            model.decode(parameters.data(), values.data(), pixels.data());
            const auto* decoded = image.decode({configuration.autoencoder.image, 1, pixels.data(), tensor.labels + i});
            ::cuda::copy_bytes(stream, ::cuda::std::span<const std::uint8_t>{decoded, bytes}, ::cuda::std::span<std::uint8_t>{rgba.data() + i * bytes, bytes});
        }
        return rgba.data();
    }
} // namespace flowdit
