module;
#include "../tokenizer/kernels.h"
#include <flowdit/cuda.h>
#include <nlohmann/json.hpp>
module flowdit.representation.latent;
import std;
namespace flowdit {
    std::string serialize_latent(const LatentConfiguration& c) {
        return nlohmann::json{{"type", "deterministic-dc-ae-v1"}, {"tokenizer", nlohmann::json::parse(serialize_tokenizer(c.tokenizer))}, {"image", {c.image.width, c.image.height, c.image.channels}}, {"shape", {c.shape.width, c.shape.height, c.shape.channels}}, {"layout", "NCHW"}, {"cache_dtype", "BF16"}, {"cache_scaling", "raw"}, {"cache_identity", c.cache_identity}}.dump();
    }
    LatentConfiguration deserialize_latent(std::string_view text) {
        const auto j = nlohmann::json::parse(text);
        if (j.at("type") != "deterministic-dc-ae-v1") throw std::runtime_error{"Unsupported latent representation"};
        return {deserialize_tokenizer(j.at("tokenizer").dump()), {j.at("image").at(0), j.at("image").at(1), j.at("image").at(2)}, {j.at("shape").at(0), j.at("shape").at(1), j.at("shape").at(2)}, j.at("cache_identity")};
    }
    LatentDecoder::LatentDecoder(::cuda::stream_ref s, LatentConfiguration c, std::uint32_t batch) : stream{s}, configuration{std::move(c)}, model{stream, configuration.tokenizer, configuration.image, TokenizerDirection::decode}, image{stream, configuration.image, 1}, values{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(configuration.shape.width) * configuration.shape.height * configuration.shape.channels, ::cuda::no_init}, pixels{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(configuration.image.width) * configuration.image.height * configuration.image.channels, ::cuda::no_init}, rgba{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(batch) * configuration.image.width * configuration.image.height * 4, ::cuda::no_init} {}
    const std::uint8_t* LatentDecoder::decode(const TensorBatch& tensor) {
        const std::size_t bytes = static_cast<std::size_t>(configuration.image.width) * configuration.image.height * 4;
        for (std::uint32_t i = 0; i < tensor.count; ++i) {
            tokenizer_kernels::scale(stream, tensor.values + i * values.size(), values.data(), values.size(), 1.f / configuration.tokenizer.scaling_factor);
            const auto* result = model.forward(values.data());
            tokenizer_kernels::unpack(stream, result, pixels.data(), pixels.size());
            const auto* decoded = image.decode({configuration.image, 1, pixels.data(), tensor.labels + i});
            ::cuda::copy_bytes(stream, ::cuda::std::span<const std::uint8_t>{decoded, bytes}, ::cuda::std::span<std::uint8_t>{rgba.data() + i * bytes, bytes});
        }
        return rgba.data();
    }
} // namespace flowdit
