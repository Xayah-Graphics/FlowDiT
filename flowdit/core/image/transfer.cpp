module;
#include "kernels.h"
#include <flowdit/cuda.h>
#include <nlohmann/json.hpp>
module flowdit.image.transfer;
import std;
namespace flowdit {
    std::string serialize_image(const ImageSpecification& image) {
        return nlohmann::json{{"name", image.name}, {"width", image.width}, {"height", image.height}, {"channels", image.channels}, {"classes", image.classes}}.dump();
    }
    ImageSpecification deserialize_image(const std::string_view text) {
        const auto json = nlohmann::json::parse(text);
        return {json.at("name").get<std::string>(), json.at("width"), json.at("height"), json.at("channels"), json.at("classes").get<std::vector<std::string>>()};
    }
    ImageTransfer::ImageTransfer(const ::cuda::stream_ref source, const TensorShape dimensions, const std::uint32_t count) : stream{source}, shape{dimensions}, batch{count} {}
    TensorBatch ImageTransfer::encode(const ImageBatch& source, const bool horizontal_flip, const std::uint64_t seed, const std::uint64_t step) {
        if (!images) {
            images.emplace(stream, ::cuda::device_default_memory_pool(stream.device()), source.pixels.size(), ::cuda::no_init);
            labels.emplace(stream, ::cuda::device_default_memory_pool(stream.device()), batch, ::cuda::no_init);
            values.emplace(stream, ::cuda::device_default_memory_pool(stream.device()), source.pixels.size(), ::cuda::no_init);
        }
        ::cuda::copy_bytes(stream, ::cuda::std::span<const std::uint8_t>{source.pixels.data(), source.pixels.size()}, *images);
        ::cuda::copy_bytes(stream, ::cuda::std::span<const std::uint32_t>{source.labels.data(), source.labels.size()}, *labels);
        kernels::encode_pixels(stream, images->data(), values->data(), batch, shape.width, shape.height, shape.channels, horizontal_flip, seed, step);
        return {shape, batch, values->data(), labels->data()};
    }
    const std::uint8_t* ImageTransfer::decode(const TensorBatch& tensor) {
        if (!rgba) rgba.emplace(stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(batch) * shape.width * shape.height * 4uz, ::cuda::no_init);
        kernels::decode_pixels(stream, tensor.values, rgba->data(), tensor.count, shape.width, shape.height, shape.channels);
        return rgba->data();
    }
} // namespace flowdit
