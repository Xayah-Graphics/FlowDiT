module;
#include "../tokenizer/kernels.h"
#include <flowdit/cuda.h>
#include <nlohmann/json.hpp>
module flowdit.representation.cache;
import flowdit.tokenizer.dcae;
import flowdit.serialization.digest;
import flowdit.image.transfer;
import std;
namespace flowdit {
    std::optional<LatentCache> LatentCache::prepare(::cuda::stream_ref stream, const Dataset& data, const std::filesystem::path& dataset, const TokenizerSpecification& tokenizer, bool flip, std::stop_token stop, const std::function<void(std::uint32_t, std::uint32_t)>& progress) {
        LatentCache result;
        result.variants      = flip ? 2 : 1;
        auto& c              = result.configuration;
        c.tokenizer          = tokenizer;
        c.image              = {data.specification.width, data.specification.height, data.specification.channels};
        c.shape              = {c.image.width / tokenizer.spatial_factor, c.image.height / tokenizer.spatial_factor, tokenizer.latent_channels};
        std::string identity = serialize_tokenizer(tokenizer) + serialize_image(data.specification) + std::format("/raw-bf16-v1/{}/{}", result.variants, static_cast<int>(data.split));
        std::set<std::uint32_t> files;
        for (std::size_t i = 0; i < data.records.size(); ++i) {
            const auto& record = data.records[i];
            identity += std::format("/{}/{}/{}", record.file, record.offset, data.labels[i]);
            files.insert(record.file);
        }
        for (const auto f : files) {
            if (stop.stop_requested()) return std::nullopt;
            identity += std::filesystem::relative(data.files[f], dataset).generic_string() + serialization::digest_file(data.files[f]);
        }
        c.cache_identity = serialization::digest(std::as_bytes(std::span{identity}));
        result.directory = dataset / ".flowdit/latents" / c.cache_identity;
        if (std::filesystem::exists(result.directory / "cache.json")) return result;
        std::filesystem::create_directories(result.directory);
        DCAE model{stream, tokenizer, c.image, TokenizerDirection::encode};
        ImageTransfer transfer{stream, c.image, 1};
        ImageBatch images;
        const std::size_t elements = static_cast<std::size_t>(c.shape.width) * c.shape.height * c.shape.channels;
        std::vector<std::uint16_t> values(elements);
        std::ofstream output{result.directory / "latents.bin", std::ios::binary | std::ios::trunc};
        output.exceptions(std::ios::badbit | std::ios::failbit);
        const auto total = static_cast<std::uint32_t>(data.labels.size()) * result.variants;
        for (std::uint32_t i = 0; i < total; ++i) {
            if (stop.stop_requested()) return std::nullopt;
            const std::uint32_t index = i / result.variants;
            data.read(std::span{&index, 1}, images);
            if (flip && i % 2)
                for (std::uint32_t channel = 0; channel < images.shape.channels; ++channel)
                    for (std::uint32_t y = 0; y < images.shape.height; ++y) {
                        const auto begin = images.pixels.begin() + (channel * images.shape.height + y) * images.shape.width;
                        std::reverse(begin, begin + images.shape.width);
                    }
            const auto* encoded = model.forward(transfer.encode(images));
            ::cuda::copy_bytes(stream, ::cuda::std::span<const std::uint16_t>{encoded, elements}, ::cuda::std::span<std::uint16_t>{values.data(), values.size()});
            stream.sync();
            output.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(std::uint16_t));
            if (progress) progress(i + 1, total);
        }
        output.close();
        std::ofstream manifest{result.directory / "cache.tmp"};
        manifest.exceptions(std::ios::badbit | std::ios::failbit);
        manifest << nlohmann::json{{"count", data.labels.size()}, {"variants", result.variants}, {"representation", nlohmann::json::parse(serialize_latent(c))}}.dump(2);
        manifest.close();
        std::filesystem::rename(result.directory / "cache.tmp", result.directory / "cache.json");
        return result;
    }
    void LatentCache::read(std::span<const std::uint32_t> indices, std::span<const std::uint32_t> views, std::vector<std::uint16_t>& values) const {
        const auto shape           = configuration.shape;
        const std::size_t elements = static_cast<std::size_t>(shape.width) * shape.height * shape.channels;
        values.resize(indices.size() * elements);
        std::ifstream file{directory / "latents.bin", std::ios::binary};
        file.exceptions(std::ios::failbit | std::ios::badbit);
        for (std::size_t i = 0; i < indices.size(); ++i) {
            file.seekg((static_cast<std::uint64_t>(indices[i]) * variants + views[i]) * elements * sizeof(std::uint16_t));
            file.read(reinterpret_cast<char*>(values.data() + i * elements), elements * sizeof(std::uint16_t));
        }
    }
    LatentBatch::LatentBatch(::cuda::stream_ref s, const LatentConfiguration& c, std::uint32_t count) : stream{s}, shape{c.shape}, batch{count}, scale{c.tokenizer.scaling_factor}, raw{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(batch) * shape.width * shape.height * shape.channels, ::cuda::no_init}, values{stream, ::cuda::device_default_memory_pool(stream.device()), raw.size(), ::cuda::no_init}, labels{stream, ::cuda::device_default_memory_pool(stream.device()), batch, ::cuda::no_init} {}
    TensorBatch LatentBatch::upload(std::span<const std::uint16_t> source, std::span<const std::uint32_t> classes) {
        ::cuda::copy_bytes(stream, ::cuda::std::span<const std::uint16_t>{source.data(), source.size()}, raw);
        ::cuda::copy_bytes(stream, ::cuda::std::span<const std::uint32_t>{classes.data(), classes.size()}, labels);
        tokenizer_kernels::unpack(stream, raw.data(), values.data(), values.size(), scale);
        return {shape, batch, values.data(), labels.data()};
    }
} // namespace flowdit
