module;
#include "../neural/spatial-kernels.h"
#include <cublas_v2.h>
#include <cudnn.h>
#include <flowdit/cuda.h>
#include <nlohmann/json.hpp>
module flowdit.representation.cache;
import flowdit.serialization.safetensors;
import flowdit.serialization.digest;
import flowdit.image.transfer;
import std;
namespace flowdit {
    std::optional<LatentCache> LatentCache::prepare(::cuda::stream_ref stream, const Dataset& data, const std::filesystem::path& dataset, const std::filesystem::path& checkpoint, bool flip, std::stop_token stop, const std::function<void(std::uint32_t, std::uint32_t)>& progress) {
        LatentCache result;
        result.variants      = flip ? 2 : 1;
        auto& c              = result.configuration;
        c.checkpoint         = std::filesystem::relative(checkpoint, dataset);
        c.identity           = serialization::digest_file(checkpoint);
        const auto metadata  = serialization::safetensors::read_metadata(checkpoint);
        c.autoencoder        = deserialize_autoencoder(metadata.at("flowdit.autoencoder"));
        std::string identity = c.identity + std::format("/posterior-v1/{}/{}", result.variants, static_cast<int>(data.split));
        std::set<std::uint32_t> files;
        for (std::size_t i = 0; i < data.records.size(); ++i) {
            const auto& record = data.records[i];
            identity += std::format("/{}/{}/{}", record.file, record.offset, data.labels[i]);
            files.insert(record.file);
        }
        for (const auto f : files) identity += data.files[f].filename().generic_string() + serialization::digest_file(data.files[f]);
        c.cache_identity = serialization::digest(std::as_bytes(std::span{identity}));
        result.directory = dataset / ".flowdit" / "latents" / c.cache_identity;
        if (std::filesystem::exists(result.directory / "cache.json")) {
            std::ifstream file{result.directory / "cache.json"};
            c = deserialize_latent(nlohmann::json::parse(file).at("representation").dump());
            return result;
        }
        std::filesystem::create_directories(result.directory);
        const auto shape          = c.autoencoder.latent_shape();
        const auto area           = shape.width * shape.height;
        const auto elements       = static_cast<std::size_t>(area) * shape.channels;
        const std::uint32_t batch = c.autoencoder.image.width >= 256 ? 2 : 32;
        Encoder model{stream, c.autoencoder, batch};
        const auto file = serialization::safetensors::read(checkpoint, std::array<std::string_view, 1>{"encoder.parameters"});
        ::cuda::device_buffer<float> parameters{stream, ::cuda::device_default_memory_pool(stream.device()), file.tensors.front().data.size() / sizeof(float), ::cuda::no_init};
        ::cuda::copy_bytes(stream, ::cuda::std::span<const std::byte>{file.tensors.front().data.data(), file.tensors.front().data.size()}, parameters);
        ::cuda::device_buffer<float> device_moments{stream, ::cuda::device_default_memory_pool(stream.device()), batch * elements * 2, ::cuda::no_init};
        ImageTransfer transfer{stream, c.autoencoder.image, batch};
        ImageBatch images;
        std::vector<std::uint32_t> indices(batch);
        std::vector<float> moments(batch * elements * 2);
        std::vector<double> sum(shape.channels), second(shape.channels);
        std::ofstream output{result.directory / "moments.bin", std::ios::binary | std::ios::trunc};
        output.exceptions(std::ios::badbit | std::ios::failbit);
        const auto total = static_cast<std::uint32_t>(data.labels.size()) * result.variants;
        for (std::uint32_t first = 0; first < total; first += batch) {
            if (stop.stop_requested()) return std::nullopt;
            const auto count = std::min(batch, total - first);
            for (std::uint32_t i = 0; i < batch; ++i) indices[i] = std::min(first + i, total - 1) / result.variants;
            data.read(indices, images);
            const auto image_area = images.shape.width * images.shape.height;
            if (flip)
                for (std::uint32_t i = 0; i < batch; ++i)
                    if (std::min(first + i, total - 1) % 2)
                        for (std::uint32_t channel = 0; channel < images.shape.channels; ++channel)
                            for (std::uint32_t y = 0; y < images.shape.height; ++y) {
                                auto begin = images.pixels.begin() + (static_cast<std::size_t>(i) * images.shape.channels + channel) * image_area + y * images.shape.width;
                                std::reverse(begin, begin + images.shape.width);
                            }
            const auto tensor = transfer.encode(images, false, 0, 0);
            model.encode(parameters.data(), tensor.values, device_moments.data());
            ::cuda::copy_bytes(stream, device_moments, ::cuda::std::span<float>{moments.data(), moments.size()});
            stream.sync();
            output.write(reinterpret_cast<const char*>(moments.data()), count * elements * 2 * sizeof(float));
            for (std::uint32_t i = 0; i < count; ++i)
                for (std::uint32_t channel = 0; channel < shape.channels; ++channel)
                    for (std::uint32_t p = 0; p < area; ++p) {
                        const auto at     = i * elements * 2 + channel * area + p;
                        const double mean = moments[at], variance = std::exp(std::clamp(moments[at + elements], -30.f, 20.f));
                        sum[channel] += mean;
                        second[channel] += mean * mean + variance;
                    }
            if (progress) progress(first + count, total);
        }
        output.close();
        c.mean.resize(shape.channels);
        c.deviation.resize(shape.channels);
        for (std::uint32_t channel = 0; channel < shape.channels; ++channel) {
            const auto count     = static_cast<double>(total) * area;
            c.mean[channel]      = static_cast<float>(sum[channel] / count);
            c.deviation[channel] = static_cast<float>(std::sqrt(second[channel] / count - (sum[channel] / count) * (sum[channel] / count)));
        }
        std::ofstream manifest{result.directory / "cache.tmp"};
        manifest.exceptions(std::ios::badbit | std::ios::failbit);
        manifest << nlohmann::json{{"count", data.labels.size()}, {"variants", result.variants}, {"representation", nlohmann::json::parse(serialize_latent(c))}}.dump(2);
        manifest.close();
        std::filesystem::rename(result.directory / "cache.tmp", result.directory / "cache.json");
        return result;
    }
    void LatentCache::read(std::span<const std::uint32_t> indices, std::span<const std::uint32_t> views, std::vector<float>& moments) const {
        const auto shape           = configuration.autoencoder.latent_shape();
        const std::size_t elements = static_cast<std::size_t>(shape.width) * shape.height * shape.channels * 2;
        moments.resize(indices.size() * elements);
        std::ifstream file{directory / "moments.bin", std::ios::binary};
        file.exceptions(std::ios::failbit | std::ios::badbit);
        for (std::size_t i = 0; i < indices.size(); ++i) {
            file.seekg((static_cast<std::uint64_t>(indices[i]) * variants + views[i]) * elements * sizeof(float));
            file.read(reinterpret_cast<char*>(moments.data() + i * elements), elements * sizeof(float));
        }
    }
    LatentBatch::LatentBatch(::cuda::stream_ref source, const LatentConfiguration& c, std::uint32_t count) : stream{source}, shape{c.autoencoder.latent_shape()}, batch{count}, moments{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(batch) * shape.width * shape.height * shape.channels * 2, ::cuda::no_init}, values{stream, ::cuda::device_default_memory_pool(stream.device()), moments.size() / 2, ::cuda::no_init}, mean{stream, ::cuda::device_default_memory_pool(stream.device()), shape.channels, ::cuda::no_init}, deviation{stream, ::cuda::device_default_memory_pool(stream.device()), shape.channels, ::cuda::no_init}, labels{stream, ::cuda::device_default_memory_pool(stream.device()), batch, ::cuda::no_init} {
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{c.mean.data(), c.mean.size()}, mean);
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{c.deviation.data(), c.deviation.size()}, deviation);
        stream.sync();
    }
    TensorBatch LatentBatch::upload(std::span<const float> source, std::span<const std::uint32_t> classes, std::uint64_t seed, std::uint64_t step) {
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{source.data(), source.size()}, moments);
        ::cuda::copy_bytes(stream, ::cuda::std::span<const std::uint32_t>{classes.data(), classes.size()}, labels);
        kernels::posterior(stream, moments.data(), values.data(), batch, static_cast<std::uint32_t>(values.size() / batch), seed, step, true);
        kernels::standardize(stream, values.data(), values.data(), mean.data(), deviation.data(), batch, shape.channels, shape.width * shape.height, false);
        return {shape, batch, values.data(), labels.data()};
    }
} // namespace flowdit
