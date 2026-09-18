module;
#include "../neural/spatial-kernels.h"
#include <cublas_v2.h>
#include <cudnn.h>
#include <flowdit/cuda.h>
module flowdit.autoencoder.perceptual;
import flowdit.serialization.safetensors;
import std;
namespace flowdit {
    PerceptualLoss::PerceptualLoss(::cuda::stream_ref source, TensorShape image, std::uint32_t count, const std::filesystem::path& path) : stream{source}, shape{image}, batch{count}, network{stream, batch, {image.width, image.height, 3}}, parameters{stream, ::cuda::device_default_memory_pool(stream.device()), 14'714'688uz, ::cuda::no_init}, linear{stream, ::cuda::device_default_memory_pool(stream.device()), 1472uz, ::cuda::no_init}, input{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(batch) * image.width * image.height * 3, ::cuda::no_init} {
        const auto file = serialization::safetensors::read(path);
        std::vector<float> values;
        const std::array<unsigned, 5> widths{64, 128, 256, 512, 512}, counts{2, 2, 3, 3, 3};
        std::size_t vgg_index{}, linear_offset{};
        for (std::size_t level = 0; level < widths.size(); ++level) {
            for (unsigned i = 0; i < counts[level]; ++i) {
                network.append(neural::SpatialOperation::convolution, widths[level]);
                for (const auto* kind : {"weight", "bias"}) {
                    const auto name    = std::format("features.{}.{}", vgg_index, kind);
                    const auto& tensor = *std::ranges::find(file.tensors, name, &serialization::safetensors::Tensor::name);
                    const auto start   = values.size();
                    values.resize(start + tensor.data.size() / sizeof(float));
                    std::memcpy(values.data() + start, tensor.data.data(), tensor.data.size());
                }
                network.append(neural::SpatialOperation::relu);
                vgg_index += 2;
            }
            features[level]    = network.layers.size() - 1;
            const auto feature = network.layers.back().shape;
            reference.emplace_back(stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(batch) * feature.width * feature.height * feature.channels, ::cuda::no_init);
            const auto& weights = *std::ranges::find(file.tensors, std::format("linear.{}", level), &serialization::safetensors::Tensor::name);
            ::cuda::copy_bytes(stream, ::cuda::std::span<const std::byte>{weights.data.data(), weights.data.size()}, ::cuda::std::span<float>{linear.data() + linear_offset, widths[level]});
            linear_offset += widths[level];
            if (level + 1 < widths.size()) {
                network.append(neural::SpatialOperation::pool);
                ++vgg_index;
            }
        }
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{values.data(), values.size()}, parameters);
        stream.sync();
    }
    void PerceptualLoss::prepare(const float* images) {
        kernels::perceptual_input(stream, images, input.data(), batch, shape.channels, shape.width * shape.height);
        network.forward(parameters.data(), input.data());
        for (std::size_t i = 0; i < features.size(); ++i) ::cuda::copy_bytes(stream, *network.layers[features[i]].values, reference[i]);
    }
    void PerceptualLoss::backward(const float* reconstructed, float* gradient, float* loss, float weight) {
        kernels::perceptual_input(stream, reconstructed, input.data(), batch, shape.channels, shape.width * shape.height);
        network.forward(parameters.data(), input.data());
        std::size_t offset{};
        for (std::size_t i = 0; i < features.size(); ++i) {
            const auto& layer = network.layers[features[i]];
            kernels::perceptual_loss(stream, reference[i].data(), layer.values->data(), linear.data() + offset, network.accumulate(features[i]), loss, batch, layer.shape.channels, layer.shape.width * layer.shape.height, weight);
            offset += layer.shape.channels;
        }
        kernels::perceptual_backward(stream, network.backward(parameters.data(), nullptr), gradient, batch, shape.channels, shape.width * shape.height);
    }
} // namespace flowdit
