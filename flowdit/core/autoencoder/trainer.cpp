module;
#include "../neural/spatial-kernels.h"
#include <cublas_v2.h>
#include <cudnn.h>
#include <flowdit/cuda.h>
module flowdit.autoencoder.trainer;
import flowdit.serialization.safetensors;
import std;
namespace flowdit {
    AutoencoderTrainer::AutoencoderTrainer(::cuda::stream_ref source, const AutoencoderConfiguration& c, std::uint32_t count, std::uint64_t seed, const neural::TrainingConfiguration& training, const std::filesystem::path& weights)
        : stream{source}, batch{count}, state{.seed = seed}, model{stream, c, batch}, parameters{stream, model.encoder.initial.size() + model.decoder.initial.size(), false}, optimizer{training}, discriminator{stream, batch, c.image}, perceptual{stream, c.image, batch, weights}, moments{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(batch) * c.latent_shape().width * c.latent_shape().height * c.latent_channels * 2, ::cuda::no_init}, latent{stream, ::cuda::device_default_memory_pool(stream.device()), moments.size() / 2, ::cuda::no_init}, reconstructed{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(batch) * c.image.width * c.image.height * c.image.channels, ::cuda::no_init}, reconstruction_gradient{stream, ::cuda::device_default_memory_pool(stream.device()), reconstructed.size(), ::cuda::no_init}, moments_gradient{stream, ::cuda::device_default_memory_pool(stream.device()), moments.size(), ::cuda::no_init},
          losses{stream, ::cuda::device_default_memory_pool(stream.device()), 5uz, ::cuda::no_init}, logits{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(batch) * c.image.width * c.image.height, ::cuda::no_init}, logits_gradient{stream, ::cuda::device_default_memory_pool(stream.device()), logits.size(), ::cuda::no_init}, device_state{stream, ::cuda::device_default_memory_pool(stream.device()), 2uz, ::cuda::no_init} {
        parameters.initialize(model.initialize(seed));
        const std::uint32_t depth = std::min(c.image.width, c.image.height) >= 128 ? 3 : 2;
        for (std::uint32_t i = 0; i < depth; ++i) {
            discriminator.append(neural::SpatialOperation::convolution, 64u << i, 4, 2, 1);
            if (i) discriminator.append(neural::SpatialOperation::normalization);
            discriminator.append(neural::SpatialOperation::leaky_relu);
        }
        discriminator.append(neural::SpatialOperation::convolution, 64u << depth, 4, 1, 1);
        discriminator.append(neural::SpatialOperation::normalization);
        discriminator.append(neural::SpatialOperation::leaky_relu);
        discriminator.append(neural::SpatialOperation::convolution, 1, 4, 1, 1);
        discriminator_parameters.emplace(stream, discriminator.initial.size(), false);
        discriminator_parameters->initialize(discriminator.initialize(seed + 2));
    }
    AutoencoderMetrics AutoencoderTrainer::optimize(const TensorBatch& input) {
        const auto& c = model.configuration;
        if (microstep == 0) ::cuda::fill_bytes(stream, losses, 0);
        const float scale      = 1.f / c.accumulation;
        const bool adversarial = state.step + 1 >= c.adversarial_start;
        model.encoder.encode(parameters.parameters.data(), input.values, moments.data());
        kernels::posterior(stream, moments.data(), latent.data(), batch, static_cast<std::uint32_t>(latent.size() / batch), state.seed, (state.step * c.accumulation + microstep) * 2, true);
        model.decoder.decode(parameters.parameters.data() + model.encoder_parameters, latent.data(), reconstructed.data());
        kernels::reconstruction_loss(stream, input.values, reconstructed.data(), reconstruction_gradient.data(), losses.data(), reconstructed.size(), scale);
        perceptual.prepare(input.values);
        perceptual.backward(reconstructed.data(), reconstruction_gradient.data(), losses.data() + 2, c.perceptual_weight * scale);
        const auto logit_count = static_cast<std::size_t>(batch) * discriminator.layers.back().shape.width * discriminator.layers.back().shape.height;
        if (adversarial) {
            kernels::unpack(stream, discriminator.forward(discriminator_parameters->parameters.data(), reconstructed.data()), logits.data(), logit_count);
            kernels::adversarial_loss(stream, logits.data(), logits_gradient.data(), losses.data() + 3, logit_count, 0, c.adversarial_weight * scale);
            kernels::sum(stream, discriminator.backward(discriminator_parameters->parameters.data(), nullptr, logits_gradient.data()), reconstruction_gradient.data(), reconstructed.size());
        }
        const auto* latent_gradient = model.decoder.backward(parameters.parameters.data() + model.encoder_parameters, parameters.gradients.data() + model.encoder_parameters, reconstruction_gradient.data());
        kernels::posterior_backward(stream, moments.data(), latent.data(), latent_gradient, moments_gradient.data(), losses.data() + 1, batch, static_cast<std::uint32_t>(latent.size() / batch), c.kl_weight, scale);
        model.encoder.backward(parameters.parameters.data(), parameters.gradients.data(), moments_gradient.data());
        if (adversarial) {
            kernels::unpack(stream, discriminator.forward(discriminator_parameters->parameters.data(), input.values), logits.data(), logit_count);
            kernels::adversarial_loss(stream, logits.data(), logits_gradient.data(), losses.data() + 4, logit_count, 1, .5f * scale);
            discriminator.backward(discriminator_parameters->parameters.data(), discriminator_parameters->gradients.data(), logits_gradient.data());
            kernels::unpack(stream, discriminator.forward(discriminator_parameters->parameters.data(), reconstructed.data()), logits.data(), logit_count);
            kernels::adversarial_loss(stream, logits.data(), logits_gradient.data(), losses.data() + 4, logit_count, 2, .5f * scale);
            discriminator.backward(discriminator_parameters->parameters.data(), discriminator_parameters->gradients.data(), logits_gradient.data());
        }
        AutoencoderMetrics metrics;
        if (++microstep == c.accumulation) {
            const std::array<std::uint64_t, 2> step{state.step + 1, state.processed_samples};
            ::cuda::copy_bytes(stream, ::cuda::std::span<const std::uint64_t>{step.data(), step.size()}, device_state);
            parameters.step(optimizer, device_state.data(), device_state.data() + 1, batch * c.accumulation);
            if (adversarial) {
                // The discriminator's Adam bias correction starts when its updates start.
                const auto discriminator_step = state.step + 2 - c.adversarial_start;
                ::cuda::copy_bytes(stream, ::cuda::std::span<const std::uint64_t>{&discriminator_step, 1}, ::cuda::std::span<std::uint64_t>{device_state.data(), 1});
                discriminator_parameters->step(optimizer, device_state.data(), device_state.data() + 1, batch * c.accumulation);
            }
            std::array<float, 5> values;
            ::cuda::copy_bytes(stream, losses, ::cuda::std::span<float>{values.data(), values.size()});
            stream.sync();
            metrics   = {values[0] * scale, values[1] * scale, values[2] * scale, values[3] * scale, values[4] * scale * .5f};
            microstep = 0;
            ++state.step;
            state.processed_samples += batch * c.accumulation;
        }
        return metrics;
    }
    const float* AutoencoderTrainer::reconstruct(const TensorBatch& input) {
        model.encoder.encode(parameters.parameters.data(), input.values, moments.data());
        kernels::posterior(stream, moments.data(), latent.data(), batch, static_cast<std::uint32_t>(latent.size() / batch), 0, 0, false);
        model.decoder.decode(parameters.parameters.data() + model.encoder_parameters, latent.data(), reconstructed.data());
        return reconstructed.data();
    }
    void AutoencoderTrainer::save(const std::filesystem::path& path, std::map<std::string, std::string> metadata) const {
        const auto main = parameters.download(), disc = discriminator_parameters->download();
        const std::array<std::uint64_t, 4> state_values{state.step, state.processed_samples, state.seed, std::bit_cast<std::uint64_t>(state.elapsed_seconds)};
        std::vector<serialization::safetensors::TensorView> tensors;
        for (const auto& [prefix, p] : std::array{std::pair{"autoencoder", &main}, std::pair{"discriminator", &disc}}) {
            if (p == &main) {
                tensors.push_back({"encoder.parameters", "F32", {model.encoder_parameters}, p->parameters.data(), model.encoder_parameters * sizeof(float)});
                const auto count = p->parameters.size() - model.encoder_parameters;
                tensors.push_back({"decoder.parameters", "F32", {count}, p->parameters.data() + model.encoder_parameters, count * sizeof(float)});
            } else tensors.push_back({"discriminator.parameters", "F32", {p->parameters.size()}, p->parameters.data(), p->parameters.size() * sizeof(float)});
            tensors.push_back({std::string{prefix} + ".first_moments", "F32", {p->first_moments.size()}, p->first_moments.data(), p->first_moments.size() * sizeof(float)});
            tensors.push_back({std::string{prefix} + ".second_moments", "F32", {p->second_moments.size()}, p->second_moments.data(), p->second_moments.size() * sizeof(float)});
        }
        tensors.push_back({"training.state", "U64", {4}, state_values.data(), sizeof(state_values)});
        metadata["flowdit.autoencoder"] = serialize_autoencoder(model.configuration);
        serialization::safetensors::write(path, "autoencoder-kl", tensors, metadata);
    }
    void AutoencoderTrainer::load(const std::filesystem::path& path) {
        const auto file = serialization::safetensors::read(path);
        for (const auto& [prefix, buffer] : std::array{std::pair{"autoencoder", &parameters}, std::pair{"discriminator", &*discriminator_parameters}}) {
            neural::ParameterState state;
            for (const auto& [name, target] : std::array{std::pair{"parameters", &state.parameters}, std::pair{"first_moments", &state.first_moments}, std::pair{"second_moments", &state.second_moments}}) {
                std::vector<std::string> names{std::string{prefix} + "." + name};
                if (buffer == &parameters && target == &state.parameters) names = {"encoder.parameters", "decoder.parameters"};
                for (const auto& tensor_name : names) {
                    const auto& tensor = *std::ranges::find(file.tensors, tensor_name, &serialization::safetensors::Tensor::name);
                    const auto offset  = target->size();
                    target->resize(offset + tensor.data.size() / sizeof(float));
                    std::memcpy(target->data() + offset, tensor.data.data(), tensor.data.size());
                }
            }
            buffer->upload(state);
        }
        const auto& values = *std::ranges::find(file.tensors, std::string{"training.state"}, &serialization::safetensors::Tensor::name);
        std::array<std::uint64_t, 4> saved;
        std::memcpy(saved.data(), values.data.data(), sizeof(saved));
        state = {saved[0], saved[1], saved[2], std::bit_cast<double>(saved[3])};
    }
} // namespace flowdit
