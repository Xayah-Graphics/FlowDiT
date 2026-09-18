module;
#include <cublas_v2.h>
#include <cudnn.h>
#include <flowdit/cuda.h>
module flowdit.runtime.session;
import flowdit.serialization.safetensors;
import flowdit.autoencoder.trainer;
import flowdit.representation.cache;
import std;
namespace flowdit {
    Session::Session(SessionObserver source) : events{std::move(source)}, worker{[this] { run(); }} {}
    Session::~Session() {
        shutdown();
        worker.join();
    }
    void Session::start(std::variant<TrainRequest, SampleRequest> request) {
        {
            const std::lock_guard lock{mutex};
            update        = {};
            update.status = {.mode = std::holds_alternative<TrainRequest>(request) ? Mode::training : Mode::sampling, .stage = Stage::loading, .busy = true, .started = std::chrono::steady_clock::now()};
            cancellation  = std::stop_source{};
            requested     = std::move(request);
        }
        condition.notify_one();
    }
    void Session::stop() {
        const std::lock_guard lock{mutex};
        cancellation.request_stop();
        condition.notify_one();
    }
    void Session::shutdown() {
        {
            const std::lock_guard lock{mutex};
            update.status.closing = true;
            cancellation.request_stop();
        }
        condition.notify_one();
    }
    SessionUpdate Session::receive() {
        const std::lock_guard lock{mutex};
        SessionUpdate result = std::move(update);
        update               = {};
        update.status        = result.status;
        return result;
    }
    void Session::report(const Stage stage) {
        {
            const std::lock_guard lock{mutex};
            update.status.stage = stage;
        }
        if (events.notify) events.notify();
    }
    bool Session::generate(const std::function<std::optional<SamplingResult>(const SamplingRequest&, const SamplingObserver&)>& sample, ::cuda::stream_ref stream, const LatentConfiguration& representation, const std::filesystem::path& dataset, SampleInfo info, bool interactive) {
        const auto count          = info.request.count;
        const std::uint32_t batch = std::min(count, 4u), chunks = (count + batch - 1) / batch;
        const std::size_t image_bytes = static_cast<std::size_t>(info.image.width) * info.image.height * 4;
        LatentDecoder decoder{stream, representation, batch, dataset};
        ::cuda::device_buffer<std::uint8_t> pixels{stream, ::cuda::device_default_memory_pool(stream.device()), count * image_bytes, ::cuda::no_init};
        ::cuda::fill_bytes(stream, pixels, 0);
        info.labels.resize(count);
        for (std::uint32_t i = 0; i < count; ++i) info.labels[i] = info.request.class_index.value_or(i % static_cast<std::uint32_t>(info.image.classes.size()));
        const FrameInfo frame{info.image, info.labels};
        for (std::uint32_t first = 0; first < count; first += batch) {
            auto request         = info.request;
            request.count        = batch;
            request.first_sample = first;
            const auto valid     = std::min(batch, count - first);
            SamplingObserver observer{.stop = cancellation.get_token(), .progress = [&, first](const SamplingProgress& progress) {
                                          {
                                              const std::lock_guard lock{mutex};
                                              update.status.sampling = {first / batch * request.step_count + progress.step, chunks * request.step_count, info.nfe + progress.nfe};
                                          }
                                          if (events.notify) events.notify();
                                      }};
            const auto publish = [&](const TensorBatch& tensor, ::cuda::stream_ref) {
                auto part           = tensor;
                part.count          = valid;
                const auto* decoded = decoder.decode(part);
                ::cuda::copy_bytes(stream, ::cuda::std::span<const std::uint8_t>{decoded, valid * image_bytes}, ::cuda::std::span<std::uint8_t>{pixels.data() + first * image_bytes, valid * image_bytes});
                if (interactive && events.image) events.image(frame, pixels.data(), stream);
            };
            if (interactive && events.image) observer.tensor = publish;
            const auto result = sample(request, observer);
            if (!result) return false;
            publish(result->tensor, stream);
            info.nfe += result->nfe;
        }
        auto output = std::make_shared<SampleOutput>(SampleOutput{.info = std::move(info)});
        output->rgba.resize(pixels.size());
        ::cuda::copy_bytes(stream, pixels, ::cuda::std::span<std::uint8_t>{output->rgba.data(), output->rgba.size()});
        stream.sync();
        output::write_sample(*output);
        {
            const std::lock_guard lock{mutex};
            update.samples.push_back(std::move(output));
        }
        return true;
    }
    void Session::run() {
        for (;;) {
            std::optional<std::variant<TrainRequest, SampleRequest>> request;
            {
                std::unique_lock lock{mutex};
                condition.wait(lock, [this] { return update.status.closing || requested.has_value(); });
                if (update.status.closing) break;
                request = std::move(requested);
                requested.reset();
            }
            try {
                std::visit([this](const auto& value) { execute(value); }, *request);
                report(cancellation.stop_requested() ? Stage::stopped : Stage::complete);
            } catch (const std::exception& error) {
                {
                    const std::lock_guard lock{mutex};
                    update.status.error = error.what();
                }
                report(Stage::failed);
                sampler.reset();
            }
            {
                const std::lock_guard lock{mutex};
                update.status.busy  = false;
                update.status.ended = std::chrono::steady_clock::now();
            }
            if (events.notify) events.notify();
        }
        sampler.reset();
        {
            const std::lock_guard lock{mutex};
            update.status.finished = true;
        }
        if (events.notify) events.notify();
    }
    void Session::train_autoencoder(const TrainRequest& request) {
        auto config = request.configuration;
        ::cuda::stream stream{::cuda::devices[0]};
        ImageTransfer transfer{stream, {config.image.width, config.image.height, config.image.channels}, config.batch};
        AutoencoderTrainer trainer{stream, config.autoencoder, config.batch, config.seed, config.optimizer, config.dataset.parent_path() / ".flowdit" / "lpips-vgg.safetensors"};
        if (!request.checkpoint.empty()) trainer.load(request.checkpoint);
        config.seed = trainer.state.seed;
        output::write_configuration(config);
        std::ofstream csv{config.output / "training.csv", request.checkpoint.empty() ? std::ios::trunc : std::ios::app};
        csv.exceptions(std::ios::failbit | std::ios::badbit);
        if (request.checkpoint.empty()) std::println(csv, "step,loss,samples_per_second,elapsed_seconds,kl,perceptual,generator,discriminator");
        const auto validation = load_dataset(config.dataset, DatasetSplit::validation);
        ImageBatch input;
        std::vector<std::uint32_t> indices(config.batch);
        const auto preview = [&] {
            report(Stage::preview_parameters);
            auto output             = std::make_shared<SampleOutput>();
            output->info            = {.path = config.output / "samples" / std::format("step-{:06}-ema.png", trainer.state.step), .request = config.preview, .reconstruction = true, .training_step = trainer.state.step, .model = config.model, .image = config.image};
            const auto pairs        = std::max(1u, config.preview.count / 2);
            const std::size_t bytes = static_cast<std::size_t>(config.image.width) * config.image.height * 4;
            output->rgba.resize(pairs * 2 * bytes);
            output->info.labels.resize(pairs * 2);
            output->info.request.count = pairs * 2;
            std::vector<std::uint8_t> originals(config.batch * bytes), reconstructions(config.batch * bytes);
            for (std::uint32_t first = 0; first < pairs; first += config.batch) {
                for (std::uint32_t i = 0; i < config.batch; ++i) indices[i] = static_cast<std::uint32_t>(std::ranges::find(validation.labels, (first + i) % static_cast<std::uint32_t>(config.image.classes.size())) - validation.labels.begin());
                validation.read(indices, input);
                const auto tensor    = transfer.encode(input, false, 0, 0);
                const auto* original = transfer.decode(tensor);
                ::cuda::copy_bytes(stream, ::cuda::std::span<const std::uint8_t>{original, originals.size()}, ::cuda::std::span<std::uint8_t>{originals.data(), originals.size()});
                const auto* reconstructed = trainer.reconstruct(tensor);
                const auto* decoded       = transfer.decode({tensor.shape, tensor.count, reconstructed, tensor.labels});
                ::cuda::copy_bytes(stream, ::cuda::std::span<const std::uint8_t>{decoded, reconstructions.size()}, ::cuda::std::span<std::uint8_t>{reconstructions.data(), reconstructions.size()});
                stream.sync();
                for (std::uint32_t i = 0; i < std::min(config.batch, pairs - first); ++i) {
                    const auto destination = (first + i) * 2;
                    std::memcpy(output->rgba.data() + destination * bytes, originals.data() + i * bytes, bytes);
                    std::memcpy(output->rgba.data() + (destination + 1) * bytes, reconstructions.data() + i * bytes, bytes);
                    output->info.labels[destination] = output->info.labels[destination + 1] = input.labels[i];
                }
            }
            std::filesystem::create_directories(output->info.path.parent_path());
            output::write_sample(*output);
            const std::lock_guard lock{mutex};
            update.samples.push_back(std::move(output));
        };
        const auto checkpoint = [&] {
            report(Stage::saving);
            const auto path = config.output / "checkpoints" / std::format("step-{:06}.safetensors", trainer.state.step);
            std::filesystem::create_directories(path.parent_path());
            trainer.save(path, {{"flowdit.image", serialize_image(config.image)}});
            const std::lock_guard lock{mutex};
            update.checkpoints.push_back(path);
        };
        preview();
        {
            const std::lock_guard lock{mutex};
            update.status.training         = trainer.state;
            update.status.training_started = std::chrono::steady_clock::now();
        }
        while (trainer.state.step < config.end_step && !cancellation.stop_requested()) {
            report(Stage::optimizing);
            const auto started = std::chrono::steady_clock::now();
            AutoencoderMetrics metrics;
            for (std::uint32_t micro = 0; micro < config.autoencoder.accumulation; ++micro) {
                std::seed_seq seed{static_cast<std::uint32_t>(config.seed), static_cast<std::uint32_t>(config.seed >> 32), static_cast<std::uint32_t>(trainer.state.step), static_cast<std::uint32_t>(trainer.state.step >> 32), micro};
                std::mt19937_64 random{seed};
                for (auto& index : indices) index = static_cast<std::uint32_t>((random() >> 32) * request.dataset->labels.size() >> 32);
                request.dataset->read(indices, input);
                metrics = trainer.optimize(transfer.encode(input, config.horizontal_flip, config.seed, trainer.state.step * config.autoencoder.accumulation + micro));
            }
            const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            trainer.state.elapsed_seconds += seconds;
            {
                const std::lock_guard lock{mutex};
                update.status.training = trainer.state;
            }
            if (trainer.state.step % config.log_interval == 0 || trainer.state.step == config.end_step) {
                const TrainingRecord record{trainer.state.step, metrics.reconstruction, config.batch * config.autoencoder.accumulation / seconds, trainer.state.elapsed_seconds, {metrics.kl, metrics.perceptual, metrics.generator, metrics.discriminator}};
                std::println(csv, "{},{},{},{},{},{},{},{}", record.step, record.loss, record.samples_per_second, record.training_seconds, metrics.kl, metrics.perceptual, metrics.generator, metrics.discriminator);
                csv.flush();
                const std::lock_guard lock{mutex};
                update.metrics.push_back(record);
            }
            if (trainer.state.step % config.preview_interval == 0 && !cancellation.stop_requested()) preview();
            if (trainer.state.step % config.save_interval == 0 && trainer.state.step < config.end_step) checkpoint();
        }
        checkpoint();
        if (!cancellation.stop_requested() && trainer.state.step % config.preview_interval != 0) preview();
    }
    void Session::execute(const TrainRequest& request) {
        sampler.reset();
        sampler_checkpoint.clear();
        auto config = request.configuration;
        if (const auto status = cudaSetDevice(0); status != cudaSuccess) throw std::runtime_error{cudaGetErrorString(status)};
        std::filesystem::create_directories(config.output);
        report(Stage::loading);
        if (config.stage == TrainingStage::autoencoder) {
            train_autoencoder(request);
            return;
        }
        ::cuda::stream stream{::cuda::devices[0]};
        report(Stage::preparing);
        const auto cache = LatentCache::prepare(stream, *request.dataset, config.dataset, config.dataset / config.autoencoder_checkpoint, config.horizontal_flip, cancellation.get_token(), [this](std::uint32_t done, std::uint32_t total) {
            {
                const std::lock_guard lock{mutex};
                update.status.prepared          = done;
                update.status.preparation_count = total;
            }
            if (events.notify) events.notify();
        });
        if (!cache) return;
        config.autoencoder = cache->configuration.autoencoder;
        config.model.shape = config.autoencoder.latent_shape();
        LatentBatch batch{stream, cache->configuration, config.batch};
        Trainer trainer{stream, config.model, config.batch, config.seed, config.optimizer};
        if (!request.checkpoint.empty()) {
            const auto previous = deserialize_latent(serialization::safetensors::read_metadata(request.checkpoint).at("flowdit.latent"));
            if (serialize_latent(previous) != serialize_latent(cache->configuration)) throw std::runtime_error{"The run's latent representation has changed"};
            trainer.load(request.checkpoint);
        }
        config.seed = trainer.state.seed;
        output::write_configuration(config);
        std::ofstream csv{config.output / "training.csv", request.checkpoint.empty() ? std::ios::trunc : std::ios::app};
        csv.exceptions(std::ios::failbit | std::ios::badbit);
        if (request.checkpoint.empty()) std::println(csv, "step,loss,samples_per_second,elapsed_seconds");
        std::vector<std::uint32_t> indices(config.batch), views(config.batch), labels(config.batch);
        std::vector<float> moments;
        const auto checkpoint = [&] {
            report(Stage::saving);
            const auto path = config.output / "checkpoints" / std::format("step-{:06}.safetensors", trainer.state.step);
            std::filesystem::create_directories(path.parent_path());
            trainer.save(path, {{"flowdit.image", serialize_image(config.image)}, {"flowdit.latent", serialize_latent(cache->configuration)}});
            const std::lock_guard lock{mutex};
            update.checkpoints.push_back(path);
        };
        const auto preview = [&] {
            SamplingRuntime runtime{stream, trainer.model, std::min(config.preview.count, 4u)};
            std::filesystem::create_directories(config.output / "samples");
            for (const auto source : {ParameterSource::parameters, ParameterSource::exponential_average}) {
                report(source == ParameterSource::parameters ? Stage::preview_parameters : Stage::preview_ema);
                const auto* parameters = source == ParameterSource::parameters ? trainer.parameter_buffer.parameters.data() : trainer.parameter_buffer.ema.data();
                const SampleInfo info{.path = config.output / "samples" / std::format("step-{:06}-{}.png", trainer.state.step, source == ParameterSource::parameters ? "parameters" : "ema"), .request = config.preview, .source = source, .training_step = trainer.state.step, .model = config.model, .image = config.image};
                if (!generate([&](const SamplingRequest& r, const SamplingObserver& o) { return runtime.sample(parameters, r, o); }, stream, cache->configuration, config.dataset, info, false)) break;
            }
        };
        {
            const std::lock_guard lock{mutex};
            update.status.training         = trainer.state;
            update.status.training_started = std::chrono::steady_clock::now();
        }
        while (trainer.state.step < config.end_step && !cancellation.stop_requested()) {
            report(Stage::optimizing);
            const auto started = std::chrono::steady_clock::now();
            std::seed_seq seed{static_cast<std::uint32_t>(config.seed), static_cast<std::uint32_t>(config.seed >> 32), static_cast<std::uint32_t>(trainer.state.step), static_cast<std::uint32_t>(trainer.state.step >> 32)};
            std::mt19937_64 random{seed};
            for (std::uint32_t i = 0; i < config.batch; ++i) {
                indices[i] = static_cast<std::uint32_t>((random() >> 32) * request.dataset->labels.size() >> 32);
                views[i]   = static_cast<std::uint32_t>(random() % cache->variants);
                labels[i]  = request.dataset->labels[indices[i]];
            }
            cache->read(indices, views, moments);
            const float loss   = trainer.optimize(batch.upload(moments, labels, config.seed, trainer.state.step + 1));
            const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            trainer.state.elapsed_seconds += seconds;
            {
                const std::lock_guard lock{mutex};
                update.status.training = trainer.state;
            }
            if (trainer.state.step % config.log_interval == 0 || trainer.state.step == config.end_step) {
                const TrainingRecord record{trainer.state.step, loss, config.batch / seconds, trainer.state.elapsed_seconds};
                std::println(csv, "{},{},{},{}", record.step, record.loss, record.samples_per_second, record.training_seconds);
                csv.flush();
                const std::lock_guard lock{mutex};
                update.metrics.push_back(record);
            }
            if (trainer.state.step % config.preview_interval == 0 && !cancellation.stop_requested()) preview();
            if (trainer.state.step % config.save_interval == 0 && trainer.state.step < config.end_step) checkpoint();
        }
        checkpoint();
        if (!cancellation.stop_requested() && trainer.state.step % config.preview_interval != 0) preview();
    }
    void Session::execute(const SampleRequest& request) {
        report(Stage::loading);
        const auto batch = std::min(request.sampling.count, 4u);
        if (!sampler || sampler_checkpoint != request.checkpoint || sampler->batch != batch) {
            sampler.reset();
            if (const auto status = cudaSetDevice(0); status != cudaSuccess) throw std::runtime_error{cudaGetErrorString(status)};
            const auto metadata = serialization::safetensors::read_metadata(request.checkpoint);
            sampler             = std::make_unique<Sampler>(deserialize_model(metadata.at("flowdit.model")), request.checkpoint, 0, batch);
            sampler_checkpoint  = request.checkpoint;
            sampler_image       = deserialize_image(metadata.at("flowdit.image"));
            sampler_latent      = deserialize_latent(metadata.at("flowdit.latent"));
        }
        const auto dataset = request.checkpoint.parent_path().parent_path().parent_path().parent_path().parent_path();
        std::filesystem::create_directories(request.output.parent_path());
        report(Stage::sampling);
        generate([&](const SamplingRequest& r, const SamplingObserver& o) { return sampler->sample(r, o); }, sampler->stream, sampler_latent, dataset, {.path = request.output, .checkpoint = request.checkpoint, .request = request.sampling, .model = sampler->configuration, .image = sampler_image}, true);
    }
} // namespace flowdit
