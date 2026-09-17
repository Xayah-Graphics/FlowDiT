module;
#include <flowdit/cuda.h>
module flowdit.runtime.session;
import flowdit.serialization.safetensors;
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
            update            = {};
            update.status     = {.mode = std::holds_alternative<TrainRequest>(request) ? Mode::training : Mode::sampling, .stage = Stage::loading, .busy = true, .started = std::chrono::steady_clock::now()};
            cancellation      = std::stop_source{};
            requested         = std::move(request);
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
    SamplingObserver Session::observe(PixelRepresentation* representation, const FrameInfo* info) {
        SamplingObserver observer{.stop = cancellation.get_token(), .progress = [this](const SamplingProgress& progress) {
                                      {
                                          const std::lock_guard lock{mutex};
                                          update.status.sampling = progress;
                                      }
                                      if (events.notify) events.notify();
                                  }};
        if (info && events.image)
            observer.tensor = [this, representation, info = *info](const TensorBatch& tensor, const ::cuda::stream_ref stream) { events.image(info, representation->decode(tensor), stream); };
        return observer;
    }
    void Session::publish_sample(const SamplingResult& result, PixelRepresentation& representation, const ::cuda::stream_ref stream, SampleInfo info, const bool preview) {
        info.nfe = result.nfe;
        info.labels.resize(result.tensor.count);
        auto sample = std::make_shared<SampleOutput>(SampleOutput{.info = std::move(info)});
        const auto& image = sample->info.image;
        sample->rgba.resize(static_cast<std::size_t>(result.tensor.count) * image.width * image.height * 4uz);
        const auto* pixels = representation.decode(result.tensor);
        if (preview && events.image) events.image({sample->info.request, image}, pixels, stream);
        ::cuda::copy_bytes(stream, ::cuda::std::span<const std::uint8_t>{pixels, sample->rgba.size()}, ::cuda::std::span<std::uint8_t>{sample->rgba.data(), sample->rgba.size()});
        ::cuda::copy_bytes(stream, ::cuda::std::span<const std::uint32_t>{result.tensor.labels, result.tensor.count}, ::cuda::std::span<std::uint32_t>{sample->info.labels.data(), sample->info.labels.size()});
        stream.sync();
        output::write_sample(*sample);
        const std::lock_guard lock{mutex};
        update.samples.push_back(std::move(sample));
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
    void Session::execute(const TrainRequest& request) {
        sampler.reset();
        sampler_checkpoint.clear();
        auto config = request.configuration;
        if (const auto result = cudaSetDevice(0); result != cudaSuccess) throw std::runtime_error{cudaGetErrorString(result)};
        std::filesystem::create_directories(config.output);
        report(Stage::loading);
        if (!request.checkpoint.empty()) {
            const auto metadata = serialization::safetensors::read_metadata(request.checkpoint);
            config.model = deserialize_model(metadata.at("flowdit.model"));
            config.image = deserialize_representation(metadata.at("flowdit.representation"));
        }
        ::cuda::stream stream{::cuda::devices[0]};
        PixelRepresentation representation{stream, {config.image.width, config.image.height, config.image.channels}, config.batch};
        Trainer trainer{stream, config.model, config.batch, config.seed, config.optimizer};
        ImageBatch input;
        std::vector<std::uint32_t> indices(config.batch);
        if (!request.checkpoint.empty()) trainer.load(request.checkpoint);
        config.seed = trainer.state.seed;
        output::write_configuration(config);
        {
            const std::lock_guard lock{mutex};
            update.status.training = trainer.state;
        }
        std::ofstream csv{config.output / "training.csv", request.checkpoint.empty() ? std::ios::trunc : std::ios::app};
        csv.exceptions(std::ios::failbit | std::ios::badbit);
        if (request.checkpoint.empty()) std::println(csv, "step,loss,samples_per_second,elapsed_seconds");
        std::uint64_t logged_iterations{};
        double accumulated_loss{}, accumulated_seconds{};
        const auto checkpoint = [&](const bool final) {
            report(Stage::saving);
            const auto path = final ? config.output / "final.safetensors" : config.output / "checkpoints" / std::format("step-{:06}.safetensors", trainer.state.step);
            std::filesystem::create_directories(path.parent_path());
            trainer.save(path, {{"flowdit.representation", serialize_representation(config.image)}});
            {
                const std::lock_guard lock{mutex};
                update.checkpoints.push_back(path);
            }
        };
        const auto record = [&] {
            const TrainingRecord row{trainer.state.step, static_cast<float>(accumulated_loss / static_cast<double>(logged_iterations)), static_cast<double>(logged_iterations * config.batch) / accumulated_seconds, trainer.state.elapsed_seconds};
            std::println(csv, "{},{},{},{}", row.step, row.loss, row.samples_per_second, row.training_seconds);
            csv.flush();
            {
                const std::lock_guard lock{mutex};
                update.metrics.push_back(row);
            }
            logged_iterations = 0;
            accumulated_loss = accumulated_seconds = 0;
        };
        while (trainer.state.step < config.end_step && !cancellation.stop_requested()) {
            report(Stage::optimizing);
            const auto iterations = std::min({static_cast<std::uint64_t>(config.execution_steps), config.end_step - trainer.state.step, config.log_interval - trainer.state.step % config.log_interval, config.preview_interval - trainer.state.step % config.preview_interval, config.save_interval - trainer.state.step % config.save_interval});
            for (std::uint64_t i = 0; i < iterations; ++i) {
                const auto started = std::chrono::steady_clock::now();
                const auto step = trainer.state.step + 1;
                std::seed_seq seed{static_cast<std::uint32_t>(config.seed), static_cast<std::uint32_t>(config.seed >> 32), static_cast<std::uint32_t>(step), static_cast<std::uint32_t>(step >> 32)};
                std::mt19937_64 random{seed};
                for (auto& index : indices) index = static_cast<std::uint32_t>((random() >> 32) * request.dataset->labels.size() >> 32);
                request.dataset->read(indices, input);
                const auto tensor = representation.encode(input, config.horizontal_flip, config.seed, step);
                const float loss = trainer.optimize(tensor);
                const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
                trainer.state.elapsed_seconds += elapsed;
                ++logged_iterations;
                accumulated_loss += loss;
                accumulated_seconds += elapsed;
            }
            {
                const std::lock_guard lock{mutex};
                update.status.training = trainer.state;
            }
            if (trainer.state.step % config.log_interval == 0 || trainer.state.step == config.end_step) record();
            if (trainer.state.step % config.preview_interval == 0 && !cancellation.stop_requested()) {
                for (const auto source : {ParameterSource::parameters, ParameterSource::exponential_average}) {
                    report(source == ParameterSource::parameters ? Stage::preview_parameters : Stage::preview_ema);
                    SamplingRuntime runtime{stream, trainer.model, config.preview.count};
                    PixelRepresentation decoder{stream, {config.image.width, config.image.height, config.image.channels}, config.preview.count};
                    const auto* parameters = source == ParameterSource::parameters ? trainer.parameter_buffer.parameters.data() : trainer.parameter_buffer.ema.data();
                    const auto result = runtime.sample(parameters, config.preview, observe());
                    if (!result) break;
                    std::filesystem::create_directories(config.output / "samples");
                    publish_sample(*result, decoder, stream, {.path = config.output / "samples" / std::format("step-{:06}-{}.png", trainer.state.step, source == ParameterSource::parameters ? "parameters" : "ema"), .request = config.preview, .source = source, .training_step = trainer.state.step, .model = config.model, .image = config.image}, false);
                }
            }
            if (trainer.state.step % RunConfiguration::save_interval == 0 && trainer.state.step < config.end_step && !cancellation.stop_requested()) checkpoint(false);
        }
        if (logged_iterations) record();
        checkpoint(trainer.state.step >= config.end_step);
    }
    void Session::execute(const SampleRequest& request) {
        report(Stage::loading);
        if (!sampler || sampler_checkpoint != request.checkpoint || sampler->batch != request.sampling.count) {
            sampler.reset();
            if (const auto result = cudaSetDevice(0); result != cudaSuccess) throw std::runtime_error{cudaGetErrorString(result)};
            const auto metadata = serialization::safetensors::read_metadata(request.checkpoint);
            sampler = std::make_unique<Sampler>(deserialize_model(metadata.at("flowdit.model")), request.checkpoint, 0, request.sampling.count);
            sampler_checkpoint = request.checkpoint;
            sampler_image = deserialize_representation(metadata.at("flowdit.representation"));
        }
        std::filesystem::create_directories(request.output.parent_path());
        report(Stage::sampling);
        const FrameInfo info{request.sampling, sampler_image};
        PixelRepresentation representation{sampler->stream, {sampler_image.width, sampler_image.height, sampler_image.channels}, request.sampling.count};
        const auto result = sampler->sample(request.sampling, observe(&representation, &info));
        if (!result) return;
        report(Stage::saving);
        publish_sample(*result, representation, sampler->stream, {.path = request.output, .checkpoint = request.checkpoint, .request = request.sampling, .model = sampler->configuration, .image = sampler_image}, true);
    }
} // namespace flowdit
