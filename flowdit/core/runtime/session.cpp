module;
#include <flowdit/cuda.h>
module flowdit.runtime.session;
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
    SamplingObserver Session::observe(const FrameInfo* info) {
        SamplingObserver observer{.stop = cancellation.get_token(), .progress = [this](const SamplingProgress& progress) {
                                      {
                                          const std::lock_guard lock{mutex};
                                          update.status.sampling = progress;
                                      }
                                      if (events.notify) events.notify();
                                  }};
        if (info && events.image)
            observer.image = [this, info = *info](const std::uint8_t* pixels, const std::uint32_t width, const std::uint32_t height, const ::cuda::stream_ref stream) { events.image(info, pixels, width, height, stream); };
        return observer;
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
        if (!request.checkpoint.empty()) config.patch_size = read_model_configuration(request.checkpoint).patch_size;
        Trainer trainer{*request.dataset, 0, config.seed, config.optimizer, config.patch_size};
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
            trainer.save(path);
            {
                const std::lock_guard lock{mutex};
                update.checkpoints.push_back(path);
            }
        };
        const auto record = [&] {
            const TrainingRecord row{trainer.state.step, static_cast<float>(accumulated_loss / static_cast<double>(logged_iterations)), static_cast<double>(logged_iterations * Trainer::batch) / accumulated_seconds, trainer.state.elapsed_seconds};
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
            const auto statistics = trainer.optimize(iterations);
            logged_iterations += iterations;
            accumulated_loss += static_cast<double>(statistics.average_loss) * iterations;
            accumulated_seconds += statistics.elapsed_seconds;
            {
                const std::lock_guard lock{mutex};
                update.status.training = trainer.state;
            }
            if (trainer.state.step % config.log_interval == 0 || trainer.state.step == config.end_step) record();
            if (trainer.state.step % config.preview_interval == 0 && !cancellation.stop_requested()) {
                for (const auto source : {ParameterSource::parameters, ParameterSource::exponential_average}) {
                    report(source == ParameterSource::parameters ? Stage::preview_parameters : Stage::preview_ema);
                    const auto observer = observe();
                    auto images         = trainer.sample(config.preview, source, observer);
                    if (!images) break;
                    std::filesystem::create_directories(config.output / "samples");
                    auto sample = std::make_shared<SampleOutput>(SampleOutput{.info = {.path = config.output / "samples" / std::format("step-{:06}-{}.png", trainer.state.step, source == ParameterSource::parameters ? "parameters" : "ema"), .request = config.preview, .source = source, .training_step = trainer.state.step, .nfe = images->nfe, .model = images->model, .labels = images->labels}, .images = std::move(*images)});
                    output::write_sample(*sample);
                    {
                        const std::lock_guard lock{mutex};
                        update.samples.push_back(std::move(sample));
                    }
                }
            }
            if (trainer.state.step % RunConfiguration::save_interval == 0 && trainer.state.step < config.end_step && !cancellation.stop_requested()) checkpoint(false);
        }
        if (logged_iterations) record();
        checkpoint(trainer.state.step >= config.end_step);
    }
    void Session::execute(const SampleRequest& request) {
        report(Stage::loading);
        if (!sampler || sampler_checkpoint != request.checkpoint) {
            sampler.reset();
            if (const auto result = cudaSetDevice(0); result != cudaSuccess) throw std::runtime_error{cudaGetErrorString(result)};
            sampler = std::make_unique<Sampler>(request.checkpoint, 0);
            sampler_checkpoint = request.checkpoint;
        }
        std::filesystem::create_directories(request.output.parent_path());
        report(Stage::sampling);
        const FrameInfo info{request.sampling, sampler->configuration};
        auto images = sampler->sample(request.sampling, observe(&info));
        if (!images) return;
        report(Stage::saving);
        auto sample = std::make_shared<SampleOutput>(SampleOutput{.info = {.path = request.output, .checkpoint = request.checkpoint, .request = request.sampling, .nfe = images->nfe, .model = images->model, .labels = images->labels}, .images = std::move(*images)});
        output::write_sample(*sample);
        const std::lock_guard lock{mutex};
        update.samples.push_back(std::move(sample));
    }
} // namespace flowdit
