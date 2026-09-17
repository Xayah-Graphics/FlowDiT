module;
#include <flowdit/cuda.h>
export module flowdit.runtime.session;
export import flowdit.io.output;
export import flowdit.dataset.load;
export import flowdit.training.trainer;
import flowdit.sampling.sampler;
import std;
export namespace flowdit {
    struct TrainRequest final {
        RunConfiguration configuration;
        std::shared_ptr<const Dataset> dataset;
        std::filesystem::path checkpoint;
    };
    struct SampleRequest final {
        std::filesystem::path checkpoint, output;
        int device{};
        ParameterSource source{ParameterSource::exponential_average};
        SamplingRequest sampling;
        bool fid{};
    };
    enum class Mode { none, training, sampling, fid };
    enum class Stage { idle, loading, optimizing, paused, preview_parameters, preview_ema, sampling, saving, complete, stopped, failed };
    inline constexpr std::array<std::string_view, 11> stage_names{"Idle", "Loading", "Training", "Paused", "Raw preview", "EMA preview", "Sampling", "Saving", "Complete", "Stopped", "Failed"};
    struct SessionStatus final {
        Mode mode{Mode::none};
        Stage stage{Stage::idle};
        bool busy{}, closing{}, finished{};
        TrainingState training;
        SamplingProgress sampling;
        std::uint32_t exported{};
        std::chrono::steady_clock::time_point started{}, ended{};
        std::string error;
    };
    struct FrameInfo final {
        std::uint64_t training_step{};
        ParameterSource source{ParameterSource::exponential_average};
        SamplingRequest request;
        SamplingProgress progress;
        ModelConfiguration model;
    };
    struct SessionObserver final {
        std::function<void()> notify;
        std::function<void(const FrameInfo&, const std::uint8_t*, std::uint32_t, std::uint32_t, ::cuda::stream_ref)> image;
    };
    struct SessionUpdate final {
        SessionStatus status;
        std::vector<TrainingRecord> metrics;
        std::vector<std::shared_ptr<const SampleOutput>> samples;
        std::vector<std::filesystem::path> checkpoints;
        std::shared_ptr<const TrainingBatch> batch;
        std::vector<std::string> messages;
    };
    struct Session final {
        explicit Session(SessionObserver events = {});
        ~Session();
        void start(std::variant<TrainRequest, SampleRequest> request);
        void pause();
        void resume();
        void save();
        void inspect();
        void stop();
        void shutdown();
        SessionUpdate receive();

    private:
        SessionObserver events;
        std::mutex mutex;
        std::condition_variable condition;
        SessionUpdate update;
        std::optional<std::variant<TrainRequest, SampleRequest>> requested;
        std::stop_source cancellation;
        std::atomic_bool pause_requested{}, save_requested{}, inspect_requested{};
        std::unique_ptr<Sampler> sampler;
        std::filesystem::path sampler_checkpoint;
        int sampler_device{};
        ParameterSource sampler_source{};
        std::jthread worker;
        void report(Stage stage, std::string message = {});
        SamplingObserver observe(const FrameInfo& info);
        void run();
        void execute(const TrainRequest& request);
        void execute(const SampleRequest& request);
    };
} // namespace flowdit
