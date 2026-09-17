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
        SamplingRequest sampling;
    };
    enum class Mode { none, training, sampling };
    enum class Stage { idle, loading, optimizing, preview_parameters, preview_ema, sampling, saving, complete, stopped, failed };
    inline constexpr std::array<std::string_view, 10> stage_names{"Idle", "Loading", "Training", "Raw preview", "EMA preview", "Sampling", "Saving", "Complete", "Stopped", "Failed"};
    struct SessionStatus final {
        Mode mode{Mode::none};
        Stage stage{Stage::idle};
        bool busy{}, closing{}, finished{};
        TrainingState training;
        SamplingProgress sampling;
        std::chrono::steady_clock::time_point started{}, ended{};
        std::string error;
    };
    struct FrameInfo final {
        SamplingRequest request;
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
    };
    struct Session final {
        explicit Session(SessionObserver events = {});
        ~Session();
        void start(std::variant<TrainRequest, SampleRequest> request);
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
        std::unique_ptr<Sampler> sampler;
        std::filesystem::path sampler_checkpoint;
        std::jthread worker;
        void report(Stage stage);
        SamplingObserver observe(const FrameInfo* info = nullptr);
        void run();
        void execute(const TrainRequest& request);
        void execute(const SampleRequest& request);
    };
} // namespace flowdit
