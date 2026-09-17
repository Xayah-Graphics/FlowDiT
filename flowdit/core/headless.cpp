module;
#include <csignal>
#include <flowdit/cuda.h>
module flowdit.headless;
import flowdit.runtime.session;
import std;
namespace flowdit::headless {
    namespace {
        volatile std::sig_atomic_t interrupted{};
        void interrupt(int) {
            interrupted = 1;
        }
        template <typename Type>
        Type number(const std::string_view text) {
            Type value{};
            std::from_chars(text.data(), text.data() + text.size(), value);
            return value;
        }
        SamplingSolver solver(const std::string_view text) {
            if (text == "euler") return SamplingSolver::euler;
            if (text == "heun") return SamplingSolver::heun;
            if (text == "rk4") return SamplingSolver::rk4;
            throw std::runtime_error{"Unknown ODE solver"};
        }
    } // namespace
    int run(const std::span<const std::string_view> arguments) {
        if (arguments.empty() || arguments.front() == "--help") {
            std::println(R"(FlowDiT
  flowdit --gui [--dataset-type cifar10|mnist] [--dataset DIRECTORY] [--output DIRECTORY] [--checkpoint FILE] [--device N]
  flowdit train DATASET OUTPUT [END_STEP] [resume] [options]
  flowdit sample CHECKPOINT OUTPUT.png [CLASS|all] [euler|heun|rk4] [STEPS] [GUIDANCE] [SEED]
  flowdit sample-fid CHECKPOINT OUTPUT_DIRECTORY [euler|heun|rk4] [STEPS] [GUIDANCE] [SEED]

Train options:
  --dataset-type cifar10|mnist
  --checkpoint FILE   --device N           --seed N
  --execution-steps N  --log-interval N     --save-interval N
  --preview-interval N --preview-steps N    --learning-rate VALUE

Training and standalone sampling are exclusive. Training retains periodic raw/EMA previews.
sample-fid exports 50,000 PNGs and manifest.csv. Ctrl+C stops at a complete step.)");
            return 0;
        }
        interrupted                 = 0;
        const auto previous_handler = std::signal(SIGINT, interrupt);
        Session session;
        const auto command = arguments.front();
        if (command == "train") {
            TrainRequest request;
            auto& config      = request.configuration;
            config.dataset    = arguments[1];
            config.output     = arguments[2];
            std::size_t index = 3;
            if (index < arguments.size() && !arguments[index].starts_with("--") && arguments[index] != "resume") config.end_step = number<std::uint64_t>(arguments[index++]);
            if (index < arguments.size() && arguments[index] == "resume") {
                const auto end_step = config.end_step;
                const auto dataset  = config.dataset;
                config              = output::read_configuration(config.output);
                config.dataset      = dataset;
                config.end_step     = end_step;
                request.checkpoint  = output::latest_checkpoint(config.output);
                ++index;
            }
            for (; index < arguments.size(); ++index) {
                const auto option = arguments[index];
                const auto value  = arguments[++index];
                if (option == "--dataset-type") config.dataset_type = dataset_kind(value);
                else if (option == "--checkpoint") request.checkpoint = value;
                else if (option == "--device") config.device = number<int>(value);
                else if (option == "--seed") config.seed = number<std::uint64_t>(value);
                else if (option == "--execution-steps") config.execution_steps = number<std::uint32_t>(value);
                else if (option == "--log-interval") config.log_interval = number<std::uint32_t>(value);
                else if (option == "--save-interval") config.save_interval = number<std::uint32_t>(value);
                else if (option == "--preview-interval") config.preview_interval = number<std::uint32_t>(value);
                else if (option == "--preview-steps") config.preview.step_count = number<std::uint32_t>(value);
                else if (option == "--learning-rate") config.optimizer.learning_rate = number<float>(value);
                else throw std::runtime_error{"Unknown training option: " + std::string{option}};
            }
            std::println("FlowDiT train | cuda:{} | batch {} | target {} | {}", config.device, Trainer::batch, config.end_step, config.output.string());
            std::cout.flush();
            request.dataset = std::make_shared<const Dataset>(load_dataset(config.dataset_type, config.dataset));
            session.start(std::move(request));
        } else if (command == "sample" || command == "sample-fid") {
            SampleRequest request{.checkpoint = arguments[1], .output = arguments[2]};
            request.fid       = command == "sample-fid";
            std::size_t index = 3;
            if (!request.fid && index < arguments.size()) {
                if (arguments[index] != "all") request.sampling.class_index = number<std::uint32_t>(arguments[index]);
                ++index;
            }
            if (index < arguments.size()) request.sampling.solver = solver(arguments[index++]);
            if (index < arguments.size()) request.sampling.step_count = number<std::uint32_t>(arguments[index++]);
            if (index < arguments.size()) request.sampling.guidance = number<float>(arguments[index++]);
            if (index < arguments.size()) request.sampling.seed = number<std::uint64_t>(arguments[index++]);
            session.start(std::move(request));
        } else throw std::runtime_error{"Unknown FlowDiT command"};
        bool closing{};
        std::string failure;
        Stage previous_stage{Stage::idle};
        std::uint32_t previous_exported{};
        for (;;) {
            if (interrupted && !closing) {
                session.stop();
                interrupted = 0;
                closing     = true;
            }
            auto update = session.receive();
            for (const auto& message : update.messages) std::println("{}", message);
            if (update.status.stage != previous_stage) std::println("{}", stage_names[static_cast<std::size_t>(update.status.stage)]);
            previous_stage = update.status.stage;
            for (const auto& record : update.metrics) std::println("step {:>7} | loss {:.6f} | {:.1f} samples/s | {:.2f}s training", record.step, record.loss, record.samples_per_second, record.training_seconds);
            for (const auto& sample : update.samples) std::println("saved {} | {} NFE | {} model forwards", sample->info.path.string(), sample->info.nfe, sample->info.nfe * 2);
            if (update.status.exported != previous_exported) std::println("exported {} / 50000", update.status.exported);
            previous_exported = update.status.exported;
            std::cout.flush();
            if (!update.status.error.empty()) failure = update.status.error;
            if (!update.status.busy) {
                session.shutdown();
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds{20});
        }
        std::signal(SIGINT, previous_handler);
        if (!failure.empty()) throw std::runtime_error{failure};
        return closing ? 130 : 0;
    }
} // namespace flowdit::headless
