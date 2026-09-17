module;
#include <csignal>
#include <flowdit/cuda.h>
module flowdit.headless;
import flowdit.runtime.session;
import flowdit.runtime.catalog;
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
  flowdit --gui
  flowdit list [DATASET]
  flowdit train DATASET [--run RUN] [--steps N] [options]
  flowdit sample DATASET [--run RUN] [--checkpoint NAME] [options]

Train options:
  --seed N           --learning-rate VALUE

Sample options:
  --class all|N       --solver euler|heun|rk4
  --steps N           --guidance VALUE      --seed N

DATASET is a directory name in the compiled data root. RUN and NAME come from list.
New training creates a run; --run resumes its highest-step checkpoint.
Sampling defaults to the newest run and its highest-step checkpoint.
All outputs are stored beneath DATASET/.flowdit/runs/RUN.
Training and standalone sampling are exclusive. Training retains periodic raw/EMA previews.
Ctrl+C stops at a complete step.)");
            std::println("\nData root: {}", Catalog::directory.string());
            return 0;
        }
        Catalog catalog;
        catalog.scan();
        const auto command = arguments.front();
        if (command == "list") {
            std::println("Data root: {}", Catalog::directory.string());
            for (const auto& [name, dataset] : catalog.datasets) {
                if (arguments.size() > 1 && name != arguments[1]) continue;
                std::println("{} | {} | {}", name, dataset.name, !dataset.error.empty() ? dataset.error : dataset.kind ? std::format("{} images", dataset.count) : "Unsupported dataset format");
                for (const auto& [run_name, run] : dataset.runs) {
                    std::println("  {}{}", run_name, run.error.empty() ? "" : " | " + run.error);
                    for (const auto& checkpoint : run.checkpoints) std::println("    {} | step {}{}", checkpoint.path.filename().string(), checkpoint.step, checkpoint.error.empty() ? "" : " | " + checkpoint.error);
                }
            }
            return 0;
        }
        const auto& dataset = catalog.datasets.at(std::string{arguments[1]});
        if (!dataset.error.empty()) throw std::runtime_error{dataset.error};
        if (!dataset.kind) throw std::runtime_error{"Unsupported dataset format: " + dataset.name};
        std::string run_name;
        for (std::size_t i = 2; i < arguments.size(); i += 2)
            if (arguments[i] == "--run") run_name = arguments[i + 1];
        interrupted                 = 0;
        const auto previous_handler = std::signal(SIGINT, interrupt);
        Session session;
        if (command == "train") {
            TrainRequest request;
            auto& config      = request.configuration;
            config = catalog.training(dataset);
            if (!run_name.empty()) {
                const auto& run = dataset.runs.at(run_name);
                if (!run.error.empty()) throw std::runtime_error{run.error};
                if (run.checkpoints.empty()) throw std::runtime_error{"No checkpoint in run " + run_name};
                const auto& checkpoint = run.checkpoints.front();
                if (!checkpoint.error.empty()) throw std::runtime_error{checkpoint.error};
                config = run.configuration;
                request.checkpoint = checkpoint.path;
            }
            for (std::size_t index = 2; index < arguments.size(); ++index) {
                const auto option = arguments[index];
                const auto value  = arguments[++index];
                if (option == "--run") continue;
                if (option == "--steps") config.end_step = number<std::uint64_t>(value);
                else if (option == "--seed") config.seed = number<std::uint64_t>(value);
                else if (option == "--learning-rate") config.optimizer.learning_rate = number<float>(value);
                else throw std::runtime_error{"Unknown training option: " + std::string{option}};
            }
            std::println("FlowDiT train | batch {} | target {} | {}", Trainer::batch, config.end_step, config.output.string());
            std::cout.flush();
            request.dataset = std::make_shared<const Dataset>(load_dataset(config.dataset_type, config.dataset));
            session.start(std::move(request));
        } else if (command == "sample") {
            if (dataset.runs.empty()) throw std::runtime_error{"No training runs for " + dataset.name};
            const auto& run = run_name.empty() ? dataset.runs.begin()->second : dataset.runs.at(run_name);
            if (!run.error.empty()) throw std::runtime_error{run.error};
            if (run.checkpoints.empty()) throw std::runtime_error{"No checkpoints in the selected run"};
            SampleRequest request;
            request.output = catalog.inference(run);
            const CheckpointEntry* selected = &run.checkpoints.front();
            for (std::size_t index = 2; index < arguments.size(); ++index) {
                const auto option = arguments[index];
                const auto value = arguments[++index];
                if (option == "--run") continue;
                if (option == "--checkpoint") {
                    const auto found = std::ranges::find(run.checkpoints, value, [](const CheckpointEntry& entry) { return entry.path.filename().string(); });
                    if (found == run.checkpoints.end()) throw std::runtime_error{"Unknown checkpoint: " + std::string{value}};
                    selected = &*found;
                } else if (option == "--class") {
                    if (value != "all") request.sampling.class_index = number<std::uint32_t>(value);
                } else if (option == "--solver") request.sampling.solver = solver(value);
                else if (option == "--steps") request.sampling.step_count = number<std::uint32_t>(value);
                else if (option == "--guidance") request.sampling.guidance = number<float>(value);
                else if (option == "--seed") request.sampling.seed = number<std::uint64_t>(value);
                else throw std::runtime_error{"Unknown sampling option: " + std::string{option}};
            }
            if (!selected->error.empty()) throw std::runtime_error{selected->error};
            request.checkpoint = selected->path;
            session.start(std::move(request));
        } else throw std::runtime_error{"Unknown FlowDiT command"};
        bool closing{};
        std::string failure;
        for (;;) {
            if (interrupted && !closing) {
                session.stop();
                interrupted = 0;
                closing     = true;
            }
            auto update = session.receive();
            for (const auto& record : update.metrics) std::println("step {:>7} | loss {:.6f} | {:.1f} samples/s | {:.2f}s training", record.step, record.loss, record.samples_per_second, record.training_seconds);
            for (const auto& sample : update.samples) std::println("Saved {}", sample->info.path.string());
            for (const auto& checkpoint : update.checkpoints) std::println("Saved {}", checkpoint.string());
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
