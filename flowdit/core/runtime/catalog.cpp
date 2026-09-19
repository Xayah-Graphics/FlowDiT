module flowdit.runtime.catalog;
import flowdit.serialization.safetensors;
import std;
namespace flowdit {
    const std::filesystem::path Catalog::directory{FLOWDIT_DATA_DIRECTORY};
    void Catalog::scan() {
        datasets.clear();
        for (const auto& folder : std::filesystem::directory_iterator{directory}) {
            if (!folder.is_directory() || folder.path().filename().string().starts_with('.')) continue;
            const auto key    = folder.path().filename().string();
            auto& dataset     = datasets[key];
            dataset.directory = folder.path();
            try {
                dataset.info = inspect_dataset(folder.path());
                if (key != "afhq_v2") dataset.training_error = "Training is currently available for AFHQ-512 only.";
                else if (dataset.info && (dataset.info->specification.width != 512 || dataset.info->specification.height != 512 || dataset.info->specification.channels != 3)) dataset.training_error = "AFHQ training requires 512 x 512 RGB images.";
                if (dataset.info) refresh(dataset);
            } catch (const std::exception& failure) {
                dataset.error = failure.what();
            }
        }
    }
    void Catalog::refresh(DatasetEntry& dataset, const std::filesystem::path& changed) {
        std::vector<std::filesystem::path> directories;
        if (changed.empty()) {
            dataset.runs.clear();
            const auto runs = dataset.directory / ".flowdit" / "runs";
            if (!std::filesystem::exists(runs)) return;
            for (const auto& folder : std::filesystem::directory_iterator{runs})
                if (folder.is_directory()) directories.push_back(folder.path());
        } else directories.push_back(changed.parent_path().parent_path());
        for (const auto& directory : directories) {
            auto& run = dataset.runs[directory.filename().string()];
            run.error.clear();
            try {
                run.configuration         = output::read_configuration(directory);
                run.configuration.dataset = dataset.directory;
                std::vector<std::filesystem::path> paths;
                if (changed.empty()) {
                    const auto checkpoints = directory / "checkpoints";
                    if (std::filesystem::exists(checkpoints))
                        for (const auto& entry : std::filesystem::directory_iterator{checkpoints})
                            if (entry.is_regular_file() && entry.path().extension() == ".safetensors") paths.push_back(entry.path());
                } else paths.push_back(changed);
                for (const auto& path : paths) {
                    auto found = std::ranges::find(run.checkpoints, path, &CheckpointEntry::path);
                    if (found == run.checkpoints.end()) found = run.checkpoints.emplace(run.checkpoints.end());
                    auto& checkpoint = *found;
                    checkpoint       = {.path = path};
                    try {
                        const auto file = serialization::safetensors::read(path, std::array<std::string_view, 1>{"training.state"});
                        if (file.metadata.at("flowdit.system") != "usit-dcae-v1") throw std::runtime_error{"Unsupported checkpoint format"};
                        checkpoint.model = deserialize_model(file.metadata.at("flowdit.model"));
                        checkpoint.image = deserialize_image(file.metadata.at("flowdit.image"));
                        std::array<std::uint64_t, 4> state;
                        std::memcpy(state.data(), file.tensors.front().data.data(), sizeof(state));
                        checkpoint.step             = state[0];
                        checkpoint.seed             = state[2];
                        checkpoint.training_seconds = std::bit_cast<double>(state[3]);
                    } catch (const std::exception& failure) {
                        checkpoint.error = failure.what();
                    }
                }
                std::ranges::sort(run.checkpoints, [](const CheckpointEntry& a, const CheckpointEntry& b) { return a.step > b.step; });
            } catch (const std::exception& failure) {
                run.error = failure.what();
            }
        }
    }
    RunConfiguration Catalog::training(const DatasetEntry& dataset) const {
        if (!dataset.training_error.empty()) throw std::runtime_error{dataset.training_error};
        auto result    = training_configuration(dataset.info->specification);
        result.dataset = dataset.directory;
        result.output  = dataset.directory / ".flowdit" / "runs" / std::format("{:%Y%m%d-%H%M%S}", std::chrono::floor<std::chrono::microseconds>(std::chrono::system_clock::now()));
        return result;
    }
    std::filesystem::path Catalog::inference(const RunEntry& run) const {
        const auto stamp = std::format("{:%Y%m%d-%H%M%S}", std::chrono::floor<std::chrono::microseconds>(std::chrono::system_clock::now()));
        return run.configuration.output / "inference" / (stamp + ".png");
    }
} // namespace flowdit
