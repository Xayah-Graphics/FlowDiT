export module flowdit.runtime.catalog;
export import flowdit.io.output;
export import flowdit.dataset.load;
import std;
export namespace flowdit {
    struct CheckpointEntry final {
        std::filesystem::path path;
        ModelConfiguration model;
        ImageSpecification image;
        std::uint64_t step{}, seed{};
        double training_seconds{};
        std::string error;
    };
    struct RunEntry final {
        RunConfiguration configuration;
        std::vector<CheckpointEntry> checkpoints;
        std::string error;
    };
    struct DatasetEntry final {
        std::filesystem::path directory;
        std::optional<DatasetInfo> info;
        std::string error;
        std::map<std::string, RunEntry, std::greater<>> runs;
    };
    struct Catalog final {
        static const std::filesystem::path directory;
        std::map<std::string, DatasetEntry> datasets;
        void scan();
        void refresh(DatasetEntry& dataset, const std::filesystem::path& changed = {});
        RunConfiguration training(const DatasetEntry& dataset) const;
        std::filesystem::path inference(const RunEntry& run) const;
    };
} // namespace flowdit
