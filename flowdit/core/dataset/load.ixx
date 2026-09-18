export module flowdit.dataset.load;
export import flowdit.dataset.types;
import std;
export namespace flowdit {
    std::optional<DatasetInfo> inspect_dataset(const std::filesystem::path& directory);
    Dataset load_dataset(const std::filesystem::path& directory, DatasetSplit split = DatasetSplit::training);
} // namespace flowdit
