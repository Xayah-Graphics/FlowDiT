export module flowdit.dataset.load;
export import flowdit.dataset.types;
import std;
export namespace flowdit {
    DatasetKind dataset_kind(std::string_view name);
    Dataset load_dataset(DatasetKind kind, const std::filesystem::path& directory);
} // namespace flowdit
