export module flowdit.dataset.load;
export import flowdit.dataset.types;
import std;
export namespace flowdit {
    Dataset load_dataset(DatasetKind kind, const std::filesystem::path& directory);
} // namespace flowdit
