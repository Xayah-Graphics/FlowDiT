export module flowdit.dataset.mnist;
export import flowdit.dataset.types;
import std;
export namespace flowdit {
    DatasetInfo inspect_mnist(const std::filesystem::path& directory);
    Dataset load_mnist(const std::filesystem::path& directory, bool test = false);
} // namespace flowdit
