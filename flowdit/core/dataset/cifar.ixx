export module flowdit.dataset.cifar;
export import flowdit.dataset.types;
import std;
export namespace flowdit {
    DatasetInfo inspect_cifar(const std::filesystem::path& directory);
    Dataset load_cifar(const std::filesystem::path& directory, bool test = false);
} // namespace flowdit
