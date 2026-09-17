export module flowdit.dataset.cifar10;
export import flowdit.dataset.types;
import std;
export namespace flowdit {
    Dataset load_cifar10(const std::filesystem::path& directory);
} // namespace flowdit
