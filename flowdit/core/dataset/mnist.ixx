export module flowdit.dataset.mnist;
export import flowdit.dataset.types;
import std;
export namespace flowdit {
    Dataset load_mnist(const std::filesystem::path& directory);
} // namespace flowdit
