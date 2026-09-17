module flowdit.dataset.load;
import flowdit.dataset.cifar10;
import flowdit.dataset.mnist;
import std;
namespace flowdit {
    Dataset load_dataset(const DatasetKind kind, const std::filesystem::path& directory) {
        switch (kind) {
        case DatasetKind::cifar10: return load_cifar10(directory);
        case DatasetKind::mnist: return load_mnist(directory);
        }
        std::unreachable();
    }
} // namespace flowdit
