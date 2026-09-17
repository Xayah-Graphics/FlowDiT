module flowdit.dataset.load;
import flowdit.dataset.cifar10;
import flowdit.dataset.mnist;
import std;
namespace flowdit {
    std::optional<DatasetInfo> inspect_dataset(const std::filesystem::path& directory) {
        if (std::filesystem::exists(directory / "data_batch_1.bin")) return inspect_cifar10();
        if (std::filesystem::exists(directory / "train-images-idx3-ubyte")) return inspect_mnist(directory);
        return std::nullopt;
    }
    Dataset load_dataset(const std::filesystem::path& directory) {
        if (std::filesystem::exists(directory / "data_batch_1.bin")) return load_cifar10(directory);
        if (std::filesystem::exists(directory / "train-images-idx3-ubyte")) return load_mnist(directory);
        throw std::runtime_error{"Unsupported dataset format: " + directory.string()};
    }
} // namespace flowdit
