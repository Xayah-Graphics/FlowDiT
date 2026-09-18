module flowdit.dataset.load;
import flowdit.dataset.cifar;
import flowdit.dataset.mnist;
import flowdit.dataset.folder;
import std;
namespace flowdit {
    std::optional<DatasetInfo> inspect_dataset(const std::filesystem::path& directory) {
        if (std::filesystem::exists(directory / "batches.meta.txt") || std::filesystem::exists(directory / "fine_label_names.txt")) return inspect_cifar(directory);
        if (std::filesystem::exists(directory / "train-images-idx3-ubyte")) return inspect_mnist(directory);
        if (std::filesystem::is_directory(directory / "train")) {
            const auto data = load_folder(directory);
            return DatasetInfo{data.specification, static_cast<std::uint32_t>(data.labels.size())};
        }
        return std::nullopt;
    }
    Dataset load_dataset(const std::filesystem::path& directory, const DatasetSplit split) {
        Dataset result;
        if (std::filesystem::exists(directory / "batches.meta.txt") || std::filesystem::exists(directory / "fine_label_names.txt")) result = load_cifar(directory, split == DatasetSplit::test);
        else if (std::filesystem::exists(directory / "train-images-idx3-ubyte")) result = load_mnist(directory, split == DatasetSplit::test);
        else if (std::filesystem::is_directory(directory / "train")) result = load_folder(directory, split == DatasetSplit::test);
        else throw std::runtime_error{"Unsupported dataset format: " + directory.string()};
        result.split = split;
        if (split != DatasetSplit::test) {
            // Fixed, stratified 5% holdout; official test files are never used to train.
            std::vector<std::uint32_t> counters(result.specification.classes.size());
            std::size_t destination{};
            for (std::size_t i = 0; i < result.labels.size(); ++i) {
                const bool held_out = counters[result.labels[i]]++ % 20 == 0;
                if (held_out != (split == DatasetSplit::validation)) continue;
                result.labels[destination]    = result.labels[i];
                result.records[destination++] = result.records[i];
            }
            result.labels.resize(destination);
            result.records.resize(destination);
        }
        return result;
    }
} // namespace flowdit
