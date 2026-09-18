module flowdit.dataset.cifar;
import std;
namespace flowdit {
    DatasetInfo inspect_cifar(const std::filesystem::path& directory) {
        const bool hundred = std::filesystem::exists(directory / "fine_label_names.txt");
        DatasetInfo result{{hundred ? "CIFAR-100" : "CIFAR-10", 32u, 32u, 3u}, 50'000u};
        std::ifstream names{directory / (hundred ? "fine_label_names.txt" : "batches.meta.txt")};
        names.exceptions(std::ios::badbit);
        for (std::string name; std::getline(names, name);) result.specification.classes.push_back(name);
        return result;
    }
    Dataset load_cifar(const std::filesystem::path& directory, const bool test) {
        const auto info          = inspect_cifar(directory);
        const bool hundred       = info.specification.classes.size() == 100;
        const std::size_t prefix = hundred ? 2 : 1, stride = 3072 + prefix;
        Dataset result{.specification = info.specification};
        if (hundred || test) result.files.push_back(directory / (hundred ? test ? "test.bin" : "train.bin" : "test_batch.bin"));
        else
            for (std::uint32_t batch = 1; batch <= 5; ++batch) result.files.push_back(directory / std::format("data_batch_{}.bin", batch));
        for (std::uint32_t f = 0; f < result.files.size(); ++f) {
            const std::size_t count = std::filesystem::file_size(result.files[f]) / stride;
            std::vector<std::uint8_t> bytes(count * stride);
            std::ifstream file{result.files[f], std::ios::binary};
            file.exceptions(std::ios::failbit | std::ios::badbit);
            file.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
            for (std::size_t i = 0; i < count; ++i) {
                result.labels.push_back(bytes[i * stride + prefix - 1]);
                result.records.push_back({f, i * stride + prefix});
            }
        }
        return result;
    }
} // namespace flowdit
