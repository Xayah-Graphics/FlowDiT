module flowdit.dataset.cifar10;
import std;
namespace flowdit {
    DatasetInfo inspect_cifar10() {
        return {{"CIFAR-10", 32u, 32u, 3u, {"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"}}, 50'000u};
    }
    Dataset load_cifar10(const std::filesystem::path& directory) {
        constexpr std::size_t image_bytes = 32uz * 32uz * 3uz;
        Dataset result{.specification = inspect_cifar10().specification, .labels = std::vector<std::uint32_t>(50'000uz), .records = std::vector<SampleRecord>(50'000uz)};
        std::vector<std::uint8_t> bytes(10'000uz * (image_bytes + 1uz));
        for (std::uint32_t batch = 0; batch < 5; ++batch) {
            result.files.push_back(directory / std::format("data_batch_{}.bin", batch + 1u));
            std::ifstream file{result.files.back(), std::ios::binary};
            file.exceptions(std::ios::failbit | std::ios::badbit);
            file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
            for (std::uint32_t image = 0; image < 10'000; ++image) {
                const std::size_t index = static_cast<std::size_t>(batch) * 10'000uz + image;
                result.labels[index] = bytes[image * (image_bytes + 1uz)];
                result.records[index] = {batch, image * (image_bytes + 1uz) + 1uz};
            }
        }
        return result;
    }
} // namespace flowdit
