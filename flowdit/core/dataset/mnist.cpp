module flowdit.dataset.mnist;
import std;
namespace flowdit {
    DatasetInfo inspect_mnist(const std::filesystem::path& directory) {
        std::ifstream images{directory / "train-images-idx3-ubyte", std::ios::binary};
        images.exceptions(std::ios::failbit | std::ios::badbit);
        std::array<std::uint32_t, 4> header{};
        images.read(reinterpret_cast<char*>(header.data()), sizeof(header));
        if constexpr (std::endian::native == std::endian::little)
            for (auto& value : header) value = std::byteswap(value);
        return {{"MNIST", header[3], header[2], 1u, {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}}, header[1]};
    }
    Dataset load_mnist(const std::filesystem::path& directory) {
        const auto info = inspect_mnist(directory);
        Dataset result{.specification = info.specification, .labels = std::vector<std::uint32_t>(info.count), .files = {directory / "train-images-idx3-ubyte"}, .records = std::vector<SampleRecord>(info.count)};
        std::ifstream labels{directory / "train-labels-idx1-ubyte", std::ios::binary};
        labels.exceptions(std::ios::failbit | std::ios::badbit);
        labels.seekg(8);
        std::vector<std::uint8_t> bytes(info.count);
        labels.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        std::ranges::copy(bytes, result.labels.begin());
        const std::uint64_t image_bytes = static_cast<std::uint64_t>(info.specification.width) * info.specification.height;
        for (std::uint32_t i = 0; i < info.count; ++i) result.records[i] = {0u, 16u + i * image_bytes};
        return result;
    }
} // namespace flowdit
