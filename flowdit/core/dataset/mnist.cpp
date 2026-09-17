module flowdit.dataset.mnist;
import std;
namespace flowdit {
    Dataset load_mnist(const std::filesystem::path& directory) {
        std::ifstream images{directory / "train-images-idx3-ubyte", std::ios::binary};
        std::ifstream labels{directory / "train-labels-idx1-ubyte", std::ios::binary};
        images.exceptions(std::ios::failbit | std::ios::badbit);
        labels.exceptions(std::ios::failbit | std::ios::badbit);
        std::array<std::uint32_t, 4> header{};
        images.read(reinterpret_cast<char*>(header.data()), sizeof(header));
        if constexpr (std::endian::native == std::endian::little)
            for (auto& value : header) value = std::byteswap(value);
        Dataset result{
            .specification = {"MNIST", header[3], header[2], 1u, {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}},
            .images        = std::vector<std::uint8_t>(static_cast<std::size_t>(header[1]) * header[2] * header[3]),
            .labels        = std::vector<std::uint32_t>(header[1]),
        };
        images.read(reinterpret_cast<char*>(result.images.data()), static_cast<std::streamsize>(result.images.size()));
        labels.seekg(8);
        std::vector<std::uint8_t> label_bytes(result.labels.size());
        labels.read(reinterpret_cast<char*>(label_bytes.data()), static_cast<std::streamsize>(label_bytes.size()));
        std::ranges::copy(label_bytes, result.labels.begin());
        return result;
    }
} // namespace flowdit
