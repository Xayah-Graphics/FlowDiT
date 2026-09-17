module flowdit.dataset.cifar10;
import std;
namespace flowdit {
    Dataset load_cifar10(const std::filesystem::path& directory) {
        constexpr std::size_t image_bytes = 32uz * 32uz * 3uz;
        Dataset result{
            .specification   = {"CIFAR-10", 32u, 32u, 3u, {"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"}},
            .images          = std::vector<std::uint8_t>(50'000uz * image_bytes),
            .labels          = std::vector<std::uint32_t>(50'000uz),
            .horizontal_flip = true,
        };
        std::array<std::uint8_t, image_bytes + 1uz> record{};
        for (std::uint32_t batch = 0u; batch < 5u; ++batch) {
            std::ifstream file{directory / std::format("data_batch_{}.bin", batch + 1u), std::ios::binary};
            file.exceptions(std::ios::failbit | std::ios::badbit);
            for (std::uint32_t image = 0u; image < 10'000u; ++image) {
                file.read(reinterpret_cast<char*>(record.data()), static_cast<std::streamsize>(record.size()));
                const std::size_t index = static_cast<std::size_t>(batch) * 10'000uz + image;
                result.labels[index]    = record[0];
                std::ranges::copy(record | std::views::drop(1), result.images.begin() + static_cast<std::ptrdiff_t>(index * image_bytes));
            }
        }
        return result;
    }
} // namespace flowdit
