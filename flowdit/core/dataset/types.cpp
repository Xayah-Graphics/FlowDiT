module;
#include <stb_image.h>
module flowdit.dataset.types;
import std;
namespace flowdit {
    void Dataset::read(const std::span<const std::uint32_t> indices, ImageBatch& batch) const {
        batch.shape                   = {specification.width, specification.height, specification.channels};
        const std::size_t image_bytes = static_cast<std::size_t>(specification.width) * specification.height * specification.channels;
        batch.pixels.resize(indices.size() * image_bytes);
        batch.labels.resize(indices.size());
        if (encoded_images) {
            for (std::size_t i = 0; i < indices.size(); ++i) {
                const auto index = indices[i];
                int width{}, height{}, channels{};
                const std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> pixels{stbi_load(files[records[index].file].string().c_str(), &width, &height, &channels, specification.channels), stbi_image_free};
                if (!pixels) throw std::runtime_error{stbi_failure_reason()};
                const auto area = specification.width * specification.height;
                for (std::uint32_t c = 0; c < specification.channels; ++c)
                    for (std::uint32_t pixel = 0; pixel < area; ++pixel) batch.pixels[i * image_bytes + c * area + pixel] = pixels.get()[pixel * specification.channels + c];
                batch.labels[i] = labels[index];
            }
            return;
        }
        std::vector<std::ifstream> streams;
        streams.reserve(files.size());
        for (const auto& path : files) {
            auto& stream = streams.emplace_back(path, std::ios::binary);
            stream.exceptions(std::ios::failbit | std::ios::badbit);
        }
        for (std::size_t i = 0; i < indices.size(); ++i) {
            const auto index   = indices[i];
            const auto& record = records[index];
            auto& stream       = streams[record.file];
            stream.seekg(static_cast<std::streamoff>(record.offset));
            stream.read(reinterpret_cast<char*>(batch.pixels.data() + i * image_bytes), static_cast<std::streamsize>(image_bytes));
            batch.labels[i] = labels[index];
        }
    }
} // namespace flowdit
