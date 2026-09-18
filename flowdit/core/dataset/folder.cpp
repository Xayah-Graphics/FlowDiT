module;
#include <stb_image.h>
module flowdit.dataset.folder;
import std;
namespace flowdit {
    Dataset load_folder(const std::filesystem::path& directory, const bool test) {
        Dataset result;
        result.specification.name = directory.filename().string();
        result.encoded_images     = true;
        for (const auto& entry : std::filesystem::directory_iterator{directory / "train"})
            if (entry.is_directory()) result.specification.classes.push_back(entry.path().filename().string());
        std::ranges::sort(result.specification.classes);
        for (std::uint32_t label = 0; label < result.specification.classes.size(); ++label) {
            std::vector<std::filesystem::path> files;
            for (const auto& entry : std::filesystem::directory_iterator{directory / (test ? "test" : "train") / result.specification.classes[label]}) {
                const auto extension = entry.path().extension().string();
                if (entry.is_regular_file() && (extension == ".png" || extension == ".jpg" || extension == ".jpeg")) files.push_back(entry.path());
            }
            std::ranges::sort(files);
            for (const auto& path : files) {
                result.records.push_back({static_cast<std::uint32_t>(result.files.size()), 0});
                result.files.push_back(path);
                result.labels.push_back(label);
            }
        }
        int width{}, height{}, channels{};
        if (!stbi_info(result.files.front().string().c_str(), &width, &height, &channels)) throw std::runtime_error{stbi_failure_reason()};
        result.specification.width    = width;
        result.specification.height   = height;
        result.specification.channels = channels;
        return result;
    }
} // namespace flowdit
