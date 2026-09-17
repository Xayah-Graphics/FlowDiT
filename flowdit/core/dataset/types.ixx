export module flowdit.dataset.types;
import std;
export namespace flowdit {
    enum class DatasetKind { cifar10, mnist };
    struct ImageSpecification final {
        std::string name;
        std::uint32_t width{}, height{}, channels{};
        std::vector<std::string> classes;
    };
    struct Dataset final {
        ImageSpecification specification;
        std::vector<std::uint8_t> images;
        std::vector<std::uint32_t> labels;
        bool horizontal_flip{};
    };
} // namespace flowdit
