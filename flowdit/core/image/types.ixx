export module flowdit.image.types;
export import flowdit.tensor.types;
import std;
export namespace flowdit {
    struct ImageSpecification final {
        std::string name;
        std::uint32_t width{}, height{}, channels{};
        std::vector<std::string> classes;
    };
    struct ImageBatch final {
        TensorShape shape;
        std::vector<std::uint8_t> pixels;
        std::vector<std::uint32_t> labels;
    };
} // namespace flowdit
