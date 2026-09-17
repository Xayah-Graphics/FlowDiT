export module flowdit.model.configuration;
export import flowdit.tensor.types;
import std;
export namespace flowdit {
    struct ModelConfiguration final {
        TensorShape shape;
        std::uint32_t class_count{};
        std::uint32_t patch_size{2u};
    };
    std::string serialize_model(const ModelConfiguration& configuration);
    ModelConfiguration deserialize_model(std::string_view text);
} // namespace flowdit
