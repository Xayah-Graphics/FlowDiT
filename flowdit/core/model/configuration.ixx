export module flowdit.model.configuration;
export import flowdit.tensor.types;
import std;
export namespace flowdit {
    struct ModelConfiguration final {
        TensorShape shape;
        std::uint32_t class_count{};
        std::uint32_t patch_size{1}, width{768}, heads{12}, side_blocks{6}, mlp_width{3072};
    };
    std::string serialize_model(const ModelConfiguration& configuration);
    ModelConfiguration deserialize_model(std::string_view text);
} // namespace flowdit
