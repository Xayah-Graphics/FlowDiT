export module flowdit.model.configuration;
export import flowdit.dataset.types;
import std;
export namespace flowdit {
    struct ModelConfiguration final {
        ImageSpecification image;
        std::uint32_t patch_size{2u};
    };
    std::string serialize_model(const ModelConfiguration& configuration);
    ModelConfiguration deserialize_model(std::string_view text);
    ModelConfiguration read_model_configuration(const std::filesystem::path& checkpoint);
} // namespace flowdit
