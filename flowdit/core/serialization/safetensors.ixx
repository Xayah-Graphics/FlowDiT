export module flowdit.serialization.safetensors;
import std;
export namespace flowdit::serialization::safetensors {
    struct TensorView final {
        std::string name;
        std::string dtype;
        std::vector<std::uint64_t> shape;
        const void* data{};
        std::uint64_t byte_count{};
    };
    struct Tensor final {
        std::string name;
        std::string dtype;
        std::vector<std::uint64_t> shape;
        std::vector<std::byte> data;
    };
    struct File final {
        std::map<std::string, std::string> metadata;
        std::vector<Tensor> tensors;
    };
    void write(const std::filesystem::path& path, std::string_view system, std::span<const TensorView> tensors, const std::map<std::string, std::string>& metadata = {});
    std::map<std::string, std::string> read_metadata(const std::filesystem::path& path);
    File read(const std::filesystem::path& path);
} // namespace flowdit::serialization::safetensors
