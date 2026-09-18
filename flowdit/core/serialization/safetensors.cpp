module;
#include <nlohmann/json.hpp>
module flowdit.serialization.safetensors;
import std;
namespace flowdit::serialization::safetensors {
    namespace {
        nlohmann::json read_header(std::ifstream& input) {
            std::uint64_t header_size{};
            input.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));
            std::string text(header_size, '\0');
            input.read(text.data(), static_cast<std::streamsize>(text.size()));
            return nlohmann::json::parse(text);
        }
    } // namespace
    void write(const std::filesystem::path& path, const std::string_view system, const std::span<const TensorView> tensors, const std::map<std::string, std::string>& metadata) {
        nlohmann::json header;
        header["__metadata__"] = {
            {"flowdit.format_version", "1"},
            {"flowdit.project_version", FLOWDIT_PROJECT_VERSION},
            {"flowdit.system", system},
        };
        for (const auto& [name, value] : metadata) header["__metadata__"][name] = value;
        std::uint64_t offset{};
        for (const TensorView& tensor : tensors) {
            header[tensor.name] = {
                {"dtype", tensor.dtype},
                {"shape", tensor.shape},
                {"data_offsets", nlohmann::json::array({offset, offset + tensor.byte_count})},
            };
            offset += tensor.byte_count;
        }
        std::string text = header.dump();
        text.append((8uz - text.size() % 8uz) % 8uz, ' ');
        const std::uint64_t header_size = text.size();
        std::ofstream output{path, std::ios::binary | std::ios::trunc};
        output.exceptions(std::ios::badbit | std::ios::failbit);
        output.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
        output.write(text.data(), static_cast<std::streamsize>(text.size()));
        for (const TensorView& tensor : tensors) output.write(static_cast<const char*>(tensor.data), static_cast<std::streamsize>(tensor.byte_count));
    }
    std::map<std::string, std::string> read_metadata(const std::filesystem::path& path) {
        std::ifstream input{path, std::ios::binary};
        input.exceptions(std::ios::failbit | std::ios::badbit);
        return read_header(input).at("__metadata__").get<std::map<std::string, std::string>>();
    }
    File read(const std::filesystem::path& path, const std::span<const std::string_view> names) {
        std::ifstream input{path, std::ios::binary};
        input.exceptions(std::ios::failbit | std::ios::badbit);
        const auto header               = read_header(input);
        const std::uint64_t data_offset = static_cast<std::uint64_t>(input.tellg());
        File result;
        result.metadata = header.at("__metadata__").get<std::map<std::string, std::string>>();
        std::vector<std::string> selected{names.begin(), names.end()};
        if (selected.empty())
            for (const auto& [name, description] : header.items())
                if (name != "__metadata__") selected.push_back(name);
        result.tensors.reserve(selected.size());
        for (const auto& name : selected) {
            const auto& description   = header.at(name);
            const std::uint64_t begin = description.at("data_offsets").at(0).get<std::uint64_t>();
            const std::uint64_t end   = description.at("data_offsets").at(1).get<std::uint64_t>();
            Tensor tensor{
                .name  = name,
                .dtype = description.at("dtype").get<std::string>(),
                .shape = description.at("shape").get<std::vector<std::uint64_t>>(),
                .data  = std::vector<std::byte>(end - begin),
            };
            input.seekg(static_cast<std::streamoff>(data_offset + begin));
            input.read(reinterpret_cast<char*>(tensor.data.data()), static_cast<std::streamsize>(tensor.data.size()));
            result.tensors.push_back(std::move(tensor));
        }
        return result;
    }
} // namespace flowdit::serialization::safetensors
