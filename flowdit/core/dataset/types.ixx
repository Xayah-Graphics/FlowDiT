export module flowdit.dataset.types;
export import flowdit.image.types;
import std;
export namespace flowdit {
    enum class DatasetSplit { training, validation, test };
    struct DatasetInfo final {
        ImageSpecification specification;
        std::uint32_t count{};
    };
    struct SampleRecord final {
        std::uint32_t file{};
        std::uint64_t offset{};
    };
    struct Dataset final {
        ImageSpecification specification;
        std::vector<std::uint32_t> labels;
        std::vector<std::filesystem::path> files;
        std::vector<SampleRecord> records;
        bool encoded_images{};
        DatasetSplit split{DatasetSplit::training};
        void read(std::span<const std::uint32_t> indices, ImageBatch& batch) const;
    };
} // namespace flowdit
