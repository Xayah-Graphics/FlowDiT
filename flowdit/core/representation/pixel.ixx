module;
#include <flowdit/cuda.h>
export module flowdit.representation.pixel;
export import flowdit.image.types;
import std;
export namespace flowdit {
    std::string serialize_representation(const ImageSpecification& image);
    ImageSpecification deserialize_representation(std::string_view text);
    struct PixelRepresentation final {
        PixelRepresentation(::cuda::stream_ref stream, TensorShape shape, std::uint32_t batch);
        TensorBatch encode(const ImageBatch& images, bool horizontal_flip, std::uint64_t seed, std::uint64_t step);
        const std::uint8_t* decode(const TensorBatch& tensor);
    private:
        ::cuda::stream_ref stream;
        TensorShape shape;
        std::uint32_t batch;
        std::optional<::cuda::device_buffer<std::uint8_t>> images;
        std::optional<::cuda::device_buffer<std::uint32_t>> labels;
        std::optional<::cuda::device_buffer<float>> values;
        std::optional<::cuda::device_buffer<std::uint8_t>> rgba;
    };
} // namespace flowdit
