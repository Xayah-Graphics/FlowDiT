module;
#include <flowdit/cuda.h>
export module flowdit.representation.cache;
export import flowdit.representation.latent;
import flowdit.dataset.types;
import std;
export namespace flowdit {
    struct LatentCache final {
        LatentConfiguration configuration;
        std::filesystem::path directory;
        std::uint32_t variants{1};
        static std::optional<LatentCache> prepare(::cuda::stream_ref stream, const Dataset& data, const std::filesystem::path& dataset, const std::filesystem::path& checkpoint, bool flip, std::stop_token stop, const std::function<void(std::uint32_t, std::uint32_t)>& progress);
        void read(std::span<const std::uint32_t> indices, std::span<const std::uint32_t> views, std::vector<float>& moments) const;
    };
    struct LatentBatch final {
        LatentBatch(::cuda::stream_ref stream, const LatentConfiguration& configuration, std::uint32_t batch);
        TensorBatch upload(std::span<const float> moments, std::span<const std::uint32_t> labels, std::uint64_t seed, std::uint64_t step);

    private:
        ::cuda::stream_ref stream;
        TensorShape shape;
        std::uint32_t batch;
        ::cuda::device_buffer<float> moments, values, mean, deviation;
        ::cuda::device_buffer<std::uint32_t> labels;
    };
} // namespace flowdit
