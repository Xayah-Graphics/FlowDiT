module;
#include <flowdit/cuda.h>
export module flowdit.autoencoder.perceptual;
import flowdit.neural.spatial;
import std;
export namespace flowdit {
    struct PerceptualLoss final {
        PerceptualLoss(::cuda::stream_ref stream, TensorShape image, std::uint32_t batch, const std::filesystem::path& weights);
        void prepare(const float* images);
        void backward(const float* reconstructed, float* gradient, float* loss, float weight);

    private:
        ::cuda::stream_ref stream;
        TensorShape shape;
        std::uint32_t batch;
        neural::SpatialNetwork network;
        std::array<std::size_t, 5> features;
        std::vector<::cuda::device_buffer<std::uint16_t>> reference;
        ::cuda::device_buffer<float> parameters, linear, input;
    };
} // namespace flowdit
