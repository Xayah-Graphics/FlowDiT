export module flowdit.tensor.types;
import std;
export namespace flowdit {
    struct TensorShape final {
        std::uint32_t width{}, height{}, channels{};
    };
    // Borrowed device memory in NCHW order; valid until the producing operation runs again.
    struct TensorBatch final {
        TensorShape shape;
        std::uint32_t count{};
        const float* values{};
        const std::uint32_t* labels{};
    };
} // namespace flowdit
