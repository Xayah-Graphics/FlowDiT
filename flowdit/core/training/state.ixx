export module flowdit.training.state;
import std;
export namespace flowdit {
    struct TrainingState final {
        std::uint64_t step{};
        std::uint64_t processed_samples{};
        std::uint64_t seed{};
        double elapsed_seconds{};
    };
} // namespace flowdit
