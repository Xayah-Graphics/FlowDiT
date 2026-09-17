module;
#include <imgui.h>
export module flowdit.editor.viewing.canvas;
export import flowdit.io.output;
import std;
export namespace flowdit::editor {
    struct Picture final {
        std::uint64_t texture{};
        SampleInfo info;
        SamplingResult images;
    };
    struct Canvas final {
        int selected{-1};
        float zoom{8}, thumbnail{72};
        bool fit{true}, nearest{true}, patches{};
        ImVec2 pan{};
        void draw(std::uint64_t texture, const SamplingResult& images, std::span<const float> times = {});
    };
} // namespace flowdit::editor
