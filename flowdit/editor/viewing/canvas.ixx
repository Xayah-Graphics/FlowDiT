module;
#include <imgui.h>
export module flowdit.editor.viewing.canvas;
export import flowdit.dataset.types;
import flowdit.editor.graphics.renderer;
import std;
export namespace flowdit::editor {
    struct Picture final {
        std::uint64_t texture{};
        ImageSpecification specification;
        std::vector<std::uint32_t> labels;
        void upload(Renderer& renderer, const ImageSpecification& image, std::span<const std::uint32_t> classes, const std::uint8_t* pixels);
    };
    struct Canvas final {
        int selected{-1};
        float zoom{1}, scroll{};
        bool fit{true}, restore_scroll{true};
        ImVec2 pan{};
        void draw(const Picture& picture);
    };
} // namespace flowdit::editor
