module;
#include <imgui.h>
module flowdit.editor.viewing.canvas;
import flowdit.editor.widgets.controls;
import std;
namespace flowdit::editor {
    void Picture::upload(Renderer& renderer, const ImageSpecification& image, const std::span<const std::uint32_t> classes, const std::uint8_t* pixels) {
        if (texture) renderer.retire(texture);
        specification = image;
        labels.assign(classes.begin(), classes.end());
        columns          = std::min(static_cast<std::uint32_t>(labels.size()), 10u);
        rows             = (static_cast<std::uint32_t>(labels.size()) + columns - 1) / columns;
        const auto width = columns * image.width, height = rows * image.height;
        std::vector<std::uint8_t> atlas(static_cast<std::size_t>(width) * height * 4uz);
        for (std::size_t i = 0; i < labels.size(); ++i)
            for (std::uint32_t y = 0; y < image.height; ++y) std::memcpy(atlas.data() + ((i / columns * image.height + y) * width + i % columns * image.width) * 4uz, pixels + (i * image.height + y) * image.width * 4uz, image.width * 4uz);
        texture = renderer.texture({width, height});
        renderer.upload(texture, atlas.data(), width, height, true);
    }
    void Canvas::draw(const Picture& picture) {
        const auto& image       = picture.specification;
        const float image_width = static_cast<float>(image.width), image_height = static_cast<float>(image.height);
        const std::uint64_t texture = picture.texture | (1ull << 32);
        const float atlas_columns = static_cast<float>(picture.columns), atlas_rows = static_cast<float>(picture.rows);
        const float dpi = ImGui::GetStyle().FontScaleDpi;
        if (selected >= 0) {
            if (text_button("Back") || (ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows) && ImGui::IsMouseClicked(ImGuiMouseButton_Right))) {
                selected       = -1;
                restore_scroll = true;
            }
            ImGui::SameLine();
            if (text_button("Fit")) fit = true;
            ImGui::SameLine();
            if (text_button("1:1")) {
                fit  = false;
                zoom = 1;
                pan  = {};
            }
            if (selected >= 0) {
                ImGui::SameLine();
                ImGui::TextDisabled("%s / %u x %u", image.classes[picture.labels[selected]].c_str(), image.width, image.height);
            }
        }
        if (selected < 0) {
            if (restore_scroll) ImGui::SetNextWindowScroll({0, scroll});
            restore_scroll = false;
            ImGui::BeginChild("grid");
            const auto available   = ImGui::GetContentRegionAvail();
            const bool preview     = picture.labels.size() > 24;
            const float gap        = (preview ? 8 : 16) * dpi;
            const int columns      = preview ? 10 : std::min(6, static_cast<int>(picture.labels.size()));
            const int rows         = (static_cast<int>(picture.labels.size()) + columns - 1) / columns;
            const float thumbnail  = std::min({112 * dpi, (available.x - (columns - 1) * gap) / columns, std::max(32 * dpi, (available.y - (rows - 1) * gap) / rows) * image_width / image_height});
            const float height     = thumbnail * image_height / image_width;
            const float row_height = height + gap;
            const float left       = ImGui::GetCursorPosX() + std::max(0.0F, (available.x - columns * thumbnail - (columns - 1) * gap) * 0.5F);
            const float top        = ImGui::GetCursorPosY() + std::max(0.0F, (available.y - rows * row_height + gap) * 0.5F);
            ImGui::SetCursorPosY(top);
            ImGuiListClipper clipper;
            clipper.Begin(rows, row_height);
            while (clipper.Step())
                for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; ++row) {
                    const float y = top + row * row_height;
                    for (int column = 0; column < columns; ++column) {
                        const int index = row * columns + column;
                        if (index >= picture.labels.size()) break;
                        ImGui::SetCursorPos({left + column * (thumbnail + gap), y});
                        ImGui::PushID(index);
                        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2{});
                        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 0);
                        if (ImGui::ImageButton("image", texture, {thumbnail, height}, {(index % picture.columns) / atlas_columns, (index / picture.columns) / atlas_rows}, {(index % picture.columns + 1) / atlas_columns, (index / picture.columns + 1) / atlas_rows})) {
                            scroll   = ImGui::GetScrollY();
                            selected = index;
                            pan      = {};
                            fit      = true;
                        }
                        ImGui::PopStyleVar(2);
                        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s / %u × %u", image.classes[picture.labels[index]].c_str(), image.width, image.height);
                        ImGui::PopID();
                    }
                    ImGui::SetCursorPosY(y + row_height - ImGui::GetStyle().ItemSpacing.y);
                    ImGui::Dummy({0, 0});
                }
            ImGui::EndChild();
            return;
        }
        ImGui::BeginChild("image-view", {}, ImGuiChildFlags_None, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
        const auto available = ImGui::GetContentRegionAvail();
        const auto origin    = ImGui::GetCursorScreenPos();
        ImGui::InvisibleButton("image-interaction", available, ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonMiddle);
        if (fit) {
            zoom = std::min(available.x / image_width, available.y / image_height);
            pan  = {};
        }
        const auto& io = ImGui::GetIO();
        if (ImGui::IsItemHovered() && io.MouseWheel != 0) {
            const float previous = zoom;
            zoom                 = std::clamp(zoom * std::exp2(io.MouseWheel * 0.5F), 0.25F, 128.0F);
            const ImVec2 center{origin.x + available.x * 0.5F, origin.y + available.y * 0.5F};
            pan = {io.MousePos.x - center.x - (io.MousePos.x - center.x - pan.x) * zoom / previous, io.MousePos.y - center.y - (io.MousePos.y - center.y - pan.y) * zoom / previous};
            fit = false;
        }
        if (ImGui::IsItemActive() && (ImGui::IsMouseDragging(ImGuiMouseButton_Left) || ImGui::IsMouseDragging(ImGuiMouseButton_Middle))) {
            fit = false;
            pan.x += io.MouseDelta.x;
            pan.y += io.MouseDelta.y;
        }
        const float width = image_width * zoom, height = image_height * zoom;
        const ImVec2 position{origin.x + (available.x - width) * 0.5F + pan.x, origin.y + (available.y - height) * 0.5F + pan.y};
        ImGui::GetWindowDrawList()->AddImage(texture, position, {position.x + width, position.y + height}, {(selected % picture.columns) / atlas_columns, (selected / picture.columns) / atlas_rows}, {(selected % picture.columns + 1) / atlas_columns, (selected / picture.columns + 1) / atlas_rows});
        ImGui::EndChild();
    }
} // namespace flowdit::editor
