module;
#include <imgui.h>
module flowdit.editor.viewing.canvas;
import std;
namespace flowdit::editor {
    void Canvas::draw(const std::uint64_t texture, const SamplingResult& images, const std::span<const float> times) {
        if (!texture) {
            ImGui::TextDisabled("No image selected");
            return;
        }
        const auto& image        = images.model.image;
        const float image_width  = static_cast<float>(image.width);
        const float image_height = static_cast<float>(image.height);
        if (selected >= 0 && ImGui::Button("Grid")) {
            selected = -1;
            pan      = {};
        }
        if (selected >= 0) ImGui::SameLine();
        ImGui::Checkbox("Nearest", &nearest);
        ImGui::SameLine();
        ImGui::Checkbox("Patch grid", &patches);
        const std::uint64_t id = texture | (nearest ? 1ull << 32 : 0);
        const float count      = static_cast<float>(images.labels.size());
        if (selected < 0) {
            ImGui::SameLine();
            ImGui::SetNextItemWidth(90);
            ImGui::SliderFloat("Size", &thumbnail, 32, 144, "%.0f");
            ImGui::BeginChild("grid");
            const float spacing = ImGui::GetStyle().ItemSpacing.x;
            const int columns   = std::max(1, static_cast<int>((ImGui::GetContentRegionAvail().x + spacing) / (thumbnail + spacing)));
            const int rows      = (static_cast<int>(images.labels.size()) + columns - 1) / columns;
            ImGuiListClipper clipper;
            clipper.Begin(rows, thumbnail * image_height / image_width + ImGui::GetTextLineHeightWithSpacing() + ImGui::GetStyle().ItemSpacing.y);
            while (clipper.Step())
                for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; ++row)
                    for (int column = 0; column < columns; ++column) {
                        const int index = row * columns + column;
                        if (index >= images.labels.size()) break;
                        if (column) ImGui::SameLine();
                        ImGui::PushID(index);
                        ImGui::BeginGroup();
                        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2{});
                        if (ImGui::ImageButton("image", id, {thumbnail, thumbnail * image_height / image_width}, {0, static_cast<float>(index) / count}, {1, static_cast<float>(index + 1) / count})) {
                            selected = index;
                            pan      = {};
                            fit      = true;
                        }
                        ImGui::PopStyleVar();
                        const std::string_view label = images.labels[index] == image.classes.size() ? std::string_view{"unconditional"} : std::string_view{image.classes[images.labels[index]]};
                        ImGui::TextUnformatted(label.data(), label.data() + label.size());
                        ImGui::EndGroup();
                        ImGui::PopID();
                    }
            ImGui::EndChild();
            return;
        }
        ImGui::SameLine();
        ImGui::Checkbox("Fit", &fit);
        ImGui::SameLine();
        if (ImGui::Button("1:1")) {
            fit  = false;
            zoom = 1;
            pan  = {};
        }
        ImGui::SameLine();
        ImGui::Text("#%d  %s", selected, images.labels[selected] == image.classes.size() ? "unconditional" : image.classes[images.labels[selected]].c_str());
        if (!times.empty()) {
            ImGui::SameLine();
            ImGui::Text("t = %.5f", times[selected]);
        }
        ImGui::BeginChild("image-view", {}, ImGuiChildFlags_None, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
        const auto available = ImGui::GetContentRegionAvail();
        const auto origin    = ImGui::GetCursorScreenPos();
        if (fit) {
            zoom = std::max(1.0F, std::floor(std::min(available.x / image_width, (available.y - 28) / image_height)));
            pan  = {};
        }
        const auto& io = ImGui::GetIO();
        if (ImGui::IsWindowHovered() && io.MouseWheel != 0) {
            const float previous = zoom;
            zoom                 = std::clamp(zoom * std::exp2(io.MouseWheel), 1.0F, 128.0F);
            const ImVec2 center{origin.x + available.x * 0.5F, origin.y + (available.y - 28) * 0.5F};
            pan = {io.MousePos.x - center.x - (io.MousePos.x - center.x - pan.x) * zoom / previous, io.MousePos.y - center.y - (io.MousePos.y - center.y - pan.y) * zoom / previous};
            fit = false;
        }
        if (ImGui::IsWindowHovered() && ImGui::IsMouseDragging(ImGuiMouseButton_Middle)) {
            fit = false;
            pan.x += io.MouseDelta.x;
            pan.y += io.MouseDelta.y;
        }
        const float width  = image_width * zoom;
        const float height = image_height * zoom;
        const ImVec2 position{origin.x + (available.x - width) * 0.5F + pan.x, origin.y + (available.y - 28 - height) * 0.5F + pan.y};
        auto* draw = ImGui::GetWindowDrawList();
        draw->AddImage(id, position, {position.x + width, position.y + height}, {0, selected / count}, {1, (selected + 1) / count});
        if (patches) {
            for (std::uint32_t x = 0; x <= image.width; x += images.model.patch_size) draw->AddLine({position.x + x * zoom, position.y}, {position.x + x * zoom, position.y + height}, IM_COL32(100, 225, 210, 160));
            for (std::uint32_t y = 0; y <= image.height; y += images.model.patch_size) draw->AddLine({position.x, position.y + y * zoom}, {position.x + width, position.y + y * zoom}, IM_COL32(100, 225, 210, 160));
        }
        const int x = static_cast<int>(std::floor((io.MousePos.x - position.x) / zoom));
        const int y = static_cast<int>(std::floor((io.MousePos.y - position.y) / zoom));
        if (ImGui::IsWindowHovered() && x >= 0 && x < image.width && y >= 0 && y < image.height && !images.rgba.empty()) {
            const auto offset = (static_cast<std::size_t>(selected) * image.width * image.height + y * image.width + x) * 4;
            const auto text   = std::format("({}, {})  RGB ({}, {}, {})  |  {:.0f}x", x, y, images.rgba[offset], images.rgba[offset + 1], images.rgba[offset + 2], zoom);
            draw->AddRectFilled({origin.x, origin.y + available.y - 26}, {origin.x + available.x, origin.y + available.y}, IM_COL32(18, 22, 28, 245));
            draw->AddText({origin.x + 6, origin.y + available.y - 22}, IM_COL32(225, 230, 235, 255), text.c_str());
        }
        ImGui::EndChild();
    }
} // namespace flowdit::editor
