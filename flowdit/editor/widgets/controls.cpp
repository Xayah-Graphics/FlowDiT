module;
#include <imgui.h>
#include <imgui_internal.h>
module flowdit.editor.widgets.controls;
import std;
namespace flowdit::editor {
    bool text_button(const char* label) {
        const float dpi = ImGui::GetStyle().FontScaleDpi;
        const auto text = ImGui::CalcTextSize(label, nullptr, true);
        const auto origin = ImGui::GetCursorScreenPos();
        const ImVec2 size{text.x + 24 * dpi, 40 * dpi};
        ImGui::PushStyleColor(ImGuiCol_NavCursor, ImVec4{});
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2{12 * dpi, (size.y - text.y) * 0.5F});
        ImGui::AlignTextToFramePadding();
        const bool pressed = ImGui::InvisibleButton(label, size, ImGuiButtonFlags_EnableNav);
        ImGui::PopStyleVar();
        ImGui::PopStyleColor();
        const bool disabled = ImGui::GetItemFlags() & ImGuiItemFlags_Disabled;
        const bool focused = !disabled && ImGui::IsItemFocused() && ImGui::GetCurrentContext()->NavCursorVisible;
        auto* storage = ImGui::GetStateStorage();
        const auto key = ImGui::GetItemID();
        const float alpha = disabled ? 0 : std::lerp(storage->GetFloat(key), ImGui::IsItemHovered() || focused ? 1.0F : 0.0F, std::min(1.0F, ImGui::GetIO().DeltaTime / 0.12F));
        storage->SetFloat(key, alpha);
        const ImVec2 position{origin.x + 12 * dpi, origin.y + (size.y - text.y) * 0.5F};
        const ImVec4 ink = disabled ? ImVec4{0.62F, 0.62F, 0.62F, 1} : ImVec4{0.57F + 0.31F * alpha, 0.58F + 0.30F * alpha, 0.64F + 0.29F * alpha, 1};
        auto* draw = ImGui::GetWindowDrawList();
        const char* end = ImGui::FindRenderedTextEnd(label);
        ImGui::PushStyleColor(ImGuiCol_Text, {0, 0, 0, 0.7F});
        ImGui::RenderTextEllipsis(draw, {position.x, position.y + dpi}, {position.x + text.x, position.y + text.y + dpi}, position.x + text.x, label, end, nullptr);
        ImGui::PopStyleColor();
        ImGui::PushStyleColor(ImGuiCol_Text, ink);
        ImGui::RenderTextEllipsis(draw, position, {position.x + text.x, position.y + text.y}, position.x + text.x, label, end, nullptr);
        ImGui::PopStyleColor();
        if (focused) draw->AddLine({position.x, position.y + text.y + 3 * dpi}, {position.x + text.x, position.y + text.y + 3 * dpi}, ImGui::GetColorU32(ink), dpi);
        return pressed;
    }
    bool tool_button(const char* label, const bool selected, const float width) {
        const float dpi = ImGui::GetStyle().FontScaleDpi;
        ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(selected ? ImGuiCol_Text : ImGuiCol_TextDisabled));
        const ImVec4 color{0.44F, 0.80F, 0.87F, 1};
        ImGui::PushStyleColor(ImGuiCol_Button, {color.x, color.y, color.z, selected ? 0.08F : 0});
        const bool pressed = ImGui::Button(label, {width, 30 * dpi});
        if (selected) {
            const auto minimum = ImGui::GetItemRectMin(), maximum = ImGui::GetItemRectMax();
            ImGui::GetWindowDrawList()->AddLine({minimum.x + 8 * dpi, maximum.y - dpi}, {maximum.x - 8 * dpi, maximum.y - dpi}, ImGui::GetColorU32(color), 2 * dpi);
        }
        ImGui::PopStyleColor(2);
        return pressed;
    }
    void number_field(const char* label, const ImGuiDataType type, void* value, const char* format, const bool stacked) {
        ImGui::PushID(label);
        ImGui::BeginGroup();
        if (stacked) ImGui::TextWrapped("%s", label);
        else {
            ImGui::AlignTextToFramePadding();
            ImGui::TextDisabled("%s", label);
        }
        const bool focus = ImGui::IsItemClicked();
        if (!stacked) ImGui::SameLine();
        ImGui::SetNextItemWidth(-1);
        if (focus) ImGui::SetKeyboardFocusHere();
        ImGui::InputScalar("##value", type, value, nullptr, nullptr, format);
        ImGui::EndGroup();
        ImGui::PopID();
    }
    std::string run_label(const std::string_view name) {
        return std::format("{}-{} · {}:{}", name.substr(4, 2), name.substr(6, 2), name.substr(9, 2), name.substr(11, 2));
    }
    bool panel_button(const char* id, const bool open) {
        const float dpi = ImGui::GetStyle().FontScaleDpi;
        const auto origin = ImGui::GetCursorScreenPos();
        const bool pressed = ImGui::InvisibleButton(id, {24 * dpi, 24 * dpi}, ImGuiButtonFlags_EnableNav);
        auto* draw = ImGui::GetWindowDrawList();
        const auto color = ImGui::GetColorU32(ImGui::IsItemHovered() || ImGui::IsItemFocused() ? ImGuiCol_Text : ImGuiCol_TextDisabled);
        const ImVec2 minimum{origin.x + 4 * dpi, origin.y + 6 * dpi}, maximum{origin.x + 20 * dpi, origin.y + 18 * dpi};
        draw->AddRect(minimum, maximum, color, 2 * dpi, 0, dpi);
        const float x = minimum.x + 5 * dpi;
        draw->AddLine({x, minimum.y}, {x, maximum.y}, color, dpi);
        if (open) draw->AddRectFilled(minimum, {x, maximum.y}, color, dpi);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s sidebar (`)", open ? "Hide" : "Show");
        return pressed;
    }
}
