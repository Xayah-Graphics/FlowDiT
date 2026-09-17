module;
#include <imgui.h>
#include <misc/cpp/imgui_stdlib.h>
module flowdit.editor.widgets.controls;
import std;
namespace flowdit::editor {
    bool path_field(const char* label, std::string& path, WindowPlatform& window, const bool directory) {
        ImGui::PushID(label);
        ImGui::TextDisabled("%s", label);
        const float button = ImGui::GetFrameHeight();
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - button - ImGui::GetStyle().ItemSpacing.x);
        bool changed = ImGui::InputText("##path", &path, ImGuiInputTextFlags_EnterReturnsTrue);
        changed |= ImGui::IsItemDeactivatedAfterEdit();
        if (ImGui::IsItemHovered() && !path.empty()) ImGui::SetTooltip("%s", path.c_str());
        ImGui::SameLine();
        if (ImGui::Button("...", {button, button})) {
            if (auto selected = window.choose_path(directory)) {
                path = std::move(*selected);
                changed = true;
            }
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(directory ? "Choose folder" : "Choose checkpoint");
        ImGui::PopID();
        return changed;
    }
}
