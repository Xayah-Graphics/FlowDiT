module;
#include <imgui.h>
export module flowdit.editor.widgets.controls;
import std;
export namespace flowdit::editor {
    bool text_button(const char* label);
    bool tool_button(const char* label, bool selected, float width);
    void number_field(const char* label, ImGuiDataType type, void* value, const char* format = nullptr, bool stacked = false);
    std::string run_label(std::string_view name);
    bool panel_button(const char* id, bool open);
}
