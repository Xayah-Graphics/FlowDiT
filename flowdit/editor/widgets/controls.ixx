export module flowdit.editor.widgets.controls;
import flowdit.editor.platform.window;
import std;
export namespace flowdit::editor {
    bool path_field(const char* label, std::string& path, WindowPlatform& window, bool directory);
}
