module;
#include <Windows.h>
#include <GLFW/glfw3.h>
export module flowdit.editor.platform.window;
import std;
export namespace flowdit::editor {
    struct WindowPlatform final {
        GLFWwindow* window{};
        HWND native_window{};
        WindowPlatform();
        ~WindowPlatform();
        WindowPlatform(const WindowPlatform&)            = delete;
        WindowPlatform& operator=(const WindowPlatform&) = delete;
    };
} // namespace flowdit::editor
