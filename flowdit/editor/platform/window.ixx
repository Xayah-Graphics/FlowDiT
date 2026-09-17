module;
#include <Windows.h>
#include <GLFW/glfw3.h>
export module flowdit.editor.platform.window;
import std;
export namespace flowdit::editor {
    struct WindowPlatform final {
        GLFWwindow* window{};
        HWND native_window{};
        std::array<float, 4> drag_region{};
        WindowPlatform();
        ~WindowPlatform();
        WindowPlatform(const WindowPlatform&)            = delete;
        WindowPlatform& operator=(const WindowPlatform&) = delete;
        std::optional<std::string> choose_path(bool directory);

    private:
        WNDPROC original_window_proc{};
        static LRESULT CALLBACK window_proc(HWND window, UINT message, WPARAM wparam, LPARAM lparam);
    };
} // namespace flowdit::editor
