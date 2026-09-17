module;
#include <Windows.h>
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
module flowdit.editor.platform.window;
import std;
namespace flowdit::editor {
    WindowPlatform::WindowPlatform() {
        if (!glfwInit()) throw std::runtime_error{"Cannot initialize GLFW"};
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_SCALE_TO_MONITOR, GLFW_TRUE);
        window = glfwCreateWindow(1440, 960, "FlowDiT", nullptr, nullptr);
        if (!window) throw std::runtime_error{"Cannot create the FlowDiT window"};
        native_window = glfwGetWin32Window(window);
        glfwSetWindowSizeLimits(window, 1080, 720, GLFW_DONT_CARE, GLFW_DONT_CARE);
    }
    WindowPlatform::~WindowPlatform() {
        glfwDestroyWindow(window);
        glfwTerminate();
    }
} // namespace flowdit::editor
