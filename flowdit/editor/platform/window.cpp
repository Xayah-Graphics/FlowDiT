module;
#include <Windows.h>
#include <windowsx.h>
#include <dwmapi.h>
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
module flowdit.editor.platform.window;
import std;
namespace flowdit::editor {
    WindowPlatform::WindowPlatform() {
        if (!glfwInit()) throw std::runtime_error{"Cannot initialize GLFW"};
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
        glfwWindowHint(GLFW_SCALE_TO_MONITOR, GLFW_TRUE);
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        window = glfwCreateWindow(1440, 960, "FlowDiT", nullptr, nullptr);
        if (!window) throw std::runtime_error{"Cannot create the FlowDiT window"};
        native_window = glfwGetWin32Window(window);
        SetPropW(native_window, L"FlowDiTWindow", this);
        original_window_proc = reinterpret_cast<WNDPROC>(SetWindowLongPtrW(native_window, GWLP_WNDPROC, reinterpret_cast<LONG_PTR>(&WindowPlatform::window_proc)));
        SetWindowLongPtrW(native_window, GWL_STYLE, WS_POPUP | WS_THICKFRAME | WS_MINIMIZEBOX | WS_MAXIMIZEBOX | WS_SYSMENU);
        constexpr BOOL dark = TRUE;
        DwmSetWindowAttribute(native_window, DWMWA_USE_IMMERSIVE_DARK_MODE, &dark, sizeof(dark));
        MONITORINFO monitor{sizeof(MONITORINFO)};
        GetMonitorInfoW(MonitorFromWindow(native_window, MONITOR_DEFAULTTONEAREST), &monitor);
        RECT bounds{};
        GetWindowRect(native_window, &bounds);
        const auto& area = monitor.rcWork;
        SetWindowPos(native_window, nullptr, area.left + (area.right - area.left - bounds.right + bounds.left) / 2, area.top + (area.bottom - area.top - bounds.bottom + bounds.top) / 2, 0, 0, SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED);
        glfwShowWindow(window);
    }
    WindowPlatform::~WindowPlatform() {
        SetWindowLongPtrW(native_window, GWLP_WNDPROC, reinterpret_cast<LONG_PTR>(original_window_proc));
        RemovePropW(native_window, L"FlowDiTWindow");
        glfwDestroyWindow(window);
        glfwTerminate();
    }
    void WindowPlatform::toggle_fullscreen() {
        if (!fullscreen) {
            GetWindowPlacement(native_window, &windowed_placement);
            windowed_style = GetWindowLongPtrW(native_window, GWL_STYLE);
            MONITORINFO monitor{sizeof(MONITORINFO)};
            GetMonitorInfoW(MonitorFromWindow(native_window, MONITOR_DEFAULTTONEAREST), &monitor);
            fullscreen = true;
            SetWindowLongPtrW(native_window, GWL_STYLE, windowed_style & ~static_cast<LONG_PTR>(WS_THICKFRAME | WS_MAXIMIZEBOX | WS_MAXIMIZE));
            constexpr DWM_WINDOW_CORNER_PREFERENCE corners = DWMWCP_DONOTROUND;
            DwmSetWindowAttribute(native_window, DWMWA_WINDOW_CORNER_PREFERENCE, &corners, sizeof(corners));
            const auto& area = monitor.rcMonitor;
            SetWindowPos(native_window, HWND_TOP, area.left, area.top, area.right - area.left, area.bottom - area.top, SWP_FRAMECHANGED | SWP_NOOWNERZORDER);
        } else {
            fullscreen = false;
            SetWindowLongPtrW(native_window, GWL_STYLE, windowed_style);
            SetWindowPlacement(native_window, &windowed_placement);
            constexpr DWM_WINDOW_CORNER_PREFERENCE corners = DWMWCP_DEFAULT;
            DwmSetWindowAttribute(native_window, DWMWA_WINDOW_CORNER_PREFERENCE, &corners, sizeof(corners));
            SetWindowPos(native_window, nullptr, 0, 0, 0, 0, SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE);
        }
    }
    LRESULT CALLBACK WindowPlatform::window_proc(HWND window, const UINT message, const WPARAM wparam, const LPARAM lparam) {
        auto& platform = *static_cast<WindowPlatform*>(GetPropW(window, L"FlowDiTWindow"));
        switch (message) {
        case WM_NCCALCSIZE:
            if (wparam) return 0;
            break;
        case WM_NCHITTEST: {
            if (platform.fullscreen) return HTCLIENT;
            POINT point{GET_X_LPARAM(lparam), GET_Y_LPARAM(lparam)};
            ScreenToClient(window, &point);
            RECT client{};
            GetClientRect(window, &client);
            if (!IsZoomed(window)) {
                const auto dpi = GetDpiForWindow(window);
                const int border = GetSystemMetricsForDpi(SM_CXSIZEFRAME, dpi) + GetSystemMetricsForDpi(SM_CXPADDEDBORDER, dpi);
                const bool left = point.x < border, right = point.x >= client.right - border;
                const bool top = point.y < border, bottom = point.y >= client.bottom - border;
                if (top && left) return HTTOPLEFT;
                if (top && right) return HTTOPRIGHT;
                if (bottom && left) return HTBOTTOMLEFT;
                if (bottom && right) return HTBOTTOMRIGHT;
                if (left) return HTLEFT;
                if (right) return HTRIGHT;
                if (top) return HTTOP;
                if (bottom) return HTBOTTOM;
            }
            const auto& region = platform.drag_region;
            if (point.x >= region[0] && point.y >= region[1] && point.x < region[2] && point.y < region[3]) return HTCAPTION;
            return HTCLIENT;
        }
        case WM_GETMINMAXINFO: {
            MONITORINFO monitor{sizeof(MONITORINFO)};
            GetMonitorInfoW(MonitorFromWindow(window, MONITOR_DEFAULTTONEAREST), &monitor);
            auto& limits = *reinterpret_cast<MINMAXINFO*>(lparam);
            const auto& area = platform.fullscreen ? monitor.rcMonitor : monitor.rcWork;
            limits.ptMaxPosition = {area.left - monitor.rcMonitor.left, area.top - monitor.rcMonitor.top};
            limits.ptMaxSize = {area.right - area.left, area.bottom - area.top};
            const auto dpi = GetDpiForWindow(window);
            limits.ptMinTrackSize = {MulDiv(960, dpi, 96), MulDiv(640, dpi, 96)};
            return 0;
        }
        case WM_SYSCOMMAND:
            if (platform.fullscreen && ((wparam & 0xFFF0) == SC_MOVE || (wparam & 0xFFF0) == SC_SIZE || (wparam & 0xFFF0) == SC_MAXIMIZE)) return 0;
            break;
        }
        return CallWindowProcW(platform.original_window_proc, window, message, wparam, lparam);
    }
} // namespace flowdit::editor
