module;
#include <Windows.h>
#include <windowsx.h>
#include <dwmapi.h>
#include <shobjidl.h>
#include <wrl/client.h>
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
module flowdit.editor.platform.window;
import std;
namespace flowdit::editor {
    WindowPlatform::WindowPlatform() {
        if (FAILED(CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED))) throw std::runtime_error{"Cannot initialize file dialogs"};
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
        CoUninitialize();
    }
    std::optional<std::string> WindowPlatform::choose_path(const bool directory) {
        Microsoft::WRL::ComPtr<IFileOpenDialog> dialog;
        if (FAILED(CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&dialog)))) throw std::runtime_error{"Cannot create file dialog"};
        dialog->SetOptions(FOS_FORCEFILESYSTEM | FOS_PATHMUSTEXIST | FOS_NOCHANGEDIR | (directory ? FOS_PICKFOLDERS : FOS_FILEMUSTEXIST));
        const COMDLG_FILTERSPEC filter{L"Checkpoint", L"*.safetensors"};
        if (!directory) dialog->SetFileTypes(1, &filter);
        const auto result = dialog->Show(native_window);
        if (result == HRESULT_FROM_WIN32(ERROR_CANCELLED)) return {};
        if (FAILED(result)) throw std::runtime_error{"Cannot open file dialog"};
        Microsoft::WRL::ComPtr<IShellItem> item;
        if (FAILED(dialog->GetResult(&item))) throw std::runtime_error{"Cannot read selected path"};
        PWSTR path{};
        if (FAILED(item->GetDisplayName(SIGDN_FILESYSPATH, &path))) throw std::runtime_error{"Cannot read selected path"};
        const std::filesystem::path selected{path};
        CoTaskMemFree(path);
        return selected.string();
    }
    LRESULT CALLBACK WindowPlatform::window_proc(HWND window, const UINT message, const WPARAM wparam, const LPARAM lparam) {
        auto& platform = *static_cast<WindowPlatform*>(GetPropW(window, L"FlowDiTWindow"));
        switch (message) {
        case WM_NCCALCSIZE:
            if (wparam) return 0;
            break;
        case WM_NCHITTEST: {
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
            limits.ptMaxPosition = {monitor.rcWork.left - monitor.rcMonitor.left, monitor.rcWork.top - monitor.rcMonitor.top};
            limits.ptMaxSize = {monitor.rcWork.right - monitor.rcWork.left, monitor.rcWork.bottom - monitor.rcWork.top};
            const auto dpi = GetDpiForWindow(window);
            limits.ptMinTrackSize = {MulDiv(960, dpi, 96), MulDiv(640, dpi, 96)};
            return 0;
        }
        }
        return CallWindowProcW(platform.original_window_proc, window, message, wparam, lparam);
    }
} // namespace flowdit::editor
