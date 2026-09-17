include_guard(GLOBAL)
flowdit_require_dependency(glfw)
FetchContent_MakeAvailable(imgui)
add_library(flowdit_imgui STATIC
        "${imgui_SOURCE_DIR}/imgui.cpp"
        "${imgui_SOURCE_DIR}/imgui_draw.cpp"
        "${imgui_SOURCE_DIR}/imgui_tables.cpp"
        "${imgui_SOURCE_DIR}/imgui_widgets.cpp"
        "${imgui_SOURCE_DIR}/misc/cpp/imgui_stdlib.cpp"
        "${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp")
add_library(flowdit::imgui ALIAS flowdit_imgui)
set_target_properties(flowdit_imgui PROPERTIES CXX_SCAN_FOR_MODULES OFF CXX_MODULE_STD OFF)
target_include_directories(flowdit_imgui PUBLIC "${imgui_SOURCE_DIR}" "${imgui_SOURCE_DIR}/backends")
target_compile_definitions(flowdit_imgui PRIVATE GLFW_INCLUDE_NONE)
target_link_libraries(flowdit_imgui PUBLIC glfw)
