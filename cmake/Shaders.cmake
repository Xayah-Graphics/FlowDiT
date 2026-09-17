set(FLOWDIT_SHADER_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/generated/editor-shaders")
set(FLOWDIT_SHADER_BINARIES)
foreach (stage IN ITEMS vertex fragment)
    set(output "${FLOWDIT_SHADER_OUTPUT_DIRECTORY}/imgui_${stage}.spv")
    add_custom_command(OUTPUT "${output}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${FLOWDIT_SHADER_OUTPUT_DIRECTORY}"
            COMMAND "${FLOWDIT_SLANG_COMPILER}" "${PROJECT_SOURCE_DIR}/flowdit/editor/shaders/imgui.slang"
            -target spirv -profile spirv_1_6 -entry imgui_${stage} -stage ${stage} -emit-spirv-directly
            -capability SPIRV_1_6+spvDescriptorHeapEXT -spirv-unified-descriptor-heap-stride -fvk-use-entrypoint-name -fvk-use-c-layout -matrix-layout-row-major -O2
            -o "${output}"
            DEPENDS "${PROJECT_SOURCE_DIR}/flowdit/editor/shaders/imgui.slang" VERBATIM)
    list(APPEND FLOWDIT_SHADER_BINARIES "${output}")
endforeach ()
add_custom_target(flowdit_shaders DEPENDS ${FLOWDIT_SHADER_BINARIES})
