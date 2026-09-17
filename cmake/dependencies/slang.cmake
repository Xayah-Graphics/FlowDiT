include_guard(GLOBAL)
flowdit_require_dependency(vulkan)
find_program(FLOWDIT_SLANG_COMPILER NAMES slangc HINTS "${FLOWDIT_VULKAN_SDK_DIRECTORY}/Bin" REQUIRED NO_DEFAULT_PATH)
