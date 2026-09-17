include_guard(GLOBAL)
find_package(Vulkan 1.4 REQUIRED GLOBAL)
list(GET Vulkan_INCLUDE_DIRS 0 FLOWDIT_VULKAN_INCLUDE_DIRECTORY)
cmake_path(GET FLOWDIT_VULKAN_INCLUDE_DIRECTORY PARENT_PATH FLOWDIT_VULKAN_SDK_DIRECTORY)
add_library(flowdit_vulkan STATIC)
add_library(flowdit::vulkan ALIAS flowdit_vulkan)
target_sources(flowdit_vulkan PUBLIC FILE_SET cxx_modules TYPE CXX_MODULES
        BASE_DIRS "${FLOWDIT_VULKAN_INCLUDE_DIRECTORY}"
        FILES "${FLOWDIT_VULKAN_INCLUDE_DIRECTORY}/vulkan/vulkan.cppm")
target_link_libraries(flowdit_vulkan PUBLIC Vulkan::Vulkan)
target_compile_definitions(flowdit_vulkan PUBLIC VK_USE_PLATFORM_WIN32_KHR NOMINMAX)
