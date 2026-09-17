# ============================================================================
# Project
# ============================================================================
cmake_minimum_required(VERSION 4.4)
set(CACHE{CMAKE_CUDA_ARCHITECTURES} TYPE STRING HELP "CUDA architectures" FORCE VALUE 120)
set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "f35a9ac6-8463-4d38-8eec-5d6008153e7d")
file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/VERSION" FLOWDIT_VERSION LIMIT_COUNT 1)
project(flowdit VERSION "${FLOWDIT_VERSION}" LANGUAGES C CXX CUDA)

# ============================================================================
# Project Configuration
# ============================================================================
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_MODULE_STD ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT WIN32 AND NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "FlowDiT supports Windows and Linux")
endif ()
option(FLOWDIT_BUILD_EDITOR "Build the Windows Vulkan Editor" ${WIN32})
if (FLOWDIT_BUILD_EDITOR AND NOT WIN32)
    message(FATAL_ERROR "The FlowDiT Editor requires Windows")
endif ()

# ============================================================================
# Compiler Configuration
# ============================================================================
if (WIN32)
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
endif ()
if (MSVC)
    add_compile_options(
            $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
            $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>
            $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
endif ()

# ============================================================================
# Dependencies
# ============================================================================
include(cmake/Dependencies.cmake)
flowdit_require_dependency(cuda)
flowdit_require_dependency(cccl)
flowdit_require_dependency(threads)
flowdit_require_dependency(nlohmann_json)
flowdit_require_dependency(stb)
include(cmake/ClionCudaFacade.cmake)

# ============================================================================
# Core
# ============================================================================
add_library(flowdit_core STATIC)
target_sources(flowdit_core PRIVATE
        flowdit/core/headless.cpp
        flowdit/core/dataset/cifar10.cpp
        flowdit/core/dataset/load.cpp
        flowdit/core/dataset/mnist.cpp
        flowdit/core/io/output.cpp
        flowdit/core/model/configuration.cpp
        flowdit/core/model/flowdit.cpp
        flowdit/core/model/kernels.cu
        flowdit/core/neural/matmul.cpp
        flowdit/core/neural/training-state.cpp
        flowdit/core/neural/training-state.cu
        flowdit/core/neural/transformer.cpp
        flowdit/core/neural/transformer.cu
        flowdit/core/runtime/session.cpp
        flowdit/core/sampling/runtime.cpp
        flowdit/core/sampling/sampler.cpp
        flowdit/core/serialization/safetensors.cpp
        flowdit/core/training/trainer.cpp
        PUBLIC FILE_SET cxx_modules TYPE CXX_MODULES FILES
        flowdit/core/headless.ixx
        flowdit/core/dataset/cifar10.ixx
        flowdit/core/dataset/load.ixx
        flowdit/core/dataset/mnist.ixx
        flowdit/core/dataset/types.ixx
        flowdit/core/io/output.ixx
        flowdit/core/model/configuration.ixx
        flowdit/core/model/flowdit.ixx
        flowdit/core/neural/matmul.ixx
        flowdit/core/neural/training-state.ixx
        flowdit/core/neural/transformer.ixx
        flowdit/core/runtime/session.ixx
        flowdit/core/sampling/types.ixx
        flowdit/core/sampling/runtime.ixx
        flowdit/core/sampling/sampler.ixx
        flowdit/core/serialization/safetensors.ixx
        flowdit/core/training/trainer.ixx)
target_include_directories(flowdit_core PUBLIC "${FLOWDIT_CLION_CUDA_INCLUDE_DIRECTORY}")
target_compile_definitions(flowdit_core PRIVATE FLOWDIT_PROJECT_VERSION="${PROJECT_VERSION}")
target_link_libraries(flowdit_core PUBLIC CCCL::libcudacxx CUDA::toolkit Threads::Threads
        PRIVATE CUDA::cublas CUDA::cublasLt CUDA::curand nlohmann_json::nlohmann_json flowdit::stb)

# ============================================================================
# Editor
# ============================================================================
if (FLOWDIT_BUILD_EDITOR)
    flowdit_require_dependency(vulkan)
    flowdit_require_dependency(slang)
    flowdit_require_dependency(imgui)
    flowdit_require_dependency(implot)
    include(cmake/Shaders.cmake)
    add_library(flowdit_editor STATIC)
    target_sources(flowdit_editor PRIVATE
            flowdit/editor/editor.cpp
            flowdit/editor/platform/window.cpp
            flowdit/editor/graphics/device.cpp
            flowdit/editor/graphics/resources.cpp
            flowdit/editor/graphics/renderer.cpp
            flowdit/editor/graphics/interop.cpp
            flowdit/editor/viewing/canvas.cpp
            flowdit/editor/panels/dataset.cpp
            flowdit/editor/panels/training.cpp
            flowdit/editor/panels/sampling.cpp
            PUBLIC FILE_SET cxx_modules TYPE CXX_MODULES FILES flowdit/editor/editor.ixx
            PRIVATE FILE_SET editor_modules TYPE CXX_MODULES FILES
            flowdit/editor/platform/window.ixx
            flowdit/editor/graphics/device.ixx
            flowdit/editor/graphics/resources.ixx
            flowdit/editor/graphics/renderer.ixx
            flowdit/editor/graphics/interop.ixx
            flowdit/editor/viewing/canvas.ixx
            flowdit/editor/panels/dataset.ixx
            flowdit/editor/panels/training.ixx
            flowdit/editor/panels/sampling.ixx)
    target_compile_definitions(flowdit_editor PRIVATE GLFW_INCLUDE_NONE GLFW_EXPOSE_NATIVE_WIN32
            FLOWDIT_SHADER_DIRECTORY="${FLOWDIT_SHADER_OUTPUT_DIRECTORY}"
            FLOWDIT_SOURCE_DIRECTORY="${PROJECT_SOURCE_DIR}")
    target_link_libraries(flowdit_editor PRIVATE flowdit_core flowdit::vulkan flowdit::imgui flowdit::implot CUDA::cudart)
    add_dependencies(flowdit_editor flowdit_shaders)
endif ()

# ============================================================================
# Application
# ============================================================================
add_executable(flowdit main.cpp)
target_link_libraries(flowdit PRIVATE flowdit_core)
if (FLOWDIT_BUILD_EDITOR)
    target_compile_definitions(flowdit PRIVATE FLOWDIT_HAS_EDITOR)
    target_link_libraries(flowdit PRIVATE flowdit_editor)
endif ()
if (MSVC)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/cxx-modules")
endif ()
