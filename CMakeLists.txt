# ============================================================================
# Project
# ============================================================================
cmake_minimum_required(VERSION 4.4)
set(CACHE{CMAKE_CUDA_ARCHITECTURES}
        TYPE STRING
        HELP "CUDA architectures"
        FORCE
        VALUE 120
)
set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "f35a9ac6-8463-4d38-8eec-5d6008153e7d")
file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/VERSION" FLOWDIT_VERSION LIMIT_COUNT 1)
project(flowdit VERSION "${FLOWDIT_VERSION}" LANGUAGES CXX CUDA)

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

option(FLOWDIT_EXAMPLES "Build examples" ON)

# ============================================================================
# Dependencies
# ============================================================================
include(cmake/Dependencies.cmake)
flowdit_require_dependency(cuda)
flowdit_require_dependency(cccl)
flowdit_require_dependency(nlohmann_json)
flowdit_require_dependency(stb)
include(cmake/ClionCudaFacade.cmake)

# ============================================================================
# Compiler Configuration
# ============================================================================
if (WIN32)
    add_compile_definitions(NOMINMAX)
endif ()

if (MSVC)
    add_compile_options(
            $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>
            $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
    )
endif ()

# ============================================================================
# Neural
# ============================================================================
add_library(flowdit-neural STATIC)
target_sources(
        flowdit-neural
        PRIVATE
        flowdit/neural/matmul.cpp
        flowdit/neural/training-state-kernels.h
        flowdit/neural/training-state.cpp
        flowdit/neural/training-state.cu
        flowdit/neural/transformer-kernels.h
        flowdit/neural/transformer.cpp
        flowdit/neural/transformer.cu
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        flowdit/neural/matmul.ixx
        flowdit/neural/training-state.ixx
        flowdit/neural/transformer.ixx
)
target_include_directories(flowdit-neural PUBLIC "${FLOWDIT_CLION_CUDA_INCLUDE_DIRECTORY}")
target_link_libraries(
        flowdit-neural
        PUBLIC
        CCCL::libcudacxx
        CUDA::toolkit
        PRIVATE
        CUDA::cublas
        CUDA::cublasLt
)

# ============================================================================
# Serialization
# ============================================================================
add_library(flowdit-serialization STATIC)
target_sources(
        flowdit-serialization
        PRIVATE
        flowdit/serialization/safetensors.cpp
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        flowdit/serialization/safetensors.ixx
)
target_compile_definitions(flowdit-serialization PRIVATE FLOWDIT_PROJECT_VERSION="${PROJECT_VERSION}")
target_link_libraries(flowdit-serialization PRIVATE nlohmann_json::nlohmann_json)

# ============================================================================
# FlowDiT
# ============================================================================
add_library(flowdit STATIC)
target_sources(
        flowdit
        PRIVATE
        flowdit/dataset/cifar10.cpp
        flowdit/kernels.h
        flowdit/kernels.cu
        flowdit/model.cpp
        flowdit/sampling.cpp
        flowdit/flowdit.cpp
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        flowdit/dataset/cifar10.ixx
        flowdit/model.ixx
        flowdit/sampling.ixx
        flowdit/flowdit.ixx
)
target_include_directories(flowdit PUBLIC "${FLOWDIT_CLION_CUDA_INCLUDE_DIRECTORY}")
target_link_libraries(
        flowdit
        PUBLIC
        CCCL::libcudacxx
        CUDA::toolkit
        flowdit-neural
        PRIVATE
        CUDA::curand
        flowdit-serialization
)

# ============================================================================
# Examples
# ============================================================================
if (FLOWDIT_EXAMPLES)
    add_executable(
            flowdit-example-cifar10-cli
            examples/cifar10/main.cpp
    )
    target_link_libraries(
            flowdit-example-cifar10-cli
            PRIVATE
            flowdit
            flowdit::stb
    )
endif ()

# MSVC interprets a bare @cmake_cxx_std.lib as a response file.
if (MSVC)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/cxx-modules")
endif ()

