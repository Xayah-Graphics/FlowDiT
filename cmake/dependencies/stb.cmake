include_guard(GLOBAL)

FetchContent_MakeAvailable(stb)

add_library(flowdit-stb INTERFACE)
add_library(flowdit::stb ALIAS flowdit-stb)
target_include_directories(flowdit-stb SYSTEM INTERFACE "${stb_SOURCE_DIR}")
