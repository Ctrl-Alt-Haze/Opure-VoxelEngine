# ============================================================================
#  Opure-VoxelEngine
#  Native C++20 / DirectX 12 LLM inference + idle-time plasticity runtime.
#
#  No third-party dependencies: Win32, D3D12 and DXC only.
#
#  Configure + build:
#     cmake -S . -B build -G "Visual Studio 17 2022" -A x64
#     cmake --build build --config Release
#
#  Shaders are compiled to DXIL at configure/build time by scripts/build_shaders.ps1
#  and embedded into the binary, so the runtime never shells out to dxc.
# ============================================================================
cmake_minimum_required(VERSION 3.24)
project(OpureVoxelEngine VERSION 1.0.0 LANGUAGES CXX)

if(NOT MSVC)
    message(FATAL_ERROR
        "Opure-VoxelEngine targets MSVC on Windows only (Direct3D 12 is required).")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ---------------------------------------------------------------------------
#  Warnings as errors. /permissive- enables standards-conformance mode, which
#  catches most MSVC extensions that would otherwise silently compile.
# ---------------------------------------------------------------------------
add_library(opure_warnings INTERFACE)
target_compile_options(opure_warnings INTERFACE
    /W4
    /WX
    /permissive-
    /Zc:__cplusplus
    /Zc:preprocessor
    /MP
    /Zi
    /utf-8
    /EHsc
    /GR
    /bigobj
    # Noise that is not actionable and originates outside our translation units.
    /wd4100   # unreferenced formal parameter
    /wd4127   # conditional expression is constant
)

# D3D12 wants its own C++ dialect settings.
add_library(opure_dx12 INTERFACE)
target_compile_options(opure_dx12 INTERFACE /D_CRT_SECURE_NO_WARNINGS /DNOMINMAX /DWIN32_LEAN_AND_MEAN)
target_link_libraries(opure_dx12 INTERFACE d3d12 dxgi dxguid)

# ---------------------------------------------------------------------------
#  Core library
# ---------------------------------------------------------------------------
file(GLOB OPURE_CORE_SOURCES   CONFIGURE_DEPENDS src/core/*.cpp)
file(GLOB OPURE_IO_SOURCES     CONFIGURE_DEPENDS src/io/*.cpp)
file(GLOB OPURE_MODEL_SOURCES  CONFIGURE_DEPENDS src/model/*.cpp)
file(GLOB OPURE_ENGINE_SOURCES CONFIGURE_DEPENDS src/engine/*.cpp)

add_library(OpureVoxelEngine STATIC
    ${OPURE_CORE_SOURCES}
    ${OPURE_IO_SOURCES}
    ${OPURE_MODEL_SOURCES}
    ${OPURE_ENGINE_SOURCES}
)

target_include_directories(OpureVoxelEngine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(OpureVoxelEngine PUBLIC opure_dx12 opure_warnings)

# The generated shader blob is produced by scripts/build_shaders.ps1 and checked
# into generated/ by the build (see the "shaders" custom target below).
target_include_directories(OpureVoxelEngine PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}/generated
)

set_target_properties(OpureVoxelEngine PROPERTIES
    VS_WEB_TARGETS   OFF
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

# ---------------------------------------------------------------------------
#  Test harness executable (Phase 1 verification + Phase 2 switch)
# ---------------------------------------------------------------------------
add_executable(opure_harness
    src/app/Opure_TestHarness.cpp
)
target_link_libraries(opure_harness PRIVATE OpureVoxelEngine opure_warnings)

# ---------------------------------------------------------------------------
#  Inspector executable -- reads a model dir and emits a support report
# ---------------------------------------------------------------------------
add_executable(opure_inspect
    src/app/Opure_Inspect.cpp
)
target_link_libraries(opure_inspect PRIVATE OpureVoxelEngine opure_warnings)

# ---------------------------------------------------------------------------
#  Shader compilation -> embedded DXIL blob
# ---------------------------------------------------------------------------
find_program(OPURE_DXC
    NAMES dxc.exe
    HINTS
        "C:/VulkanSDK/1.4.357.0/Bin"
        "$ENV{VULKAN_SDK}/Bin"
        "C:/Program Files (x86)/Windows Kits/10/bin"
    PATH_SUFFIXES x64
)

if(OPURE_DXC)
    message(STATUS "Opure-VoxelEngine: DXC found at ${OPURE_DXC}")
    set(OPURE_DXC_PATH "${OPURE_DXC}" CACHE FILEPATH "DXC executable" FORCE)
else()
    message(WARNING
        "Opure-VoxelEngine: dxc.exe not found. Shaders will not be rebuilt; "
        "run scripts/build_shaders.ps1 once dxc is on PATH.")
    set(OPURE_DXC_PATH "" CACHE FILEPATH "DXC executable" FORCE)
endif()

add_custom_target(opure_shaders
    COMMAND ${CMAKE_COMMAND} -E echo "Building Opure shaders -> DXIL blob"
    COMMAND powershell -NoProfile -ExecutionPolicy Bypass
            -File "${CMAKE_CURRENT_SOURCE_DIR}/scripts/build_shaders.ps1"
            -DxcPath "${OPURE_DXC_PATH}"
            -ShaderDir "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
            -OutFile "${CMAKE_CURRENT_BINARY_DIR}/generated/Opure_ShaderBlob.h"
    DEPENDS ${OPURE_CORE_SOURCES}
    COMMENT "Compiling HLSL compute shaders to an embedded DXIL blob"
    VERBATIM
)
# Allow the harness to be built standalone while the shader set is being brought
# up. Once every manifest shader exists, the dependency is enforced again.
file(GLOB OPURE_SHADER_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.usf")
list(LENGTH OPURE_SHADER_FILES _opure_shader_count)
if(_opure_shader_count LESS 10)
    add_custom_target(opure_shaders_optional)
    add_dependencies(opure_harness opure_shaders_optional)
else()
    add_dependencies(opure_harness opure_shaders)
endif()

# ---------------------------------------------------------------------------
#  Opure TUI executable (interactive chat) -- links engine directly
# ---------------------------------------------------------------------------
add_executable(opure_chat
    src/app/Opure_Chat.cpp
)
target_link_libraries(opure_chat PRIVATE OpureVoxelEngine opure_warnings)

# ---------------------------------------------------------------------------
#  Opure Server executable -- simple HTTP for UE5 plugin
# ---------------------------------------------------------------------------
add_executable(opure_server
    src/app/Opure_Server.cpp
)
target_link_libraries(opure_server PRIVATE OpureVoxelEngine opure_warnings)