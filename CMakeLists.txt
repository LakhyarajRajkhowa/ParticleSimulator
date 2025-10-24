cmake_minimum_required(VERSION 3.10)
project(ParticleSimulator LANGUAGES CXX CUDA)

# ────────────────────────────────────────────────
#  Basic setup
# ────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ────────────────────────────────────────────────
#  Source and header file discovery
# ────────────────────────────────────────────────
file(GLOB_RECURSE CPP_HEADER_FILES "src/*.h")
file(GLOB_RECURSE CUDA_HEADER_FILES "src/*.cuh")
file(GLOB_RECURSE CPP_SOURCES "src/*.cpp")
file(GLOB_RECURSE CUDA_SOURCES "src/*.cu")
file(GLOB_RECURSE LENGINE_SOURCES "Lengine/*.cpp")
file(GLOB_RECURSE LENGINE_HEADER_FILES "Lengine/*.h")

# ────────────────────────────────────────────────
#  Output directories (/bin for all configs)
# ────────────────────────────────────────────────
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

foreach(OUTPUTCONFIG IN LISTS CMAKE_CONFIGURATION_TYPES)
    string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG_UPPER)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UPPER} ${CMAKE_SOURCE_DIR}/bin)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UPPER} ${CMAKE_SOURCE_DIR}/bin)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UPPER} ${CMAKE_SOURCE_DIR}/bin)
endforeach()

# ────────────────────────────────────────────────
#  Third-party directories (SDL2 / ImGui / CUDA)
# ────────────────────────────────────────────────
set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/include/imgui)
set(SDL2_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include/SDL2)
set(SDL2_LIB_DIR ${CMAKE_SOURCE_DIR}/lib)

# ────────────────────────────────────────────────
#  ImGui source files
# ────────────────────────────────────────────────
set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/backends/imgui_impl_sdl2.cpp
    ${IMGUI_DIR}/backends/imgui_impl_sdlrenderer2.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

# ────────────────────────────────────────────────
#  Find CUDA Toolkit
# ────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

# ────────────────────────────────────────────────
#  Source grouping for Visual Studio filters
# ────────────────────────────────────────────────
source_group("Header Files\\C++" FILES ${CPP_HEADER_FILES})
source_group("Header Files\\CUDA" FILES ${CUDA_HEADER_FILES})
source_group("Source Files\\C++" FILES ${CPP_SOURCES})
source_group("Source Files\\CUDA" FILES ${CUDA_SOURCES})
source_group("External\\ImGui" FILES ${IMGUI_SOURCES})
source_group("Lengine\\Headers" FILES ${LENGINE_HEADER_FILES})
source_group("Lengine\\Source" FILES ${LENGINE_SOURCES})

# ────────────────────────────────────────────────
#  Create the executable
# ────────────────────────────────────────────────
add_executable(${PROJECT_NAME}
    ${CPP_HEADER_FILES}
    ${CUDA_HEADER_FILES}
    ${CPP_SOURCES}
    ${CUDA_SOURCES}
    ${IMGUI_SOURCES}
    ${LENGINE_HEADER_FILES}
    ${LENGINE_SOURCES}
)

# ────────────────────────────────────────────────
#  Include directories
# ────────────────────────────────────────────────
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    ${SDL2_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
    
)

# ────────────────────────────────────────────────
#  Link directories and libraries
# ────────────────────────────────────────────────
target_link_directories(${PROJECT_NAME} PRIVATE 
    ${SDL2_LIB_DIR}
)

target_link_libraries(${PROJECT_NAME}
    SDL2main
    SDL2
    glew32
    opengl32
    CUDA::cudart
)

# ────────────────────────────────────────────────
#  (Optional) Compiler warnings
# ────────────────────────────────────────────────
# target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4 /WX>)
