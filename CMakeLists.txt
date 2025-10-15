cmake_minimum_required(VERSION 3.10)
project(ParticleSimulator LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

# Collect sources
file(GLOB_RECURSE CPP_SOURCES "src/*.cpp")
file(GLOB_RECURSE CUDA_SOURCES "src/*.cu")

# Output all builds to /bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
foreach(OUTPUTCONFIG IN LISTS CMAKE_CONFIGURATION_TYPES)
    string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG_UPPER)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UPPER} ${CMAKE_SOURCE_DIR}/bin)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UPPER} ${CMAKE_SOURCE_DIR}/bin)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UPPER} ${CMAKE_SOURCE_DIR}/bin)
endforeach()

# ImGui paths
set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/include/imgui)

# SDL2 paths
set(SDL2_LIB_DIR ${CMAKE_SOURCE_DIR}/lib)
set(SDL2_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include/SDL2)

# ImGui sources
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

# Add CUDA toolkit
find_package(CUDAToolkit REQUIRED)

# Create executable
add_executable(${PROJECT_NAME}
    ${CPP_SOURCES}
    ${CUDA_SOURCES}
    ${IMGUI_SOURCES}
)

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    ${SDL2_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)
# Link directories (Windows only)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)

# Link libraries
target_link_libraries(${PROJECT_NAME}
    SDL2main
    SDL2
    glew32
    opengl32
    CUDA::cudart
)

# Optional: treat warnings as errors for safety
# target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4 /WX>)
