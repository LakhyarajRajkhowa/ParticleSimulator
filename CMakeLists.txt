cmake_minimum_required(VERSION 3.10)
project(ParticleSimulator)

set(CMAKE_CXX_STANDARD 17)

# Add source files
file(GLOB_RECURSE SOURCES "src/*.cpp")

# Add include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Add library directory (only for Windows)
link_directories(${CMAKE_SOURCE_DIR}/lib)

# Create the executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Link SDL2 (Windows)
target_link_libraries(${PROJECT_NAME} SDL2main SDL2)
