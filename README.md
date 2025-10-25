# ParticleSimulator with SDL2 

A simple 3-D Particle Simulator engine made in C++ using OpenGL for rendering.  
It uses Verlet integration to simulate realistic particle physics.  

https://github.com/user-attachments/assets/90b5d821-f079-413b-930a-71137e0ce872

## Controls
  - **ENTER** : Spawn Particles
  - **W**/**A**/**S**/**D**: Control Camera
  - **R**: Rotate camera 
  - **UP**: Apply upward acceleration (gravity up)
  - **DOWN**: Apply downward acceleration (gravity down)
  - **LEFT**: Apply leftward acceleration (gravity left)
  - **RIGHT**: Apply rightward acceleration (gravity right)
    
## Features 
  - Real-time 3D rendering with OpenGL
  - Optimised perfomance with CUDA multithreading
  
## Prerequisites
- CUDA Toolkit 13.0 or later
- A C++ compiler (e.g. g++, clang, or MSVC)
- CMake (version 3.10+)

## Build Instructions (Windows)

```bash
git clone https://github.com/LakhyarajRajkhowa/ParticleSimulator.git
cd ParticleSimulator
mkdir build
cd build
cmake ..
cmake --build .

## Run Instructions (Windows)

- run the ParticleSimulator.exe in the bin folder

