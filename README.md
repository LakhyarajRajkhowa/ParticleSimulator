# ParticleSimulator with SDL2 

A simple 2-D Particle Simulator engine made in C++ using SDL2 for rendering.  
It uses Verlet integration to simulate realistic particle physics.  
Gravity can be dynamically adjusted in real time using the keyboard.

<img width="1871" height="928" alt="Image" src="https://github.com/user-attachments/assets/790680f7-7aa7-4865-a35a-5b2ba8fec23a" />

## Features

- Particle simulation using Verlet integration
- Interactive gravity control using keyboard:
  - **W**: Apply upward acceleration (gravity up)
  - **S**: Apply downward acceleration (gravity down)
  - **A**: Apply leftward acceleration (gravity left)
  - **D**: Apply rightward acceleration (gravity right)
- Real-time 2D rendering with SDL2
- Modular and extensible C++ code structure

## Prerequisites

- SDL2 installed (or use provided `lib/` folder)
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

- Copy the "run/SDL2.dll" program and paste in the build folder near the .exe program.
- Run the .exe program
