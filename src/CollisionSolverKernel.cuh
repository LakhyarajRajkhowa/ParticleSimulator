#pragma once
#include "VerletObjectCUDA.cuh"
#include "device_launch_parameters.h"
#include "ObjectManager.h"


constexpr int GRID_SIZE_GPU = 6 * ObjectManager::objectRadius;

const int GRID_WIDTH = ObjectManager::screenWidth / GRID_SIZE_GPU;
const int GRID_HEIGHT = ObjectManager::screenHeight / GRID_SIZE_GPU;
constexpr int MAX_PER_CELL = 64;
constexpr int DISPLACEMENT_ARRAY_SIZE = GRID_WIDTH * GRID_HEIGHT;



void resetGridGPU();

__global__ void buildGridKernel(VerletObjectCUDA* particles, int N);
__global__ void solveCollisionByGridKernel(VerletObjectCUDA* particles, int N, float response_coef, float attraction_coef, float repulsion_coef);
__global__ void applyDisplacementsKernel(VerletObjectCUDA* particles, int N);
__global__ void applyBoundaryCollisionKernel(VerletObjectCUDA* particles, int N, int screenWidth, int screenHeight, float restitution);

