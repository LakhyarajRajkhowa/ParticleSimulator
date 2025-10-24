#pragma once
#include "VerletObjectCUDA.cuh"
#include "device_launch_parameters.h"
#include "ObjectManager.h"


constexpr int GRID_SIZE_GPU = 6 * ObjectManager::objectRadius;

const int GRID_WIDTH = ObjectManager::boxWidth / GRID_SIZE_GPU;
const int GRID_HEIGHT = ObjectManager::boxHeight / GRID_SIZE_GPU;
const int GRID_DEPTH = ObjectManager::boxDepth / GRID_SIZE_GPU;
constexpr int MAX_PER_CELL = 256;
constexpr int DISPLACEMENT_ARRAY_SIZE = GRID_WIDTH * GRID_HEIGHT * GRID_DEPTH;


void resetGrid3D();
__global__ void buildGridKernel3D(VerletObjectCUDA* particles, int N);
__global__ void solveCollisionByGridKernel3D(VerletObjectCUDA* particles, int N, float response_coef, float attraction_coef, float repulsion_coef);
__global__ void applyDisplacementsKernel3D(VerletObjectCUDA* particles, int N);
__global__ void applyBoundaryCollisionKernel(VerletObjectCUDA* particles,
	uint64_t N,
	uint16_t boxWidth,
	uint16_t boxHeight,
	uint16_t boxDepth,
	float restitution
);

