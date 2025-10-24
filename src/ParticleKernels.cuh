// ParticleKernels.h
#pragma once


#include "CollisionSolverKernel.cuh"

#include "ObjectManager.h"
#include "Interaction.cuh"
#include "addColors.cuh"


void launchUpdateParticlesKernel(VerletObjectCUDA* d_particles, float* d_vbo, int N, float dt);
void applyGravityGPU(VerletObjectCUDA* d_particles, int N, float2 gravity);
void applyBoundaryCollisionGPU(VerletObjectCUDA* d_particles, int N, int screenWidth, int screenHeight, float restitution);
void solveCollisionByGridGPU(VerletObjectCUDA* d_particles, int N, float response_coef, float attraction_coef , float repuslsion_coef);
void solveInteraction(VerletObjectCUDA* particles, int N, float2 mouseCoords, float radius, float intensity, float dt);