// ParticleKernels.h
#pragma once


#include "CollisionSolverKernel.cuh"

#include "ObjectManager.h"
#include "Interaction.cuh"
#include "addColors.cuh"


void launchUpdateParticlesKernel(VerletObjectCUDA* d_particles, float* d_vbo, uint64_t N, float dt);
void applyGravityGPU(VerletObjectCUDA* d_particles, uint64_t N, float3 gravity);
void applyBoundaryCollisionGPU(VerletObjectCUDA* d_particles, uint64_t N, uint16_t boxWidth, uint16_t boxHeight, uint16_t boxDepth, float restitution);
void solveCollisionByGridGPU(VerletObjectCUDA* d_particles, uint64_t N, float response_coef, float attraction_coef , float repuslsion_coef);
void solveInteraction(VerletObjectCUDA* particles, uint64_t N, float2 mouseCoords, float radius, float intensity, float dt);