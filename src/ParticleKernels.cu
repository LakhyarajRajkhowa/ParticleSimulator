#pragma once
#include "ParticleKernels.cuh"
#include <iostream>


__global__ void updateParticlesKernel(VerletObjectCUDA* particles, float* vboPtr, int N, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    VerletObjectCUDA& p = particles[i];

    p.updatePosition(dt);
 
    int idx = i * 6;
    vboPtr[idx + 0] = p.current_position.x;
    vboPtr[idx + 1] = p.current_position.y;
    vboPtr[idx + 2] = p.color.x; 
    vboPtr[idx + 3] = p.color.y; 
    vboPtr[idx + 4] = p.color.z; 
    vboPtr[idx + 5] = p.radius;
    
};

__global__ void applyGravityKernel(VerletObjectCUDA* particles, int N, float2 gravity)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    particles[i].accelerate(gravity);
}




 void launchUpdateParticlesKernel(VerletObjectCUDA* d_particles, float* d_vbo, int N, float dt)
{
     
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    updateParticlesKernel << <blocks, threads >> > (d_particles, d_vbo, N, dt);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[CUDA] Kernel launch error: " << cudaGetErrorString(err) << "\n";
    }

   
}
 void applyGravityGPU(VerletObjectCUDA* d_particles, int N, float2 gravity)
 {
    
     int threads = 256;
     int blocks = (N + threads - 1) / threads;

     applyGravityKernel << <blocks, threads >> > (d_particles, N, gravity);

     cudaError_t err = cudaGetLastError();
     if (err != cudaSuccess) {
         std::cerr << "[CUDA] applyGravityKernel error: " << cudaGetErrorString(err) << "\n";
     }  
 }
 void applyBoundaryCollisionGPU(VerletObjectCUDA* d_particles, int N, int screenWidth, int screenHeight, float restitution)
 {
     int threads = 256;
     int blocks = (N + threads - 1) / threads;

     applyBoundaryCollisionKernel << <blocks, threads >> > (d_particles, N, screenWidth, screenHeight, restitution);

     cudaError_t err = cudaGetLastError();
     if (err != cudaSuccess) {
         std::cerr << "[CUDA] applyBorderKernel error: " << cudaGetErrorString(err) << "\n";
     }  
 }

 void solveCollisionByGridGPU(VerletObjectCUDA* d_particles, int N, float response_coef, float attraction_coef, float repulsion_coef) {
   
     int threads = 256;
     int blocks = (N + threads - 1) / threads;

     resetGridGPU();
     buildGridKernel << <blocks, threads >> > (d_particles, N);
     solveCollisionByGridKernel << <blocks, threads >> > (d_particles, N, response_coef, attraction_coef, repulsion_coef);
     applyDisplacementsKernel << <blocks, threads >> > (d_particles, N);

     cudaError_t err = cudaGetLastError();
     if (err != cudaSuccess)
         std::cerr << "[CUDA] solveCollisionByGridGPU error: " << cudaGetErrorString(err) << "\n";
 }

 void solveInteraction(VerletObjectCUDA* d_particles, int N, float2 mouseCoords, float radius, float intensity, float dt) {
     int threads = 256;
     int blocks = (N + threads - 1) / threads;
     moveObjects << <blocks, threads >> > (d_particles, N, mouseCoords, radius, intensity, dt);

     cudaError_t err = cudaGetLastError();
     if (err != cudaSuccess) {
         std::cerr << "[CUDA] solveInteraction error: " << cudaGetErrorString(err) << "\n";
     }
 }