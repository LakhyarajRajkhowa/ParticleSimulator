#include "solver.h"


void Solver::update()
{
    int N = objectManager.getGPUObjectsCount();
    for (Uint32 i = 0; i < sub_steps; ++i)
    {        
        if (objectManager.d_particles == nullptr || N <= 0) return;
        applyGravityGPU(objectManager.d_particles, N, make_float3(objectManager.gravity.x, objectManager.gravity.y, objectManager.gravity.z));
        applyBoundaryCollisionGPU(objectManager.d_particles, N, objectManager.boxWidth, objectManager.boxHeight, objectManager.boxDepth, objectManager.restitution);      
        solveCollisionByGridGPU(objectManager.d_particles, N, objectManager.reponse_coef, objectManager.attraction_coef, objectManager.repulsion_coef);
        updatePositionGPU();

    }
}

void Solver::updatePositionGPU()
{
    uint64_t N = objectManager.getGPUObjectsCount();
    if (N == 0) return;

    float* d_vbo = nullptr;
    size_t num_bytes = 0;
    cudaGraphicsMapResources(1, &objectManager.cudaVBOResource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&d_vbo, &num_bytes, objectManager.cudaVBOResource);

    size_t expectedBytes = objectManager.MAX_PARTICLES * sizeof(float);
    if (num_bytes < expectedBytes) {

        std::cerr << "[CUDA] VBO size mismatch! num_bytes=" << num_bytes
            << ", expected=" << expectedBytes << "\n";
    }
   
    
    launchUpdateParticlesKernel(objectManager.d_particles, d_vbo, N, dt_sub/ ImGui::GetIO().Framerate);


    cudaGraphicsUnmapResources(1, &objectManager.cudaVBOResource, 0);
}


