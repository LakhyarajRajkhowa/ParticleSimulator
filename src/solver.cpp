#include "solver.h"

void Solver::updateCPU()
{

    float dt_sub = dt / sub_steps;

    for (Uint32 i = 0; i < sub_steps; ++i)
    {
        // CPU version
        applyGravityCPU();
        collisionHandler.applyBorder(SCREEN_HEIGHT, SCREEN_WIDTH, objectManager);
        collisionHandler.solveCollisionByGrid(objectManager);
        updatePositionCPU();

		
    }

}
void Solver::updateGPU()
{
    int N = objectManager.getGPUObjectsCount();
    for (Uint32 i = 0; i < sub_steps; ++i)
    {
        
        // GPU version
        if (objectManager.d_particles == nullptr || N <= 0) return;
        applyGravityGPU(objectManager.d_particles, N, make_float2(gravity.x, gravity.y));
        applyBoundaryCollisionGPU(objectManager.d_particles, N, SCREEN_WIDTH, SCREEN_HEIGHT, objectManager.restitution);      
        solveCollisionByGridGPU(objectManager.d_particles, N, objectManager.reponse_coef, objectManager.attraction_coef, objectManager.repulsion_coef);
        updatePositionGPU();


    }
}

void Solver::updatePositionGPU()
{
    int N = objectManager.getGPUObjectsCount();
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

void Solver::updatePositionCPU()
{
    for (auto& obj : objectManager.getObjectsCPU()) {
        obj.updatePosition(dt);
    }
}


void Solver::applyGravityCPU()
{
    for (auto& obj : objectManager.getObjectsCPU())
    {
        obj.accelerate(gravity);
    }
}
