#include "ObjectManager.h"



void ObjectManager::addObjectGPU(float2 position, float radius, float3 color, float2 velocity, float dt) {
    if (!d_particles) {
        std::cerr << "[GPU] Error: GPU buffer not allocated!\n";
        return;
    }
    if (gpuObjectsCount >= MAX_PARTICLES) {
        std::cerr << "[GPU] Max particles reached!\n";
        return;
    }

    VerletObjectCUDA temp;
    temp.current_position.x = position.x;
	temp.current_position.y = position.y;
    temp.old_position.x =position.x - velocity.x * dt;
	temp.old_position.y = position.y - velocity.y * dt;
    temp.acceleration = make_float3(0.0f, 0.0f, 0.0f);
    temp.radius = radius;
    temp.color = color;

    cudaMemcpy(&d_particles[gpuObjectsCount], &temp, sizeof(VerletObjectCUDA), cudaMemcpyHostToDevice);

    gpuObjectsCount++;
}

std::vector<VerletObjectCUDA> ObjectManager::getObjectsGPU() {
    std::vector<VerletObjectCUDA> hostParticles(gpuObjectsCount);

    if (!d_particles) {
        std::cerr << "[GPU] Error: GPU buffer not allocated!\n";
        return {};
    }
    if (gpuObjectsCount <= 0) {
        return {};
    }

    cudaError_t err = cudaMemcpy(
        hostParticles.data(),
        d_particles,
        gpuObjectsCount * sizeof(VerletObjectCUDA),
        cudaMemcpyDeviceToHost
    );

    if (err != cudaSuccess) {
        std::cerr << "[GPU] cudaMemcpy (DeviceToHost) failed: "
            << cudaGetErrorString(err) << std::endl;
        return {};
    }

    

    return hostParticles;
}
