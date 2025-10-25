#include "ObjectManager.h"



void ObjectManager::addObjectGPU(float3 position, float radius, float3 color, float3 velocity, float dt) {
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
	temp.current_position.z = position.z;
    temp.old_position.x =position.x - velocity.x * dt;
	temp.old_position.y = position.y - velocity.y * dt;
    temp.old_position.z = position.z - velocity.z * dt;

    temp.acceleration = make_float3(0.0f, 0.0f, 0.0f);
    temp.radius = radius;
    temp.color = color;
	temp.escpaedBox = false;

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
