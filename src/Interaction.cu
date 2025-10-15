#include "Interaction.cuh"


// Use the mouse pointer to move particles
__global__ void moveObjects(VerletObjectCUDA* particles, int N, float2 mouseCoords, float radius, float intensity, float dt) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    VerletObjectCUDA& p = particles[i];

    float2 dir = make_float2(p.current_position.x - mouseCoords.x,
        p.current_position.y - mouseCoords.y);

    float dist = sqrtf(dir.x * dir.x + dir.y * dir.y);
    if (dist < radius && dist > 1e-5f)  
    {
        dir.x /= dist;
        dir.y /= dist;

        float t = dist / radius;
        float strength = intensity * (1.0f - t * t); 

        float2 vel = make_float2(p.current_position.x - p.old_position.x,
            p.current_position.y - p.old_position.y);

        vel.x += dir.x * strength * dt;
        vel.y += dir.y * strength * dt;

        p.old_position = p.current_position;
        p.current_position.x += vel.x;
        p.current_position.y += vel.y;
    }
}


