#include "VerletObjectCUDA.cuh"
#include "Color.h"
__host__ __device__ inline float3 addRainbowColor(float t)
{
    t = fmodf(t, 1.0f);

    Color color;
    float r = 0.5f + 0.5f * sinf(2.0f * 3.14159265f * (t + 0.0f));
    float g = 0.5f + 0.5f * sinf(2.0f * 3.14159265f * (t + 0.33f));
    float b = 0.5f + 0.5f * sinf(2.0f * 3.14159265f * (t + 0.66f));

    return make_float3(r, g, b);
}

__host__ __device__ inline float3 addVelocityColor(float2 velocity, float max_speed) {

    float speed = sqrtf(velocity.x * velocity.x + velocity.y * velocity.y);

    float MAX_SPEED = 1.5f;
    float t = fminf(speed / MAX_SPEED, 1.0f);

    float3 color;
    color.x = t;
    color.y = 1.0f - fabsf(t - 0.5f) * 2.0f;
    color.z = 1.0f - t;


}
