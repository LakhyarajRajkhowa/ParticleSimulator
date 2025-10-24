#pragma once
#include <cuda_runtime.h>
#include <math.h>
#include "device_launch_parameters.h"

#include <iostream>
struct VerletObjectCUDA
{
    float3 current_position;
    float3 old_position;
    float3 acceleration;
    float radius;
    float3 color; 

    // VERLET INTERGATION
    __device__ void updatePosition(float dt)
    {
        float2 velocity = make_float2(
            (current_position.x - old_position.x),
            (current_position.y - old_position.y)
        );

        old_position = current_position;

        current_position.x += velocity.x + acceleration.x * dt * dt;
        current_position.y += velocity.y + acceleration.y * dt * dt;

        acceleration = make_float3(0.0f, 0.0f, 0.0f);

    }

    __device__ void accelerate(float2 a)
    {
        acceleration.x += a.x;
        acceleration.y += a.y;
    }

    __device__ void setVelocity(float2 v, float dt)
    {
        old_position.x = current_position.x - v.x * dt;
        old_position.y = current_position.y - v.y * dt;
    }

    __device__ float2 getVelocity()
    {
        return make_float2(
            current_position.x - old_position.x,
            current_position.y - old_position.y
        );
    }
};
