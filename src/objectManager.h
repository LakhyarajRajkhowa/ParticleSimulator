#pragma once

#include <vector>
#include <iostream>
#include <glm/vec2.hpp>
#include "VerletObjectCUDA.cuh"
#include <imgui/imgui.h>

class ObjectManager
{
private:

public:
    ObjectManager() = default;

	static constexpr int MAX_PARTICLES = 1000000; // 1 million
    static constexpr float objectRadius = 2.5f; 

    static constexpr float screenWidth = 1500;
    static constexpr float screenHeight = 700;

    float dt = 0.75f;
    bool spawn = false;


	// methods for GPU objects
    cudaGraphicsResource* cudaVBOResource = nullptr;
    VerletObjectCUDA* d_particles = nullptr;
    std::vector<VerletObjectCUDA> getObjectsGPU();
    void addObjectGPU(float2 position, float radius, float3 color, float2 velocity, float dt);  
    int getGPUObjectsCount() const { return gpuObjectsCount; }
    int gpuObjectsCount = 0;

	// Physics constants
    glm::vec2 gravity = { 0.0f, 1000.0f };
	float restitution = 0.50f;
    float reponse_coef = 0.75f;
	float attraction_coef = 0.0f;
    float repulsion_coef = 0.0f;

   // Interactions
    float bubbleRadius = 10 * objectRadius; // bubble to move objects by left mouse click
    float bubbleIntensity = 100.0f; 
};
