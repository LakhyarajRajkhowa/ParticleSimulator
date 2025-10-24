#pragma once

#include <vector>
#include <iostream>
#include <glm/glm.hpp>
#include "VerletObjectCUDA.cuh"
#include <imgui/imgui.h>

extern bool spawn ;

class ObjectManager
{
private:

public:
    ObjectManager() = default;

	static constexpr uint64_t MAX_PARTICLES = 1000000; // 1 million
    static constexpr float objectRadius = 10.0f; // crashes on 2.5 , idk why 

    static constexpr float screenWidth = 1500;
    static constexpr float screenHeight = 700;
    
    static constexpr uint16_t boxWidth = 600;
    static constexpr uint16_t boxHeight = 200;
    static constexpr uint16_t boxDepth = 200;

    float dt = 0.75f;


	// methods for GPU objects
    cudaGraphicsResource* cudaVBOResource = nullptr;
    VerletObjectCUDA* d_particles = nullptr;
    std::vector<VerletObjectCUDA> getObjectsGPU();
    void addObjectGPU(float3 position, float radius, float3 color, float3 velocity, float dt);  
    uint64_t getGPUObjectsCount() const { return gpuObjectsCount; }
    uint64_t gpuObjectsCount = 0;

	// Physics constants
    glm::vec3 gravity = { 0.0f, -1000.0f, 0.0f };
	float restitution = 0.50f;
    float reponse_coef = 0.75f;
	float attraction_coef = 0.0f;
    float repulsion_coef = 0.0f;

   // Interactions
    float bubbleRadius = 10 * objectRadius; // bubble to move objects by left mouse click
    float bubbleIntensity = 100.0f; 
};
