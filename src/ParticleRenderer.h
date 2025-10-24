#pragma once
#include <SDL2/SDL.h>
#include <GL/glew.h>
#include <string>

#include "objectManager.h"

#include <cuda_gl_interop.h>
#include "ParticleKernels.cuh"


#include "../Lengine/GLSLProgram.h"

extern bool isFluid;


class ParticleRenderer {

public:
    ParticleRenderer(ObjectManager& objMgr) : objectManager(objMgr) {}

    void initCudaInterop();
    void initParticleBuffers();
    void renderGPU();
    int destroy();

private:
    ObjectManager& objectManager;
    GLuint particleVAO = 0;
    GLuint particleVBO = 0;


    std::string vertexShaderPath = "../shaders/particle.vert";
    std::string fragmentShaderPath = "../shaders/particle.frag";

    uint64_t MAX_PARTICLES = ObjectManager::MAX_PARTICLES;

    float renderSize = ObjectManager::objectRadius * 2.0f;

    std::vector<float> particleData;

};
