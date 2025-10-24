#pragma once
#include <SDL2/SDL.h>
#include <GL/glew.h>
#include <string>

#include "objectManager.h"

#include <cuda_gl_interop.h>
#include "ParticleKernels.cuh"


#include "../Lengine/GLSLProgram.h"
#include "../Lengine/Camera3d.h"
extern bool isFluid;


class Render {
    ObjectManager& objectManager;
public:
    Render(ObjectManager& objMgr) : objectManager(objMgr) {}
    
    void renderUI();
    void cleanupCudaInterop();
    int destroy();


private:

    GLuint particleVAO = 0;
    GLuint particleVBO = 0;

	std::string vertexShaderPath = "../shaders/particle.vert";
	std::string fragmentShaderPath = "../shaders/particle.frag";

	float renderSize = ObjectManager::objectRadius * 2.0f;
	Lengine::Camera3d camera3D;

    std::vector<float> particleData;

    void addImGuiParameter(const char* label);

	uint64_t MAX_PARTICLES = ObjectManager::MAX_PARTICLES;
};
