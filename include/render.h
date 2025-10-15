#pragma once
#include <SDL2/SDL.h>
#include <GL/glew.h>
#include <string>

#include "objectManager.h"

#include <cuda_gl_interop.h>
#include "ParticleKernels.cuh"

extern bool isFluid;


class Render {
    ObjectManager& objectManager;
public:
    Render(ObjectManager& objMgr) : objectManager(objMgr) {}

    int createWindow(std::string windowName, int screenWidth, int screenHeight, unsigned int currentFlags);
    void initCudaInterop();

    void initParticleBuffersCPU();
    void initParticleBuffersGPU();

    void renderUI();

    void renderCPU();
    void renderGPU();

    void present();

    void cleanupCudaInterop();
    int destroy();


private:

	std::string vertexShaderPath = "../shaders/particle.vert";
	std::string fragmentShaderPath = "../shaders/particle.frag";

    SDL_Window* _sdlWindow = nullptr;
    SDL_GLContext glContext;

    GLuint shaderProgram;

	float renderSize = ObjectManager::objectRadius * 2.0f;

    GLuint particleVAO = 0;
    GLuint particleVBO = 0;

    std::vector<float> particleData;

    void addImGuiParameter(const char* label);

	int MAX_PARTICLES = ObjectManager::MAX_PARTICLES;
};
