#pragma once
#include <SDL2/SDL.h>
#include <GL/glew.h>
#include <string>

#include "objectManager.h"

#include <cuda_gl_interop.h>
#include "ParticleKernels.cuh"


#include "../Lengine/GLSLProgram.h"

extern bool isFluid;


class BoxRenderer {

public:
    BoxRenderer(ObjectManager& objMgr) : objectManager(objMgr) {}

    void initBuffers(uint16_t boxWidth, uint16_t boxHeight, uint16_t boxDepth);
    void render();
    int destroy();

private:
    ObjectManager& objectManager;
    GLuint boxVAO = 0;
    GLuint boxVBO = 0;

    std::vector<float> boxData;
	uint16_t vertexCount = 0;

};
