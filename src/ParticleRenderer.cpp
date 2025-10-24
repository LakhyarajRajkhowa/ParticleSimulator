#include "ParticleRenderer.h"
#include <GL/glew.h>
#include <vector>
#include <iostream>



void ParticleRenderer::initCudaInterop()
{
    cudaGraphicsGLRegisterBuffer(&objectManager.cudaVBOResource, particleVBO, cudaGraphicsMapFlagsWriteDiscard);
}

void ParticleRenderer::initParticleBuffers() {
    particleData.resize(MAX_PARTICLES * 7);

    glGenVertexArrays(1, &particleVAO);
    glGenBuffers(1, &particleVBO);

    glBindVertexArray(particleVAO);
    glBindBuffer(GL_ARRAY_BUFFER, particleVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 7 * MAX_PARTICLES, nullptr, GL_DYNAMIC_DRAW);

    // Positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // Radius
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    glBindVertexArray(0);
}


void ParticleRenderer::renderGPU()
{

    uint64_t N = objectManager.getGPUObjectsCount();
    if (N == 0) return;

    glBindVertexArray(particleVAO);
    glBindBuffer(GL_ARRAY_BUFFER, particleVBO);
    glPointSize(renderSize);
    glDrawArrays(GL_POINTS, 0, N);
    glBindVertexArray(0);
}



int ParticleRenderer::destroy() {
    glDeleteBuffers(1, &particleVBO);
    glDeleteVertexArrays(1, &particleVAO);
    return 0;
}

