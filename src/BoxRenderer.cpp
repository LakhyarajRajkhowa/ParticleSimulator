#include "BoxRenderer.h"
#include <GL/glew.h>

void BoxRenderer::initBuffers(uint16_t boxWidth, uint16_t boxHeight, uint16_t boxDepth) {
    // 8 corner vertices of the cuboid
    float vertices[] = {
        // front face (z = 0)
        0,0,0,            boxWidth,0,0,
        boxWidth,0,0,     boxWidth,boxHeight,0,
        boxWidth,boxHeight,0, 0,boxHeight,0,
        0,boxHeight,0,    0,0,0,

        // back face (z = boxDepth)
        0,0,boxDepth,        boxWidth,0,boxDepth,
        boxWidth,0,boxDepth, boxWidth,boxHeight,boxDepth,
        boxWidth,boxHeight,boxDepth, 0,boxHeight,boxDepth,
        0,boxHeight,boxDepth, 0,0,boxDepth,

        // connecting edges
        0,0,0,       0,0,boxDepth,
        boxWidth,0,0,   boxWidth,0,boxDepth,
        boxWidth,boxHeight,0, boxWidth,boxHeight,boxDepth,
        0,boxHeight,0, 0,boxHeight,boxDepth
    };

    vertexCount = 24; // 12 edges * 2 vertices each

    glGenVertexArrays(1, &boxVAO);
    glGenBuffers(1, &boxVBO);

    glBindVertexArray(boxVAO);
    glBindBuffer(GL_ARRAY_BUFFER, boxVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

    glBindVertexArray(0);
}

void BoxRenderer::render() {
    glBindVertexArray(boxVAO);
    glDrawArrays(GL_LINES, 0, vertexCount);
    glBindVertexArray(0);
}

int BoxRenderer::destroy() {
    glDeleteBuffers(1, &boxVBO);
    glDeleteVertexArrays(1, &boxVAO);
    return 0;
}
