#include "BoxRenderer.h"
#include <GL/glew.h>

void BoxRenderer::initBuffers(uint16_t boxWidth, uint16_t boxHeight, uint16_t boxDepth) {
    float w = static_cast<float>(boxWidth);
    float h = static_cast<float>(boxHeight);
    float d = static_cast<float>(boxDepth);

    // 36 vertices (6 faces × 2 triangles × 3 vertices)
    float vertices[] = {
        // Front face (z = 0)
        0, 0, 0,   w, 0, 0,   w, h, 0,
        0, 0, 0,   w, h, 0,   0, h, 0,

        // Back face (z = d)
        0, 0, d,   w, h, d,   w, 0, d,
        0, 0, d,   0, h, d,   w, h, d,

        // Left face (x = 0)
        0, 0, 0,   0, h, 0,   0, h, d,
        0, 0, 0,   0, h, d,   0, 0, d,

        // Right face (x = w)
        w, 0, 0,   w, 0, d,   w, h, d,
        w, 0, 0,   w, h, d,   w, h, 0,

        

        // Bottom face (y = 0)
        0, 0, 0,   0, 0, d,   w, 0, d,
        0, 0, 0,   w, 0, d,   w, 0, 0
    };

    vertexCount = 36; // 12 triangles * 3 vertices

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
    glDrawArrays(GL_TRIANGLES, 0, vertexCount);
    glBindVertexArray(0);
}

int BoxRenderer::destroy() {
    glDeleteBuffers(1, &boxVBO);
    glDeleteVertexArrays(1, &boxVAO);
    return 0;
}
