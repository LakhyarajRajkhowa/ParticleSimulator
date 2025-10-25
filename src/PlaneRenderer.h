#pragma once
#include <GL/glew.h>
#include <cstdint>

class PlaneRenderer {
public:
    PlaneRenderer() : VAO(0), VBO(0), vertexCount(0) {}

    void initBuffers(uint8_t plane_axis);
    void render();
    int destroy();

private:
    GLuint VAO, VBO;
    GLsizei vertexCount;
};
