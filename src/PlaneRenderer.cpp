#include "PlaneRenderer.h"
#include <GL/glew.h>
#include <vector>

/*
x-plane : -1
y-plane : 0
z_plane : 1
*/
void PlaneRenderer::initBuffers(uint8_t plane_axis) {
    // We'll use a large "infinite" extent
    float l = 10000.0f;

    std::vector<float> vertices;

    if (plane_axis == -1) {
        // X-Plane (YZ plane, x = 0)
        vertices = {
            0, -l, -l,
            0,  l, -l,
            0,  l,  l,

            0, -l, -l,
            0,  l,  l,
            0, -l,  l
        };
    }
    else if (plane_axis == 0) {
        // Y-Plane (XZ plane, y = 0)
        vertices = {
            -l, 0, -l,
             l, 0, -l,
             l, 0,  l,

            -l, 0, -l,
             l, 0,  l,
            -l, 0,  l
        };
    }
    else if (plane_axis == 1) {
        // Z-Plane (XY plane, z = 0)
        vertices = {
            -l, -l, 0,
             l, -l, 0,
             l,  l, 0,

            -l, -l, 0,
             l,  l, 0,
            -l,  l, 0
        };
    }

    vertexCount = static_cast<GLsizei>(vertices.size() / 3);

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

    glBindVertexArray(0);
}

void PlaneRenderer::render() {
    glBindVertexArray(VAO);
    glDrawArrays(GL_TRIANGLES, 0, vertexCount);
    glBindVertexArray(0);
}

int PlaneRenderer::destroy() {
    glDeleteBuffers(1, &VBO);
    glDeleteVertexArrays(1, &VAO);
    return 0;
}

