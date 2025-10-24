#version 330 core
layout (location = 0) in vec3 aPos;     // 3D world position
layout (location = 1) in vec3 aColor;
layout (location = 2) in float aRadius;

uniform mat4 uProjection;
uniform mat4 uView; // camera view matrix

out vec3 fragColor;
out float radius;

void main()
{
    // Transform particle position into camera space
    gl_Position = uProjection * uView * vec4(aPos, 1.0);

    // Set size in pixels
    gl_PointSize = aRadius * 2.0; // diameter in pixels

    fragColor = aColor;
    radius = aRadius;
}
