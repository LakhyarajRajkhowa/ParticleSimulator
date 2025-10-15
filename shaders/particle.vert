#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in float aRadius;

uniform mat4 uProjection;

out vec3 fragColor;

void main()
{
    gl_Position = uProjection * vec4(aPos, 0.0, 1.0);
    gl_PointSize = aRadius * 2.0;   // diameter
    fragColor = aColor;
}
