#version 330 core
in vec3 fragColor;
in float radius;
out vec4 outColor;

void main()
{
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = dot(coord, coord);

    if (dist > 0.25)
        discard;

    // Simple lighting for spherical look
    float light = sqrt(1.0 - 4.0 * dist);
    outColor = vec4(fragColor * light, 1.0);
}
