#version 330 core
in vec3 fragColor;
out vec4 outColor;

void main()
{
    // gl_PointCoord gives the coordinate inside the point [0,1]
    vec2 coord = gl_PointCoord - vec2(0.5);
    if (dot(coord, coord) > 0.25)
        discard; // outside circle — discard the fragment

    outColor = vec4(fragColor, 1.0);
}
