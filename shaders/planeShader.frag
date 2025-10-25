#version 330 core
out vec4 FragColor;
in vec3 FragPos;
uniform vec4 uColor; // rgba

void main()
{
    FragColor = uColor;
}
