#pragma once
#include <cstdint>
#include <GL/glew.h>

namespace Lengine {

    struct  GLTexture
    {
        GLuint id;
        uint32_t width;
        uint32_t height;
    };
}
