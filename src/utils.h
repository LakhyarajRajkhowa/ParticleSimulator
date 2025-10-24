#pragma once

#include <random>
#include <cmath>

#include "Color.h"

#define PI  3.1415926535897932384626433832795028841971693993751058209749445923078164062



static inline Color getRainbow(float t)
{

    const float r = sin(t);
    const float g = sin(t + 0.33f * 2.0f * PI);
    const float b = sin(t + 0.66f * 2.0f * PI);
    return { static_cast<uint8_t>(255.0f * r * r),
            static_cast<uint8_t>(255.0f * g * g),
            static_cast<uint8_t>(255.0f * b * b) };
}

float3 inline normalizeColor(uint8_t r, uint8_t g, uint8_t b) {
    return { r / 255.0f, g / 255.0f, b / 255.0f };
}