#pragma once

#include <random>
#include <cmath>

#include "Color.h"

const double PI = 2 * acos(0);

int inline randomNumber(int start, int end)
{
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> distrib(start, end);
    return distrib(gen);
}

static inline Color getRainbow(float t)
{

    const float r = sin(t);
    const float g = sin(t + 0.33f * 2.0f * PI);
    const float b = sin(t + 0.66f * 2.0f * PI);
    return { static_cast<uint8_t>(255.0f * r * r),
            static_cast<uint8_t>(255.0f * g * g),
            static_cast<uint8_t>(255.0f * b * b) };
}

float3 inline normalizeColor(int r, int g, int b) {
    return { r / 255.0f, g / 255.0f, b / 255.0f };
}