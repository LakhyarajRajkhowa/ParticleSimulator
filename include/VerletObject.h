#pragma once

#include "Vec2.h"
#include "Color.h"

using namespace std;

struct VerletObject
{
    Vec2 current_position;
    Vec2 old_position;
    Vec2 velocity = current_position - old_position;
    Vec2 acceleration;
    float radius;
    Color color;

    void updatePosition(float &dt)
    {
        const Vec2 velocity = current_position - old_position;
        old_position = current_position;
        current_position = current_position + velocity + acceleration * dt * dt;
        acceleration = {};
    }

    void setVelocity(Vec2 v, float dt)
    {
        old_position = current_position - (v * dt);
    }

    Vec2 getVelocity()
    {
        return current_position- old_position;
    }

    void accelerate(Vec2 a)
    {
        acceleration += a;
    }

    VerletObject(const Vec2 &position = Vec2(0, 0), float radius = 1.0f, const Color &initialColor = Color())
        : current_position(position), old_position(position), acceleration(Vec2(0, 0)), radius(radius), color(initialColor)
    {
    }
};
