#ifndef VEC2_H
#define VEC2_H

#include <cmath>
#include <ostream>

class Vec2
{
public:
    float x, y;

    // Constructors
    Vec2() : x(0), y(0) {}
    Vec2(float x, float y) : x(x), y(y) {}

    // Addition
    Vec2 operator+(const Vec2 &other) const
    {
        return Vec2(x + other.x, y + other.y);
    }

    Vec2& operator+=(const Vec2& other) {
        x += other.x;
        y += other.y;
        return *this;
    }

    // Subtraction
    Vec2 operator-(const Vec2 &other) const
    {
        return Vec2(x - other.x, y - other.y);
    }

    Vec2& operator-=(const Vec2& other) {
        x -= other.x;
        y -= other.y;
        return *this;
    }

    // Scalar multiplication
    Vec2 operator*(float scalar) const
    {
        return Vec2(x * scalar, y * scalar);
    }

    // Scalar division
    Vec2 operator/(float scalar) const
    {
        return Vec2(x / scalar, y / scalar);
    }

    // Dot product
    float dot(const Vec2 &other) const
    {
        return x * other.x + y * other.y;
    }

    // Magnitude
    float magnitude() const
    {
        return std::sqrt(x * x + y * y);
    }

    // Normalize
    Vec2 normalized() const
    {
        float mag = magnitude();
        if (mag == 0)
            return Vec2(0, 0); // Return zero vector if magnitude is zero
        return *this / mag;
    }

    // Rotate by angle (in radians)
    Vec2 rotated(float angle) const
    {
        float cosTheta = std::cos(angle);
        float sinTheta = std::sin(angle);
        return Vec2(x * cosTheta - y * sinTheta, x * sinTheta + y * cosTheta);
    }

    // Stream output (for debugging)
    friend std::ostream &operator<<(std::ostream &os, const Vec2 &vec)
    {
        os << "(" << vec.x << ", " << vec.y << ")";
        return os;
    }
};

#endif // VEC2_H
