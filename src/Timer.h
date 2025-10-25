#pragma once
#include <SDL2/SDL.h>

inline Uint32 gStartTime = 0; // Global start time

// Call this once after SDL_Init()
inline void InitTimer() {
    gStartTime = SDL_GetTicks();
}

// Call this anywhere to get elapsed time in seconds
inline float GetTimeSeconds() {
    Uint32 current = SDL_GetTicks();
    return (current - gStartTime) / 1000.0f;
}

