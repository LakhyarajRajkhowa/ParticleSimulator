#pragma once

#include <SDL2/SDL.h>

struct Color
{
    Uint8 r, g, b; // Red, Green, Blue, Alpha components

    // Constructor
    Color(Uint8 red = 255, Uint8 green = 255, Uint8 blue = 255)
        : r(red), g(green), b(blue) {}

     void apply(SDL_Renderer *renderer) const
    {
        SDL_SetRenderDrawColor(renderer, r, g, b, 255);
    }
   
};
