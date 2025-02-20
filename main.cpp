
#define SDL_MAIN_HANDLED


#include <SDL2/SDL.h>
#include <vector>
#include <iostream>
#include <random>
#include <cmath>
#include "Solver.h"
#include "Color.h"
#include "Timing.h"


using namespace std;

const int SCREEN_WIDTH = 1500;
const int SCREEN_HEIGHT = 750;
const int FRAME_RATE = 100;
const double PI = 2 * acos(0);

const int maxFPS = 60;

void handleInputs(bool& running, Vec2& gravity);
void SDL_RenderFillCircle(SDL_Renderer* renderer, int x_centre, int y_centre, int radius);
void fillBackground(SDL_Renderer* renderer, int x_centre, int y_centre, int radius);
static Color getRainbow(float t);
int randomNumber(int start, int end);

int main(int argc, char* argv[])
{
    if (SDL_Init(SDL_INIT_VIDEO) != 0)
    {
        cerr << "Error initializing SDL: " << SDL_GetError() << endl;
        return 1;
    }

    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 8); // 4x MSAA

    SDL_Window* window = SDL_CreateWindow(
        "2D Physics Simulation",
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        SCREEN_WIDTH,
        SCREEN_HEIGHT,
        SDL_WINDOW_SHOWN);

    if (!window)
    {
        cerr << "Error creating SDL window: " << SDL_GetError() << endl;
        SDL_Quit();
        return 1;
    }

    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer)
    {
        cerr << "Error creating SDL renderer: " << SDL_GetError() << endl;
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }
    int fps = 60;
    bool running = true;
    float dt = 1.0f / fps;
    Vec2 gravity = { 0.0f, 8.0f };
    int time = 1;
    int count = 0;
    Uint32 lastTime = SDL_GetTicks();
    Uint32 frameCount = 0;

    Solver solver;
    Color color;

    Lengine::FpsLimiter fpsLimiter;

    fpsLimiter.setMaxFPS(maxFPS);

    
    while (running)
    {
        fpsLimiter.begin();
        
       

        float t = (float)time / maxFPS;
        if (time % 1 == 0 && fps > 30 && solver.maxObject(3000))
        {
            auto& object = solver.addObject({ SCREEN_HEIGHT /10 + abs(SCREEN_WIDTH / 2 * cos(t)), SCREEN_HEIGHT / 2 }, randomNumber(4, 7), getRainbow(t));
            object.setVelocity({ 0, 100 }, dt);
        }

        handleInputs(running, gravity);
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);


        for (auto& obj : solver.getObjects())
        {

            obj.color.apply(renderer);
            SDL_RenderFillCircle(renderer, obj.current_position.x, obj.current_position.y, obj.radius);
        }

        SDL_RenderPresent(renderer);

        //SDL_Delay(1000 / FRAME_RATE);
        solver.update(SCREEN_HEIGHT, SCREEN_WIDTH, dt, gravity);
        time++;


        if (count == 60) {
            cout << "FPS: " << fps << endl;
            solver.getObjectCount();
            count = 0;
        }
        count++;
        fps = fpsLimiter.end();
    }

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}

void handleInputs(bool& running, Vec2 &gravity)
{
    SDL_Event event;
    while (SDL_PollEvent(&event))
    {
        if (event.type == SDL_QUIT)
            running = false;
        else if (event.type == SDL_KEYDOWN)
        {
            switch (event.key.keysym.sym)
            {
            case SDLK_w:
                gravity.y -= 0.1f;
                break;
            case SDLK_a:
                gravity.x -= 0.1f;
                break;
            case SDLK_s:
                gravity.y += 0.1f;
                break;
            case SDLK_d:
                gravity.x += 0.1f;
                break;
            }
        }
    }
}

void SDL_RenderFillCircle(SDL_Renderer* renderer, int centerX, int centerY, int radius)
{
    for (int y = -radius; y <= radius; ++y)
    {
        int dx = static_cast<int>(std::sqrt(radius * radius - y * y)); // Calculate x extent for the given y
        SDL_RenderDrawLine(renderer, centerX - dx, centerY + y, centerX + dx, centerY + y);
    }
}

void fillBackground(SDL_Renderer* renderer, int x_centre, int y_centre, int radius)
{
    for (int y = 0; y < SCREEN_HEIGHT; y++)
    {
        for (int x = 0; x < SCREEN_WIDTH; x++)
        {
            if ((x - x_centre) * (x - x_centre) + (y - y_centre) * (y - y_centre) >= radius * radius && (x - x_centre) * (x - x_centre) + (y - y_centre) * (y - y_centre) <= (radius + 5) * (radius + 5))
            {
                SDL_RenderDrawPoint(renderer, x, y);
            }
        }
    }
}

int randomNumber(int start, int end)
{
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> distrib(start, end);
    return distrib(gen);
}

static Color getRainbow(float t)
{
    const float r = sin(t);
    const float g = sin(t + 0.33f * 2.0f * PI);
    const float b = sin(t + 0.66f * 2.0f * PI);
    return { static_cast<uint8_t>(255.0f * r * r),
            static_cast<uint8_t>(255.0f * g * g),
            static_cast<uint8_t>(255.0f * b * b) };
}




