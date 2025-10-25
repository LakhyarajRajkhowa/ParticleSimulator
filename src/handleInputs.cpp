#include "handleInputs.h"
#include "../Lengine/InputManager.h"
#include <SDL2/SDL.h>
#include <glm/glm.hpp>
#include <imgui.h>

bool spawn = false;
bool rotateCamera = false;
void HandleInputs::handleMouseResponse() {
    glm::vec2 mouseCoords = inputManager.getMouseCoords();  // Use InputManager
    if (inputManager.isMouseButtonDown(SDL_BUTTON_LEFT)) {
     
        if (objectManager.d_particles == nullptr || objectManager.getGPUObjectsCount() <= 0)
            return;

        solveInteraction(
            objectManager.d_particles,
            objectManager.getGPUObjectsCount(),
            make_float2(mouseCoords.x, mouseCoords.y),
            objectManager.bubbleRadius,
            objectManager.bubbleIntensity,
            objectManager.dt / ImGui::GetIO().Framerate
        ); 
    }

    int mouseX, mouseY;
    SDL_GetRelativeMouseState(&mouseX, &mouseY);
    camera3D.processMouse((float)mouseX, (float)mouseY);
    SDL_SetRelativeMouseMode(SDL_TRUE);
    
   
}

void HandleInputs::handleKeyboardResponse(bool& running) {
    for (SDL_Keycode key : {SDLK_ESCAPE, SDLK_RETURN, SDLK_UP, SDLK_DOWN, SDLK_LEFT, SDLK_RIGHT}) {
        if (inputManager.isKeyDown(key)) {
            switch (key) {
            case SDLK_UP:
				objectManager.gravity.y += 10.0f;
                break;
            case SDLK_DOWN:
				objectManager.gravity.y -= 10.0f;
                break;
            case SDLK_LEFT:
				objectManager.gravity.x -= 10.0f;
                break;
            case SDLK_RIGHT:
				objectManager.gravity.x += 10.0f;
                break;
            }
        }
    }

    if (inputManager.isKeyPressed(SDLK_ESCAPE)) 
    {
        running = false; 
    } 
    if (inputManager.isKeyPressed(SDLK_RETURN))
    { 
        spawn = !spawn;
    }
    if (inputManager.isKeyPressed(SDLK_r))
    {
        rotateCamera = !rotateCamera;
    }
}


void HandleInputs::handleInputs(bool& running) {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        ImGui_ImplSDL2_ProcessEvent(&event);

        switch (event.type)
        {
        case SDL_QUIT:
            running = false;
            break;
        case SDL_KEYDOWN:
            inputManager.pressKey(event.key.keysym.sym);
            
            break;
        case SDL_KEYUP:
            inputManager.releaseKey(event.key.keysym.sym);
            break;
        case SDL_MOUSEBUTTONDOWN:
            inputManager.pressKey(event.button.button);
            break;
        case SDL_MOUSEBUTTONUP:
            inputManager.releaseKey(event.button.button);
            break;
        case SDL_MOUSEMOTION:
            inputManager.setMouseCoords(event.motion.x, event.motion.y);
            break;
        
        }

    }

    handleMouseResponse();
    handleKeyboardResponse(running);
}

void HandleInputs::spawnParticle(const glm::vec3& spawnCoords, float speed) {
    if (!spawn) return;

    static float t = 0.0f;
    t += 0.1f; // for rainbow color cycling

    // Example: launch angle for X-Y plane, random Z direction
    float angleXY = atan(3.0f / 4.0f); // you can randomize this if you want
    float angleZ = 1.0f;               // straight in Z, modify if needed

    float3 pos = make_float3(spawnCoords.x, spawnCoords.y, spawnCoords.z);
    float3 vel = make_float3(
        -speed * cos(angleXY) * cos(angleZ),
        speed * sin(angleXY) * cos(angleZ),
        -speed * sin(angleZ)
    );

    float3 color = addRainbowColor(t); // your existing color function
    float radius = ObjectManager::objectRadius;

    // Add the particle to GPU memory
    objectManager.addObjectGPU(pos, radius, color, vel, objectManager.dt / ImGui::GetIO().Framerate);
}

void HandleInputs::spawnParticlesArray(const glm::vec3& spawnCoords, int arraySize, float speed) {
    for (int i = 0; i < arraySize; i++) {
        // Slight offset in Y and Z for visual separation
        glm::vec3 offset = glm::vec3(0.0f, i * ObjectManager::objectRadius * 1.5f, i * ObjectManager::objectRadius * 1.5f);
        spawnParticle(spawnCoords + offset, speed);
    }
}