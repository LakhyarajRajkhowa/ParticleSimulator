#include "handleInputs.h"
#include "../Lengine/InputManager.h"
#include <SDL2/SDL.h>
#include <glm/glm.hpp>
#include <imgui.h>

bool HandleInputs::handleMouseResponse() {
    glm::vec2 mouseCoords = inputManager.getMouseCoords();  // Use InputManager
    if (inputManager.isMouseButtonDown(SDL_BUTTON_LEFT)) {
     
        if (objectManager.d_particles == nullptr || objectManager.getGPUObjectsCount() <= 0)
            return false;

        solveInteraction(
            objectManager.d_particles,
            objectManager.getGPUObjectsCount(),
            make_float2(mouseCoords.x, mouseCoords.y),
            objectManager.bubbleRadius,
            objectManager.bubbleIntensity,
            objectManager.dt / ImGui::GetIO().Framerate
        );

        return true;
    }
    return false;
}

void HandleInputs::handleKeyboardResponse(bool& running) {
    if (inputManager.isKeyPressed(SDLK_ESCAPE)) {
        running = false;
    }

    if (inputManager.isKeyDown(SDLK_w)) gravity.y -= 20.0f;
    if (inputManager.isKeyDown(SDLK_a)) gravity.x -= 20.0f;
    if (inputManager.isKeyDown(SDLK_s)) gravity.y += 20.0f;
    if (inputManager.isKeyDown(SDLK_d)) gravity.x += 20.0f;
}

void HandleInputs::handleInputs(bool& running) {
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
        ImGui_ImplSDL2_ProcessEvent(&e);

        // Update InputManager
        if (e.type == SDL_KEYDOWN) inputManager.pressKey(e.key.keysym.sym);
        if (e.type == SDL_KEYUP)   inputManager.releaseKey(e.key.keysym.sym);

        if (e.type == SDL_MOUSEMOTION) {
            inputManager.setMouseCoords((float)e.motion.x, (float)e.motion.y);
        }
        if (e.type == SDL_MOUSEBUTTONDOWN)
            inputManager.pressMouseButton(e.button.button);

        if (e.type == SDL_MOUSEBUTTONUP)
            inputManager.releaseMouseButton(e.button.button);

        if (e.type == SDL_MOUSEMOTION)
            inputManager.setMouseCoords((float)e.motion.x, (float)e.motion.y);
        if (e.type == SDL_QUIT) running = false;
    }

    inputManager.update();  // Call once per frame to update previous key states
    handleMouseResponse();
    handleKeyboardResponse(running);
}

void HandleInputs::spawnParticle(const glm::vec2 screenCoords, float speed) {
    if (!objectManager.spawn) return;

    static float t = 0.0f;
    t += 0.1f;

    float angle = atan(3.0f / 4.0f);

    float2 pos = make_float2(screenCoords.x, screenCoords.y);
    float2 vel = make_float2(speed * cos(angle), -speed * sin(angle));
    float3 color = addRainbowColor(t);
    float radius = ObjectManager::objectRadius;

    objectManager.addObjectGPU(pos, radius, color, vel, objectManager.dt / ImGui::GetIO().Framerate);
}

void HandleInputs::spawnParticlesArray(const glm::vec2 screenCoords, int arraySize) {
    for (int i = 0; i < arraySize; i++) {
        spawnParticle({ screenCoords.x, screenCoords.y + i * (ObjectManager::objectRadius * 2) }, 100);
    }
}
