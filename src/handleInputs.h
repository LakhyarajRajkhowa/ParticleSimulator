#pragma once

#include <SDL2/SDL.h>
#include <imgui/backends/imgui_impl_sdl2.h>

#include "objectManager.h"
#include "ParticleKernels.cuh"

#include "utils.h"

#include "../Lengine/InputManager.h"

class HandleInputs
{
	ObjectManager& objectManager;
public:
	HandleInputs(ObjectManager& objMgr) : objectManager(objMgr) {}

	void handleInputs( bool& running);
	bool handleMouseResponse();
	void handleKeyboardResponse(bool& running);

	void spawnParticle( const glm::vec2 screenCoords, float speed);
	void spawnParticlesArray( const glm::vec2 screenCoords, int arraySize);
	
private:
	Lengine::InputManager inputManager;
	glm::vec2& gravity = (objectManager.gravity);

};