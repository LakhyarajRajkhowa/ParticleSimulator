#pragma once

#include <SDL2/SDL.h>
#include <imgui/backends/imgui_impl_sdl2.h>

#include "objectManager.h"
#include "ParticleKernels.cuh"

#include "utils.h"

class HandleInputs
{
	ObjectManager& objectManager;
public:
	HandleInputs(ObjectManager& objMgr) : objectManager(objMgr) {}

	void handleInputs( bool& running);
	bool handleMouseResponseCPU();
	bool handleMouseResponseGPU();

	void handleKeyboardResponse(SDL_Event& e, bool& running);

	void handleMouseDraggingLeft( int& selectedParticle, bool& dragging);
	void handleMouseDraggingRight( bool& rightMouseDown);

	void spawnParticle( const Vec2 screenCoords, float speed);
	void spawnParticlesArray( const Vec2 screenCoords, int arraySize);
	
private:
	Vec2& gravity = (objectManager.gravity);

};