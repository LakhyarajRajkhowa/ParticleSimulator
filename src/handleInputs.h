#pragma once

#include <SDL2/SDL.h>
#include <imgui/backends/imgui_impl_sdl2.h>

#include "objectManager.h"
#include "ParticleKernels.cuh"

#include "utils.h"

#include "../Lengine/InputManager.h"
#include "../Lengine/Camera3d.h"

class HandleInputs
{

public:
	HandleInputs(ObjectManager& objMgr, Lengine::InputManager& inputMgr, Lengine::Camera3d& cam3d)
		: objectManager(objMgr), inputManager(inputMgr), camera3D(cam3d) {
	}
	void handleInputs( bool& running);
	void handleMouseResponse();
	void handleKeyboardResponse(bool& running);

	void spawnParticle(const glm::vec3& spawnCoords, float speed);
	void spawnParticlesArray(const glm::vec3& spawnCoords, int arraySize, float speed);
private:
	ObjectManager& objectManager;
	Lengine::InputManager& inputManager;
	Lengine::Camera3d& camera3D;
	glm::vec3& gravity = (objectManager.gravity);

};