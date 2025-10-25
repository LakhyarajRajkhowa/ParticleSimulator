#pragma once

#include "solver.h"
#include "handleInputs.h"
#include "Render.h"
#include "ParticleRenderer.h"
#include "BoxRenderer.h"
#include "PlaneRenderer.h"
#include "Timer.h"

#include "../Lengine/Lengine.h"
#include "../Lengine/Window.h"
#include "../Lengine/GLSLProgram.h"
#include "../Lengine/Camera3d.h"
#include "../Lengine/InputManager.h"	

#include "config.h"
class Simulator
{
private:

	Lengine::Window mainWindow;
	Lengine::GLSLProgram particleShader;
	Lengine::GLSLProgram boxShader;	
	Lengine::GLSLProgram planeShader;
	Lengine::Camera3d camera3D;
	Lengine::InputManager inputManager;

	void initParticleShaders();
	void initBoxShaders();
	void initplaneShaders();
	void loop();
	void cudaInit();

	ObjectManager objectManager;
	Render mainRenderer;
	ParticleRenderer particleRenderer;
	BoxRenderer boxRenderer;
	PlaneRenderer planeRenderer;
	HandleInputs inputHandler;
	Solver particleSolver;

	Config config;

	void spawnParticles();
	void renderParticles();
	void renderBox();
	void renderPlane();

	float spawnTimer = 0.0f;
	float cameraRotationAngle = 0.0f;
	const float spawnInterval = 0.1f;


public:
	Simulator();
	~Simulator();
	void run();


};

