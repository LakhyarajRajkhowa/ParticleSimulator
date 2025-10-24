#pragma once

#include "solver.h"
#include "handleInputs.h"
#include "render.h"

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
	Lengine::GLSLProgram textureProgram;
	Lengine::Camera3d camera3D;
	Lengine::InputManager _inputManager;

	void initShaders();
	void loop();
	void cudaInit();

	ObjectManager objectManager;
	Render renderer;
	HandleInputs inputHandler;
	Solver solver;

	Config config;


public:
	Simulator();
	~Simulator();
	void run();


};

