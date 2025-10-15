#pragma once

#include "solver.h"
#include "handleInputs.h"
#include "render.h"

extern bool CPU ;


class Simulator
{
private:

	ObjectManager objectManager;
	Render render;
	HandleInputs inputHandler;
	Solver solver;



public:
	Simulator();
	~Simulator();
	void run();
	void loop();
	void cudaInit();

};

