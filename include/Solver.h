#pragma once

#include "GridCollisionSolver.h"
#include "objectManager.h"

#include "ParticleKernels.cuh"

class Solver
{
    ObjectManager& objectManager;
private:
	GridCollisionSolver collisionHandler;
    const Uint32 sub_steps = 8;
    Vec2& gravity = objectManager.gravity;
    float dt = objectManager.dt;
    float dt_sub = dt / sub_steps;
    float SCREEN_WIDTH = objectManager.screenWidth;
    float SCREEN_HEIGHT = objectManager.screenHeight;
   
public:
    Solver(ObjectManager& objMgr) : objectManager(objMgr) {}

    void updateCPU();
    void updateGPU();

    void updatePositionCPU();
    void updatePositionGPU();

    void applyGravityCPU();

  
};

