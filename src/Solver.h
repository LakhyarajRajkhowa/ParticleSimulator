#pragma once

#include "objectManager.h"

#include "ParticleKernels.cuh"

class Solver
{
    ObjectManager& objectManager;
private:
    const Uint32 sub_steps = 8;
    glm::vec2& gravity = objectManager.gravity;
    float dt = objectManager.dt;
    float dt_sub = dt / sub_steps;
    float SCREEN_WIDTH = objectManager.screenWidth;
    float SCREEN_HEIGHT = objectManager.screenHeight;
   
public:
    Solver(ObjectManager& objMgr) : objectManager(objMgr) {}

    void updateGPU();


    void updatePositionGPU();


  
};

