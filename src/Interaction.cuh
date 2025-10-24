#pragma once

#include "VerletObjectCUDA.cuh"
#include "ObjectManager.h"


__global__ void moveObjects(VerletObjectCUDA* particles, int N, float2 mouseCoords, float radius, float intensity, float dt);
		
