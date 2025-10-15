#pragma once

#include "objectManager.h"
#include <unordered_map>
#include <vector>

const float GRID_SIZE = 3 * ObjectManager::objectRadius;

class GridCollisionSolver
{
private:

    struct GridHash
    {
        size_t operator()(const pair<int, int>& cell) const
        {
            return hash<int>()(cell.first) ^ hash<int>()(cell.second);
        }
    };

    pair<int, int> getGridCell(float x, float y)
    {
        return { static_cast<int>(floor(x / GRID_SIZE)),
                static_cast<int>(floor(y / GRID_SIZE)) };
    }

    unordered_map<pair<int, int>, vector<VerletObject*>, GridHash> grid;

public:

    void solveCollisionByGrid(ObjectManager& objectManager);

    void applyBorder(int SCREEN_HEIGHT, int SCREEN_WIDTH, ObjectManager& objectManager);
   
};



