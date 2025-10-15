#include "GridCollisionSolver.h"

void GridCollisionSolver::solveCollisionByGrid(ObjectManager& objectManager)
{
    grid.clear();
    const float response_coef = 0.75f;

    for (auto& obj : objectManager.getObjectsCPU())
    {
        auto cell = getGridCell(obj.current_position.x, obj.current_position.y);
        grid[cell].push_back(&obj);
    }

    for (auto& obj : objectManager.getObjectsCPU())
    {

        // Check neighboring cells for collisions
        auto cell = getGridCell(obj.current_position.x, obj.current_position.y);
        for (int dx = -1; dx <= 1; ++dx)
        {
            for (int dy = -1; dy <= 1; ++dy)
            {
                auto neighborCell = make_pair(cell.first + dx, cell.second + dy);
                if (grid.count(neighborCell))
                {
                    for (auto neighbor : grid[neighborCell])
                    {
                        if (neighbor != &obj)
                        {
                            const Vec2 v = obj.current_position - neighbor->current_position;
                            const float dist2 = v.dot(v);
                            const float min_dist = neighbor->radius + obj.radius;
                            // Check overlapping
                            if (dist2 < min_dist * min_dist)
                            {
                                const float dist = sqrt(dist2);
                                const Vec2 n = v / dist;
                                const float mass_ratio_1 = obj.radius / min_dist;
                                const float mass_ratio_2 = neighbor->radius / min_dist;
                                const float delta = 0.5f * response_coef * (dist - min_dist);
                                // Update positions
                                obj.current_position -= n * (mass_ratio_1 * delta);
                                neighbor->current_position += n * (mass_ratio_2 * delta);
                            }
                        }
                    }
                }
            }
        }
    }
}


void GridCollisionSolver::applyBorder(int SCREEN_HEIGHT, int SCREEN_WIDTH, ObjectManager& objectManager) {
    const float restitution = 0.50f; // coefficient of restitution (energy retained after bounce)

    for (auto& obj : objectManager.getObjectsCPU()) {
        Vec2 pos = obj.current_position;
        Vec2 vel = obj.getVelocity();

        // Bounce off left/right walls
        if (pos.x < obj.radius) {
            pos.x = obj.radius;
            vel.x = -vel.x * restitution; // invert and lose some energy
        }
        else if (pos.x + obj.radius > SCREEN_WIDTH) {
            pos.x = SCREEN_WIDTH - obj.radius;
            vel.x = -vel.x * restitution;
        }

        // Bounce off top/bottom walls
        if (pos.y < obj.radius) {
            pos.y = obj.radius;
            vel.y = -vel.y * restitution;
        }
        else if (pos.y + obj.radius > SCREEN_HEIGHT) {
            pos.y = SCREEN_HEIGHT - obj.radius;
            vel.y = -vel.y * restitution;
        }

        obj.current_position = pos;
        obj.setVelocity(vel, 1.0);
    }
}
