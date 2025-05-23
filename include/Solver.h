#pragma once

#include "VerletObject.h"
#include <vector>
#include <unordered_map>

class Solver
{

private:
    vector<VerletObject> objects = {};
    const float GRID_SIZE = 15.0f;

public:
    Solver() = default;
    VerletObject &addObject(Vec2 position, float radius, Color color)
    {

         objects.emplace_back(position, radius, color);
        return objects.back();
    }
    void getObjectCount()
    {
        cout << "No. of Objects: " << objects.size() << endl;
    }
    bool maxObject(int max) {
        if (objects.size() > max) {
            return false;
        }
        return true;
    }
    void update(int SCREEN_HEIGHT, int SCREEN_WIDTH, float dt, Vec2 gravity)
    {
        const Uint32 sub_steps = 8;
        const float sub_dt = dt / (float)(sub_steps);

       
        for (Uint32 i = sub_steps; i--;)
        {
            applyGravity(dt, gravity);
           // applyConstraints(SCREEN_HEIGHT, SCREEN_WIDTH);
            applyBorder(SCREEN_HEIGHT, SCREEN_WIDTH);
            solveCollisionByGrid();
            updatePosition(dt);
        }
    }

    void updatePosition(float &dt)
    {
        for (auto &obj : objects)
        {
            obj.updatePosition(dt);
        }
    }

    // void setObjectVelocity( Vec2 v, float &dt)
    // {
    //         obj.setVelocity(v, dt);

    // }
    void applyGravity(float &dt, Vec2 gravity)
    {
        for (auto &obj : objects)
        {
            obj.accelerate(gravity);
        }
    }

    

    void applyConstraints(int SCREEN_HEIGHT, int SCREEN_WIDTH)
    {
        const Vec2 position = {(float)SCREEN_HEIGHT / 2, (float)SCREEN_WIDTH / 2};
        const float radius = (float)SCREEN_HEIGHT / 2 - 100;
        for (auto &obj : objects)
        {
            const Vec2 to_obj = obj.current_position - position;
            const float dist = to_obj.magnitude();

            if (dist > radius - obj.radius)
            {
                const Vec2 n = to_obj / dist;
                obj.current_position = position + n * (radius - obj.radius);
            }
        }
    }

    void applyConstraintsScreen(int SCREEN_HEIGHT, int SCREEN_WIDTH)
    {
        const Vec2 position = {(float)SCREEN_HEIGHT / 2, (float)SCREEN_WIDTH / 2};
        for (auto &obj : objects)
        {
            const Vec2 to_obj = obj.current_position - position;
            const Vec2 coeff = {0.1, 0.1};
            const float dist = to_obj.magnitude();

            if (dist > obj.current_position.x + obj.radius > SCREEN_WIDTH || obj.current_position.x + obj.radius< 0 || obj.current_position.y + obj.radius> SCREEN_HEIGHT || obj.current_position.y + obj.radius< 0)
            {
                const Vec2 n = to_obj / dist;
                const Vec2 N = (n * (dist - obj.radius));
                obj.current_position = position +  N;
            }

        //    if ( obj.current_position.x + obj.radius > SCREEN_WIDTH) obj.current_position.x = obj.old_position.x  ;
        //    if (obj.current_position.x + obj.radius< 0 ) obj.current_position.x  =obj.old_position.x ;
        //     if (obj.current_position.y + obj.radius> SCREEN_HEIGHT) obj.current_position.y =obj.old_position.y ;
        //     if (obj.current_position.y + obj.radius< 0 ) obj.current_position.y =obj.old_position.y ;
        }
    }

     void applyBorder(int SCREEN_HEIGHT, int SCREEN_WIDTH) {
        for (auto & obj : objects) {
            const float dampening = 0.75f;
            const Vec2 pos  = obj.current_position;
            Vec2 npos = obj.current_position;
            Vec2 vel  = obj.getVelocity();
            Vec2 dy = {vel.x * dampening, -vel.y};
            Vec2 dx = {-vel.x, vel.y * dampening};
            if (pos.x < obj.radius || pos.x + obj.radius > SCREEN_WIDTH) { // Bounce off left/right
                if (pos.x < obj.radius) npos.x = obj.radius;
                if (pos.x + obj.radius > SCREEN_WIDTH) npos.x = SCREEN_WIDTH - obj.radius;
                obj.current_position = npos;
                obj.setVelocity(dx, 1.0);
            }
            if (pos.y < obj.radius || pos.y + obj.radius > SCREEN_HEIGHT) { // Bounce off top/bottom
                if (pos.y < obj.radius) npos.y = obj.radius;
                if (pos.y + obj.radius > SCREEN_HEIGHT) npos.y = SCREEN_HEIGHT - obj.radius;
                obj.current_position = npos;
                obj.setVelocity(dy, 1.0);
            }
        }
    }

    void solveCollision()
    {
        const float response_coef = 0.75f;
        const uint64_t objects_count = objects.size();
        for (int i = 0; i < objects_count; i++)
        {
            for (int j = i+1; j < objects_count; j++)
            {
                if (i == j)
                    continue;
                else
                {
                    const Vec2 v = objects[i].current_position - objects[j].current_position;
                    const float dist2 = v.dot(v);
                    const float min_dist = objects[i].radius + objects[j].radius;
                    // Check overlapping
                    if (dist2 < min_dist * min_dist)
                    {
                        const float dist = sqrt(dist2);
                        const Vec2 n = v / dist;
                        const float mass_ratio_1 = objects[i].radius / min_dist;
                        const float mass_ratio_2 = objects[j].radius / min_dist;
                        const float delta = 0.5f * response_coef * (dist - min_dist);
                        // Update positions
                        objects[i].current_position -= n * (mass_ratio_1 * delta);
                        objects[j].current_position += n * (mass_ratio_2 * delta);
                    }
                }
            }
        }
    }

    struct GridHash
    {
        size_t operator()(const pair<int, int> &cell) const
        {
            return hash<int>()(cell.first) ^ hash<int>()(cell.second);
        }
    };

    pair<int, int> getGridCell(float x, float y)
    {
        return {static_cast<int>(floor(x / GRID_SIZE)),
                static_cast<int>(floor(y / GRID_SIZE))};
    }

    const vector<VerletObject> &getObjects() const
    {
        return objects;
    }

    unordered_map<pair<int, int>, vector<VerletObject *>, GridHash> grid;

    void solveCollisionByGrid()
    {
        grid.clear();
        const float response_coef = 0.75f;
        for (auto &obj : objects)
        {
            auto cell = getGridCell(obj.current_position.x, obj.current_position.y);
            grid[cell].push_back(&obj);
        }

        for (auto &obj : objects)
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
};
