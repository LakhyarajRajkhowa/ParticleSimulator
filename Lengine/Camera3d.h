#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "../Lengine/InputManager.h"

namespace Lengine {
    class Camera3d {
    public:
        Camera3d();
        ~Camera3d();
        void init(float width, float height, InputManager* _inputManager, glm::vec3 cameraPos, float FOV);
        glm::vec3& getCameraPosition() { return position; }
        glm::vec3& getCameraDirection() { return front; }

        glm::mat4 getViewMatrix();
        glm::mat4 getProjectionMatrix();
        void update(float deltaTime);
        void processMouse(float xoffset, float yoffset);
        void rotateCamera(const glm::vec3& center, float radius, float angularSpeed, float deltaTime);

    private:
        glm::vec3 position;
        glm::vec3 front;
        glm::vec3 up;
        float yaw, pitch;
        float fov;
        float aspectRatio;

        glm::vec3 direction;
        InputManager* _inputManager;

        void applyGravity();
        bool _applyGravity;
    };

}