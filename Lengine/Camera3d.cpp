#include "../Lengine/Camera3d.h"
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <SDL2/SDL.h>

namespace Lengine {
    float _speed = 0.1f;
    Camera3d::Camera3d() {
        _applyGravity = false;
    }
    Camera3d::~Camera3d() {}

    void Camera3d::init(float width, float height, InputManager* inputManager, glm::vec3 cameraPos, float FOV) {
        position = cameraPos;
        yaw = -90.0f;
        pitch = 0.0f;
        fov = FOV;
        aspectRatio = width / height;
        _inputManager = inputManager;

        direction.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        direction.y = sin(glm::radians(pitch));
        direction.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
        front = glm::normalize(direction);

        up = glm::vec3(0.0f, 1.0f, 0.0f);
    }

    
    glm::mat4 Camera3d::getViewMatrix() {
        return glm::lookAt(position, position + front, up);
    }

    glm::mat4 Camera3d::getProjectionMatrix() {
        return glm::perspective(glm::radians(fov), aspectRatio, 0.50f, 1000.0f);
    }

   
    void Camera3d::update(float deltaTime ) {
        const float speed = 0.1f * deltaTime;

           
        for (SDL_Keycode key : { SDLK_w, SDLK_s, SDLK_a, SDLK_d, SDLK_SPACE, SDLK_LSHIFT, SDLK_g, SDLK_h , SDLK_UP}) {
            if (_inputManager->isKeyDown(key)) {
                switch (key) {
                case SDLK_UP:
                    position += glm::normalize(front) * speed * 10.0f;
                    break;
                case SDLK_w:
                    position += glm::normalize(front) * speed ;
                    break;
                case SDLK_s:
                    position -= glm::normalize(front) * speed;
                    break;
                case SDLK_a:
                    position -= glm::normalize(glm::cross(front, up)) * speed;
                    break;
                case SDLK_d:
                    position += glm::normalize(glm::cross(front, up)) * speed;
                    break;
                case SDLK_SPACE:
                    position += glm::normalize(up) * speed;
                    break;
                case SDLK_LSHIFT:
                    position -= glm::normalize(up) * speed;
                    break;
                }
            }
            if (_inputManager->isKeyPressed(key)) {
                switch (key) {
                case SDLK_g:
                    _applyGravity = true;
                    break;
                case SDLK_h:
                    _applyGravity = false;
                    break;
                }
            }
        }
    }

    void Camera3d::processMouse(float xoffset, float yoffset) {
        float sensitivity = 0.1f;
        xoffset *= sensitivity;
        yoffset *= sensitivity;

        yaw += xoffset;
        pitch -= yoffset;
        if (pitch > 89.0f) pitch = 89.0f;
        if (pitch < -89.0f) pitch = -89.0f;


        direction.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        direction.y = sin(glm::radians(pitch));
        direction.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
        front = glm::normalize(direction);
      
    }
    
    void Camera3d::applyGravity() {
        _speed += 0.01;
        position.y -= _speed;
    }

}
