#pragma once
#include <string>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>

class Config {
public:
    int screenWidth = 1280;
    int screenHeight = 720;
    bool fullscreen = false;
    int framerateLimit = 60;
    int particleCount = 5000;
    float bubbleRadius = 5.0f;
    float bubbleIntensity = 1.0f;

    bool loadFromFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open config file: " << filename << std::endl;
            return false;
        }

        nlohmann::json j;
        file >> j;

        screenWidth = j.value("screenWidth", screenWidth);
        screenHeight = j.value("screenHeight", screenHeight);
        fullscreen = j.value("fullscreen", fullscreen);
        

        return true;
    }
};
