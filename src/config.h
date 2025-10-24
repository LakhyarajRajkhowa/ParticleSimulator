#pragma once
#include <string>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>

class Config {
public:
    uint32_t screenWidth = 1280;
    uint32_t screenHeight = 720;
    bool fullscreen = false;
    uint16_t framerateLimit = 60;
    uint64_t particleCount = 5000;
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
