#include "Render.h"
#include <SDL2/SDL.h>
#include <GL/glew.h>
#include <imgui/backends/imgui_impl_sdl2.h>
#include <imgui/backends/imgui_impl_opengl3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

bool isFluid = false;

// Helper function to compile shader
GLuint compileShader(const std::string& vertPath, const std::string& fragPath) {
    auto readFile = [](const std::string& path) {
        std::ifstream file(path);
        std::stringstream ss;
        ss << file.rdbuf();
        return ss.str();
        };

    std::string vertCode = readFile(vertPath);
    std::string fragCode = readFile(fragPath);
    const char* vShaderCode = vertCode.c_str();
    const char* fShaderCode = fragCode.c_str();

    GLuint vertex = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex, 1, &vShaderCode, nullptr);
    glCompileShader(vertex);

    GLint success;
    glGetShaderiv(vertex, GL_COMPILE_STATUS, &success);
    if (!success) std::cerr << "Vertex Shader compilation failed!\n";

    GLuint fragment = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment, 1, &fShaderCode, nullptr);
    glCompileShader(fragment);
    glGetShaderiv(fragment, GL_COMPILE_STATUS, &success);
    if (!success) std::cerr << "Fragment Shader compilation failed!\n";

    GLuint program = glCreateProgram();
    glAttachShader(program, vertex);
    glAttachShader(program, fragment);
    glLinkProgram(program);

    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) std::cerr << "Shader program linking failed!\n";

    glDeleteShader(vertex);
    glDeleteShader(fragment);

    return program;
}

int Render::createWindow(std::string windowName, int screenWidth, int screenHeight, unsigned int currentFlags) {
    // Flags for SDL window 
    Uint32 flags = SDL_WINDOW_OPENGL;
    if (currentFlags & SDL_WINDOW_FULLSCREEN) {
        flags |= SDL_WINDOW_FULLSCREEN;
    }
    if (currentFlags & SDL_WINDOW_FULLSCREEN_DESKTOP) {
        flags |= SDL_WINDOW_FULLSCREEN_DESKTOP;
    }
    if (currentFlags & SDL_WINDOW_RESIZABLE) {
        flags |= SDL_WINDOW_RESIZABLE;
    }

    // Create SDL window
    _sdlWindow = SDL_CreateWindow(windowName.c_str(), SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        screenWidth, screenHeight, flags);

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

    glContext = SDL_GL_CreateContext(_sdlWindow);
    SDL_GL_SetSwapInterval(1);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW\n";
        return -1;
    }

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // ImGui setup
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplSDL2_InitForOpenGL(_sdlWindow, glContext);
    ImGui_ImplOpenGL3_Init("#version 330");

    // Shader setup
    shaderProgram = compileShader(vertexShaderPath, fragmentShaderPath);


    return 0;
}
void Render::initCudaInterop()
{
    cudaGraphicsGLRegisterBuffer(&objectManager.cudaVBOResource, particleVBO, cudaGraphicsMapFlagsWriteDiscard);
    std::cout << "[CUDA-OpenGL] Interop initialized for " << objectManager.MAX_PARTICLES << " particles.\n";
}
void Render::initParticleBuffersCPU() {
    particleData.resize(MAX_PARTICLES * 5);
    glGenVertexArrays(1, &particleVAO);
    glGenBuffers(1, &particleVBO);
    glBindVertexArray(particleVAO);
    glBindBuffer(GL_ARRAY_BUFFER, particleVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 5 * MAX_PARTICLES, nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);
}

void Render::initParticleBuffersGPU() {
   // "Particle" VAO/VBO
  particleData.resize(MAX_PARTICLES * 6);

  // During Render::create()
  glGenVertexArrays(1, &particleVAO);
  glGenBuffers(1, &particleVBO);

  glBindVertexArray(particleVAO);
  glBindBuffer(GL_ARRAY_BUFFER, particleVBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * MAX_PARTICLES, nullptr, GL_DYNAMIC_DRAW);

  // Positions
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(0);

  // Colors
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(2 * sizeof(float)));
  glEnableVertexAttribArray(1);

  // Radius
  glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(5 * sizeof(float)));
  glEnableVertexAttribArray(2);

  glBindVertexArray(0);
}


void Render::addImGuiParameter(const char* label) {
   
    ImGui::Begin(label);

 
    ImGui::Text("Spawn: ");
    ImGui::SameLine();
    if (ImGui::Button(objectManager.spawn ? "ON" : "OFF", ImVec2(60, 0))) {
        objectManager.spawn = !objectManager.spawn; // Toggle the state
    }
   
       
    ImGui::Text("Reset Gravity: ");
    ImGui::SameLine();
    if (ImGui::Button("RESET")) {
        objectManager.gravity = Vec2(0.0f, 1000.0f);
       
    }

    if (isFluid) {
        if (ImGui::SliderFloat("Restituion", &objectManager.restitution, 0.0f, 1.0f));
        if (ImGui::SliderFloat("Response Coefficient", &objectManager.reponse_coef, 0.0f, 1.0f));
        if (ImGui::SliderFloat("Atrraction Coefficient", &objectManager.attraction_coef, 0.0f, 2.0f));
        if (ImGui::SliderFloat("Repulsion Coefficient", &objectManager.repulsion_coef, 0.0f, 2.0f));
        if (ImGui::SliderFloat("Bubble Intensity", &objectManager.bubbleIntensity, 0.0f, 1000.0f));
    }
   
    ImGui::End();

}

void Render::renderUI() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplSDL2_NewFrame();
    ImGui::NewFrame();

	addImGuiParameter("Parameters");

    // FPS Display
    ImGui::SetNextWindowPos(ImVec2(1350, 10));
    ImGui::SetNextWindowBgAlpha(0.35f);
    ImGui::Begin("FPS", nullptr,
        ImGuiWindowFlags_NoDecoration |
        ImGuiWindowFlags_AlwaysAutoResize |
        ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_NoFocusOnAppearing |
        ImGuiWindowFlags_NoNav);

    ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
    ImGui::End();

	// Object count display
    ImGui::SetNextWindowPos(ImVec2(1350, 50));
    ImGui::SetNextWindowBgAlpha(0.35f);
    ImGui::Begin("Objects CPU", nullptr,
        ImGuiWindowFlags_NoDecoration |
        ImGuiWindowFlags_AlwaysAutoResize |
        ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_NoFocusOnAppearing |
        ImGuiWindowFlags_NoNav);

    ImGui::Text("Objects CPU: %d", static_cast<int>(objectManager.getConstObjectsCPU().size()));
    ImGui::End();


    ImGui::SetNextWindowPos(ImVec2(1350, 90));
    ImGui::SetNextWindowBgAlpha(0.35f);
    ImGui::Begin("Objects GPU", nullptr,
        ImGuiWindowFlags_NoDecoration |
        ImGuiWindowFlags_AlwaysAutoResize |
        ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_NoFocusOnAppearing |
        ImGuiWindowFlags_NoNav);
    ImGui::Text("Objects GPU: %d", static_cast<int>(objectManager.getGPUObjectsCount()));  
    ImGui::End();



    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);

}

void Render::renderCPU() {
   
    glUseProgram(shaderProgram);

    glm::mat4 projection = glm::ortho(0.0f, (float)objectManager.screenWidth, (float)objectManager.screenHeight, 0.0f, -1.0f, 1.0f);
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "uProjection"), 1, GL_FALSE, &projection[0][0]);

    for (size_t i = 0; i < objectManager.getConstObjectsCPU().size(); i++) {
        auto& obj = objectManager.getConstObjectsCPU()[i];
        particleData[i * 5 + 0] = obj.current_position.x;
        particleData[i * 5 + 1] = obj.current_position.y;
        particleData[i * 5 + 2] = obj.color.r / 255.0f;
        particleData[i * 5 + 3] = obj.color.g / 255.0f;
        particleData[i * 5 + 4] = obj.color.b / 255.0f;
       

    }

    glBindVertexArray(particleVAO);
    glBindBuffer(GL_ARRAY_BUFFER, particleVBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, particleData.size() * sizeof(float), particleData.data());

    glUniform3f(glGetUniformLocation(shaderProgram, "uColor"), 1.0f, 1.0f, 1.0f); // white
    glPointSize(objectManager.objectRadius * 2.0f); // increase size of particles
    glDrawArrays(GL_POINTS, 0, (objectManager.getConstObjectsCPU().size()));
    glBindVertexArray(0);
}
void Render::renderGPU()
{

    int N = objectManager.getGPUObjectsCount();
    if (N == 0) return;

    glUseProgram(shaderProgram);

    // Projection
    glm::mat4 projection = glm::ortho(0.0f, (float)objectManager.screenWidth,
        (float)objectManager.screenHeight, 0.0f, -1.0f, 1.0f);
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "uProjection"), 1, GL_FALSE, &projection[0][0]);

    glBindVertexArray(particleVAO);
    glBindBuffer(GL_ARRAY_BUFFER, particleVBO);
    glUniform3f(glGetUniformLocation(shaderProgram, "uColor"), 1.0f, 1.0f, 1.0f); // white
    glPointSize(renderSize);
    glDrawArrays(GL_POINTS, 0, N);
    glBindVertexArray(0);
}

void Render::present() {
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    SDL_GL_SwapWindow(_sdlWindow);
}

void Render::cleanupCudaInterop()
{
    if (objectManager.cudaVBOResource)
    {
        cudaGraphicsUnregisterResource(objectManager.cudaVBOResource);
        objectManager.cudaVBOResource = nullptr;
    }
}


int Render::destroy() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();

    glDeleteBuffers(1, &particleVBO);
    glDeleteVertexArrays(1, &particleVAO);
    glDeleteProgram(shaderProgram);

    SDL_GL_DeleteContext(glContext);
    SDL_DestroyWindow(_sdlWindow);
    SDL_Quit();
    return 0;
}




