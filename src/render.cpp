#include "Render.h"
#include <GL/glew.h>
#include <imgui/backends/imgui_impl_sdl2.h>
#include <imgui/backends/imgui_impl_opengl3.h>





void Render::addImGuiParameter(const char* label) {
   
    ImGui::Begin(label);

 
    ImGui::Text("Spawn: ");
    ImGui::SameLine();
    if (ImGui::Button(spawn ? "ON" : "OFF", ImVec2(60, 0))) {
        spawn = !spawn; // Toggle the state
    }
   
       
    ImGui::Text("Reset Gravity: ");
    ImGui::SameLine();
    if (ImGui::Button("RESET")) {
        objectManager.gravity = glm::vec3(0.0f, -1000.0f, 0.0f);
       
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

    ImGui::SetNextWindowPos(ImVec2(1350, 50));
    ImGui::SetNextWindowBgAlpha(0.35f);
    ImGui::Begin("Objects GPU", nullptr,
        ImGuiWindowFlags_NoDecoration |
        ImGuiWindowFlags_AlwaysAutoResize |
        ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_NoFocusOnAppearing |
        ImGuiWindowFlags_NoNav);
    ImGui::Text("Objects GPU: %d", static_cast<int>(objectManager.getGPUObjectsCount()));  
    ImGui::End();

    ImGui::SetNextWindowPos(ImVec2(1050, 90));
    ImGui::SetNextWindowBgAlpha(0.35f);
    ImGui::Begin("Camera Position", nullptr,
        ImGuiWindowFlags_NoDecoration |
        ImGuiWindowFlags_AlwaysAutoResize |
        ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_NoFocusOnAppearing |
        ImGuiWindowFlags_NoNav);
    ImGui::Text("Camera Position: (%.00001f, %.00001f, %.00001f)", (camera3D.getCameraPosition().x),
       (camera3D.getCameraPosition().y),
        (camera3D.getCameraPosition().z));
    ImGui::End();

    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);

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
    return 0;
}




