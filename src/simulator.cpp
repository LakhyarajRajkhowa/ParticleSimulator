#include "simulator.h"

Simulator::Simulator()
    : particleRenderer(objectManager),
	boxRenderer(objectManager),
	mainRenderer(objectManager, camera3D),
    inputHandler(objectManager, inputManager, camera3D),
    particleSolver(objectManager)
    
{
    Lengine::init();
     //if (!config.loadFromFile("../config.json")) {
     //   std::cout << "Using default config values.\n";
     // }
}

Simulator::~Simulator() {}


void Simulator::run() {
        InitTimer();
		mainWindow.create("PARTICLE SIMULATOR", objectManager.screenWidth, objectManager.screenHeight, 0);
        camera3D.init(objectManager.screenWidth, objectManager.screenHeight, &inputManager, glm::ivec3(objectManager.boxHeight , objectManager.boxWidth / 2, objectManager.boxDepth ), 45.0f);
        
        glClearColor(0, 0, 0, 1);

        initParticleShaders();
        initBoxShaders();
       
        particleRenderer.initParticleBuffers();
        particleRenderer.initCudaInterop();
		boxRenderer.initBuffers(objectManager.boxWidth, objectManager.boxHeight, objectManager.boxDepth);
        cudaInit();
        loop();

}
void Simulator::loop() {
    bool running = true;


    while (running) {
        

        inputManager.update();
        inputHandler.handleInputs(running);
        camera3D.update(objectManager.dt / ImGui::GetIO().Framerate);

        if(rotateCamera) camera3D.rotateCamera(glm::vec3(objectManager.boxWidth/2 , objectManager.boxHeight / 2, objectManager.boxDepth / 2 ), GetTimeSeconds() , objectManager.boxWidth * 0.75, 25.0f);
        mainRenderer.renderUI();

        // Redering particles
		spawnParticles();
        particleSolver.update();
        renderParticles();
       
		// Render box
        renderBox();

       

        mainWindow.swapBuffer();
        
    }

    particleRenderer.destroy();
	boxRenderer.destroy();
    mainRenderer.cleanupCudaInterop();
	mainRenderer.destroy();

    mainWindow.quitWindow();
    
}

void Simulator::initParticleShaders() {
    particleShader.compileShaders("../shaders/particle.vert", "../shaders/particle.frag");
    particleShader.linkShaders();
}
void Simulator::initBoxShaders() {
    boxShader.compileShaders("../shaders/boxShader.vert", "../shaders/boxShader.frag");
    boxShader.linkShaders();
}
void Simulator::initplaneShaders() {
    planeShader.compileShaders("../shaders/planeShader.vert", "../shaders/planeShader.frag");
    planeShader.linkShaders();
}
void Simulator::cudaInit() {
    cudaMalloc(&objectManager.d_particles, objectManager.MAX_PARTICLES * sizeof(VerletObjectCUDA));
}

void Simulator::renderParticles() {


    particleShader.use();
    particleShader.setMat4("uProjection", camera3D.getProjectionMatrix());
    particleShader.setMat4("uView", camera3D.getViewMatrix());

    particleRenderer.renderGPU();
    particleShader.unuse();

}

void Simulator::renderBox() {

    boxShader.use();
    boxShader.setMat4("uProjection", camera3D.getProjectionMatrix());
    boxShader.setMat4("uView", camera3D.getViewMatrix());
	boxShader.setMat4("uModel", glm::mat4(1.0f));
    boxShader.setVec4("uColor", glm::vec4(0.1f, 0.1f, 0.1f, 0.2f));
    boxRenderer.render();
    boxShader.unuse();

}

void Simulator::renderPlane() {

    planeShader.use();
    planeShader.setMat4("uProjection", camera3D.getProjectionMatrix());
    planeShader.setMat4("uView", camera3D.getViewMatrix());
    planeShader.setMat4("uModel", glm::mat4(1.0f));
    planeShader.setVec4("uColor", glm::vec4(0.9f, 0.9f, 0.9f, 0.5f));
    planeRenderer.render();
    planeShader.unuse();

}

void Simulator::spawnParticles() {

    float deltaTime = objectManager.dt / ImGui::GetIO().Framerate;
    spawnTimer += deltaTime;
    if (spawnTimer >= spawnInterval) {
        inputHandler.spawnParticlesArray({ objectManager.boxWidth * 0.90f,
            objectManager.boxHeight * 0.80f,
            objectManager.boxDepth * 0.50f},
            10,
            50.0f
        );
        spawnTimer = 0.0f;
    }
}