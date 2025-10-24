#include "simulator.h"

Simulator::Simulator()
    : renderer(objectManager),       
    inputHandler(objectManager),
    solver(objectManager)
    
{
    Lengine::init();
     //if (!config.loadFromFile("../config.json")) {
     //   std::cout << "Using default config values.\n";
     // }
}

Simulator::~Simulator() {}


void Simulator::run() {
		mainWindow.create("PARTICLE SIMULATOR", objectManager.screenWidth, objectManager.screenHeight, 0);
        camera3D.init(objectManager.screenWidth, objectManager.screenHeight, &_inputManager, glm::ivec3(100, 50, 100), 45.0f);
        initShaders();
        renderer.initParticleBuffers();
        renderer.initCudaInterop();
        cudaInit();
        loop();

}
void Simulator::loop() {

    bool running = true;

    while (running) {
        inputHandler.handleInputs( running);
        inputHandler.spawnParticlesArray( { 50,200 }, 5);
        solver.updateGPU();
	    camera3D.update(objectManager.dt / ImGui::GetIO().Framerate);
        renderer.renderUI();

	    textureProgram.use();
        glm::mat4 projection = glm::ortho(0.0f, (float)objectManager.screenWidth,
            (float)objectManager.screenHeight, 0.0f, -1.0f, 1.0f);
        textureProgram.setMat4("uProjection", projection);
           
        renderer.renderGPU();
        renderer.present();
	    textureProgram.unuse();
	    mainWindow.swapBuffer();
        SDL_Delay(12); // bcoz the particles are spawing too fast!!!
    }
    renderer.cleanupCudaInterop();
	mainWindow.quitWindow();
    renderer.destroy();
    
}

void Simulator::initShaders() {
    textureProgram.compileShaders("../shaders/particle.vert", "../shaders/particle.frag");
    textureProgram.linkShaders();
}
void Simulator::cudaInit() {
    cudaMalloc(&objectManager.d_particles, objectManager.MAX_PARTICLES * sizeof(VerletObjectCUDA));
}