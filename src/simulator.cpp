#include "simulator.h"

Simulator::Simulator()
    : render(objectManager),       
    inputHandler(objectManager),
    solver(objectManager)
    
{}

Simulator::~Simulator() {}

bool CPU = false;

void Simulator::run() {

    //  CPU
    if (CPU) {
        render.createWindow("PARTICLE SIMULATOR", objectManager.screenWidth, objectManager.screenHeight, 2);
        render.initParticleBuffersCPU();
        loop();
    }
 

    //  GPU
    if (!CPU) {
        render.createWindow("PARTICLE SIMULATOR", objectManager.screenWidth, objectManager.screenHeight, 2);
        render.initParticleBuffersGPU();
        render.initCudaInterop();
        cudaInit();
        loop();
    }
}
void Simulator::loop() {

    bool running = true;

    while (running) {
        inputHandler.handleInputs( running);
        inputHandler.spawnParticlesArray( { 50,200 }, 5);

        // CPU
        if (CPU) {
            inputHandler.handleMouseResponseCPU();
            solver.updateCPU();
            render.renderUI();
            render.renderCPU();
        }


        // GPU
        if (!CPU) {
            inputHandler.handleMouseResponseGPU();
            solver.updateGPU();
            render.renderUI();
            render.renderGPU();
        }


        render.present();

        SDL_Delay(6); // bcoz the particles are spawing too fast!!!
    }
   if(!CPU) render.cleanupCudaInterop();
    render.destroy();
    

}

void Simulator::cudaInit() {
    cudaMalloc(&objectManager.d_particles, objectManager.MAX_PARTICLES * sizeof(VerletObjectCUDA));
}