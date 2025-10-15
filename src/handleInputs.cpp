#include "handleInputs.h"



bool HandleInputs::handleMouseResponseGPU() {
    int mx, my;
    Uint32 mouseState = SDL_GetMouseState(&mx, &my);
    static  float t = 0.0f;

	Vec2 mouseCoords = { (float)mx, (float)my };
    if (mouseState & SDL_BUTTON(SDL_BUTTON_LEFT))
    {
      


        if (objectManager.d_particles == nullptr || objectManager.getGPUObjectsCount() <= 0) return false;
        solveInteraction(objectManager.d_particles,
            objectManager.getGPUObjectsCount(),
            make_float2(float(mx), float(my)),
            objectManager.bubbleRadius,
            objectManager.bubbleIntensity,
            objectManager.dt/ImGui::GetIO().Framerate);

        return true;
    }
    return false;
}
bool HandleInputs::handleMouseResponseCPU() {
    int mx, my;
    Uint32 mouseState = SDL_GetMouseState(&mx, &my);
    static  float t = 0.0f;
    
    if (mouseState & SDL_BUTTON(SDL_BUTTON_LEFT))
    {
        t += 0.1f;
		objectManager.addObjectCPU({ (float)mx, (float)my }, 
            objectManager.objectRadius,
            getRainbow(t));
        objectManager.getObjectsCPU().back().setVelocity({ 0, 200 }, objectManager.dt/ ImGui::GetIO().Framerate);
        return true;
    }


    return false;
}

void HandleInputs::handleKeyboardResponse(SDL_Event& e, bool& running) {
   
	switch (e.type) {
	case SDL_QUIT:
		running = false;
		break;
	case SDL_KEYDOWN:
        switch (e.key.keysym.sym)
        {
        case SDLK_ESCAPE:
            running = false;
            break;
        case SDLK_w:
            gravity.y -= 20.0f;
            break;
        case SDLK_a:
            gravity.x -= 20.0f;
            break;
        case SDLK_s:
            gravity.y += 20.0f;
            break;
        case SDLK_d:
            gravity.x += 20.0f;
            break;
        }
	}
}

void HandleInputs::handleInputs( bool& running) {
    
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
        ImGui_ImplSDL2_ProcessEvent(&e);
        handleKeyboardResponse(e, running);
    }

}
void HandleInputs::spawnParticle( const Vec2 screenCoords, float speed) {
    float angle = atan(3 / 4);
    if (objectManager.spawn) {
        static float t = 0.0f;
        t += 0.1f;

        float2 pos = make_float2(screenCoords.x, screenCoords.y);
        float2 vel = make_float2(speed * acos(angle), -(speed * asin(angle))); // shoot particles at arctan(3/4) degress
        float3 color = addRainbowColor(t);  
        float radius = ObjectManager::objectRadius;

        objectManager.addObjectGPU(pos, radius, color, vel, objectManager.dt/ ImGui::GetIO().Framerate);
    }
}

void HandleInputs::spawnParticlesArray( const Vec2 screenCoords, int arraySize) {
    for(int i = 0; i < arraySize; i++)
		spawnParticle( { screenCoords.x , screenCoords.y + i * (ObjectManager::objectRadius * 2) }, 100);
}
