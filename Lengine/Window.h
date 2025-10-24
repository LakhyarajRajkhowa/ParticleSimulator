#pragma once
#include <SDL2/SDL.h>
#include <GL/glew.h>
#include <string>

namespace Lengine {

	enum  windowFlags { INVISIBLE = 0x1, FULLSCREEN = 0x2, BORDERLESS = 0x4 };

	class Window
	{
	public:
		Window();
		~Window();

		int create(std::string windowName, int screenWidth, int screenHeight, unsigned int currentFlags);

		void swapBuffer();
		void quitWindow();

		uint32_t getScreenWidth() { return _screenWidth; }
		uint32_t getScreenHeight() { return _screenHeight; }

		SDL_Window* getWindow() { return _sdlWindow; }

	private:
		
		SDL_Window* _sdlWindow;
		SDL_GLContext glContext;
		uint32_t _screenWidth, _screenHeight;

	};
}

