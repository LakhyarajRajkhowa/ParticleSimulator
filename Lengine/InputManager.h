#pragma once

#include <unordered_map>
#include <glm/glm.hpp>

namespace Lengine {
	class InputManager
	{
	public:
		InputManager();
		~InputManager();

		void update();

		void pressKey(unsigned int keyID);
		void releaseKey(unsigned int keyID);
		void setMouseCoords(float x, float y);
		bool isKeyPressed(unsigned int keyID);
		bool isKeyDown(unsigned int keyID);
		
		void pressMouseButton(unsigned int buttonID);
		void releaseMouseButton(unsigned int buttonID);
		bool isMouseButtonDown(unsigned int buttonID);

		bool isMouseButtonPressed(unsigned int buttonID);
		bool wasMouseButtonDown(unsigned int buttonID);

		glm::vec2 getMouseCoords() const { return _mouseCoords; }


	private:
		bool wasKeyDown(unsigned int keyID);
		std::unordered_map<unsigned int, bool> _keyMap;
		std::unordered_map<unsigned int, bool> _previousKeyMap;

		std::unordered_map<unsigned int, bool> _mouseMap;
		std::unordered_map<unsigned int, bool> _previousMouseMap;

		glm::vec2 _mouseCoords;

	};
}


