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

		void pressKey(uint32_t keyID);
		void releaseKey(uint32_t keyID);
		void setMouseCoords(float x, float y);
		bool isKeyPressed(uint32_t keyID);
		bool isKeyDown(uint32_t keyID);
		
		void pressMouseButton(uint8_t buttonID);
		void releaseMouseButton(uint8_t buttonID);
		bool isMouseButtonDown(uint8_t buttonID);

		bool isMouseButtonPressed(uint8_t buttonID);
		bool wasMouseButtonDown(uint8_t buttonID);

		glm::vec2 getMouseCoords() const { return _mouseCoords; }


	private:
		bool wasKeyDown(uint32_t keyID);
		std::unordered_map<uint32_t, bool> _keyMap;
		std::unordered_map<uint32_t, bool> _previousKeyMap;

		std::unordered_map<uint8_t, bool> _mouseMap;
		std::unordered_map<uint8_t, bool> _previousMouseMap;

		glm::vec2 _mouseCoords;

	};
}


