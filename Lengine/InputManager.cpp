#include "../Lengine/InputManager.h"

namespace Lengine {


	InputManager::InputManager() : _mouseCoords(0, 0) {}
	InputManager::~InputManager() {}

	void InputManager::update() {
		for (auto& it : _keyMap) {
			_previousKeyMap[it.first] = it.second;
		}
	}


	void InputManager::pressKey(uint32_t keyID) {
		_keyMap[keyID] = true;
	}

	void InputManager::releaseKey(uint32_t keyID) {
		_keyMap[keyID] = false;
	}

	void InputManager::setMouseCoords(float x, float y) {
		_mouseCoords.x = x;
		_mouseCoords.y = y;
	}


	bool InputManager::isKeyDown(uint32_t keyID) {
		auto it = _keyMap.find(keyID);
		if (it != _keyMap.end()){
			return it->second;
		}
		else {
			return false;
		}
	}

	bool InputManager::isKeyPressed(uint32_t keyID) {
		if (isKeyDown(keyID) && !wasKeyDown(keyID)) {
			return true;
		}
		return false;
	}

	bool InputManager::wasKeyDown(uint32_t keyID) {
		auto it = _previousKeyMap.find(keyID);
		if (it != _previousKeyMap.end()) {
			return it->second;
		}
		else {
			return false;
		}
	}

	void InputManager::pressMouseButton(uint8_t buttonID) {
		_mouseMap[buttonID] = true;
	}

	void InputManager::releaseMouseButton(uint8_t buttonID) {
		_mouseMap[buttonID] = false;
	}

	bool InputManager::isMouseButtonDown(uint8_t buttonID) {
		auto it = _mouseMap.find(buttonID);
		return (it != _mouseMap.end()) ? it->second : false;
	}

	bool InputManager::isMouseButtonPressed(uint8_t buttonID) {
		return isMouseButtonDown(buttonID) && !wasMouseButtonDown(buttonID);
	}

	bool InputManager::wasMouseButtonDown(uint8_t buttonID) {
		auto it = _previousMouseMap.find(buttonID);
		return (it != _previousMouseMap.end()) ? it->second : false;
	}

	
}

