#include "../Lengine/InputManager.h"

namespace Lengine {


	InputManager::InputManager() : _mouseCoords(0, 0) {}
	InputManager::~InputManager() {}

	void InputManager::update() {
		for (auto& it : _keyMap) {
			_previousKeyMap[it.first] = it.second;
		}
	}


	void InputManager::pressKey(unsigned int keyID) {
		_keyMap[keyID] = true;
	}

	void InputManager::releaseKey(unsigned int keyID) {
		_keyMap[keyID] = false;
	}

	void InputManager::setMouseCoords(float x, float y) {
		_mouseCoords.x = x;
		_mouseCoords.y = y;
	}


	bool InputManager::isKeyDown(unsigned int keyID) {
		auto it = _keyMap.find(keyID);
		if (it != _keyMap.end()){
			return it->second;
		}
		else {
			return false;
		}
	}

	bool InputManager::isKeyPressed(unsigned int keyID) {
		if (isKeyDown(keyID) && !wasKeyDown(keyID)) {
			return true;
		}
		return false;
	}

	bool InputManager::wasKeyDown(unsigned int keyID) {
		auto it = _previousKeyMap.find(keyID);
		if (it != _previousKeyMap.end()) {
			return it->second;
		}
		else {
			return false;
		}
	}

	void InputManager::pressMouseButton(unsigned int buttonID) {
		_mouseMap[buttonID] = true;
	}

	void InputManager::releaseMouseButton(unsigned int buttonID) {
		_mouseMap[buttonID] = false;
	}

	bool InputManager::isMouseButtonDown(unsigned int buttonID) {
		auto it = _mouseMap.find(buttonID);
		return (it != _mouseMap.end()) ? it->second : false;
	}

	bool InputManager::isMouseButtonPressed(unsigned int buttonID) {
		return isMouseButtonDown(buttonID) && !wasMouseButtonDown(buttonID);
	}

	bool InputManager::wasMouseButtonDown(unsigned int buttonID) {
		auto it = _previousMouseMap.find(buttonID);
		return (it != _previousMouseMap.end()) ? it->second : false;
	}

	
}

