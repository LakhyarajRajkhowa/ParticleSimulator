#pragma once

namespace Lengine {
	class FpsLimiter {
	public:
		FpsLimiter();

		void init(float maxtFPS);
		void setMaxFPS(float maxFPS);
		void begin();

		int end();

	private:
		void calculateFPS();

		unsigned int _startTicks;

		float _maxFPS;
		float _fps;
		float _frameTime;


	};
}
