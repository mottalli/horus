#pragma once

#if defined(_MSC_VER)
#include <time.h>
#else
#include <sys/time.h>
#endif

namespace horus {

class Clock {
public:
	Clock();
	virtual ~Clock();

	void start();
	double stop();
	inline double time() const { return _time; };
private:
	double _time;
#if !defined(_MSC_VER)
	timeval _tic, _toc;
#endif

};

}
