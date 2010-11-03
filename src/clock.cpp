#include "clock.h"

Clock::Clock() {
    _time = 0;
}

Clock::~Clock() {
}


void Clock::start() {
#if !defined(_MSC_VER)
	gettimeofday(&_tic, 0);
#endif
}

double Clock::stop() {
#if !defined(_MSC_VER)
	gettimeofday(&_toc, 0);
	_time = 1000.0*(_toc.tv_sec-_tic.tv_sec) + (_toc.tv_usec-_tic.tv_usec)/1000.0;
#else
	_time = 0.0;
#endif

    return  _time;
}
