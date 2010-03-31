#include "clock.h"

Clock::Clock() {
    _time = 0;
}

Clock::~Clock() {
}


void Clock::start() {
	gettimeofday(&_tic, 0);
}

double Clock::stop() {
	gettimeofday(&_toc, 0);
	_time = 1000.0*(_toc.tv_sec-_tic.tv_sec) + (_toc.tv_usec-_tic.tv_usec)/1000.0;

    return  _time;
}
