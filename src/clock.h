#pragma once

#include <sys/time.h>

class Clock {
public:
    Clock();
    virtual ~Clock();

    void start();
    double stop();
	inline double time() const { return _time; };
private:
    double _time;
	struct timeval _tic, _toc;

};


