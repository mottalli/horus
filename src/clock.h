#pragma once

#include <ctime>

class Clock {
public:
    Clock();
    virtual ~Clock();

    void start();
    double stop();
	inline double time() const { return _time; };
private:
    double _time;
    std::clock_t _tic, _toc;

};


