/* 
 * File:   clock.h
 * Author: marcelo
 *
 * Created on January 26, 2009, 2:35 AM
 */

#ifndef _CLOCK_H
#define	_CLOCK_H

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

#endif	/* _CLOCK_H */

