/* 
 * File:   clock.cpp
 * Author: marcelo
 * 
 * Created on January 26, 2009, 2:35 AM
 */

#include "clock.h"

Clock::Clock() {
    _time = 0;
}

Clock::~Clock() {
}


void Clock::start() {
    _tic = std::clock();
}

double Clock::stop() {
    _toc = std::clock();
    _time = double(_toc-_tic)/double(CLOCKS_PER_SEC);

    return  _time;
}
