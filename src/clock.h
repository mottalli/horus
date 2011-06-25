#pragma once

#ifdef WIN32

#include <boost/timer.hpp>
namespace horus {
typedef boost::timer Timer;		// boost::timer does not have enough resolution on Linux!
}

#else
#include <sys/time.h>

namespace horus {
class Timer {
public:
	Timer()
	{
		this->restart();
	}

	inline double elapsed()
	{
		struct timeval end;
		gettimeofday(&end, NULL);

		double seconds, microseconds, miliseconds;
		seconds = (double)(end.tv_sec - start.tv_sec);
		microseconds = (double)(end.tv_usec - start.tv_usec);
		miliseconds = 1000.0*seconds + microseconds/1000;

		return miliseconds;
	}

	inline void restart()
	{
		gettimeofday(&start, NULL);
	}
private:
	struct timeval start;
};

}
#endif	/* WIN32 */
