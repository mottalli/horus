#pragma once

#include <opencv2/opencv.hpp>
#if defined(_MSC_VER)
#define _USE_MATH_DEFINES 1
#include <math.h>
#else
#include <cmath>
#endif
#include <iostream>
#include <vector>
#include <list>
#include <limits.h>
#include <stdint.h>
#include <cassert>
#include <stdexcept>
#include <algorithm>

// "Lo que no cuesta, no vale"

#include "types.h"

namespace horus {
	#ifdef USE_CUDA
	static const bool HORUS_CUDA_SUPPORT = true;
	#else
	static const bool HORUS_CUDA_SUPPORT = false;
	#endif
}
