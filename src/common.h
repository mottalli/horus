#pragma once

#include <opencv/cv.h>
#include <opencv/cvaux.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>
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

using namespace std;
using namespace cv;

// "Lo que no cuesta, no vale"


#include "types.h"

#define SAME_SIZE(im1, im2) (im1->width==im2->width && im1->height == im2->height)

#ifdef USE_CUDA
static const bool HORUS_CUDA_SUPPORT = true;
#else
static const bool HORUS_CUDA_SUPPORT = false;
#endif
