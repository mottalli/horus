#pragma once

#include <opencv/cv.h>
#include <opencv/cvaux.hpp>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <limits.h>
#include <stdint.h>
#include <cassert>
#include <stdexcept>

using namespace std;
using namespace cv;


#include "types.h"
#include "parameters.h"

#define SAME_SIZE(im1, im2) (im1->width==im2->width && im1->height == im2->height)

#ifdef USE_CUDA
static const bool HORUS_CUDA_SUPPORT = true;
#else
static const bool HORUS_CUDA_SUPPORT = false;
#endif

//TODO: Remove this
#define TO_IPLIMAGE(mat) (&((IplImage)mat))
#define TO_CVMAT(mat) (&((CvMat)mat))
