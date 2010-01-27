/*
 * File:   common.h
 * Author: marcelo
 *
 * Created on January 22, 2009, 2:25 PM
 */

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


#include "types.h"
#include "parameters.h"

#define SAME_SIZE(im1, im2) (im1->width==im2->width && im1->height == im2->height)


