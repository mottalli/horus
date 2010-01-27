/*
 * File:   segmentationresult.h
 * Author: marcelo
 *
 * Created on January 21, 2009, 9:33 PM
 */

#pragma once

#include "types.h"

class SegmentationResult {
public:
    SegmentationResult();
    virtual ~SegmentationResult();

    Contour irisContour;
    Contour pupilContour;
    Circle pupilCircle;
    Circle irisCircle;
    Parabola upperEyelid;
    Parabola lowerEyelid;

	double pupilContourQuality;

    bool eyelidsSegmented;
private:

};


