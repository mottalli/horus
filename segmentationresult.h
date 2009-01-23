/* 
 * File:   segmentationresult.h
 * Author: marcelo
 *
 * Created on January 21, 2009, 9:33 PM
 */

#ifndef _SEGMENTATIONRESULT_H
#define	_SEGMENTATIONRESULT_H

#include "types.h"

class SegmentationResult {
public:
    SegmentationResult();
    virtual ~SegmentationResult();

    Contour irisContour;
    Contour pupilContour;
private:

};

#endif	/* _SEGMENTATIONRESULT_H */

