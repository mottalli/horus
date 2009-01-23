/* 
 * File:   irissegmentator.h
 * Author: marcelo
 *
 * Created on January 21, 2009, 8:39 PM
 */

#ifndef _IRISSEGMENTATOR_H
#define	_IRISSEGMENTATOR_H

#include "types.h"

class IrisSegmentator {
public:
    IrisSegmentator();
    virtual ~IrisSegmentator();

    Contour segmentIris(const Image* image, const ContourAndCloseCircle& pupilSegmentation);

private:

};

#endif	/* _IRISSEGMENTATOR_H */

