/*
 * File:   segmentator.h
 * Author: marcelo
 *
 * Created on January 21, 2009, 8:37 PM
 */

#ifndef _SERIALIZER_H
#define	_SERIALIZER_H

#include "common.h"
#include "segmentationresult.h"

namespace Serializer
{
	std::string serializeSegmentationResult(const SegmentationResult& sr);
	std::string serializeContour(const Contour& contour);
	std::string serializeParabola(const Parabola& parabola);

	SegmentationResult unserializeSegmentationResult(const std::string& s);
	Contour unserializeContour(std::istringstream& stream);
	Parabola unserializeParabola(std::istringstream& stream);
};

#endif	/* _SERIALIZER_H */
