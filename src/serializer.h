/*
 * File:   segmentator.h
 * Author: marcelo
 *
 * Created on January 21, 2009, 8:37 PM
 */

#pragma once

#include "common.h"
#include "segmentationresult.h"
#include "iristemplate.h"

namespace Serializer
{
	std::string serializeSegmentationResult(const SegmentationResult& sr);
	std::string serializeContour(const Contour& contour);
	std::string serializeParabola(const Parabola& parabola);

	SegmentationResult unserializeSegmentationResult(const std::string& s);
	Contour unserializeContour(std::istringstream& stream);
	Parabola unserializeParabola(std::istringstream& stream);

	std::string serializeIrisTemplate(const IrisTemplate& irisTemplate);
	IrisTemplate unserializeIrisTemplate(const std::string& serializedTemplate);
};

