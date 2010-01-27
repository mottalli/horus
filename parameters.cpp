	/*
 * File:   parameters.cpp
 * Author: marcelo
 *
 * Created on January 22, 2009, 2:15 PM
 */

#include "parameters.h"

Parameters* Parameters::instance = (Parameters*)0;

Parameters::Parameters() {
	bufferWidth = 320;
	muPupil = 5.0;
	sigmaPupil = 5.0;

	pupilAdjustmentRingWidth = 256;
	pupilAdjustmentRingHeight = 100;

	irisAdjustmentRingWidth = 512;
	irisAdjustmentRingHeight = 90;

	segmentEyelids = true;

	infraredThreshold = 200;

	parabolicDetectorStep = 10;

	templateWidth = 32*8;		// Must be a multiple of 32
	templateHeight = 20;
	normalizationWidth = 2*templateWidth;
	normalizationHeight = 2*templateHeight;

	interlacedVideo = true;
	focusThreshold = 40;
	correlationThreshold = 92;
	expectedIrisDiameter = 250;
	segmentationScoreThreshold = 1.7;
	minimumContourQuality = 60;
	
	pupilIrisGrayDiff = 20;
	pupilIrisZScore = 3;
}

Parameters::~Parameters() {
}

Parameters* Parameters::getParameters() {
	if (Parameters::instance == 0) {
		Parameters::instance = new Parameters();
	}

	return Parameters::instance;
}

