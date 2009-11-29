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

	segmentEyelids = false;

	infraredThreshold = 200;

	parabolicDetectorStep = 10;

	templateWidth = 320;
	templateHeight = 28;

	interlacedVideo = true;
	focusThreshold = 50;
	correlationThreshold = 92;
	expectedIrisDiameter = 250;
	segmentationScoreThreshold = 1.7;
	minimumContourQuality = 70;
}

Parameters::~Parameters() {
}

Parameters* Parameters::getParameters() {
	if (Parameters::instance == 0) {
		Parameters::instance = new Parameters();
	}

	return Parameters::instance;
}

