/*
 * File:   parameters.h
 * Author: marcelo
 *
 * Created on January 22, 2009, 2:15 PM
 */

#ifndef _PARAMETERS_H
#define	_PARAMETERS_H

/**
 * Singleton class
 */
class Parameters {
public:
	virtual ~Parameters();
	static Parameters* getParameters();

	// Segmentation parameters
	unsigned bufferWidth;
	double muPupil;
	double sigmaPupil;

	int pupilAdjustmentRingWidth, pupilAdjustmentRingHeight;
	int irisAdjustmentRingWidth, irisAdjustmentRingHeight;

	int infraredThreshold;

	int parabolicDetectorStep;

	// Video processing parameters
	bool interlacedVideo;
	int focusThreshold;
	double segmentationScoreThreshold;
	int correlationThreshold;
	int expectedIrisDiameter;

	// Template generation parameters
	int templateWidth;
	int templateHeight;

private:
	Parameters();
	static Parameters* instance;
};

#endif	/* _PARAMETERS_H */

