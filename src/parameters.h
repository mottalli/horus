/*
 * File:   parameters.h
 * Author: marcelo
 *
 * Created on January 22, 2009, 2:15 PM
 */

#pragma once

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
	int minimumPupilRadius, maximumPupilRadius;

	int pupilAdjustmentRingWidth, pupilAdjustmentRingHeight;
	int irisAdjustmentRingWidth, irisAdjustmentRingHeight;

	bool segmentEyelids;

	int infraredThreshold;

	int parabolicDetectorStep;

	// Video processing parameters
	bool interlacedVideo;
	int focusThreshold;
	double segmentationScoreThreshold;
	int correlationThreshold;
	int expectedIrisDiameter;
	int minimumContourQuality;
	int pupilIrisGrayDiff;
	int pupilIrisZScore;
	int bestFrameWaitCount;
	int waitingFrames;

	// Template generation parameters
	int normalizationWidth;
	int normalizationHeight;
	//int templateWidth;
	//int templateHeight;

private:
	Parameters();
	static Parameters* instance;
};


