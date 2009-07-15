/*
 * File:   pupilsegmentator.h
 * Author: marcelo
 *
 * Created on January 21, 2009, 8:39 PM
*/

#ifndef _PUPILSEGMENTATOR_H
#define	_PUPILSEGMENTATOR_H

#include "common.h"


class PupilSegmentator {
public:
	PupilSegmentator();
	virtual ~PupilSegmentator();

	ContourAndCloseCircle segmentPupil(const Image* image);

	struct {
		Image* similarityImage;
		Image* equalizedImage;
		Image* adjustmentRing;
		Image* adjustmentRingGradient;
		Image* workingImage;
		CvMat* adjustmentSnake;
		CvMat* LUT;
		double resizeFactor;
	} buffers;

private:
	void setupBuffers(const Image* image);
	void similarityTransform();
	Circle approximatePupil(const Image* image);
	Circle cascadedIntegroDifferentialOperator(const Image* image);

	typedef struct {
		int maxRad;
		int maxStep;
	} MaxAvgRadiusResult;
	MaxAvgRadiusResult maxAvgRadius(const Image* image, int x, int y, int radmin, int radmax, int radstep);

	uint8_t circleAverage(const Image* image, int x, int y, int radius);
	Contour adjustPupilContour(const Image* image, const Circle& approximateCircle);

	double _lastSigma, _lastMu;
};

#endif	/* _PUPILSEGMENTATOR_H */

