/*
 * qualitychecker.cpp
 *
 *  Created on: Jun 19, 2009
 *      Author: marcelo
 */

#include "qualitychecker.h"
#include <cmath>

QualityChecker::QualityChecker()
{
}

QualityChecker::~QualityChecker(){
}

double QualityChecker::interlacedCorrelation(const Image* frame)
{
	CvSize size = cvGetSize(frame);
	Image* even = cvCreateImage(cvSize(size.width, size.height/2), IPL_DEPTH_8U, 1);
	Image* odd = cvCreateImage(cvSize(size.width, size.height/2), IPL_DEPTH_8U, 1);
	CvMat tmpodd, tmpeven, tmpdest;

	for (int i = 0; i < size.height/2; i++) {
		cvGetSubRect(frame, &tmpeven, cvRect(0, 2*i, size.width, 1));
		cvGetSubRect(frame, &tmpodd, cvRect(0, 2*i+1, size.width, 1));

		cvGetSubRect(even, &tmpdest, cvRect(0, i, size.width, 1));
		cvCopy(&tmpeven, &tmpdest);
		cvGetSubRect(odd, &tmpdest, cvRect(0, i, size.width, 1));
		cvCopy(&tmpodd, &tmpdest);
	}

	// TODO: Use buffers
	Image* bufX = cvCreateImage(cvGetSize(even), IPL_DEPTH_32F, 1);
	Image* bufY = cvCreateImage(cvGetSize(odd), IPL_DEPTH_32F, 1);
	Image* bufMul = cvCreateImage(cvGetSize(even), IPL_DEPTH_32F, 1);

	double meanX = 0, stdX = 0, meanY = 0, stdY = 0;
	double mean;

	cvConvert(even, bufX);
	cvConvert(odd, bufY);

	cvMean_StdDev(bufX, &meanX, &stdX);
	cvMean_StdDev(bufY, &meanY, &stdY);

	cvSubS(bufX, cvScalar(meanX,0,0,0), bufX);
	cvSubS(bufY, cvScalar(meanY,0,0,0), bufY);
	cvMul(bufX, bufY, bufMul);

	mean = cvMean(bufMul);

	cvReleaseImage(&bufX);
	cvReleaseImage(&bufY);
	cvReleaseImage(&bufMul);
	cvReleaseImage(&odd);
	cvReleaseImage(&even);

	return 100.0*mean/(stdX*stdY);

}

double QualityChecker::checkFocus(const Image* image)
{
	Image* src = cvCreateImage(cvGetSize(image), IPL_DEPTH_32F, 1);
	cvConvert(image, src);
	cvSobel(src, src, 0, 1, 3);
	cvAbs(src, src);
	double s = cvSum(src).val[0];
	double c = 3e+06;

	cvReleaseImage(&src);
	return 100.0*s*s/(s*s+c*c);

}

/**
 * Checks if the segmentation is correct (heuristics - not 100% reliable)
 */
double QualityChecker::segmentationScore(const Image* image, const SegmentationResult& sr)
{
	CvMat portionMat;
	IplImage portion;
	double d = 1.5;
	double r = d*sr.irisCircle.radius;
	int x0 = std::max(0.0, sr.irisCircle.xc-r);
	int x1 = std::min(double(image->width), sr.irisCircle.xc+r);
	int y0 = std::max(0, sr.irisCircle.yc-20);
	int y1 = std::min(image->height, sr.irisCircle.yc+20);

	cvGetSubRect(image, &portionMat, cvRect(x0, y0, x1-x0, y1-y0));
	cvGetImage(&portionMat, &portion);

	int xpupil = sr.pupilCircle.xc-x0, ypupil = sr.pupilCircle.yc-y0;
	int xiris = sr.irisCircle.xc-x0, yiris = sr.irisCircle.yc-y0;
	int rpupil2 = sr.pupilCircle.radius*sr.pupilCircle.radius;
	int riris2 = sr.irisCircle.radius*sr.irisCircle.radius;

	double pupilSum = 0, irisSum = 0, scleraSum = 0;
	int pupilCount = 0, irisCount = 0, scleraCount = 0;

	// Computes the mean for each part
	for (int y = 0; y < portion.height; y++) {
		const uint8_t* row = (const uint8_t*)portion.imageData + y*portion.widthStep;
		for (int x = 0; x < portion.width; x++) {
			double val = double(row[x]);

			// Ignore reflections
			if (val > 200) continue;

			int dx2,dy2;

			// Inside pupil?
			dx2 = (x-xpupil)*(x-xpupil);
			dy2 = (y-ypupil)*(y-ypupil);
			if (dx2+dy2 < rpupil2) {
				pupilSum += val;
				pupilCount++;
			} else {
				// Inside iris?
				dx2 = (x-xiris)*(x-xiris);
				dy2 = (y-yiris)*(y-yiris);
				if (dx2+dy2 < riris2) {
					irisSum += val;
					irisCount++;
				} else {
					// Inside sclera
					scleraSum += val;
					scleraCount++;
				}
			}
		}
	}


	double meanPupil = pupilSum/double(pupilCount);
	double meanIris = irisSum/double(irisCount);
	double meanSclera = scleraSum/double(scleraCount);

	// Computes the deviation
	pupilSum = 0;
	irisSum = 0;
	scleraSum = 0;
	for (int y = 0; y < portion.height; y++) {
		const uint8_t* row = (const uint8_t*)portion.imageData + y*portion.widthStep;
		for (int x = 0; x < portion.width; x++) {
			double val = double(row[x]);
			if (val > 200) continue;

			int dx2,dy2;

			// Inside pupil?
			dx2 = (x-xpupil)*(x-xpupil);
			dy2 = (y-ypupil)*(y-ypupil);
			if (dx2+dy2 < rpupil2) {
				pupilSum += (val-meanPupil)*(val-meanPupil);
			} else {
				// Inside iris?
				dx2 = (x-xiris)*(x-xiris);
				dy2 = (y-yiris)*(y-yiris);
				if (dx2+dy2 < riris2) {
					irisSum += (val-meanIris)*(val-meanIris);
				} else {
					// Inside sclera
					scleraSum += (val-meanSclera)*(val-meanSclera);
				}
			}
		}
	}

	double varPupil = pupilSum/double(pupilCount);
	double varIris = irisSum/double(irisCount);
	double varSclera = scleraSum/double(scleraCount);

	double zScorePupilIris = abs(meanPupil-meanIris) / sqrt((varPupil+varIris)/2.0);
	//double zScoreIrisSclera = abs(meanSclera-meanIris) / sqrt((varSclera+varIris)/2.0);

	double c = 2.0;
	double s = zScorePupilIris;

	return 100.0*s*s/(s*s+c*c);
}

/**
 * Checks the quality of the image
 */
bool QualityChecker::checkIrisQuality(const Image* image, const SegmentationResult& segmentationResult)
{
	//TODO
	return true;
}
