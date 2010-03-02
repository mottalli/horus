/*
 * decorator.cpp
 *
 *  Created on: Jun 4, 2009
 *      Author: marcelo
 */

#include "decorator.h"
#include "irisencoder.h"
#include <cmath>

Decorator::Decorator()
{
	this->irisColor = CV_RGB(0,255,0);
	this->pupilColor = CV_RGB(255,0,0);
	this->upperEyelidColor = CV_RGB(0,0,255);
	this->lowerEyelidColor = CV_RGB(0,0,255);
}

void Decorator::drawSegmentationResult(IplImage* image, const SegmentationResult& segmentationResult)
{
	this->drawContour(image, segmentationResult.irisContour, this->irisColor);
	this->drawContour(image, segmentationResult.pupilContour, this->pupilColor);

	Circle irisCircle = segmentationResult.irisCircle;

	if (segmentationResult.eyelidsSegmented) {
		int xMin = segmentationResult.irisCircle.xc-segmentationResult.irisCircle.radius;
		int xMax = segmentationResult.irisCircle.xc+segmentationResult.irisCircle.radius;
		this->drawParabola(image, segmentationResult.upperEyelid, xMin, xMax, this->upperEyelidColor);
		this->drawParabola(image, segmentationResult.lowerEyelid, xMin, xMax, this->lowerEyelidColor);
	}

	/*cvCircle(image, cvPoint(segmentationResult.irisCircle.xc,segmentationResult.irisCircle.yc), segmentationResult.irisCircle.radius, CV_RGB(255,255,255), 1);
	cvCircle(image, cvPoint(segmentationResult.pupilCircle.xc,segmentationResult.pupilCircle.yc), segmentationResult.pupilCircle.radius, CV_RGB(255,255,255), 1);*/
}

void Decorator::drawEncodingZone(IplImage* image, const SegmentationResult& segmentationResult)
{
	// Most of the code taken from irisencoder.cpp
	Parameters* parameters = Parameters::getParameters();
	int normalizedWidth = parameters->normalizationWidth, normalizedHeight = parameters->normalizationHeight;
	CvPoint p0, p1;

	double theta0 = IrisEncoder::THETA0;
	double theta1 = IrisEncoder::THETA1;
	double radiusToUse = IrisEncoder::RADIUS_TO_USE;		// Only use three-quarters of the radius
	
	const Contour& pupilContour = segmentationResult.pupilContour;
	const Contour& irisContour = segmentationResult.irisContour;

	
	for (int x = 0; x < normalizedWidth; x++) {
		double theta = (double(x)/double(normalizedWidth)) * (theta1-theta0) + theta0;
		if (theta < 0) theta = 2.0 * M_PI + theta;
		assert(theta >= 0 && theta <= 2.0*M_PI);
		double w = (theta/(2.0*M_PI))*double(pupilContour.size());
		p0 = pupilContour[int(std::floor(w))];
		p1 = pupilContour[int(std::ceil(w)) % pupilContour.size()];
		double prop = w-std::floor(w);
		double xfrom = double(p0.x) + double(p1.x-p0.x)*prop;
		double yfrom = double(p0.y) + double(p1.y-p0.y)*prop;
		w = (theta/(2.0*M_PI))*double(irisContour.size());
		p0 = irisContour[int(std::floor(w))];
		p1 = irisContour[int(std::ceil(w)) % irisContour.size()];
		prop = w-std::floor(w);
		double xto = double(p0.x) + double(p1.x-p0.x)*prop;
		double yto = double(p0.y) + double(p1.y-p0.y)*prop;
		for (int y = 0; y < normalizedHeight; y++) {
			w = (double(y)/double(normalizedHeight-1)) * radiusToUse;
			double ximage = xfrom + w*(xto-xfrom);
			double yimage = yfrom + w*(yto-yfrom);

			int ximage0 = int(std::floor(ximage));
			int ximage1 = int(std::ceil(ximage));
			int yimage0 = int(std::floor(yimage));
			int yimage1 = int(std::ceil(yimage));			

			if (ximage0 < 0 || ximage1 >= image->width || yimage0 < 0 || yimage1 >= image->height) {
			} else if (segmentationResult.eyelidsSegmented && (yimage <= segmentationResult.upperEyelid.value(ximage) || yimage >= segmentationResult.lowerEyelid.value(ximage))) {
			} else {
				cvSet2D(image, yimage, ximage, CV_RGB(255,255,0));
			}
		}
	}
}


void Decorator::drawContour(IplImage* image, const Contour& contour, CvScalar color)
{
	if (contour.size() < 2) return;

	const CvPoint p0 = contour[0];
	int n = contour.size();

	CvPoint lastPoint = p0;

	for (int i = 1; i < n; i++) {
		const CvPoint p = contour[i];

		if (false) {

		} else {
			cvLine(image, lastPoint, p, color, 1);
		}
		lastPoint = p;
	}

	cvLine(image, lastPoint, p0, color, 1);
}

void Decorator::drawParabola(IplImage* image, const Parabola& parabola, int xMin, int xMax, CvScalar color)
{
	if (xMin < 0) xMin = 1;
	if (xMax < 0 || xMax >= image->width) xMax = image->width-1;

	CvPoint lastPoint = cvPoint(xMin, int(parabola.value(xMin)));
	for (int x = xMin+1; x <= xMax; x++) {
		CvPoint point = cvPoint(x, int(parabola.value(x)));

		cvLine(image, lastPoint, point, color, 1);
		lastPoint = point;
	}
}

void Decorator::drawTemplate(IplImage* image, const IrisTemplate& irisTemplate)
{
	IplImage* imgTemplate = irisTemplate.getTemplateImage();
	IplImage* imgMask = irisTemplate.getNoiseMaskImage();

	CvSize size = cvGetSize(imgTemplate);

	CvMat region;
	CvPoint topleftTemplate, topleftMask;

	topleftTemplate = cvPoint(10, 10);
	topleftMask = cvPoint(topleftTemplate.x, image->height-10-size.height);

	cvGetSubRect(image, &region, cvRect(topleftTemplate.x, topleftTemplate.y, size.width, size.height));
	if (image->nChannels == 3) {
		cvMerge(imgTemplate, NULL, NULL, NULL, &region);
		cvMerge(NULL, imgTemplate, NULL, NULL, &region);
		cvMerge(NULL, NULL, imgTemplate, NULL, &region);
	} else {
		cvCopy(imgTemplate, &region);
	}
	cvRectangle(image, topleftTemplate, cvPoint(topleftTemplate.x+size.width-1, topleftTemplate.y+size.height-1), CV_RGB(0,0,0), 1);

	cvGetSubRect(image, &region, cvRect(topleftMask.x, topleftMask.y, size.width, size.height));
	if (image->nChannels == 3) {
		cvMerge(imgMask, NULL, NULL, NULL, &region);
		cvMerge(NULL, imgMask, NULL, NULL, &region);
		cvMerge(NULL, NULL, imgMask, NULL, &region);
	} else {
		cvCopy(imgMask, &region);
	}
	cvRectangle(image, topleftMask, cvPoint(topleftMask.x+size.width-1, topleftMask.y+size.height-1), CV_RGB(0,0,0), 1);

	cvReleaseImage(&imgTemplate);
	cvReleaseImage(&imgMask);
}
