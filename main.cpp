/*
 * File:   main.cpp
 * Author: marcelo
 *
 * Created on January 22, 2009, 2:01 PM
 */

#include <iostream>

#include "common.h"
#include "segmentator.h"
#include "decorator.h"
#include "irisencoder.h"
#include "parameters.h"
#include "videoprocessor.h"
#include "templatecomparator.h"
#include "qualitychecker.h"
#include "tools.h"

using namespace std;

double correlation(Image* X, Image* Y);
void processImage(Image* image);

Segmentator segmentator;
QualityChecker qualityChecker;
Decorator decorator;

CvFont FONT;
char BUFFER[1000];


int main(int argc, char** argv) {
	CvCapture* capture = cvCaptureFromCAM(0);
	if (!capture) {
		cout << "No se puede capturar" << endl;
		return 1;
	}

	cvInitFont(&FONT, CV_FONT_HERSHEY_PLAIN, 1, 2);

	IplImage* framecolor = 0;
	framecolor = cvQueryFrame(capture);

	if (!framecolor) {
		cout << "No se pudo capturar frame" << endl;
		return 1;
	} else {
		cout << "Primer frame capturado (" << framecolor->width << "x" << framecolor->height << ")" << endl;
		cout << "Depth: " << framecolor->depth << endl;
		cout << "nChannels: " << framecolor->nChannels << endl;
		cout << "Origen: " << framecolor->origin << endl;
	}

	CvSize size = cvGetSize(framecolor);

	Image* frame = cvCreateImage(size, IPL_DEPTH_8U, 1);
	cvNamedWindow("video");

	while (true) {
		framecolor = cvQueryFrame(capture);
		cvCvtColor(framecolor, frame, CV_BGR2GRAY);

		CvMat tmp;
		int delta = 10;
		cvGetSubRect(frame, &tmp, cvRect(delta, 0, frame->width-2*delta, frame->height));
		Image porcion;
		cvGetImage(&tmp, &porcion);

		processImage(&porcion);
		char k = cvWaitKey(10);
		if (k == 'q') {
			break;
		}
	}
}

void processImage(Image* image)
{
	SegmentationResult sr = segmentator.segmentImage(image);

	//////////
	//cvNamedWindow("porcion");
	CvMat porcionMat;
	IplImage porcion;
	double d = 1.5;
	double r = d*sr.irisCircle.radius;
	int x0 = max(0.0, sr.irisCircle.xc-r);
	int x1 = min(double(image->width), sr.irisCircle.xc+r);
	int y0 = max(0, sr.irisCircle.yc-20);
	int y1 = min(image->height, sr.irisCircle.yc+20);

	if ((x1-x0) % 2 == 1) x1--;
	if ((y1-y0) % 2 == 1) y1--;

	cvGetSubRect(image, &porcionMat, cvRect(x0, y0, x1-x0, y1-y0));
	cvGetImage(&porcionMat, &porcion);
	//cvShowImage("porcion", &porcion);

	int xpupil = sr.pupilCircle.xc-x0, ypupil = sr.pupilCircle.yc-y0;
	int xiris = sr.irisCircle.xc-x0, yiris = sr.irisCircle.yc-y0;
	int rpupil2 = sr.pupilCircle.radius*sr.pupilCircle.radius;
	int riris2 = sr.irisCircle.radius*sr.irisCircle.radius;

	double pupilSum = 0, irisSum = 0, scleraSum = 0;
	int pupilCount = 0, irisCount = 0, scleraCount = 0;

	// Computes the mean for each part
	for (int y = 0; y < porcion.height; y++) {
		const uint8_t* row = (const uint8_t*)porcion.imageData + y*porcion.widthStep;
		for (int x = 0; x < porcion.width; x++) {
			double val = double(row[x]);

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
	for (int y = 0; y < porcion.height; y++) {
		const uint8_t* row = (const uint8_t*)porcion.imageData + y*porcion.widthStep;
		for (int x = 0; x < porcion.width; x++) {
			double val = double(row[x]);
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


	double focus = qualityChecker.checkFocus(image);
	double corr = qualityChecker.interlacedCorrelation(image);

	double zScorePupilIris = abs(meanPupil-meanIris) / sqrt((varPupil+varIris)/2.0);
	double zScoreIrisSclera = abs(meanSclera-meanIris) / sqrt((varSclera+varIris)/2.0);


	Tools::drawHistogram(&porcion);
	//////////

	int ytext = image->height-90;
	int linea = 19;
	int xtext = image->width-300;
	sprintf(BUFFER, "Focus: %.3f", focus);
	cvPutText(image, BUFFER, cvPoint(xtext, ytext+1*linea), &FONT, CV_RGB(0,0,0));
	sprintf(BUFFER, "Corr: %.3f", corr);
	cvPutText(image, BUFFER, cvPoint(xtext, ytext+2*linea), &FONT, CV_RGB(0,0,0));
	sprintf(BUFFER, "Means: %.2f, %.2f, %.2f", meanPupil, meanIris, meanSclera);
	cvPutText(image, BUFFER, cvPoint(xtext, ytext+3*linea), &FONT, CV_RGB(0,0,0));
	sprintf(BUFFER, "Z-scores: %.2f, %.2f", zScorePupilIris, zScoreIrisSclera);
	cvPutText(image, BUFFER, cvPoint(xtext, ytext+4*linea), &FONT, CV_RGB(0,0,0));


	if (focus > 61 && corr > 90 && zScorePupilIris > 2.5 &&  zScoreIrisSclera > 1.5) {
		decorator.irisColor = decorator.pupilColor = CV_RGB(255,255,255);
		decorator.drawSegmentationResult(image, sr);
		cvNamedWindow("mejorImagen");
		cvShowImage("mejorImagen", image);
	} else {
		decorator.irisColor = decorator.pupilColor = CV_RGB(0,0,0);
		decorator.drawSegmentationResult(image, sr);
	}

	cvShowImage("video", image);

}
