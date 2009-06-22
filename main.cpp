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

using namespace std;

double correlation(Image* X, Image* Y);
void processImage(Image* image);

Segmentator segmentator;
QualityChecker qualityChecker;
Decorator decorator;


int main(int argc, char** argv) {
	CvCapture* capture = cvCaptureFromCAM(0);
	if (!capture) {
		cout << "No se puede capturar" << endl;
		return 1;
	}

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

	double focus = qualityChecker.checkFocus(image);
	double corr = qualityChecker.interlacedCorrelation(image);

	cout << focus << ',' << corr << endl;
	if (focus > 50 && corr >= 0.95) {
		cvCircle(image, cvPoint(600, 400), 10, CV_RGB(255,255,255));
	}


	decorator.drawSegmentationResult(image, sr);
	cvShowImage("video", image);

}
