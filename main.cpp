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
void captured(Image* image);

Segmentator segmentator;
QualityChecker qualityChecker;
Decorator decorator;
VideoProcessor videoProcessor;
IrisEncoder irisEncoder;

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
	//videoProcessor.processFrame(image);
	//cout << qualityChecker.checkFocus(image) << endl;
	SegmentationResult sr = segmentator.segmentImage(image);
	decorator.drawSegmentationResult(image, sr);
	cvShowImage("video", image);
}

void captured(Image* image)
{
	IrisTemplate irisTemplate = irisEncoder.generateTemplate(image, videoProcessor.lastSegmentationResult);

	IplImage* imgTemplate = irisTemplate.getTemplateImage();
	IplImage* imgMask = irisTemplate.getNoiseMaskImage();

	cvNamedWindow("template");
	cvNamedWindow("mask");
	cvShowImage("template", imgTemplate);
	cvShowImage("mask", imgMask);

	cvReleaseImage(&imgTemplate);
	cvReleaseImage(&imgMask);
}
