/*
 * File:   main.cpp
 * Author: marcelo
 *
 * Created on January 22, 2009, 2:01 PM
 */

#include <iostream>
#include <stdio.h>

#include "common.h"
#include "segmentator.h"
#include "decorator.h"
#include "irisencoder.h"
#include "irisdctencoder.h"
#include "parameters.h"
#include "videoprocessor.h"
#include "templatecomparator.h"
#include "qualitychecker.h"
#include "tools.h"
#include "serializer.h"

using namespace std;

double correlation(IplImage* X, IplImage* Y);
void processImage(IplImage* image);
void captured();

Segmentator segmentator;
QualityChecker qualityChecker;
Decorator decorator;
VideoProcessor videoProcessor;
LogGaborEncoder irisEncoder;
IrisDCTEncoder irisDCTEncoder;
Parameters* parameters = Parameters::getParameters();

CvFont FONT;
char BUFFER[1000];

int main(int argc, char** argv) {
	const char* imagePath = "/home/marcelo/iris/BBDD/UBA/marcelo_der_1.bmp";
	IplImage* image = cvLoadImage(imagePath, 1);

    SegmentationResult res = segmentator.segmentImage(image);
    decorator.drawSegmentationResult(image, res);

	IrisTemplate irisTemplate = irisEncoder.generateTemplate(image, res);
	//IrisTemplate irisTemplate = irisDCTEncoder.generateTemplate(image, res);
	//irisTemplate = irisDCTEncoder.generateTemplate(image, res);

	cvNamedWindow("Test");
	decorator.drawTemplate(image, irisTemplate);
	decorator.drawEncodingZone(image, res);
	cvShowImage("Test", image);

	cvWaitKey(0);

	return 0;
}

int main1(int argc, char** argv) {
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

	IplImage* frame = cvCreateImage(size, IPL_DEPTH_8U, 1);

	while (true) {
		framecolor = cvQueryFrame(capture);
		cvCvtColor(framecolor, frame, CV_BGR2GRAY);

		CvMat tmp;
		int delta = 10;
		cvGetSubRect(frame, &tmp, cvRect(delta, 0, frame->width-2*delta, frame->height));
		IplImage porcion;
		cvGetImage(&tmp, &porcion);

		processImage(&porcion);
		char k = cvWaitKey(10);
		if (k == 'q') {
			break;
		}
	}

	return 0;
}

void processImage(IplImage* image)
{
	VideoProcessor::VideoStatus vs = videoProcessor.processFrame(image);
	switch (vs) {
	case VideoProcessor::UNPROCESSED:
		sprintf(BUFFER, "UNPROCESSED");
		break;
	case VideoProcessor::DEFOCUSED:
		sprintf(BUFFER, "DEFOCUSED");
		break;
	case VideoProcessor::INTERLACED:
		sprintf(BUFFER, "INTERLACED");
		break;
	case VideoProcessor::FOCUSED_NO_IRIS:
		sprintf(BUFFER, "FOCUSED_NO_IRIS");
		break;
	case VideoProcessor::IRIS_LOW_QUALITY:
		sprintf(BUFFER, "IRIS_LOW_QUALITY");
		break;
	case VideoProcessor::IRIS_TOO_CLOSE:
		sprintf(BUFFER, "IRIS_TOO_CLOSE");
		break;
	case VideoProcessor::IRIS_TOO_FAR:
		sprintf(BUFFER, "IRIS_TOO_FAR");
		break;
	case VideoProcessor::FOCUSED_IRIS:
		sprintf(BUFFER, "FOCUSED_IRIS");
		break;
	case VideoProcessor::GOT_TEMPLATE:
		sprintf(BUFFER, "GOT_TEMPLATE");
		captured();
		break;
	}
	cvPutText(image, BUFFER, cvPoint(400, 300), &FONT, CV_RGB(255,255,255));

	/*sprintf(BUFFER, "Focus: %.2f", videoProcessor.lastFocusScore);
	cvPutText(image, BUFFER, cvPoint(400, 330), &FONT, CV_RGB(255,255,255));
	sprintf(BUFFER, "S. score: %.2f", videoProcessor.lastSegmentationScore);
	cvPutText(image, BUFFER, cvPoint(400, 360), &FONT, CV_RGB(255,255,255));*/

	cvNamedWindow("video");
	cvShowImage("video", image);

}

void captured()
{
	/*IrisTemplate irisTemplate = videoProcessor.getTemplate();

	IplImage* image = cvCloneImage(videoProcessor.bestFrame);
	decorator.drawTemplate(image, irisTemplate);
	decorator.drawSegmentationResult(image, videoProcessor.lastSegmentationResult);


	cvNamedWindow("template");
	cvShowImage("template", image);
	cvReleaseImage(&image);*/
}
