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
#include "parameters.h"
#include "videoprocessor.h"
#include "templatecomparator.h"
#include "qualitychecker.h"
#include "tools.h"

using namespace std;

double correlation(Image* X, Image* Y);
void processImage(Image* image);
void captured();

Segmentator segmentator;
QualityChecker qualityChecker;
Decorator decorator;
VideoProcessor videoProcessor;
IrisEncoder irisEncoder;
Parameters* parameters = Parameters::getParameters();

CvFont FONT;
char BUFFER[1000];

int main(int argc, char** argv) {
	const char* imagePath = "/home/marcelo/Mis_Documentos/Facu/Tesis/Bases de datos/Bath/0019/R/0009.jpg";
	IplImage* image = cvLoadImage(imagePath, 0);

    SegmentationResult res = segmentator.segmentImage(image);
    decorator.drawSegmentationResult(image, res);

	IrisTemplate irisTemplate = irisEncoder.generateTemplate(image, res);

	IplImage* noiseMask = irisTemplate.getNoiseMaskImage();
	IplImage* templateImage = irisTemplate.getTemplateImage();

	cvNamedWindow("template");
	cvShowImage("template", templateImage);
	cvNamedWindow("mask");
	cvShowImage("mask", noiseMask);

	/*cvNamedWindow("image");
    cvShowImage("image", image);

	cvNamedWindow("debug1");
	IplImage* foo = cvCreateImage(cvGetSize(segmentator._irisSegmentator.buffers.adjustmentRingGradient), IPL_DEPTH_8U, 1);
    cvNormalize(segmentator._irisSegmentator.buffers.adjustmentRingGradient, foo, 0, 255, CV_MINMAX);
    for (int x = 0; x < segmentator._irisSegmentator.buffers.adjustmentSnake->width; x++) {
		cvCircle(foo, cvPoint(x, cvGetReal2D(segmentator._irisSegmentator.buffers.adjustmentSnake, 0, x)), 1, CV_RGB(255,255,255), 1);
    }
	cvShowImage("debug1", foo);
	cvReleaseImage(&foo);

	cvNamedWindow("debug2");
	foo = cvCreateImage(cvGetSize(segmentator._pupilSegmentator.buffers.adjustmentRingGradient), IPL_DEPTH_8U, 1);
	cvNormalize(segmentator._pupilSegmentator.buffers.adjustmentRingGradient, foo, 0, 255, CV_MINMAX);
	for (int x = 0; x < segmentator._pupilSegmentator.buffers.adjustmentSnake->width; x++) {
		cvCircle(foo, cvPoint(x, cvGetReal2D(segmentator._pupilSegmentator.buffers.adjustmentSnake, 0, x)), 1, CV_RGB(255,255,255), 1);
	}
	cvShowImage("debug2", foo);
	cvReleaseImage(&foo);*/


	while (true) {
		char k = cvWaitKey(0);
		if (k == 'q') {
			break;
		}
	}

	cvReleaseImage(&noiseMask);
	cvReleaseImage(&templateImage);

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

	Image* frame = cvCreateImage(size, IPL_DEPTH_8U, 1);

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

	return 0;
}

void processImage(Image* image)
{
	VideoProcessor::VideoStatus vs = videoProcessor.processFrame(image);
	switch (vs) {
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

	sprintf(BUFFER, "Focus: %.2f", videoProcessor.lastFocusScore);
	cvPutText(image, BUFFER, cvPoint(400, 330), &FONT, CV_RGB(255,255,255));
	sprintf(BUFFER, "S. score: %.2f", videoProcessor.lastSegmentationScore);
	cvPutText(image, BUFFER, cvPoint(400, 360), &FONT, CV_RGB(255,255,255));

	cvNamedWindow("video");
	cvShowImage("video", image);

}

void captured()
{
	IrisTemplate irisTemplate = videoProcessor.getTemplate();

	IplImage* image = cvCloneImage(videoProcessor.buffers.bestFrame);
	decorator.drawTemplate(image, irisTemplate);
	decorator.drawSegmentationResult(image, videoProcessor.lastSegmentationResult);


	cvNamedWindow("template");
	cvShowImage("template", image);
	cvReleaseImage(&image);
}
