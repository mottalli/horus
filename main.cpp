/*
 * File:   main.cpp
 * Author: marcelo
 *
 * Created on January 22, 2009, 2:01 PM
 */

#include <iostream>

#include "segmentator.h"
#include "decorator.h"

using namespace std;

int main(int argc, char** argv) {
	/*IplImage* imagen = cvLoadImage(argv[1]);

	Segmentator segmentator;
	SegmentationResult res = segmentator.segmentImage(imagen);

	Decorator decorator;
	decorator.drawSegmentationResult(imagen, res);

	cvNamedWindow("imagen");
	cvShowImage("imagen", imagen);

	cvWaitKey(0);
	cvReleaseImage(&imagen);*/
	CvCapture* capture = cvCaptureFromCAM(0);
	if (!capture) {
		cout << "No se puede capturar" << endl;
		return 1;
	}

	IplImage* frame = 0;
	frame = cvQueryFrame(capture);

	if (!frame) {
		cout << "No se pudo capturar frame" << endl;
		return 1;
	} else {
		cout << "Primer frame capturado (" << frame->width << "x" << frame->height << ")" << endl;
		cout << "Depth: " << frame->depth << endl;
		cout << "nChannels: " << frame->nChannels << endl;
		cout << "Origen: " << frame->origin << endl;
	}

	cvNamedWindow("Video");
	Segmentator segmentator;
	Decorator decorator;

	Parameters* parameters = Parameters::getParameters();

	parameters->muPupil = 10;
	parameters->sigmaPupil = 10;


	while (true) {
		frame = cvQueryFrame(capture);
		if (!frame) {
			break;
		}

		cout << "Frame" << endl;

		SegmentationResult res = segmentator.segmentImage(frame);
		decorator.drawSegmentationResult(frame, res);
		cvShowImage("Video", frame);
		//cvShowImage("Video", segmentator._pupilSegmentator.buffers.similarityImage);
		cvWaitKey(10);
	}
}

