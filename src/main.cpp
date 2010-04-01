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
#include "gaborencoder.h"
#include "parameters.h"
#include "videoprocessor.h"
#include "templatecomparator.h"
#include "qualitychecker.h"
#include "tools.h"
#include "serializer.h"
#include "tools.h"

using namespace std;

double correlation(IplImage* X, IplImage* Y);
void processImage(IplImage* image);
void captured();

Segmentator segmentator;
QualityChecker qualityChecker;
Decorator decorator;
VideoProcessor videoProcessor;
LogGaborEncoder logGaborEncoder;
//IrisDCTEncoder irisDCTEncoder;
Parameters* parameters = Parameters::getParameters();
GaborEncoder gaborEncoder;


int main(int argc, char** argv) {
	const char* imagePath = "/home/marcelo/iris/BBDD/UBA/marcelo_der_1.bmp";
	Mat image = imread(imagePath);

	SegmentationResult segmentationResult = segmentator.segmentImage(image);

	IrisTemplate template_ = logGaborEncoder.generateTemplate(image, segmentationResult);
	//IrisTemplate template_ = gaborEncoder.generateTemplate(image, segmentationResult);

	/*IplImage* imTemplate = template_.getTemplateImage();
	cvNamedWindow("templ");
	cvShowImage("templ", beautificarTemplate(imTemplate));*/

	decorator.drawSegmentationResult(image, segmentationResult);
	decorator.drawEncodingZone(image, segmentationResult);
	decorator.drawTemplate(image, template_);

	namedWindow("imagen", 1);
	imshow("imagen", image);

	cout << "T: " << segmentator.segmentationTime << " ms" << endl;


	char k;
	do {
		k = cvWaitKey(0);
	} while (k != 'q');

	return 0;
}

int main1(int argc, char** argv) {
	VideoCapture capture(0);

	Mat frame;
	char k;

	namedWindow("video", 1);

	while (true) {
		capture >> frame;

		SegmentationResult segmentationResult = segmentator.segmentImage(frame);
		IrisTemplate template_ = gaborEncoder.generateTemplate(&((IplImage)frame), segmentationResult);

		decorator.drawSegmentationResult(frame, segmentationResult);
		decorator.drawTemplate(frame, template_);


		cout << "T: " << segmentator.segmentationTime << " ms" << endl;

		imshow("video", frame);

		if ('q' == (k = waitKey(20)) ) {
			break;
		}
	}
}
