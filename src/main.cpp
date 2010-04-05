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
Decorator decorator;
LogGaborEncoder logGaborEncoder;
//IrisDCTEncoder irisDCTEncoder;
Parameters* parameters = Parameters::getParameters();
GaborEncoder gaborEncoder;


int main(int argc, char** argv) {
	const char* imagePath1 = "/home/marcelo/iris/BBDD/UBA/marcelo_der_1.bmp";
	const char* imagePath2 = "/home/marcelo/iris/BBDD/UBA/marcelo_der_2.bmp";
	Mat image1 = imread(imagePath1);
	Mat image2 = imread(imagePath2);

	SegmentationResult segmentationResult1 = segmentator.segmentImage(image1);
	SegmentationResult segmentationResult2 = segmentator.segmentImage(image2);

	IrisTemplate template1 = logGaborEncoder.generateTemplate(image1, segmentationResult1);
	IrisTemplate template2 = logGaborEncoder.generateTemplate(image2, segmentationResult2);
	/*IrisTemplate template1 = gaborEncoder.generateTemplate(image1, segmentationResult1);
	IrisTemplate template2 = gaborEncoder.generateTemplate(image2, segmentationResult2);*/

	decorator.drawSegmentationResult(image1, segmentationResult1);
	decorator.drawEncodingZone(image1, segmentationResult1);
	decorator.drawTemplate(image1, template1);

	decorator.drawSegmentationResult(image2, segmentationResult2);
	decorator.drawEncodingZone(image2, segmentationResult2);
	decorator.drawTemplate(image2, template2);


	namedWindow("imagen1", 1);
	imshow("imagen1", image1);
	namedWindow("imagen2", 1);
	imshow("imagen2", image2);

	TemplateComparator comparator(template1);
	cout << "HD: " << comparator.compare(template2) << endl;

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
