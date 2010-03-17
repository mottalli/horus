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

int main()
{
	IplImage* texture = cvLoadImage("/home/marcelo/Desktop/prueba.bmp");
	const char* imagePath = "/home/marcelo/iris/BBDD/UBA/marcelo_der_1.bmp";

	IplImage* image = cvLoadImage(imagePath, 1);

	SegmentationResult res = segmentator.segmentImage(image);
	/*decorator.drawEncodingZone(image, res);
	decorator.drawSegmentationResult(image, res);*/
	Tools::superimposeTexture(image, texture, res, IrisEncoder::THETA0, IrisEncoder::THETA1, IrisEncoder::RADIUS_TO_USE);

	cvNamedWindow("Test");
	cvShowImage("Test", image);
	while (true){
		char c = cvWaitKey(0);
		if (c == 'q') break;

	}

	return 0;

}

int main1(int argc, char** argv) {
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
