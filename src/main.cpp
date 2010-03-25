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
IrisDCTEncoder irisDCTEncoder;
Parameters* parameters = Parameters::getParameters();
GaborEncoder gaborEncoder;


int main(int argc, char** argv) {
	const char* imagePath = "/home/marcelo/iris/BBDD/UBA/marcelo_der_2.bmp";
	IplImage* image = cvLoadImage(imagePath, 0);

    SegmentationResult res = segmentator.segmentImage(image);


	IrisTemplate template_ = logGaborEncoder.generateTemplate(image, res);
	//IrisTemplate template_ = gaborEncoder.generateTemplate(image, res);

	/*IplImage* imTemplate = template_.getTemplateImage();
	cvNamedWindow("templ");
	cvShowImage("templ", beautificarTemplate(imTemplate));*/

	decorator.drawSegmentationResult(image, res);
	decorator.drawTemplate(image, template_);

	cvNamedWindow("imagen");
	cvShowImage("imagen", image);


	char k;
	do {
		k = cvWaitKey(0);
	} while (k != 'q');

	return 0;
}
