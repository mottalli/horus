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
LogGaborEncoder irisEncoder;
IrisDCTEncoder irisDCTEncoder;
Parameters* parameters = Parameters::getParameters();


IplImage* beautificarTemplate(const IplImage* src)
{
	IplImage* beautif = cvCreateImage(cvSize(src->width+2, 3*src->height+2), IPL_DEPTH_8U, 1);
	cvSet(beautif, cvScalar(128));
	int width = src->width;
	for (int i = 0; i < src->height; i++) {
		CvMat tmpsrc, tmpdest;
		cvGetSubRect(src, &tmpsrc, cvRect(0, i, width, 1));
		cvGetSubRect(beautif, &tmpdest, cvRect(1, 3*i+2, width, 1));
		cvCopy(&tmpsrc, &tmpdest);
	}

	IplImage* resizeada = cvCreateImage(cvSize(beautif->width*2, beautif->height*3), IPL_DEPTH_8U, 1);
	cvResize(beautif, resizeada, CV_INTER_NN);

	cvReleaseImage(&beautif);

	return resizeada;
}

int main(int argc, char** argv) {
	const char* imagePath = "/home/marcelo/iris/BBDD/UBA/marcelo_der_2.bmp";
	IplImage* image = cvLoadImage(imagePath, 0);

    SegmentationResult res = segmentator.segmentImage(image);
	GaborEncoder gaborEncoder;

	IrisTemplate template_ = gaborEncoder.generateTemplate(image, res);

	IplImage* imTemplate = template_.getTemplateImage();
	cvNamedWindow("templ");
	cvShowImage("templ", beautificarTemplate(imTemplate));


	/*IplImage* normalized = cvCreateImage(cvSize(512, 96), IPL_DEPTH_8U, 1);
	irisEncoder.normalizeIris(image, normalized, NULL, res, 0, 2.0*M_PI, 1);

	GaborFilter f(11, 11, 0.5, 0.5, 1.5, 1.5, GaborFilter::FILTER_IMAG);
	IplImage* filtered = cvCreateImage(cvGetSize(normalized), IPL_DEPTH_8U, 1);
	CvMat* foo = cvCreateMat(1,1,CV_8U);
	f.applyFilter(normalized, filtered, foo, foo);

	IplImage* im = cvCreateImage(cvGetSize(filtered), IPL_DEPTH_8U, 1);
	cvNormalize(filtered, im, 0, 255, CV_MINMAX);

	cvNamedWindow("test");
	cvShowImage("test", im);*/

	char k;
	do {
		k = cvWaitKey(0);
	} while (k != 'q');

	return 0;
}
