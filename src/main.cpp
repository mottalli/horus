/*
 * File:   main.cpp
 * Author: marcelo
 *
 * Created on January 22, 2009, 2:01 PM
 */

#include <iostream>
#include <stdio.h>
#include <sstream>

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
string statusToString(VideoProcessor::VideoStatus status);

Segmentator segmentator;
Decorator decorator;
LogGaborEncoder logGaborEncoder;
//IrisDCTEncoder irisDCTEncoder;
Parameters* parameters = Parameters::getParameters();
GaborEncoder gaborEncoder;
VideoProcessor videoProcessor;


int main1(int argc, char** argv) {
	const char* imagePath1 = "/home/marcelo/iris/BBDD/UBA/marcelo_der_1.bmp";
	const char* imagePath2 = "/home/marcelo/iris/BBDD/UBA/marcelo_der_2.bmp";
	Mat image1 = imread(imagePath1);
	Mat image2 = imread(imagePath2);

	SegmentationResult segmentationResult1 = segmentator.segmentImage(image1);
	SegmentationResult segmentationResult2 = segmentator.segmentImage(image2);

	segmentator.segmentEyelids(image1, segmentationResult1);
	segmentator.segmentEyelids(image2, segmentationResult2);

	IrisTemplate template1 = logGaborEncoder.generateTemplate(image1, segmentationResult1);
	IrisTemplate template2 = logGaborEncoder.generateTemplate(image2, segmentationResult2);
	//IrisTemplate template1 = gaborEncoder.generateTemplate(image1, segmentationResult1);
	//IrisTemplate template2 = gaborEncoder.generateTemplate(image2, segmentationResult2);

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

int main(int argc, char** argv) {
	VideoCapture capture(0);

	Mat frame;
	char k;

	namedWindow("video", 1);
	namedWindow("debug1", 1);
	namedWindow("debug2", 1);

	while (true) {
		capture >> frame;

		const int dx= 60, dy=20;
		frame = frame(Rect(dx, dy, frame.cols-dx, frame.rows-dy));

		VideoProcessor::VideoStatus status = videoProcessor.processFrame(frame);

		Mat frameOriginal = frame.clone();

		std::stringstream strStatus;

		strStatus << statusToString(status) << " ";
		strStatus << "Foco: " << videoProcessor.lastFocusScore;
		putText(frame, strStatus.str(), Point(20, 20), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,255,255), 2);

		decorator.drawSegmentationResult(frame, videoProcessor.lastSegmentationResult);
		imshow("video", frame);

		//imshow("debug1", videoProcessor.segmentator.pupilSegmentator.similarityImage);
		//imshow("debug2", videoProcessor.segmentator.pupilSegmentator.equalizedImage);

		k = waitKey(20);

		if (k == 'q') {
			break;
		} else if (k == 's') {
			imwrite("/home/marcelo/Desktop/iris_capturado.jpg", frameOriginal);
		}
	}
}

int main3(int argc, char** argv) {
	Mat imagen = imread("/home/marcelo/Desktop/iris_capturado.jpg");

	//VideoProcessor::VideoStatus status = videoProcessor.processFrame(frame);
}

string statusToString(VideoProcessor::VideoStatus status)
{
	string strStatus;
	switch (status) {
	case VideoProcessor::UNPROCESSED:
		strStatus = "UNPROCESSED";
		break;
	case VideoProcessor::DEFOCUSED:
		strStatus = "DEFOCUSED";
		break;
	case VideoProcessor::INTERLACED:
		strStatus = "INTERLACED";
		break;
	case VideoProcessor::FOCUSED_NO_IRIS:
		strStatus = "FOCUSED_NO_IRIS";
		break;
	case VideoProcessor::IRIS_LOW_QUALITY:
		strStatus = "IRIS_LOW_QUALITY";
		break;
	case VideoProcessor::IRIS_TOO_CLOSE:
		strStatus = "IRIS_TOO_CLOSE";
		break;
	case VideoProcessor::IRIS_TOO_FAR:
		strStatus = "IRIS_TOO_FAR";
		break;
	case VideoProcessor::FOCUSED_IRIS:
		strStatus = "FOCUSED_IRIS";
		break;
	case VideoProcessor::GOT_TEMPLATE:
		strStatus = "GOT_TEMPLATE";
		break;
	}

	return strStatus;
}
