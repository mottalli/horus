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
Mat_<uint8_t> normalizarImagen(const Mat& imagen);


Segmentator segmentator;
Decorator decorator;
LogGaborEncoder logGaborEncoder;
//IrisDCTEncoder irisDCTEncoder;
Parameters* parameters = Parameters::getParameters();
GaborEncoder gaborEncoder;
VideoProcessor videoProcessor;
QualityChecker qualityChecker;


int main1(int, char**) {
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

int main2(int, char**) {
	VideoCapture capture(0);
	parameters->bestFrameWaitCount = 0;

	Mat frame;
	char k;

	namedWindow("video", 1);
	namedWindow("debug1", 1);
	namedWindow("debug2", 1);

	videoProcessor.setWaitingFrames(0);

	while (true) {
		capture >> frame;

		const int dx= 60, dy=20;
		frame = frame(Rect(dx, dy, frame.cols-dx, frame.rows-dy));

		VideoProcessor::VideoStatus status = videoProcessor.processFrame(frame);

		Mat frameOriginal = frame.clone();

		std::stringstream strStatus;

		strStatus << statusToString(status) << " ";
		strStatus << "Foco: " << videoProcessor.lastFocusScore << " ";
		strStatus << "Cal. iris: " << videoProcessor.lastIrisQuality << " ";
		putText(frame, strStatus.str(), Point(20, frame.rows-40), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,255,255), 2);

		if (status >= VideoProcessor::FOCUSED_NO_IRIS) {
			decorator.drawSegmentationResult(frame, videoProcessor.lastSegmentationResult);
		}

		if (status == VideoProcessor::GOT_TEMPLATE) {
			IrisTemplate irisTemplate = logGaborEncoder.generateTemplate(frame, videoProcessor.lastSegmentationResult);
			decorator.drawTemplate(frame, irisTemplate);
		}

		imshow("video", frame);

		//imshow("debug1", videoProcessor.segmentator.pupilSegmentator.similarityImage);
		//imshow("debug2", videoProcessor.segmentator.pupilSegmentator.equalizedImage);

		k = waitKey(10);

		if (k == 'q') {
			break;
		} else if (k == 's') {
			imwrite("/home/marcelo/Desktop/iris_capturado.jpg", frameOriginal);
		}
	}

	return 0;
}

int main(int, char**) {
	Mat imagen = imread("/home/marcelo/iris/horus/ui/_base/984/984.jpg", 1);
	Mat imagenBW;

	cvtColor(imagen, imagenBW, CV_BGR2GRAY);

	SegmentationResult sr = segmentator.segmentImage(imagenBW);
	IrisTemplate irisTemplate = logGaborEncoder.generateTemplate(imagenBW, sr);

	cout << "Foco: " << qualityChecker.checkFocus(imagenBW) << endl;
	cout << "Calidad de iris: " << qualityChecker.getIrisQuality(imagenBW, sr) << endl;

	Mat tmp;

	// -- Imagen c/similaridad --
	namedWindow("similaridad", 1);
	imshow("similaridad", segmentator.pupilSegmentator.similarityImage);

	// -- Anillo de ajuste --
	namedWindow("ajuste", 1);
	const Mat_<float>& snake = segmentator.pupilSegmentator.adjustmentSnake;
	cvtColor(segmentator.pupilSegmentator.adjustmentRing, tmp, CV_GRAY2BGR);
	for (int x = 0; x < snake.cols; x++) {
		circle(tmp, Point(x, snake(0, x)), 1, CV_RGB(255,0,0));
	}
	imshow("ajuste", tmp);

	// -- Gradiente anillo de ajuste --
	namedWindow("gradiente", 1);
	imshow("gradiente", normalizarImagen(segmentator.pupilSegmentator.adjustmentRingGradient));


	// -- Imagen segmentada --
	decorator.drawSegmentationResult(imagen, sr);
	decorator.drawTemplate(imagen, irisTemplate);
	namedWindow("decorada", 1);
	imshow("decorada", imagen);

	for (;;) {
		char k = waitKey(0);
		if (k == 'q') {
			break;
		}
	}

	return 0;
}

int main4(int, char**) {
	VideoCapture capture(0);
	parameters->bestFrameWaitCount = 0;

	Mat imagen, imagenBW, tmp;
	char k;

	while (true) {
		capture >> imagen;

		const int dx= 60, dy=20;
		imagen= imagen(Rect(dx, dy, imagen.cols-dx, imagen.rows-dy));

		cvtColor(imagen, imagenBW, CV_BGR2GRAY);

		SegmentationResult sr = segmentator.segmentImage(imagenBW);

		// -- Imagen c/similaridad --
		namedWindow("similaridad", 1);
		imshow("similaridad", segmentator.pupilSegmentator.similarityImage);

		// -- Anillo de ajuste --
		namedWindow("ajuste", 1);
		const Mat_<float>& snake = segmentator.pupilSegmentator.adjustmentSnake;
		cvtColor(segmentator.pupilSegmentator.adjustmentRing, tmp, CV_GRAY2BGR);
		for (int x = 0; x < snake.cols; x++) {
			circle(tmp, Point(x, snake(0, x)), 1, CV_RGB(255,0,0));
		}
		imshow("ajuste", tmp);

		// -- Gradiente anillo de ajuste --
		namedWindow("gradiente", 1);
		imshow("gradiente", normalizarImagen(segmentator.pupilSegmentator.adjustmentRingGradient));


		// -- Imagen segmentada --
		decorator.drawSegmentationResult(imagen, sr);
		namedWindow("decorada", 1);
		imshow("decorada", imagen);

		k = waitKey(10);
		if (k == 'q') {
			break;
		}
	}

	return 0;

}

int main5(int, char**)
{
	//Mat_<uint8_t> imagen = imread("/home/marcelo/iris/horus/ui/_base/982.jpg", 0);
	Mat frame;
	Mat_<uint8_t> imagen;

	VideoCapture capture(0);

	int x0, x1, y0, y1, x, y;
	unsigned int mean;

	while (true) {
		capture >> frame;

		cvtColor(frame, imagen, CV_BGR2GRAY);

		for (x0 = 0, mean = 0; mean < 100 && x0 < imagen.cols; x0++) {
			for (y = 0; y < imagen.rows; y++) {
				mean += int(imagen(y, x0));
			}
			mean = mean/imagen.rows;
		}

		for (x1 = x0+1; mean >= 100 && x1 < imagen.cols; x1++) {
			for (y = 0; y < imagen.rows; y++) {
				mean += int(imagen(y, x1));
			}
			mean = mean/imagen.rows;
		}

		for (y0 = 0, mean = 0; mean < 100 && y0 < imagen.rows; y0++) {
			for (x = 0; x < imagen.cols; x++) {
				mean += int(imagen(y0, x));
			}
			mean = mean/imagen.cols;
		}

		for (y1 = y0+1; mean >= 100 && y1 < imagen.rows; y1++) {
			for (x = 0; x < imagen.cols; x++) {
				mean += int(imagen(y1, x));
			}
			mean = mean/imagen.cols;
		}

		line(imagen, Point(x0, 0), Point(x0, imagen.rows-1), CV_RGB(255,255,255), 1);
		line(imagen, Point(x1, 0), Point(x1, imagen.rows-1), CV_RGB(255,255,255), 1);
		line(imagen, Point(0, y0), Point(imagen.cols-1, y0), CV_RGB(255,255,255), 1);
		line(imagen, Point(0, y1), Point(imagen.cols-1, y1), CV_RGB(255,255,255), 1);


		namedWindow("imagen", 1);
		imshow("imagen", imagen);

		if (waitKey(10) == 'q') {
			break;
		}
	}

}





Mat_<uint8_t> normalizarImagen(const Mat& imagen)
{
	Mat res;
	normalize(imagen, res, 0, 255, NORM_MINMAX);

	return res;
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
