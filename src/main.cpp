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

vector<string> archivos;


int main1(int, char**)
{

	vector<string> imagenes = archivos;
	for (vector<string>::iterator it = imagenes.begin(); it != imagenes.end(); it++) {
		Mat_<uint8_t> imagen = imread(*it, 0);
		cout << *it << endl;

		SegmentationResult sr = segmentator.segmentImage(imagen);

		int x0 = sr.irisCircle.xc, y0 = sr.irisCircle.yc, radius = sr.irisCircle.radius;
		Rect rect(x0-radius, y0-radius, 2*radius, 2*radius);

		Mat_<uint8_t> porcion = imagen(rect).clone();
		threshold(porcion, porcion, parameters->infraredThreshold, 255, THRESH_BINARY);


		decorator.drawSegmentationResult(imagen, sr);

		namedWindow("porcion", 1);
		imshow("porcion", porcion);

		namedWindow("ojo", 1);
		imshow("ojo", imagen(rect));

		namedWindow("imagen", 1);
		imshow("imagen", imagen);

		while (char(waitKey(0)) != 'q') ;
	}
}

int main2(int, char**)
{
	VideoCapture cap(0);
	Mat frame;
	Mat_<uint8_t> frameBW;
	Mat_<uint8_t> previousTemplate, currentTemplate;
	namedWindow("video", 1);

	parameters->waitingFrames = 0;
	parameters->bestFrameWaitCount = 1;

	while (true) {
		cap >> frame;
		cvtColor(frame, frameBW, CV_BGR2GRAY);

		VideoProcessor::VideoStatus status = videoProcessor.processFrame(frameBW);
		cout << statusToString(status) << endl;

		rectangle(frameBW, videoProcessor.eyeROI, CV_RGB(255,255,255));
		decorator.drawSegmentationResult(frameBW, videoProcessor.lastSegmentationResult);

		if (status == VideoProcessor::GOT_TEMPLATE) {
			IrisTemplate irisTemplate = videoProcessor.getTemplate();
			currentTemplate = irisTemplate.getTemplateImage();

			decorator.drawTemplate(frameBW, irisTemplate);
			Tools::superimposeTexture(frameBW, currentTemplate, videoProcessor.getTemplateSegmentation(), IrisEncoder::THETA0, IrisEncoder::THETA1, IrisEncoder::RADIUS_TO_USE);

			if (!previousTemplate.empty()) {
				namedWindow("diff");
				Mat tmp;
				bitwise_xor(currentTemplate, previousTemplate, tmp);
				imshow("diff", tmp);

			}

			previousTemplate = currentTemplate.clone();

		}

		imshow("video", frameBW);
		if (char(waitKey(30)) == 'q') break;
	}
}

int main7(int, char**)
{
	for (vector<string>::iterator it = archivos.begin(); it != archivos.end(); it++) {
		cout << *it << endl;
		string archivo = *it;

		Mat imagen = imread(archivo, 1);
		Mat_<uint8_t> imagenBW, normalizada(Size(512,40)), mascaraNormalizada(Size(512,40)), sobel;

		cvtColor(imagen, imagenBW, CV_BGR2GRAY);

		SegmentationResult sr = segmentator.segmentImage(imagenBW);
		IrisEncoder::normalizeIris(imagenBW, normalizada, mascaraNormalizada, sr, IrisEncoder::THETA0, IrisEncoder::THETA1, IrisEncoder::RADIUS_TO_USE);

		int x0 = sr.irisCircle.xc-sr.irisCircle.radius;
		int y0 = sr.irisCircle.yc-sr.irisCircle.radius;
		int x1 = x0 + 2*sr.irisCircle.radius;
		int y1 = y0 + 2*sr.irisCircle.radius;

		float delta = 0.1;
		x0 = (1-delta)*x0;
		y0 = (1-delta)*y0;
		x1 = (1+delta)*x1;
		y1 = (1+delta)*y1;
		Rect porcion(Point(x0, y0), Point(x1, y1));

		//Sobel(normalizada, sobel, CV_8U, 1, 1, 7);
		Sobel(imagenBW(porcion), sobel, CV_8U, 1, 1, 7);

		MatConstIterator_<uint8_t> it1, it2;
		int total2 = 0, sobel2 = 0;
		for (it1 = normalizada.begin(), it2=sobel.begin(); it1 != normalizada.end(); it1++, it2++) {
			int v1 = int(*it1), v2 = int(*it2);

			total2 += v1*v1;
			sobel2 += v2*v2;
		}

		float foco = float(sobel2)/float(sobel2 + total2);
		cout << "Foco: " << foco << endl;



		decorator.drawSegmentationResult(imagenBW, sr);
		decorator.drawEncodingZone(imagenBW, sr);
		rectangle(imagenBW, porcion, CV_RGB(255,255,255));

		namedWindow("imagen");
		imshow("imagen", imagenBW);

		namedWindow("normalizada");
		imshow("normalizada", normalizada);

		namedWindow("sobel");
		imshow("sobel", sobel);

		while (true) if (char(waitKey(0)) == 'q') break;
	}
}

string statusToString(VideoProcessor::VideoStatus status)
{
	string strStatus;
	switch (status) {
	case VideoProcessor::UNPROCESSED:
		strStatus = "UNPROCESSED";
		break;
	case VideoProcessor::NO_EYE:
		strStatus = "NO_EYE";
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

int main(int argc, char** argv)
{
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/977/977.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/1002/1002_2.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/1002/1002_1.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/1002/1002.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/987/987.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/988/988.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/997/997.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/997/997_5.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/997/997_1.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/997/997_2.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/997/997_4.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/997/997_3.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/1004/1004_1.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/1004/1004_3.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/1004/1004.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/1004/1004_2.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/1003/1003.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/1003/1003_1.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/984/984.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/1000/1000_1.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/1000/1000_2.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/1000/1000.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/981/981_1.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/981/981_4.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/981/981_2.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/981/981.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/981/981_5.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/981/981_3.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/989/989.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/982/982.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/980/980_1.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/980/980_2.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/980/980.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/978/978_6.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/978/978_3.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/978/978_4.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/978/978_7.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/978/978_2.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/978/978.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/978/978_5.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/978/978_1.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/992/992.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/998/998_3.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/998/998_4.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/998/998_1.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/998/998_2.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/998/998.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/979/979.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/990/990.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/996/996_4.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/996/996.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/996/996_1.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/996/996_3.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/996/996_5.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/996/996_2.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/985/985.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/1001/1001.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/1001/1001_1.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/983/983.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/993/993.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/999/999_3.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/999/999_4.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/999/999_1.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/999/999.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/999/999_2.jpg");
	archivos.push_back("/home/marcelo/iris/horus/ui/_base/986/986.jpg");

	// CAMBIAR ESTA LLAMADA
	return main2(argc, argv);
}
