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
GaborEncoder gaborEncoder;
VideoProcessor videoProcessor;
QualityChecker qualityChecker;

vector<string> archivos;


/**
 * PROCESAR DOS IMÁGENES
 */
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

/**
 * ANÁLISIS DE FOCO Y CALIDAD
 */
int main2(int, char**) {
	VideoCapture capture(0);

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

		cout << "Tiempo de segmentacion: " << videoProcessor.segmentator.segmentationTime << endl;

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

int main3(int, char**) {

	vector<string> aProcesar;

	//aProcesar = archivos;
	aProcesar.push_back("/home/marcelo/iris/horus/ui/_base/999/999_3.jpg");


	for (vector<string>::iterator it = aProcesar.begin(); it != aProcesar.end(); it++) {
		cout << *it << endl;
		Mat imagen = imread(*it, 1);
		Mat_<uint8_t> imagenBW;

		cvtColor(imagen, imagenBW, CV_BGR2GRAY);
		//Tools::stretchHistogram(imagenBW, imagenBW);

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
		decorator.drawSegmentationResult(imagenBW, sr);
		decorator.drawTemplate(imagenBW, irisTemplate);
		namedWindow("decorada", 1);
		imshow("decorada", imagenBW);

		while (true) if (char(waitKey(0)) == 'q') break;
	}


	return 0;
}

int main4(int, char**) {
	VideoCapture capture(0);

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

		k = waitKey(30);
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
	int mean;

	const int THRESH = 60;

	while (true) {
		capture >> frame;

		cvtColor(frame, imagen, CV_BGR2GRAY);
		GaussianBlur(imagen, imagen, Size(7,7), 0);

		for (x0 = 0, mean = 0; mean < THRESH && x0 < imagen.cols; x0++) {
			for (y = 0; y < imagen.rows; y++) {
				mean += int(imagen(y, x0));
			}
			mean = mean/imagen.rows;
		}

		cout << "x0 " << mean << endl;

		for (x1 = imagen.cols, mean=0; mean < THRESH && x1 > x0+1; x1--) {
			for (y = 0; y < imagen.rows; y++) {
				mean += int(imagen(y, x1));
			}
			mean = mean/imagen.rows;
		}

		cout << "x1 " << mean << endl;

		for (y0 = 0, mean = 0; mean < THRESH && y0 < imagen.rows; y0++) {
			for (x = 0; x < imagen.cols; x++) {
				mean += int(imagen(y0, x));
			}
			mean = mean/imagen.cols;
		}

		cout << "y0 " << mean << endl;

		for (y1 = imagen.rows, mean=0; mean < THRESH && y1 > y0+1; y1--) {
			for (x = 0; x < imagen.cols; x++) {
				mean += int(imagen(y1, x));
			}
			mean = mean/imagen.cols;
		}

		cout << "y1 " << mean << endl;

		line(imagen, Point(x0, 0), Point(x0, imagen.rows-1), CV_RGB(255,255,255), 1);
		line(imagen, Point(x1, 0), Point(x1, imagen.rows-1), CV_RGB(255,255,255), 1);
		line(imagen, Point(0, y0), Point(imagen.cols-1, y0), CV_RGB(255,255,255), 1);
		line(imagen, Point(0, y1), Point(imagen.cols-1, y1), CV_RGB(255,255,255), 1);


		namedWindow("imagen", 1);
		imshow("imagen", imagen);

		if (char(waitKey(30)) == 'q') {
			break;
		}
	}

	return 0;
}

int main6(int, char**)
{
	for (vector<string>::iterator it = archivos.begin(); it != archivos.end(); it++) {
		string archivo = *it;

		Mat imagen = imread(archivo, 1);
		Mat_<uint8_t> imagenBW, normalizada(Size(512,40)), mascaraNormalizada(Size(512,40)), sobel;

		cvtColor(imagen, imagenBW, CV_BGR2GRAY);

		SegmentationResult sr = segmentator.segmentImage(imagen);
		IrisEncoder::normalizeIris(imagenBW, normalizada, mascaraNormalizada, sr, IrisEncoder::THETA0, IrisEncoder::THETA1, IrisEncoder::RADIUS_TO_USE);

		Sobel(normalizada, sobel, CV_8U, 1, 1, 7);

		MatConstIterator_<uint8_t> it1, it2;
		int total2 = 0, sobel2 = 0;
		for (it1 = normalizada.begin(), it2=sobel.begin(); it1 != normalizada.end(); it1++, it2++) {
			int v1 = int(*it1), v2 = int(*it2);

			total2 += v1*v1;
			sobel2 += v2*v2;
		}

		float foco = float(sobel2)/float(sobel2 + total2);
		cout << "Foco: " << foco << endl;



		decorator.drawSegmentationResult(imagen, sr);
		decorator.drawEncodingZone(imagen, sr);

		namedWindow("imagen");
		imshow("imagen", imagen);

		namedWindow("normalizada");
		imshow("normalizada", normalizada);

		namedWindow("sobel");
		imshow("sobel", sobel);

		while (true) if (char(waitKey(0)) == 'q') break;
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

// PROCESAR UNA UNICA IMAGEN
int main8(int, char**)
{
	Mat imagen = imread(archivos[3]);
	Mat imagenBW;

	cvtColor(imagen, imagenBW, CV_BGR2GRAY);

	SegmentationResult sr = segmentator.segmentImage(imagenBW);
	IrisTemplate logGaborTemplate = logGaborEncoder.generateTemplate(imagenBW, sr);

	decorator.drawSegmentationResult(imagen, sr);
	decorator.drawEncodingZone(imagen, sr);
	decorator.drawTemplate(imagen, logGaborTemplate);

	namedWindow("imagen", 1);
	imshow("imagen", imagen);

	while (char(waitKey(0)) != 'q') {};

	Mat_<uint8_t> binaryMatrix = logGaborTemplate.getUnpackedTemplate();
	//Mat_<uint8_t> binaryMatrix = logGaborTemplate.getUnpackedMask();
	uint8_t v = 0;
	for (int y = 0; y < binaryMatrix.rows; y++) {
		int acum = 0;
		for (int x = 0; x < binaryMatrix.cols; x++) {
			if (binaryMatrix(y, x) != v) {
				cout << acum << ' ';
				v = binaryMatrix(y, x);
				acum = 0;
			} else {
				acum++;
			}
		}
		cout << endl;
	}

	return 0;
}

int main8(int, char**)
{
	/*Mat imagen1 = imread(archivos[9], 0);
	Mat imagen2 = imread(archivos[3], 0);*/

	Mat imagen1 = imread(archivos[9], 0);
	Mat imagen2 = imread(archivos[3], 0);

	SegmentationResult sr1 = segmentator.segmentImage(imagen1);
	SegmentationResult sr2 = segmentator.segmentImage(imagen2);

	IrisTemplate template1 = logGaborEncoder.generateTemplate(imagen1, sr1);
	IrisTemplate template2 = logGaborEncoder.generateTemplate(imagen2, sr2);
	TemplateComparator comparator1(template1);
	TemplateComparator comparator2(template2);

	Mat textura1(Size(512, 80), CV_8UC1), textura2(Size(512, 80), CV_8UC1), mascara1, mascara2;

	IrisEncoder::normalizeIris(imagen1, textura1, mascara1, sr1);
	IrisEncoder::normalizeIris(imagen2, textura2, mascara2, sr2);

	imshow("imagen1", imagen1);
	imshow("imagen2", imagen2);

	Tools::superimposeTexture(imagen1, textura2, sr1);
	Tools::superimposeTexture(imagen2, textura1, sr2);

	imshow("superpuesta1", imagen1);
	imshow("superpuesta2", imagen2);

	sr1 = segmentator.segmentImage(imagen1);
	sr2 = segmentator.segmentImage(imagen2);

	IrisTemplate templateSup1 = logGaborEncoder.generateTemplate(imagen1, sr1);
	IrisTemplate templateSup2 = logGaborEncoder.generateTemplate(imagen2, sr2);
	TemplateComparator comparatorSup1(templateSup1);
	TemplateComparator comparatorSup2(templateSup2);

	cout << "HD template 1, template 2:" << comparator1.compare(template2) << endl;
	cout << "HD template sup 1, template 1:" << comparatorSup1.compare(template1) << endl;
	cout << "HD template sup 1, template 2:" << comparatorSup1.compare(template2) << endl;
	cout << "HD template sup 2, template 1:" << comparatorSup2.compare(template1) << endl;
	cout << "HD template sup 2, template 2:" << comparatorSup2.compare(template2) << endl;

	while (true) if (char(waitKey(0)) == 'q') break;

	return 0;
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
	return main8(argc, argv);
}

