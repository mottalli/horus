/*
 * File:   main.cpp
 * Author: marcelo
 *
 * Created on January 22, 2009, 2:01 PM
 */

#include <iostream>
#include <strstream>

#include "segmentator.h"
#include "decorator.h"
#include "irisencoder.h"
#include "parameters.h"
#include "videoprocessor.h"

using namespace std;

int main(int argc, char** argv) {
#if 0
	IplImage* imagen = cvLoadImage(argv[1]);

	Segmentator segmentator;
	SegmentationResult res = segmentator.segmentImage(imagen);
	IrisEncoder encoder;
	Parameters* parameters = Parameters::getParameters();
	parameters->templateWidth = 1024;
	parameters->templateHeight = 1024/5;

	encoder.generateTemplate(imagen, res);

	Decorator decorator;
	decorator.drawSegmentationResult(imagen, res);

	cvNamedWindow("imagen");
	cvShowImage("imagen", imagen);

	cvNamedWindow("normalizada");
	cvShowImage("normalizada", encoder.buffers.normalizedTexture);

	cvWaitKey(0);
	cvReleaseImage(&imagen);
#elif 1
	//const string path_video = "/home/marcelo/Mis_Documentos/Facu/Tesis/Bases de datos/Videos/marcelo1";
	string path_video = string(argv[1]);
	//const string path_video = "/home/marcelo/Mis_Documentos/Facu/Tesis/Bases de datos/Videos/bursztyn1";
	IplImage* frame;
	Segmentator segmentator;
	IrisEncoder encoder;
	Decorator decorator;
	VideoProcessor videoprocessor;
	Parameters* parameters = Parameters::getParameters();
	//parameters->templateWidth = 1024;
	//parameters->templateHeight = 1024/5;

	cvNamedWindow("imagen");
	cvMoveWindow("imagen", 900, 10);
	cvNamedWindow("normalizada");
	cvMoveWindow("normalizada", 900, 600);
	cvNamedWindow("filtrada");
	cvMoveWindow("filtrada", 900, 700);
	cvNamedWindow("histo");
	cvMoveWindow("histo", 1000, 700);

	for (unsigned numero_imagen = 1; ; numero_imagen++) {
		char path[100];
		sprintf(path, "%s/%i.jpg", path_video.c_str(), numero_imagen);
		cout << path << endl;
		frame = cvLoadImage(path, 0);
		if (!frame) {
			cout << "NO!" << endl;
			break;
		}

		SegmentationResult res = segmentator.segmentImage(frame);
		IrisTemplate irisTemplate = encoder.generateTemplate(frame, res);

		decorator.drawSegmentationResult(frame,  res);

		cvShowImage("normalizada", encoder.buffers.normalizedTexture);

		Image* foo1 = cvCreateImage(cvGetSize(encoder.buffers.filteredTexture), IPL_DEPTH_32F, 1);
		Image* foo2 = cvCreateImage(cvGetSize(encoder.buffers.filteredTexture), IPL_DEPTH_8U, 1);
		cvSplit(encoder.buffers.filteredTexture, NULL, foo1, NULL, NULL);
		//cvNormalize(foo1, foo2, 0, 255, CV_MINMAX);
		cvThreshold(foo1, foo2, 0, 255, CV_THRESH_BINARY);
		cvShowImage("filtrada", irisTemplate.getTemplate());



		Image* texture = foo1;
		int hist_size[] = {256};
		CvHistogram* hist = cvCreateHist(1, hist_size, CV_HIST_ARRAY, NULL, 1);
		double min, max;
		cvMinMaxLoc(texture, &max, &min, NULL, NULL);
		float ranges[] = { -20, 20 };
		float* foo[] = {ranges};
		cvSetHistBinRanges(hist, foo, true);
		IplImage* planes[] = {texture};
		cvCalcHist(planes, hist, false, NULL);
		float min_value, max_value;
		cvGetMinMaxHistValue(hist, &min_value, &max_value);
		Image* imgHistogram = cvCreateImage(cvSize(256, 50),8,3);
		cvRectangle(imgHistogram, cvPoint(0,0), cvPoint(256,50), CV_RGB(255,255,255),-1);
		for (int j = 0; j < 256; j++) {
			double value = cvQueryHistValue_1D(hist, j);
			double normalized = cvRound(value*50.0/max_value);
			cvLine(imgHistogram,cvPoint(j,50), cvPoint(j,50-normalized), CV_RGB(0,0,0));
		}

		CvScalar avg, std;
		cvAvgSdv(texture, &avg, &std);
		double m = avg.val[0];
		double s = std.val[0];
		cvLine(imgHistogram, cvPoint(m+s,0), cvPoint(m+s,50), CV_RGB(255,0,0));
		cvLine(imgHistogram, cvPoint(m-s,0), cvPoint(m-s,50), CV_RGB(255,0,0));
		std::cout << s << std::endl;

		cvShowImage("histo", imgHistogram);
		cvReleaseImage(&imgHistogram);
		cvReleaseImage(&foo1);
		cvReleaseImage(&foo2);

		cvShowImage("imagen", frame);
		cvWaitKey(10);
	}


#elif 0
	/*CvCapture* capture = cvCaptureFromCAM(0);
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
	}*/
#endif
}

