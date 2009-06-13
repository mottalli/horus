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
	/*const string path_video = "/home/marcelo/Mis_Documentos/Facu/Tesis/Bases de datos/Videos/marcelo1";
	const unsigned cantidad_imagenes = 227;*/
	const string path_video = "/home/marcelo/Mis_Documentos/Facu/Tesis/Bases de datos/Videos/bursztyn1";
	IplImage* frame;
	Segmentator segmentator;
	IrisEncoder encoder;
	Decorator decorator;
	VideoProcessor videoprocessor;
	Parameters* parameters = Parameters::getParameters();
	parameters->templateWidth = 1024;
	parameters->templateHeight = 1024/5;

	cvNamedWindow("imagen");
	cvNamedWindow("normalizada");

	for (unsigned numero_imagen = 1; ; numero_imagen++) {
		char path[100];
		sprintf(path, "%s/%i.jpg", path_video.c_str(), numero_imagen);
		cout << path << endl;
		frame = cvLoadImage(path);
		if (!frame) {
			cout << "NO!" << endl;
			break;
		}

		SegmentationResult res = segmentator.segmentImage(frame);
		encoder.generateTemplate(frame, res);

		decorator.drawSegmentationResult(frame,  res);


		cvShowImage("normalizada", encoder.buffers.normalizedTexture);

		cout << videoprocessor.imageQuality(encoder.buffers.normalizedTexture) << endl;

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

