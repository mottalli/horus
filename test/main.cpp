#include <iostream>
#include <bitset>
#include <opencv/highgui.h>
#include <opencv/cv.h>
#include <iomanip>
#include <boost/thread.hpp>

#include "horus.h"

using namespace std;
using namespace cv;
using namespace horus;

Segmentator segmentator;
VideoProcessor videoProcessor;
Decorator decorator;
LogGaborEncoder encoder;

void procesarImagen(Mat& imagen)
{
	Mat imagenBW;
	cvtColor(imagen, imagenBW, CV_BGR2GRAY);
	SegmentationResult sr = segmentator.segmentImage(imagenBW);
	decorator.drawSegmentationResult(imagen, sr);

	IrisTemplate irisTemplate = encoder.generateTemplate(imagenBW, sr);
	decorator.drawTemplate(imagen, irisTemplate);

	imshow("imagen", imagen);
}

int main(int, char**)
{
	/*VideoCapture cap(0);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 720);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 576);

	Mat tmp, frame;

	do {
		cap >> tmp;
		flip(tmp, tmp, 1);
		frame = tmp(Rect(10, 0, tmp.cols-2*10, tmp.rows));

		procesarImagen(frame);

;
	} while (char(waitKey(5)) != 'q');*/

	Mat imagen = imread("/home/marcelo/iris/horus/base-iris/15_2.jpg", 1);
	procesarImagen(imagen);
	while (char(waitKey(5)) != 'q') {};

	return 0;
}
