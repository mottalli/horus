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

int main(int, char**)
{
	VideoCapture cap(0);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 720);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 576);

	Segmentator segmentator;
	VideoProcessor videoProcessor;
	Decorator decorator;
	LogGaborEncoder encoder;

	Mat tmp, frame;
	GrayscaleImage frameBW;
	//Mat1b imagen = imread("/home/marcelo/iris/horus/base-iris/15_2.jpg", 0);

	do {
		cap >> tmp;
		flip(tmp, tmp, 1);
		frame = tmp(Rect(10, 0, tmp.cols-2*10, tmp.rows));
		cvtColor(frame, frameBW, CV_BGR2GRAY);

		SegmentationResult sr = segmentator.segmentImage(frameBW);
		decorator.drawSegmentationResult(frame, sr);
		//cout << segmentator.segmentationTime << endl;

		IrisTemplate irisTemplate = encoder.generateTemplate(frameBW, sr);
		decorator.drawTemplate(frame, irisTemplate);

		imshow("video", frame);
	} while (char(waitKey(5)) != 'q');


	return 0;
}
