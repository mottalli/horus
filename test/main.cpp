#include <iostream>
#include <opencv/highgui.h>
#include <opencv/cv.h>

#include "segmentator.h"
#include "videoprocessor.h"
#include "decorator.h"

using namespace std;
using namespace cv;

Segmentator segmentator;
VideoProcessor videoProcessor;
Decorator decorator;


int main(int, char**)
{
	VideoCapture cap("/home/marcelo/iris/BBDD/Videos/bursztyn1/20080501-230748.mpg");
	Mat frame;

	namedWindow("video");
	namedWindow("template");
	videoProcessor.setWaitingFrames(0);

	while (true) {
		cap >> frame;

		if (frame.empty()) break;

		VideoProcessor::VideoStatus status = videoProcessor.processFrame(frame);
		if (status >= VideoProcessor::FOCUSED_IRIS) {
			decorator.drawSegmentationResult(frame, videoProcessor.lastSegmentationResult);
			decorator.drawTemplate(frame, videoProcessor.lastTemplate);
		}

		if (status == VideoProcessor::GOT_TEMPLATE) {
			imshow("template", frame);
		}

		imshow("video", frame);

		if ( char(waitKey(20)) == 'q') break;
	}
}
