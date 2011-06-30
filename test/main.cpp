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
vector<IrisTemplate> templates;

void procesarImagen(Mat& imagen)
{
	GrayscaleImage imagenBW;
	cvtColor(imagen, imagenBW, CV_BGR2GRAY);
	SegmentationResult sr = segmentator.segmentImage(imagenBW);
	decorator.drawSegmentationResult(imagen, sr);

	IrisTemplate irisTemplate = encoder.generateTemplate(imagenBW, sr);
	decorator.drawTemplate(imagen, irisTemplate);
	templates.push_back(irisTemplate);

	size_t n = 10;
	vector<IrisTemplate> lastTemplates(n);
	if (templates.size() < n) {
		lastTemplates = templates;
	} else {
		std::copy(templates.end()-n, templates.end(), lastTemplates.begin());
	}

	IrisTemplate averageTemplate = IrisEncoder::averageTemplates(lastTemplates);
	decorator.drawTemplate(imagen, averageTemplate, Point(15, 400));

	tools::superimposeTexture(imagenBW, averageTemplate.getTemplateImage(), sr,
							  IrisEncoder::THETA0, IrisEncoder::THETA1, IrisEncoder::MIN_RADIUS_TO_USE, IrisEncoder::MAX_RADIUS_TO_USE, false);

	imshow("imagen", imagenBW);

	imshow("equalized", segmentator.pupilSegmentator.equalizedImage);
	imshow("similarity", segmentator.pupilSegmentator.similarityImage);
}

int main(int, char**)
{
	VideoCapture cap(0);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 720);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 576);

	Mat tmp, frame;
	int dx = 10, dy = 10;

	do {
		cap >> tmp;
		flip(tmp, tmp, 1);
		frame = tmp(Rect(dx, dy, tmp.cols-2*dx, tmp.rows-2*dy));

		procesarImagen(frame);

;
	} while (char(waitKey(5)) != 'q');

	/*Mat imagen = imread("/home/marcelo/iris/horus/base-iris/15_2.jpg", 1);
	procesarImagen(imagen);
	while (char(waitKey(5)) != 'q') {};*/

	return 0;
}
