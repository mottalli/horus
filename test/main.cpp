#include <iostream>
#include <bitset>
#include <opencv/highgui.h>
#include <opencv/cv.h>
#include <iomanip>

#include "segmentator.h"
#include "videoprocessor.h"
#include "decorator.h"
#include "templatecomparator.h"

using namespace std;
using namespace cv;

Segmentator segmentator;
VideoProcessor videoProcessor;
Decorator decorator;

void drawFocusScores(const vector<double>& focusScores, Mat image, Rect rect, double threshold = 0);
void calcularCuadroDistancias(const vector<IrisTemplate>& templates);

const string GREEN = "\033[32m";
const string RED = "\033[31m";
const string BLACK = "\033[0m";

int main(int, char**)
{
	//VideoCapture cap("/home/marcelo/iris/BBDD/Videos/bursztyn1/20080501-230748.mpg");
	//VideoCapture cap("/home/marcelo/iris/BBDD/Videos/marcelo1/marcelo1.mpg");
	//VideoCapture cap("/home/marcelo/iris/BBDD/Videos/marta1/20080702-232946.mpg");
	VideoCapture cap("/home/marcelo/iris/BBDD/Videos/norberto1/20080501-230608.mpg");
	//VideoCapture cap("/home/marcelo/iris/BBDD/Videos/norberto2/20080501-231028.mpg");
	Mat frame;

	namedWindow("video");
	namedWindow("template");
	videoProcessor.setWaitingFrames(0);

	vector<double> focusScores;
	vector<IrisTemplate> templates;

	while (true) {
		cap >> frame;

		if (frame.empty()) break;

		VideoProcessor::VideoStatus status = videoProcessor.processFrame(frame);
		if (status >= VideoProcessor::FOCUSED_IRIS) {
			decorator.drawSegmentationResult(frame, videoProcessor.lastSegmentationResult);
			decorator.drawTemplate(frame, videoProcessor.lastTemplate);
		}


		if (status == VideoProcessor::GOT_TEMPLATE) {
			IrisTemplate irisTemplate = videoProcessor.getAverageTemplate();
			decorator.drawTemplate(frame, irisTemplate);
			imshow("template", frame);
			templates.push_back(irisTemplate);
		}

		focusScores.push_back(videoProcessor.lastFocusScore);
		drawFocusScores(focusScores, frame, Rect(200, 500, 300, 50), videoProcessor.parameters.focusThreshold);

		imshow("video", frame);


		if ( char(waitKey(20)) == 'q') break;
	}

	calcularCuadroDistancias(templates);

	//waitKey(0);
}

void drawFocusScores(const vector<double>& focusScores, Mat image, Rect rect, double threshold)
{
	image(rect) = Scalar(255,255,255);
	rectangle(image, rect, Scalar(0,0,0), 1);

	if (threshold > 0) {
		double y = rect.height - ((threshold/100.0) * rect.height);
		double yimg = rect.y + y;
		line(image, Point(rect.x, yimg), Point(rect.x+rect.width, yimg), Scalar(255,0,0));
	}

	Point lastPoint;

	for (int i = focusScores.size()-1; i >= 0 && (rect.width - (focusScores.size()-1-i)) > 0; i--) {
		double focusScore = focusScores[i];
		double x = rect.width - (focusScores.size()-1-i);
		double y = rect.height - ((focusScore/100.0) * rect.height);

		Point pimg = rect.tl() + Point(x,y);
		if (lastPoint.x == 0 && lastPoint.y == 0) {
			lastPoint = pimg;
		}

		line(image, lastPoint, pimg, Scalar(0,0,255));
		lastPoint = pimg;
	}
}

void calcularCuadroDistancias(const vector<IrisTemplate>& templates)
{
	cout.setf(ios::fixed, ios::floatfield);

	for (int i = 0; i < templates.size(); i++) {
		TemplateComparator comparator(templates[i]);
		for (int j = 0; j < templates.size(); j++) {
			if (i == j) {
				cout << setw(10) << '-';
				continue;
			}

			double dist = comparator.compare(templates[j]);

			cout << setw(10) << setprecision(3) << (dist < 0.35 ? GREEN : RED) << dist << BLACK;
		}
		cout << endl;
	}
}
