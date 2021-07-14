#include <iostream>
#include <opencv/highgui.h>
#include <opencv/cv.h>
#include <iomanip>

#include "../src/horus/horus.h"

using namespace std;
using namespace cv;
using namespace horus;

Segmentator segmentator;
VideoProcessor videoProcessor;
Decorator decorator;
LogGaborEncoder encoder;

pair<IrisTemplate, TemplateComparator> procesarImagen(Mat& imagen)
{
	GrayscaleImage imagenBW;
    if (imagen.channels() == 3) {
    	cvtColor(imagen, imagenBW, CV_BGR2GRAY);
    } else {
        imagenBW = imagen.clone();
    }
	//horus::tools::stretchHistogram(imagenBW, imagenBW, 0.01, 0.01);
	SegmentationResult sr = segmentator.segmentImage(imagenBW);
	decorator.drawSegmentationResult(imagen, sr);

    std::cout << sr.irisCircle.center.x << "," << sr.irisCircle.center.y << "-" << sr.irisCircle.radius << std::endl;

	IrisTemplate irisTemplate = encoder.generateTemplate(imagenBW, sr);
	decorator.drawTemplate(imagen, irisTemplate);

#if 1
    imshow("imagen", imagen);
    imshow("similarity", segmentator.pupilSegmentator.similarityImage);
    while (char(waitKey(5)) != 'q') {}
#endif

    TemplateComparator comparator(irisTemplate);
    return make_pair(irisTemplate, comparator);
}

#if 0
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
	} while (char(waitKey(5)) != 'q');

	return 0;
}
#else
int main(int argc, char** argv)
{
    vector<TemplateComparator> comparators;
    vector<IrisTemplate> templates;
    for (int i = 1; i < argc; i++) {
        cv::Mat image = cv::imread(argv[i]);
        if (image.empty()) {
            throw std::runtime_error("Unable to open image!");
        }

        pair<IrisTemplate, TemplateComparator> proc = procesarImagen(image);
        templates.push_back(proc.first);
        comparators.push_back(proc.second);
    }

    int w = 10;

    size_t n = comparators.size();
    cout << "/" << setw(w);
    for (size_t i = 0; i < n; i++)
        cout << i+1 << setw(w);
    cout << endl;

    for (size_t i = 0; i < n; i++) {
        cout << setw(0) << i+1 << setw(w);

        for (size_t j = 0; j < n; j++) {
            if (i == j) {
                cout << "-" << setw(w);
                continue;
            }

            double distance = comparators[j].compare(templates[i]);


            cout << setprecision(5) << distance << setw(w);
        }

        cout << endl;
    }
}

#endif
