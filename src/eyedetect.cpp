#include <fstream>

#include "eyedetect.h"
#include "eyedetect_haarcascades.h"

using namespace horus;

EyeDetect::EyeDetect()
{
	char *tempFilename = tempnam(NULL, "haarcascade");
	ofstream tempFile(tempFilename);
	if (!tempFile.is_open()) {
		throw runtime_error(string("Couldn't open temp file ") + tempFilename);
	}

	tempFile << haarcascadeEye;
	tempFile << endl;
	tempFile.close();

	this->eyeClassifier.load(string(tempFilename));
	if (this->eyeClassifier.empty()) {
		throw runtime_error(string("Unable to load eye classifier from file ") + tempFilename);
	}

	unlink(tempFilename);

	this->eyeRect = Rect();
}

EyeDetect::~EyeDetect()
{

}

bool EyeDetect::detectEye(const GrayscaleImage& image)
{
	pyrDown(image, this->pyramid);

	this->eyeRect = Rect();

	vector<Rect> classifications;
	Size minSize = Size(double(this->pyramid.cols)*this->parameters.minSizeProp, double(this->pyramid.rows)*this->parameters.minSizeProp);
	this->eyeClassifier.detectMultiScale(this->pyramid, classifications, this->parameters.scaleFactor, this->parameters.minNeighbors, 0, minSize);

	if (classifications.size() == 0) {
		return false;
	}

	Rect c = classifications[0];
	int x = 2*c.x;
	int y = 2*c.y;
	int w = 2*c.width;
	int h = 2*c.height;
	if (x+w >= image.cols) w = image.cols-x-1;
	if (y+h >= image.rows) h = image.rows-y-1;

	this->eyeRect = Rect(x,y,w,h);

	return true;
}
