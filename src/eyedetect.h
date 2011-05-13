#pragma once
#include "common.h"

class EyeDetectParameters
{
public:
	double scaleFactor;
	int minNeighbors;
	double minSizeProp;

	EyeDetectParameters()
	{
		scaleFactor = 2;
		minNeighbors = 3;
		minSizeProp = 0.25;
	}
};

class EyeDetect
{
public:
	EyeDetect();
	~EyeDetect();

	bool detectEye(const GrayscaleImage& image);
	Rect eyeRect;
	EyeDetectParameters parameters;

private:
	CascadeClassifier eyeClassifier;
	GrayscaleImage pyramid;
};
