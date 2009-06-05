/*
 * File:   main.cpp
 * Author: marcelo
 *
 * Created on January 22, 2009, 2:01 PM
 */

#include <iostream>

#include "segmentator.h"
#include "decorator.h"

/*
 *
 */
int main(int argc, char** argv) {
	IplImage* imagen = cvLoadImage(argv[1]);

	Segmentator segmentator;
	SegmentationResult res = segmentator.segmentImage(imagen);

	Decorator decorator;
	decorator.drawSegmentationResult(imagen, res);

	cvNamedWindow("imagen");
	cvShowImage("imagen", imagen);

	cvWaitKey(0);
	cvReleaseImage(&imagen);
}

