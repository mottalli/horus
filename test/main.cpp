#include <iostream>
#include <opencv/highgui.h>
#include <opencv/cv.h>

#include "segmentator.h"

using namespace std;
using namespace cv;

Segmentator segmentator;

int main(int, char**)
{
	cout << segmentator.pupilSegmentator.parameters.infraredThreshold << endl;
}
