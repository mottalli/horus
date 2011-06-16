#include <iostream>
#include <bitset>
#include <opencv/highgui.h>
#include <opencv/cv.h>
#include <iomanip>
#include <boost/foreach.hpp>

#include "horus.h"

using namespace std;
using namespace cv;
using namespace horus;

Segmentator segmentator;
VideoProcessor videoProcessor;
Decorator decorator;

int main(int, char**)
{
	vector<double> v(10);
	v[0] = 0;
	v[1] = 2;
	v[2] = 3;
	v[3] = 3;
	v[4] = 3;
	v[5] = 3;
	v[6] = 6;
	v[7] = 4;
	v[8] = 5;
	v[9] = 9;

	horus::tools::Histogram h(v, 5);
	horus::tools::Histogram hc = h.cumulative();

	BOOST_FOREACH(double v, h.values) {
		cout << v << ' ';
	}
	cout << endl;

	BOOST_FOREACH(double v, hc.values) {
		cout << v << ' ';
	}
	cout << endl;

	BOOST_FOREACH(double v, h.values) {
		cout << v << endl;
	}

	for (int i = 0; i < 10; i++) {
		cout << h.binFor(i) << ' ';
	}
	cout << endl;
}
