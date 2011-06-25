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
using namespace boost;

//Segmentator segmentator;
//VideoProcessor videoProcessor;
//Decorator decorator;

int main(int, char**)
{
	thread_group tg;

	for (int i = 0; i < 10; i++) {
		tg.create_thread([i]() {
			cout << i << endl;
		});
	}

	tg.join_all();

	return 0;
}
