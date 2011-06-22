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

Segmentator segmentator;
VideoProcessor videoProcessor;
Decorator decorator;

template<class T>
class ThreadDispatcher
{
public:
	ThreadDispatcher(T& obj, size_t nThreads) {
		for (size_t i = 0; i < nThreads; i++) {
			T copy = obj;
			copy.threadIdx = i;
			copy.threadCount = nThreads;

			group.create_thread(copy);
		}
	}

	inline void join() { group.join_all(); }

	thread_group group;
};

struct ThreadableBlock
{
	unsigned threadIdx;
	unsigned threadCount;
	virtual void operator() () = 0;
};

struct AlgoritmoProcesador : ThreadableBlock
{
	Mat matriz;

	AlgoritmoProcesador(Mat& m) : matriz(m) {}

	void operator() ()
	{
		unsigned totalBytes = matriz.cols*matriz.rows;
		int n = totalBytes / threadCount;
		uint8_t* p0 = matriz.data + n*threadIdx;

		uint8_t dest = double(threadIdx*255)/double(threadCount-1);
		double alpha = 0.5;
		for (size_t i = 0; i < n; i++) {
			uint8_t src = p0[i];
			p0[i] = alpha*src + (1.0-alpha)*dest;
		}
	}

};

int main(int, char**)
{
	VideoCapture cap(0);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

	Mat frame_, frame, frameBW;
	int dx = 10;

	namedWindow("video");

	while (true) {
		cap >> frame_;
		frame = frame_(Rect(dx, 0, frame_.cols-2*dx, frame_.rows));

		cvtColor(frame, frameBW, CV_BGR2GRAY);

		assert(frameBW.isContinuous());
		assert(frame.type() == CV_8UC1);
		AlgoritmoProcesador procesador(frameBW);
		ThreadDispatcher<AlgoritmoProcesador> td(procesador, 20);

		td.join();

		imshow("video", frameBW);

		if (char(waitKey(20)) == 'q') break;
	}
}
