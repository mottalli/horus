#include <iostream>
#include <bitset>
#include <opencv/highgui.h>
#include <opencv/cv.h>
#include <iomanip>

#include "horus.h"
#include "irisdatabasecuda.h"

using namespace std;
using namespace cv;
using namespace horus;

Segmentator segmentator;
VideoProcessor videoProcessor;
Decorator decorator;

class A
{
public:
	void X() { this->Y(); }
protected:
	virtual void Y() { cout << "A" << endl; }
};

class B : public A
{
public:
	void X() { A::X(); }
protected:
	virtual void Y() { cout << "B" << endl; }
};

int main(int, char**)
{
	B b;
	b.X();
}
