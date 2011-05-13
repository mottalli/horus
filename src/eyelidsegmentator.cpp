#include "eyelidsegmentator.h"

EyelidSegmentator::EyelidSegmentator()
{
}

EyelidSegmentator::~EyelidSegmentator()
{
}

std::pair<Parabola, Parabola> EyelidSegmentator::segmentEyelids(const GrayscaleImage& image, const Circle& pupilCircle, const Circle& irisCircle)
{
	int r = irisCircle.radius * 1.5;
	int x0 = max(0, irisCircle.xc-r);
	int x1 = min(image.cols-1, irisCircle.xc+r);
	int y0upper = max(0, irisCircle.yc-r);
	int y1upper = irisCircle.yc;
	int y0lower= irisCircle.yc;
	int y1lower = min(image.rows-1, irisCircle.yc+r);

	Sobel(image, this->gradient, this->gradient.type(), 0, 1, 3);
	blur(this->gradient, this->gradient, Size(7,7));

	Parabola upperEyelid = this->segmentUpper(image, this->gradient, x0, y0upper, x1, y1upper, pupilCircle, irisCircle);
	Parabola lowerEyelid = this->segmentLower(image, this->gradient, x0, y0lower, x1, y1lower, pupilCircle, irisCircle);

	std::pair<Parabola, Parabola> result;
	result.first = upperEyelid;
	result.second = lowerEyelid;

	return result;
}

Parabola EyelidSegmentator::segmentUpper(const GrayscaleImage& image, const Mat_<float>& gradient, int x0, int y0, int x1, int y1, const Circle&, const Circle&)
{
	Parabola bestParabola;
	double maxGrad = INT_MIN;

	for (int p = 150; p < 300; p += 50) {
		pair<Parabola, double> res = this->findParabola(image, gradient, p, x0, y0, x1, y1);
		if (res.second > maxGrad) {
			maxGrad = res.second;
			bestParabola = res.first;
		}
	}

	return bestParabola;
}

Parabola EyelidSegmentator::segmentLower(const GrayscaleImage& image, const Mat_<float>& gradient, int x0, int y0, int x1, int y1, const Circle&, const Circle&)
{
	Parabola bestParabola;
	double maxGrad = INT_MIN;

	for (int p = -300; p < -150; p += 50) {
		pair<Parabola, double> res = this->findParabola(image, gradient, p, x0, y0, x1, y1);
		if (res.second > maxGrad) {
			maxGrad = res.second;
			bestParabola = res.first;
		}
	}

	return bestParabola;
}

std::pair<Parabola, double> EyelidSegmentator::findParabola(const GrayscaleImage& image, const Mat_<float>& gradient, int p, int x0, int y0, int x1, int y1)
{
	int step = this->parameters.parabolicDetectorStep;

	assert(y1 >= y0);
	assert(x1 >= x0);

	Mat_<float> M((y1-y0)/step + 1, (x1-x0)/step + 1);

	for (int i = 0; i < M.rows; i++) {
		int y = y0+i*step;
		for (int j = 0; j < M.cols; j++) {
			int x = x0+j*step;
			double avg = this->parabolaAverage(gradient, image, Parabola(x, y, p));
			M(i, j) = avg;
		}
	}

	Size s(M.rows, M.cols);
	Mat_<float> Dx(s), Dy(s), D(s);

	Sobel(M, Dx, Dx.type(), 1, 0);
	Sobel(M, Dy, Dy.type(), 0, 1);

	// D = Dx^2 + Dy^2
	multiply(Dx, Dx, Dx);
	multiply(Dy, Dy, Dy);
	D = Dx + Dy;

	blur(D, D, Size(3,3));

	Point maxPos;
	Point minPos;
	double max;
	minMaxLoc(D, &max, NULL, &minPos, &maxPos);

	int x = x0+maxPos.x*step;
	int y = y0+maxPos.y*step;

	return std::pair<Parabola, double>(Parabola(x, y, p), max);
}

double EyelidSegmentator::parabolaAverage(const Mat_<float>& gradient, const GrayscaleImage& originalImage, const Parabola& parabola)
{
	double S = 0;
	int n = 0;
	double x, y;

	for (x = 0; x < gradient.cols; x += gradient.cols/100 + 1) {
		y = parabola.value(x);
		if (y < 0 || y >= gradient.rows) {
			continue;
		}

		const float* rowGradient = gradient[y];
		const uint8_t* rowImage = originalImage[y];

		uint8_t v = rowImage[int(x)];
		if (v < 80 || v > 250) {
			// Try to avoid the pupil and the infrared reflection
			continue;
		}

		S += rowGradient[int(x)];
		n++;
	}

	if (!n) {
		return 0;		// No values were processed
	} else {
		return S/double(n);
	}
}
