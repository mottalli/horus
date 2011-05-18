/*
 * tools.cpp
 *
 *  Created on: Jun 15, 2009
 *      Author: marcelo
 */

#include "tools.h"

inline uint8_t setBit(uint8_t b, int bit, bool value);
inline bool getBit(uint8_t b, int bit);

// Pack the binary in src into bits
void Tools::packBits(const GrayscaleImage& src, GrayscaleImage& dest)
{
	assert( (src.cols % 8) == 0);
	dest.create(src.rows, src.cols/8);

	for (int y = 0; y < src.rows; y++) {
		const uint8_t* srcrow = src.ptr(y);

		int xsrc = 0;
		for (int bytenum = 0; bytenum < dest.cols; bytenum++) {
			uint8_t& destbyte =  dest(y, bytenum);
			uint8_t byteval = 0;
			for (int bit = 0; bit < 8; bit++) {
				bool value = (srcrow[xsrc] > 0 ? true : false);
				byteval = setBit(byteval, bit, value);
				xsrc++;
			}
			destbyte = byteval;
		}
	}
}

void Tools::unpackBits(const GrayscaleImage& src, GrayscaleImage& dest, int trueval)
{
	dest.create(src.rows, src.cols*8);

	for (int y = 0; y < src.rows; y++) {
		int xdest = 0;
		for (int xsrc = 0; xsrc < src.cols; xsrc++) {
			uint8_t byte = src(y, xsrc);
			for (int bit = 0; bit < 8; bit++) {
				dest(y, xdest) = (getBit(byte, bit) ? trueval : 0);
				xdest++;
			}
		}
	}
}

void Tools::drawHistogram(const IplImage* img)
{
	int bins = 256;
	int hsize[] = { bins };

	float xranges[] = { 0, 256 };
	float* ranges[] =  { xranges };

	IplImage* copy = cvCloneImage(img);
	IplImage* planes[] = { copy };

	CvHistogram* hist = cvCreateHist(1, hsize, CV_HIST_ARRAY, ranges, 1);
	cvCalcHist(planes, hist, false);

	float min_value = 0, max_value = 0;
	cvGetMinMaxHistValue(hist, &min_value, &max_value);

	IplImage* imgHist = cvCreateImage(cvSize(bins, 50), IPL_DEPTH_8U, 1);
	cvSet(imgHist, cvScalar(255,0,0,0));
	for (int i = 0; i < bins; i++) {
		float value = cvQueryHistValue_1D(hist, i);
		int normalized = cvRound(imgHist->height*(value/max_value));
		cvLine(imgHist, Point(i,imgHist->height), Point(i, imgHist->height-normalized), CV_RGB(0,0,0));
	}

	cvNamedWindow("histogram");
	cvShowImage("histogram", imgHist);

	cvReleaseImage(&imgHist);
	cvReleaseHist(&hist);
	cvReleaseImage(&copy);
}

/*
 10000000: 128
 01000000: 64
 00100000: 32
 00010000: 16
 00001000: 8
 00000100: 4
 00000010: 2
 00000001: 1
 */

static uint8_t BIT_MASK[] = {0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01 };

uint8_t setBit(uint8_t b, int bit, bool value)
{
	if (value) {
		// Set to 1
		return b | BIT_MASK[bit];
	} else {
		// Set to 0
		return b & (~BIT_MASK[bit]);
	}
}

bool getBit(uint8_t b, int bit)
{
	return (b & BIT_MASK[bit]) ? true : false;
}

vector< pair<Point, Point> > Tools::iterateIris(const SegmentationResult& segmentation, int width, int height, double theta0, double theta1, double radiusMin, double radiusMax)
{
	vector< pair<Point, Point> > res(width*height);
	const Contour& pupilContour = segmentation.pupilContour;
	const Contour& irisContour = segmentation.irisContour;

	assert(height > 1);

	Point p0, p1;
	for (int x = 0; x < width; x++) {
		double theta = (double(x)/double(width)) * (theta1-theta0) + theta0;
		if (theta < 0) theta = 2.0 * M_PI + theta;
		assert(theta >= 0 && theta <= 2.0*M_PI);
		double w = (theta/(2.0*M_PI))*double(pupilContour.size());
		p0 = pupilContour[int(floor(w)) % pupilContour.size()];
		p1 = pupilContour[int(ceil(w)) % pupilContour.size()];

		double prop = w-floor(w);
		double xfrom = double(p0.x) + double(p1.x-p0.x)*prop;
		double yfrom = double(p0.y) + double(p1.y-p0.y)*prop;

		w = (theta/(2.0*M_PI))*double(irisContour.size());
		p0 = irisContour[int(floor(w)) % irisContour.size()];
		p1 = irisContour[int(ceil(w)) % irisContour.size()];
		prop = w-floor(w);
		double xto = double(p0.x) + double(p1.x-p0.x)*prop;
		double yto = double(p0.y) + double(p1.y-p0.y)*prop;

		for (int y = 0; y < height; y++) {
			w = (double(y)/double(height-1)) * (radiusMax-radiusMin) + radiusMin;
			double ximage = xfrom + w*(xto-xfrom);
			double yimage = yfrom + w*(yto-yfrom);

			res[x*height+y] = pair<Point, Point>(Point(x, y), Point(ximage, yimage));
		}
	}

	return res;
}

void Tools::superimposeTexture(GrayscaleImage& image, const GrayscaleImage& texture, const SegmentationResult& segmentation, double theta0, double theta1, double radius, bool blend, double blendStart)
{
	assert(texture.type() == CV_8U);
	assert(image.type() == CV_8U);

	vector< pair<Point, Point> > irisIt = Tools::iterateIris(segmentation, texture.cols, texture.rows, theta0, theta1, radius);
	for (size_t i = 0; i < irisIt.size(); i++) {
		int xsrc = irisIt[i].first.x, ysrc = irisIt[i].first.y;
		int xdest = floor(irisIt[i].second.x + 0.5), ydest = floor(irisIt[i].second.y + 0.5);

		double orig = image(ydest, xdest);
		double new_ = texture(ysrc, xsrc);

		if (blend && ysrc >= (texture.rows*blendStart)) {
			double q = 1.0 - ( double(ysrc-texture.rows*blendStart)/double(texture.rows-texture.rows*blendStart) );
			new_ = q*new_ + (1.0-q)*orig;
			//new_ = 255;
		}

		image(ydest, xdest) = uint8_t(new_);
	}
}

void Tools::extractRing(const GrayscaleImage& src, GrayscaleImage& dest, int x0, int y0, int radiusMin, int radiusMax)
{
	assert(src.channels() == 1 && dest.channels() == 1);
	assert(radiusMin < radiusMax);

	int xsrc, ysrc, xdest, ydest;
	double stepRadius = double(radiusMax-radiusMin)/double(dest.rows-1);
	double stepTheta = (2.0*M_PI) / double(dest.cols-1);

	for (ydest = 0; ydest < dest.rows; ydest++) {
		double radius = double(radiusMin) + (stepRadius * double(ydest));
		for (xdest = 0; xdest < dest.cols; xdest++) {
			double theta = stepTheta * double(xdest);

			xsrc = int(double(x0) + radius*cos(theta));
			ysrc = int(double(y0) + radius*sin(theta));

			if (xsrc < 0 || xsrc >= src.cols || ysrc < 0 || ysrc >= src.rows) {
				dest(ydest, xdest) = 0;
			} else {
				dest(ydest, xdest) = src(ysrc, xsrc);
			}
		}
	}
}

void Tools::smoothSnakeFourier(Mat_<float>& snake, int coefficients)
{
	dft(snake, snake, CV_DXT_FORWARD);
	for (int u = coefficients; u < snake.cols-coefficients; u++) {
		snake(0, u) = 0;
	}
	dft(snake, snake, CV_DXT_INV_SCALE);
}

Circle Tools::approximateCircle(const Contour& contour)
{
	Circle result;

	int n = contour.size();

	int sumX = 0, sumY = 0;
	for (Contour::const_iterator it = contour.begin(); it != contour.end(); it++) {
		sumX += (*it).x;
		sumY += (*it).y;
	}
	result.xc = sumX/n;
	result.yc = sumY/n;

	int bestRadius = 0;
	int x,y;
	for (Contour::const_iterator it = contour.begin(); it != contour.end(); it++) {
		x = (*it).x;
		y = (*it).y;
		if ( (x-result.xc)*(x-result.xc)+(y-result.yc)*(y-result.yc) > bestRadius*bestRadius) {
			bestRadius = int(sqrt((float)(x-result.xc)*(x-result.xc)+(y-result.yc)*(y-result.yc)));
		}
	}

	result.radius = bestRadius;

	return result;
}

void Tools::stretchHistogram(const Image& image, Image& dest, float marginMin, float marginMax)
{
	assert(image.depth() == CV_8U);

	if (dest.size() != image.size() || image.type() != dest.type()) {
		dest.create(image.size(), image.type());
	}

	vector<Mat> chansSrc(image.channels()), chansDest(dest.channels());

	split(image, chansSrc);
	split(dest, chansDest);

	vector<int> hist(256, 0);
	unsigned int total = image.rows*image.cols;

	for (size_t c = 0; c < chansSrc.size(); c++) {
		GrayscaleImage chanSrc = chansSrc[c];
		GrayscaleImage chanDest = chansDest[c];

		// Quick & dirty way to calculate the histogram
		for (GrayscaleImage::const_iterator it = chanSrc.begin(); it != chanSrc.end(); it++) {
			hist[*it]++;
		}

		unsigned int sum;
		unsigned char x0, x1;
		for (x0 = 0, sum=0; sum <= marginMin*float(total); x0++) {
			sum += hist[x0];
		}

		for (x1 = 255, sum=0; sum <= marginMax*float(total); x1--) {
			sum += hist[x1];
		}

		for (GrayscaleImage::const_iterator it = chanSrc.begin(); it != chanSrc.end(); it++) {
			int q = int((float((*it)- x0)/float(x1-x0))*255.0);
			q = max(min(q,255), 0);
			chanDest(it.pos()) = q;
		}
	}

	merge(chansDest, dest);
}

GrayscaleImage Tools::normalizeImage(const GrayscaleImage& image, uint8_t min, uint8_t max)
{
	GrayscaleImage res;
	normalize(image, res, min, max, NORM_MINMAX);
	return res;
}

void Tools::toGrayscale(const Image& src, GrayscaleImage& dest, bool cloneIfAlreadyGray) {
	assert(src.type() == CV_8UC1 || src.type() == CV_8UC3);

	if (src.type() == CV_8UC1) {
		if (cloneIfAlreadyGray) {
			dest = src.clone();
		} else {
			dest = src;
		}
	} else if (src.type() == CV_8UC3) {
		cvtColor(src, dest, CV_BGR2GRAY);
	}
}
