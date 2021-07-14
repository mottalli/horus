/*
 * tools.cpp
 *
 *  Created on: Jun 15, 2009
 *      Author: marcelo
 */

#include "tools.h"

using namespace horus;
using namespace horus::tools;
using namespace std;

// Pack the binary in src into bits
void horus::tools::packBits(const GrayscaleImage& src, GrayscaleImage& dest)
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

void horus::tools::unpackBits(const GrayscaleImage& src, GrayscaleImage& dest, int trueval)
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

/*vector< pair<Point, Point> > horus::tools::iterateIris(const SegmentationResult& segmentation, int width, int height, double theta0, double theta1, double radiusMin, double radiusMax)
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

			//res[x*height+y] = pair<Point, Point>(Point(x, y), Point(ximage, yimage));			// This causes some weird issues! (Ubuntu 11.04, gcc 4.5)
			res.push_back(pair<Point, Point>(Point(x, y), Point(ximage, yimage)));
		}
	}

	return res;
}*/

void horus::tools::superimposeTexture(GrayscaleImage& image, const GrayscaleImage& texture, const SegmentationResult& segmentation, double theta0, double theta1, double minRadius, double maxRadius, bool blend, double blendStart)
{
	assert(texture.type() == CV_8U);
	assert(image.type() == CV_8U);

	SegmentationResult::iterator it = segmentation.iterateIris(texture.size(), theta0, theta1, minRadius, maxRadius);
	do {
		int xsrc = it.texturePoint.x, ysrc = it.texturePoint.y;
		int xdest = floor(it.imagePoint.x + 0.5), ydest = floor(it.imagePoint.y + 0.5);

		if (xdest < 0 || xdest >= image.cols || ydest < 0 || ydest >= image.rows) {
			continue;
		}

		double orig = image(ydest, xdest);
		double new_ = texture(ysrc, xsrc);

		if (blend && ysrc >= (texture.rows*blendStart)) {
			double q = 1.0 - ( double(ysrc-texture.rows*blendStart)/double(texture.rows-texture.rows*blendStart) );
			new_ = q*new_ + (1.0-q)*orig;
		}

		image(ydest, xdest) = uint8_t(new_);
	} while (it.next());
}

void horus::tools::extractRing(const GrayscaleImage& src, GrayscaleImage& dest, int x0, int y0, int radiusMin, int radiusMax)
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

void horus::tools::smoothSnakeFourier(Mat_<float>& snake, int coefficients)
{
	dft(snake, snake, CV_DXT_FORWARD);
	for (int u = coefficients; u < snake.cols-coefficients; u++) {
		snake(0, u) = 0;
	}
	dft(snake, snake, CV_DXT_INV_SCALE);
}

Circle horus::tools::approximateCircle(const Contour& contour)
{
	Circle result;

	int n = contour.size();

	// Calculate the centroid of the contour
	int sumX = 0, sumY = 0;
	for (Contour::const_iterator it = contour.begin(); it != contour.end(); it++) {
		sumX += (*it).x;
		sumY += (*it).y;
	}
	result.center.x = sumX/n;
	result.center.y = sumY/n;

	int bestRadius = 0;
	int x,y;
	for (Contour::const_iterator it = contour.begin(); it != contour.end(); it++) {
		x = (*it).x;
		y = (*it).y;
		if ( (x-result.center.x)*(x-result.center.x)+(y-result.center.y)*(y-result.center.y) > bestRadius*bestRadius) {
			bestRadius = int(sqrt((float)(x-result.center.x)*(x-result.center.x)+(y-result.center.y)*(y-result.center.y)));
		}
	}

	result.radius = bestRadius;

	return result;
}

void horus::tools::stretchHistogram(const Image& image, Image& dest, float marginMin, float marginMax)
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

GrayscaleImage horus::tools::normalizeImage(const Mat& image, uint8_t min, uint8_t max)
{
	assert(image.channels() == 1);

	Mat tmp;
	GrayscaleImage res;
	normalize(image, tmp, min, max, NORM_MINMAX);
	tmp.convertTo(res, CV_8UC1);
	return res;
}

void horus::tools::toGrayscale(const Image& src, GrayscaleImage& dest, bool cloneIfAlreadyGray) {
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

void horus::tools::superimposeImage(const Image& imageSrc, Image& imageDest, Point p, bool drawBorder)
{
	Rect r = Rect(p.x, p.y, imageSrc.cols, imageSrc.rows);
	assert(r.br().x < imageDest.cols && r.br().y < imageDest.rows);		// Inside the image

	Image destRect = imageDest(r);

	if (imageSrc.type() == imageDest.type()) {
		imageSrc.copyTo(destRect);
	} else if (imageSrc.channels() == 3) {
		assert(imageDest.channels() == 1);
		vector<Mat> channels(3, imageSrc);
		merge(channels, destRect);
	} else {
		assert(imageSrc.channels() == 1 && imageDest.channels() == 3);
		cvtColor(imageSrc, destRect, CV_GRAY2BGR);
	}

	if (drawBorder) {
		Point tl(r.tl().x-1, r.tl().y-1);
		Point br(r.br().x, r.br().y);
		rectangle(imageDest, tl, br, CV_RGB(0,0,0), 1);
	}
}

/**
 * 0000 0
 * 0001 1
 * -------
 * 0010 1 = 0+1
 * 0011 2 = 1+1
 * -------
 * 0100 1 = 0+1
 * 0101 2 = 1+1
 * 0110 2 = 0+1+1
 * 0111 3 = 1+1+1
 * -------
 * 1000 1 = 0+1
 * 1001 2 = 1+1
 * 1010 2 = 0+1+1
 * 1011 3 = 1+1+1
 * 1100 2 = 0+1+1
 * 1101 3 = 1+1+1
 * 1110 3 = 0+1+1+1
 * 1111 4 = 1+1+1+1
 * --------
 * etc...
 */
int horus::tools::countNonZeroBits(const Mat& mat)
{
	assert(mat.depth() == CV_8U);

	static bool initialized = false;
	static int nonZeroBits[256];		// Note: this could be hard-coded but it's too long and the algorithm to calculate it
										// is quite simple and we only do it once
	if (!initialized) {
		// Calculate the number of bits set to 1 on an 8-bit value
		nonZeroBits[0] = 0;
		nonZeroBits[1] = 1;

		// Sorry about non-meaningful variable names :)
		int p = 2;
		while (p < 256) {
			for (int q = 0; q < p; q++) {
				nonZeroBits[p+q] = nonZeroBits[q]+1;
			}
			p = 2*p;
		}

		initialized = true;
	}

	int res = 0;

	for (int y = 0; y < mat.rows; y++) {
		const uint8_t* row = mat.ptr(y);
		int x;
		for (x = 0; x < mat.cols-3; x += 4) {		// Optimization: aligned to 4 byes, extracted from cvCountNonZero
			uint8_t val0 = row[x], val1 = row[x+1], val2 = row[x+2], val3 = row[x+3];
			res += nonZeroBits[val0] + nonZeroBits[val1] + nonZeroBits[val2] + nonZeroBits[val3];
		}
		for (; x < mat.cols; x++) {
			res += nonZeroBits[ row[x] ];
		}
	}
	return res;
}
