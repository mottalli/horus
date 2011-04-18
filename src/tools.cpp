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
void Tools::packBits(const Mat_<uint8_t>& src, Mat_<uint8_t>& dest)
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

void Tools::unpackBits(const Mat_<uint8_t>& src, Mat_<uint8_t>& dest, int trueval)
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

static const std::string base64_chars =
			 "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
			 "abcdefghijklmnopqrstuvwxyz"
			 "0123456789+/";

std::string Tools::base64EncodeMat(const Mat& mat)
{
	int width = mat.cols, height = mat.rows;
	assert(mat.depth() == CV_8U);

	uint8_t* buffer = new uint8_t[2*sizeof(int16_t) + width*height];		// Stores width, height and data

	// Store width and height
	int16_t* header = (int16_t*)buffer;
	header[0] = width;
	header[1] = height;

	uint8_t* p = buffer + 2*sizeof(int16_t);		// Pointer to the actual data past the width and height

	for (int y = 0; y < height; y++) {
		memcpy(p + y*width, mat.ptr(y), width);		// Copy one line
	}

	std::string base64 = Tools::base64Encode(buffer, width*height+2*sizeof(int16_t));

	delete[] buffer;

	return base64;
}

Mat Tools::base64DecodeMat(const std::string &s)
{
	int width, height;

	std::string decoded = Tools::base64Decode(s);
	uint8_t* buffer = (uint8_t*)decoded.c_str();

	int16_t* header = (int16_t*)buffer;
	width = header[0];
	height = header[1];

	uint8_t* p = buffer + 2*sizeof(int16_t);		// Pointer to the actual data past the width and height

	Mat res(height, width, CV_8U);

	for (int y = 0; y < height; y++) {
		memcpy(res.ptr(y), p + y*width, width);		// Copy one line
	}

	return res;
}

static inline bool is_base64(unsigned char c) {
  return (isalnum(c) || (c == '+') || (c == '/'));
}

std::string Tools::base64Encode(unsigned char const* bytes_to_encode, unsigned int in_len)
{
  std::string ret;
  int i = 0;
  int j = 0;
  unsigned char char_array_3[3];
  unsigned char char_array_4[4];

  while (in_len--) {
	char_array_3[i++] = *(bytes_to_encode++);
	if (i == 3) {
	  char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
	  char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
	  char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
	  char_array_4[3] = char_array_3[2] & 0x3f;

	  for(i = 0; (i <4) ; i++)
		ret += base64_chars[char_array_4[i]];
	  i = 0;
	}
  }

  if (i)
  {
	for(j = i; j < 3; j++)
	  char_array_3[j] = '\0';

	char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
	char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
	char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
	char_array_4[3] = char_array_3[2] & 0x3f;

	for (j = 0; (j < i + 1); j++)
	  ret += base64_chars[char_array_4[j]];

	while((i++ < 3))
	  ret += '=';

  }

  return ret;

}

std::string Tools::base64Decode(std::string const& encoded_string) {
  int in_len = encoded_string.size();
  int i = 0;
  int j = 0;
  int in_ = 0;
  unsigned char char_array_4[4], char_array_3[3];
  std::string ret;

  while (in_len-- && ( encoded_string[in_] != '=') && is_base64(encoded_string[in_])) {
	char_array_4[i++] = encoded_string[in_]; in_++;
	if (i ==4) {
	  for (i = 0; i <4; i++)
		char_array_4[i] = base64_chars.find(char_array_4[i]);

	  char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
	  char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
	  char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

	  for (i = 0; (i < 3); i++)
		ret += char_array_3[i];
	  i = 0;
	}
  }

  if (i) {
	for (j = i; j <4; j++)
	  char_array_4[j] = 0;

	for (j = 0; j <4; j++)
	  char_array_4[j] = base64_chars.find(char_array_4[j]);

	char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
	char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
	char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

	for (j = 0; (j < i - 1); j++) ret += char_array_3[j];
  }

  return ret;
}

std::vector< std::pair<Point, Point> > Tools::iterateIris(const SegmentationResult& segmentation, int width, int height, double theta0, double theta1, double radius)
{
	std::vector< std::pair<Point, Point> > res;
	const Contour& pupilContour = segmentation.pupilContour;
	const Contour& irisContour = segmentation.irisContour;

	Point p0, p1;
	for (int x = 0; x < width; x++) {
		double theta = (double(x)/double(width)) * (theta1-theta0) + theta0;
		if (theta < 0) theta = 2.0 * M_PI + theta;
		assert(theta >= 0 && theta <= 2.0*M_PI);
		double w = (theta/(2.0*M_PI))*double(pupilContour.size());
		p0 = pupilContour[int(std::floor(w))];
		p1 = pupilContour[int(std::ceil(w)) % pupilContour.size()];

		double prop = w-std::floor(w);
		double xfrom = double(p0.x) + double(p1.x-p0.x)*prop;
		double yfrom = double(p0.y) + double(p1.y-p0.y)*prop;

		w = (theta/(2.0*M_PI))*double(irisContour.size());
		p0 = irisContour[int(std::floor(w))];
		p1 = irisContour[int(std::ceil(w)) % irisContour.size()];
		prop = w-std::floor(w);
		double xto = double(p0.x) + double(p1.x-p0.x)*prop;
		double yto = double(p0.y) + double(p1.y-p0.y)*prop;

		for (int y = 0; y < height; y++) {
			w = (double(y)/double(height-1)) * radius;
			double ximage = xfrom + w*(xto-xfrom);
			double yimage = yfrom + w*(yto-yfrom);

			res.push_back(std::pair<Point, Point>(Point(x, y), Point(ximage, yimage)));
		}
	}

	return res;
}

void Tools::superimposeTexture(Mat& image, const Mat& texture, const SegmentationResult& segmentation, double theta0, double theta1, double radius, bool blend, double blendStart)
{
	assert(texture.type() == CV_8U);
	assert(image.type() == CV_8U);

	std::vector< std::pair<Point, Point> > irisIt = Tools::iterateIris(segmentation, texture.cols, texture.rows, theta0, theta1, radius);
	for (size_t i = 0; i < irisIt.size(); i++) {
		int xsrc = irisIt[i].first.x, ysrc = irisIt[i].first.y;
		int xdest = std::floor(irisIt[i].second.x + 0.5), ydest = std::floor(irisIt[i].second.y + 0.5);

		double orig = double(image.at<uint8_t>(ydest, xdest));
		double new_ = double(texture.at<uint8_t>(ysrc, xsrc));

		if (blend && ysrc >= (texture.rows*blendStart)) {
			double q = 1.0 - ( double(ysrc-texture.rows*blendStart)/double(texture.rows-texture.rows*blendStart) );
			new_ = q*new_ + (1.0-q)*orig;
			//new_ = 255;
		}

		image.at<uint8_t>(ydest, xdest) = uint8_t(new_);
	}
}

void Tools::extractRing(const Mat_<uint8_t>& src, Mat_<uint8_t>& dest, int x0, int y0, int radiusMin, int radiusMax)
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

void Tools::stretchHistogram(const Mat_<uint8_t>& image, Mat_<uint8_t>& dest, float marginMin, float marginMax)
{
	if (dest.size() != image.size()) {
		dest.create(image.size());
	}

	// Quick & dirty way to calculate the histogram
	unsigned int hist[256];
	memset(hist, 0, sizeof(hist));

	unsigned int total = image.rows*image.cols;

	/*for (int y = 0; y < image.rows; y++) {
		const uint8_t* ptr = image.ptr(y);
		for (int x = 0; x < image.cols; x++) {
		hist[ptr[x]]++;
	}
	}*/

	for (MatConstIterator_<uint8_t> it = image.begin(); it != image.end(); it++) {
		hist[*it]++;
	}

	unsigned int sum;
	unsigned char x0, x1;
	for (x0 = 0, sum=0; sum <= marginMin*float(total); x0++) {
		sum += hist[x0];
	}

	for (x1 = 255,sum=0; sum <= marginMax*float(total); x1--) {
		sum += hist[x1];
	}

	for (int y = 0; y < image.rows; y++) {
		for (int x = 0; x < image.cols; x++) {
			int q = (float(image(y, x)-x0)/float(x1-x0)) * 255;
			q = max(min(q, 255), 0);
			dest(y, x) = q;
		}
	}
}

Mat_<uint8_t> Tools::normalizeImage(const Mat& image, uint8_t min, uint8_t max)
{
	Mat res;
	normalize(image, res, min, max, NORM_MINMAX);
	return res;
}
