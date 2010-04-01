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
void Tools::packBits(const CvMat* src, CvMat* dest)
{
	assert(src->width / 8 == dest->width);
	assert(src->height == dest->height);

	for (int y = 0; y < src->height; y++) {
		const uint8_t* srcrow = &(src->data.ptr[y*src->step]);

		int xsrc = 0;
		for (int bytenum = 0; bytenum < dest->width; bytenum++) {
			uint8_t *destbyte =  &(dest->data.ptr[y*dest->step+bytenum]);
			uint8_t byteval = 0;
			for (int bit = 0; bit < 8; bit++) {
				bool value = (srcrow[xsrc] > 0 ? true : false);
				byteval = setBit(byteval, bit, value);
				xsrc++;
			}
			*destbyte = byteval;
		}
	}
}

void Tools::unpackBits(const CvMat* src, CvMat* dest, int trueval)
{
	assert(src->width * 8 == dest->width);
	assert(src->height == dest->height);

	for (int y = 0; y < src->height; y++) {
		int xdest = 0;
		for (int xsrc = 0; xsrc < src->width; xsrc++) {
			uint8_t byte = src->data.ptr[y*src->step+xsrc];
			for (int bit = 0; bit < 8; bit++) {
				cvSetReal2D(dest, y, xdest, getBit(byte, bit) ? trueval : 0);
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
		cvLine(imgHist, cvPoint(i,imgHist->height), cvPoint(i, imgHist->height-normalized), CV_RGB(0,0,0));
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

std::string Tools::base64EncodeMat(const CvMat* mat)
{
	int width = mat->width, height = mat->height;
	assert(sizeof(uint8_t) == 1);
	uint8_t* buffer = new uint8_t[2*sizeof(int16_t) + width*height];		// Stores width, height and data

	// Store width and height
	int16_t* header = (int16_t*)buffer;
	header[0] = width;
	header[1] = height;

	uint8_t* p = buffer + 2*sizeof(int16_t);		// Pointer to the actual data past the width and height

	for (int y = 0; y < height; y++) {
		memcpy(p + y*width, mat->data.ptr + y*mat->step, width);		// Copy one line
	}

	std::string base64 = Tools::base64Encode(buffer, width*height+2*sizeof(int16_t));

	delete[] buffer;

	return base64;
}

CvMat* Tools::base64DecodeMat(const std::string &s)
{
	int width, height;

	std::string decoded = Tools::base64Decode(s);
	uint8_t* buffer = (uint8_t*)decoded.c_str();

	int16_t* header = (int16_t*)buffer;
	width = header[0];
	height = header[1];

	uint8_t* p = buffer + 2*sizeof(int16_t);		// Pointer to the actual data past the width and height

	CvMat* res = cvCreateMat(height, width, CV_8U);
	for (int y = 0; y < height; y++) {
		memcpy(res->data.ptr + y*res->step, p + y*width, width);		// Copy one line
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

std::vector< std::pair<CvPoint, CvPoint> > Tools::iterateIris(const SegmentationResult& segmentation, int width, int height, double theta0, double theta1, double radius)
{
	std::vector< std::pair<CvPoint, CvPoint> > res;
	const Contour& pupilContour = segmentation.pupilContour;
	const Contour& irisContour = segmentation.irisContour;

	CvPoint p0, p1;
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

			res.push_back(std::pair<CvPoint, CvPoint>(cvPoint(x, y), cvPoint(ximage, yimage)));
		}
	}

	return res;
}

void Tools::superimposeTexture(IplImage* image, const IplImage* texture, const SegmentationResult& segmentation, double theta0, double theta1, double radius)
{
	std::vector< std::pair<CvPoint, CvPoint> > irisIt = Tools::iterateIris(segmentation, texture->width, texture->height, theta0, theta1, radius);
	for (size_t i = 0; i < irisIt.size(); i++) {
		int xsrc = irisIt[i].first.x, ysrc = irisIt[i].first.y;
		int xdest = std::floor(irisIt[i].second.x + 0.5), ydest = std::floor(irisIt[i].second.y + 0.5);
		cvSet2D(image, ydest, xdest, cvGet2D(texture, ysrc, xsrc));
	}
}

void Tools::updateSize(IplImage** image, CvSize size, int depth, int channels)
{
	if (*image == NULL || (*image)->width != size.width || (*image)->height != size.height || (*image)->depth != depth) {
		if (*image != NULL) {
			cvReleaseImage(image);
		}
		*image = cvCreateImage(size, depth, channels);
	}
}

void Tools::updateSize(CvMat** mat, CvSize size, int depth)
{
	if (*mat == NULL || (*mat)->cols != size.width || (*mat)->cols != size.height || (*mat)->type != depth) {
		if (*mat != NULL) {
			cvReleaseMat(mat);
		}
		*mat = cvCreateMat(size.height, size.width, depth);
	}
}
