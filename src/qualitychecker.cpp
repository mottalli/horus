#include "qualitychecker.h"
#include <cmath>

QualityChecker::QualityChecker()
{
}

QualityChecker::~QualityChecker(){
}

double QualityChecker::interlacedCorrelation(const GrayscaleImage& frame)
{
	Size size = frame.size();

	this->oddFrame.create(Size(size.width, size.height/2));
	this->evenFrame.create(Size(size.width, size.height/2));

	for (int i = 0; i < size.height/2; i++) {
		Mat rdest = this->evenFrame.row(i);
		frame.row(2*i).copyTo(rdest);
		rdest = this->oddFrame.row(i);
		frame.row(2*i+1).copyTo(rdest);
	}

	this->evenFrame.convertTo(this->bufX, this->bufX.type());
	this->oddFrame.convertTo(this->bufY, this->bufY.type());

	Scalar meanX, meanY, stdX, stdY;

	meanStdDev(this->bufX, meanX, stdX);
	meanStdDev(this->bufY, meanY, stdY);

	subtract(this->bufX, meanX, this->bufX);
	subtract(this->bufY, meanX, this->bufY);
	multiply(this->bufX, this->bufY, this->bufMul);

	double mean_ = mean(this->bufMul).val[0];

	return 100.0*mean_/(stdX.val[0]*stdY.val[0]);

}

double QualityChecker::checkFocus(const Image& image)
{
	Sobel(image, this->bufSobel, CV_32F, 0, 1, 3);
	this->bufSobel = cv::abs(this->bufSobel);


	double s = sum(this->bufSobel).val[0];
	double c = 3e+06;

	return 100.0*s*s/(s*s+c*c);
}

/**
 * Checks if there is an iris on the image and/or if the segmentation is correct (heuristics - not 100% reliable)
 */
QualityChecker::ValidationHeuristics QualityChecker::validateIris(const GrayscaleImage& image, const SegmentationResult& sr)
{
	const Circle& irisCircle = sr.irisCircle;
	double r = irisCircle.radius;

	// Check if the iris is outside the image (or very close to the border)
	int delta = image.cols/20;
	bool outside = false;
	outside = outside || (irisCircle.xc-r < delta);
	outside = outside || (irisCircle.xc+r > image.cols-delta);
	outside = outside || (irisCircle.yc-r < delta);
	outside = outside || (irisCircle.yc+r > image.rows-delta);
	if (outside) {
		return OUTSIDE_IMAGE;
	}

	double pupilRatio = double(sr.pupilCircle.radius)/double(sr.irisCircle.radius);
	if (pupilRatio > 0.65) {		// An unusually large pupil usually means a segmentation error. Happens when the iris is close to the
		return PUPIL_TOO_BIG;		// edge of the image
	}

	int x0 = std::max(0.0, sr.irisCircle.xc-r);
	int x1 = std::min(image.cols-1, int(sr.irisCircle.xc+r));
	int y0 = std::max(0, sr.irisCircle.yc-20);
	int y1 = std::min(image.rows-1, sr.irisCircle.yc+20);

	const Mat portion = image(Rect(x0, y0, x1-x0, y1-y0));

	int xpupil = sr.pupilCircle.xc-x0, ypupil = sr.pupilCircle.yc-y0;
	int xiris = sr.irisCircle.xc-x0, yiris = sr.irisCircle.yc-y0;
	int rpupil2 = sr.pupilCircle.radius*sr.pupilCircle.radius;
	int riris2 = sr.irisCircle.radius*sr.irisCircle.radius;

	double pupilSum = 0, irisSum = 0;
	int pupilCount = 0, irisCount = 0;

	// Computes the mean for each part
	for (int y = 0; y < portion.rows; y++) {
		const uint8_t* row = portion.ptr(y);
		for (int x = 0; x < portion.cols; x++) {
			double val = double(row[x]);

			// Ignore reflections
			if (val > 200) continue;

			int dx2,dy2;

			// Inside pupil?
			dx2 = (x-xpupil)*(x-xpupil);
			dy2 = (y-ypupil)*(y-ypupil);
			if (dx2+dy2 < rpupil2) {
				pupilSum += val;
				pupilCount++;
			} else {
				// Inside iris?
				dx2 = (x-xiris)*(x-xiris);
				dy2 = (y-yiris)*(y-yiris);
				if (dx2+dy2 < riris2) {
					irisSum += val;
					irisCount++;
				}
			}
		}
	}

	if (pupilCount == 0 || irisCount == 0) {
		return NO_COUNT;
	}

	double meanPupil = pupilSum/double(pupilCount);
	double meanIris = irisSum/double(irisCount);
	
	if (meanIris-meanPupil < this->parameters.pupilIrisGrayDiff) {
		// not enough contrast between pupil and iris
		return LOW_CONTRAST;
	}

	// Computes the deviation
	pupilSum = 0;
	irisSum = 0;
	for (int y = 0; y < portion.rows; y++) {
		const uint8_t* row = portion.ptr(y);
		for (int x = 0; x < portion.cols; x++) {
			double val = double(row[x]);
			if (val > 200) continue;

			int dx2,dy2;

			// Inside pupil?
			dx2 = (x-xpupil)*(x-xpupil);
			dy2 = (y-ypupil)*(y-ypupil);
			if (dx2+dy2 < rpupil2) {
				pupilSum += (val-meanPupil)*(val-meanPupil);
			} else {
				// Inside iris?
				dx2 = (x-xiris)*(x-xiris);
				dy2 = (y-yiris)*(y-yiris);
				if (dx2+dy2 < riris2) {
					irisSum += (val-meanIris)*(val-meanIris);
				}
			}
		}
	}

	double varPupil = pupilSum/double(pupilCount);
	double varIris = irisSum/double(irisCount);

	double zScorePupilIris = abs(meanPupil-meanIris) / sqrt((varPupil+varIris)/2.0);
	if (zScorePupilIris < this->parameters.pupilIrisZScore) {
		//return LOW_ZSCORE;
	}
	
	return HAS_IRIS;
}

/**
 * Checks the quality of the image
 */
double QualityChecker::getIrisQuality(const GrayscaleImage& /*image*/, const SegmentationResult& segmentationResult)
{
	return segmentationResult.pupilContourQuality;
}

double QualityChecker::irisTemplateQuality(const IrisTemplate& /*irisTemplate*/)
{
	// TODO
	return 100.0;
}

double QualityChecker::matchQuality(const TemplateComparator& /*comparator*/)
{
	// TODO
	return 100.0;
}
