#include "loggaborencoder.h"
#include "tools.h"

#include <cmath>

LogGabor1DFilter::LogGabor1DFilter()
{
	this->f0 = 1/32.0;
	this->sigmaOnF = 0.5;
}


LogGabor1DFilter::LogGabor1DFilter(double f0, double sigmanOnF, FilterType type):
	f0(f0), sigmaOnF(sigmanOnF), type(type)
{
}

LogGabor1DFilter::~LogGabor1DFilter()
{
}

void LogGabor1DFilter::applyFilter(const GrayscaleImage& image, Mat1d& dest, const GrayscaleImage& mask, GrayscaleImage& destMask)
{
	assert(image.size() == mask.size());

	this->initializeFilter(image);

	Mat1d imageFloat;
	image.convertTo(imageFloat, imageFloat.type());
	//blur(imageFloat, imageFloat, Size(3,3));			// Blur a bit (improves result for some reason)

	assert(this->filter.channels() == 2);
	assert(image.cols == this->filter.cols);

	// Calculate the Fourier spectrum for each row of the input image
	dft(imageFloat, this->filterResult, DFT_ROWS + DFT_COMPLEX_OUTPUT, imageFloat.rows);


	vector<Mat1d> ps;
	split(this->filterResult, ps);
	assert(ps.size() == 2);
	Mat1d& preal = ps[0];
	Mat1d& pimag = ps[1];
	imshow("real", Tools::normalizeImage(preal));
	imshow("imag", Tools::normalizeImage(pimag));


	// Convolve each row of the image with the filter by multiplying the spectrums
	mulSpectrums(this->filter, this->filterResult, this->filterResult, DFT_ROWS);

	// Perform the inverse transform
	dft(this->filterResult, this->filterResult, DFT_INVERSE | DFT_ROWS, 0);

	assert(this->filterResult.channels() == 2);

	// Split real and imaginary parts
	vector<Mat1d> parts;
	split(this->filterResult, parts);
	assert(parts.size() == 2);
	Mat1d& real = parts[0];
	Mat1d& imag = parts[1];

	dest = (this->type == FILTER_REAL ? real : imag);

	// Filter out elements with low response to the filter (low magnitude)
	GrayscaleImage responseMask;
	Mat1d mag;
	magnitude(real, imag, mag);
	compare(mag, 0.01, responseMask, CMP_GE);
	bitwise_and(mask, responseMask, destMask);
}

void LogGabor1DFilter::initializeFilter(const GrayscaleImage image)
{
	double q = 2.0*log(this->sigmaOnF)*log(this->sigmaOnF);
	this->filter.create(image.size());

	Mat_<Complexd> row(1, image.cols);

	const double x0 = 0.0, x1 = 0.5;

	for (int i = 0; i < image.cols; i++) {
		double r = ((x1-x0)/double(image.cols-1)) * double(i);
		double value = exp( -(log(r/f0)*log(r/f0)) / q);
		row(0, i) = Complexd(value, 0);
	}

	for (int y = 0; y < image.rows; y++) {
		Mat_<Complexd> destRow = this->filter.row(y);
		row.copyTo(destRow);
	}
}

LogGaborEncoder::LogGaborEncoder()
{
	this->filterBank.push_back(LogGabor1DFilter(1.0/32.0, 0.5, LogGabor1DFilter::FILTER_IMAG));
	//this->filterBank.push_back(LogGabor1DFilter(1/16.0, 0.7, LogGabor1DFilter::FILTER_IMAG));
}

LogGaborEncoder::~LogGaborEncoder()
{
}

IrisTemplate LogGaborEncoder::encodeTexture(const GrayscaleImage& texture, const GrayscaleImage& mask)
{
	Size templateSize = LogGaborEncoder::getTemplateSize();
	Size textureSize = texture.size();

	assert(texture.size() == mask.size());
	assert(textureSize == templateSize);		// This precondition holds because IrisEncoder generates a texture
												// of the size specified by the function getNormalizationSize

	// A slots holds the results of all the filters for a single image pixel, distributed in
	// the horizontal direction.
	size_t nFilters = this->filterBank.size();
	size_t nSlots = templateSize.width / nFilters;
	int slotSize = nFilters;

	this->resultTemplate.create(templateSize);
	this->resultMask.create(templateSize);

	for (size_t f = 0; f < nFilters; f++) {
		LogGabor1DFilter& filter = this->filterBank[f];
		filter.applyFilter(texture, this->filteredTexture, mask, this->filteredMask);

		for (size_t s = 0; s < nSlots; s++) {
			int xtemplate = s*slotSize + f;
			int xtexture = (textureSize.width/nSlots) * s;
			for (int ytemplate = 0; ytemplate < templateSize.height; ytemplate++) {
				int ytexture = (textureSize.height/templateSize.height) * ytemplate;
				assert(xtexture < textureSize.width && ytexture < textureSize.height);

				unsigned char templateBit = (this->filteredTexture(ytexture, xtexture) > 0 ? 1 : 0);
				unsigned char maskBit1 = this->filteredMask(ytexture, xtexture);
				unsigned char maskBit2 = (abs(this->filteredTexture(ytexture, xtexture)) < 0.001 ? 0 : 1);

				resultTemplate(ytemplate, xtemplate) = templateBit;
				resultMask(ytemplate, xtemplate) = maskBit1 & maskBit2;
			}
		}
	}

	IrisTemplate result(resultTemplate, resultMask, this->getEncoderSignature());

	return result;
}

string LogGaborEncoder::getEncoderSignature() const
{
	ostringstream signature;
	signature << "LG:" << this->filterBank.size() << ':';
	for (vector<LogGabor1DFilter>::const_iterator it = this->filterBank.begin(); it != this->filterBank.end(); it++) {
		const LogGabor1DFilter& f = (*it);
		signature << f.f0 << '-' << f.sigmaOnF << '-' << f.type;
	}
	return signature.str();
}
