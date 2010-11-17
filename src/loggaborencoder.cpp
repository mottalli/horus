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

void LogGabor1DFilter::applyFilter(const Mat_<uint8_t>& image, Mat_<float>& dest, const Mat_<uint8_t>& mask, Mat_<uint8_t>& destMask)
{
	assert(image.size() == mask.size());

	this->initializeFilter(image);

	Mat_<float> imageFloat;
	image.convertTo(imageFloat, imageFloat.type());
	blur(imageFloat, imageFloat, Size(3,3));			// Blur a bit (improves result for some reason)

	this->filterResult.create(image.size());
	assert(this->filter.channels() == 2);

	// Calculate the Fourier spectrum for each row of the input image
	dft(imageFloat, this->filterResult, DFT_ROWS | DFT_COMPLEX_OUTPUT, this->filterResult.rows);
	// Convolve each row of the image with the filter by multiplying the spectrums
	mulSpectrums(this->filter, this->filterResult, this->filterResult, DFT_ROWS);

	// Perform the inverse transform
	dft(this->filterResult, this->filterResult, DFT_INVERSE | DFT_ROWS, this->filterResult.rows);

	assert(this->filterResult.channels() == 2);

	// Split real and imaginary parts
	vector< Mat_<float> > parts;
	split(this->filterResult, parts);
	assert(parts.size() == 2);
	Mat_<float>& real = parts[0];
	Mat_<float>& imag = parts[1];

	dest = (this->type == FILTER_REAL ? real : imag);

	// Filter out elements with low response to the filter
	Mat_<uint8_t> responseMask;
	Mat absreal, absimag, absResponse;
	multiply(real, real, absreal);
	multiply(imag, imag, absimag);
	add(absreal, absimag, absResponse);

	compare(absResponse, 0.01, responseMask, CMP_GE);
	bitwise_and(mask, responseMask, destMask);
}

void LogGabor1DFilter::initializeFilter(const Mat_<uint8_t> image)
{
	float q = 2.0*log(this->sigmaOnF)*log(this->sigmaOnF);
	this->filter.create(image.size());

	Mat_< complex<float> > row(1, image.cols);

	const float x0 = 0.0, x1 = 0.5;

	for (int i = 0; i < image.cols; i++) {
		float r = ((x1-x0)/float(image.cols-1)) * float(i);
		float value = exp( -(log(r/f0)*log(r/f0)) / q);
		row(0, i) = complex<float>(value, 0);
	}

	for (int y = 0; y < image.rows; y++) {
		Mat destRow = this->filter.row(y);
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

IrisTemplate LogGaborEncoder::encodeTexture(const Mat_<uint8_t>& texture, const Mat_<uint8_t>& mask)
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

	IrisTemplate result(resultTemplate, resultMask);

	return result;
}
