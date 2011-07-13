#include "loggaborencoder.h"
#include "tools.h"

using namespace horus;
using namespace std;

LogGabor1DFilter::LogGabor1DFilter()
{
	this->f0 = 1/32.0;
	this->sigmaOnF = 0.5;
	this->type = LogGabor1DFilter::FILTER_IMAG;
}


LogGabor1DFilter::LogGabor1DFilter(double f0_, double sigmanOnF_, FilterType type_):
	f0(f0_), sigmaOnF(sigmanOnF_), type(type_)
{
}

LogGabor1DFilter::~LogGabor1DFilter()
{
}

void LogGabor1DFilter::applyFilter(const GrayscaleImage& image, Mat1d& dest, const GrayscaleImage& mask, Mat1b& destMask) const
{
	assert(image.size() == mask.size());

	if (this->realFilter.cols != image.cols) {
		// Filters not initialized
		std::pair<Mat1d, Mat1d> filters = LogGabor1DFilter::createSpatialFilter(image.cols, this->f0, this->sigmaOnF);
		this->realFilter = filters.first;
		this->imagFilter = filters.second;
	}


	Mat1d& filter = (this->type == FILTER_REAL ? this->realFilter : this->imagFilter);

	Mat1d flippedFilter;			// Flip the filter because filter2D calculates correlation, not convolution
	flip(filter, flippedFilter, 1);

	Mat1d imageDouble;
	image.convertTo(imageDouble, imageDouble.type());

	// Do the convolution between the texture and the log-gabor filter in the spatial domain
	filter2D(imageDouble, dest, dest.type(), flippedFilter, Point(-1,-1), 0, BORDER_DEFAULT);

	// Don't change the mask
	mask.copyTo(destMask);
}

Mat_<Complexd> LogGabor1DFilter::createFrequencyFilter(size_t size, double f0, double sigmaOnF)
{
	Mat_<Complexd> filter(1, size);

	double q = 2.0*log(sigmaOnF)*log(sigmaOnF);
	const double x0 = 0.0, x1 = 0.5;

	for (size_t i = 0; i < size; i++) {
		double r = ((x1-x0)/double(size-1)) * double(i);
		double value = exp( -(log(r/f0)*log(r/f0)) / q);
		filter(0, i) = Complexd(value, 0.0);
	}

	return filter;
}

std::pair<Mat1d, Mat1d> LogGabor1DFilter::createSpatialFilter(size_t size, double f0, double sigmaOnF)
{
	Mat_<Complexd> frequencyFilter = LogGabor1DFilter::createFrequencyFilter(size, f0, sigmaOnF);

	Mat_<Complexd> tmp;
	vector<Mat1d> parts;
	idft(frequencyFilter, tmp, DFT_COMPLEX_OUTPUT | DFT_ROWS);
	assert(tmp.channels() == 2 && tmp.rows == 1 && tmp.cols == (int)size);
	split(tmp, parts);			// Split into real and imaginary filters
	Mat1d& real = parts[0];
	Mat1d& imag = parts[1];

	// Functor that shifts the left and right parts of the vector
	// (similar to Matlab's "fftshift")
	auto fftshift = [](Mat1d& m) {
		Rect rLeft(0, 0, m.cols/2, m.rows);
		Rect rRight(m.cols/2, 0, m.cols/2, m.rows);
		Mat1d partLeft = m(rLeft).clone();
		Mat1d partRight = m(rRight).clone();
		Mat1d resRight = m(rRight);
		partLeft.copyTo(resRight);
		Mat1d resLeft = m(rLeft);
		partRight.copyTo(resLeft);
	};

	fftshift(real);
	fftshift(imag);

	return std::pair<Mat1d, Mat1d>(real, imag);
}

LogGaborEncoder::LogGaborEncoder()
{
	this->filterBank.push_back(LogGabor1DFilter(1.0/32.0, 0.5, LogGabor1DFilter::FILTER_IMAG));
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

	GrayscaleImage resultTemplate(templateSize), resultMask(templateSize);
	Mat1d filteredTexture(textureSize);
	Mat1b filteredMask(textureSize);

	for (size_t f = 0; f < nFilters; f++) {
		LogGabor1DFilter& filter = this->filterBank[f];
		filter.applyFilter(texture, filteredTexture, mask, filteredMask);

		for (size_t s = 0; s < nSlots; s++) {
			int xtemplate = s*slotSize + f;
			int xtexture = (textureSize.width/nSlots) * s;
			for (int ytemplate = 0; ytemplate < templateSize.height; ytemplate++) {
				int ytexture = (textureSize.height/templateSize.height) * ytemplate;
				assert(xtexture < textureSize.width && ytexture < textureSize.height);

				unsigned char templateBit = (filteredTexture(ytexture, xtexture) > 0 ? 1 : 0);
				unsigned char maskBit1 = filteredMask(ytexture, xtexture);
				unsigned char maskBit2 = (abs(filteredTexture(ytexture, xtexture)) < 0.001 ? 0 : 1);

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
