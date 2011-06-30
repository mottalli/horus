#include "loggaborencoder.h"
#include "tools.h"

using namespace horus;
using namespace std;

LogGabor1DFilter::LogGabor1DFilter()
{
	this->f0 = 1/32.0;
	this->sigmaOnF = 0.5;
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

	if (this->filter.empty()) {
		this->filter = LogGabor1DFilter::createRowFilter(image.size(), this->f0, this->sigmaOnF);
		assert(this->filter.channels() == 2);
	}

	Mat_<Complexd> imageFourier, multipliedSpectrums, filterResult;

	Mat1d imageFloat;
	image.convertTo(imageFloat, imageFloat.type());
	//blur(imageFloat, imageFloat, Size(3,3));			// Blur a bit (improves result for some reason)

	assert(this->filter.channels() == 2);
	assert(image.cols == this->filter.cols);

	// Calculate the Fourier spectrum for each row of the input image
	dft(imageFloat, imageFourier, DFT_ROWS | DFT_COMPLEX_OUTPUT);
	// Convolve each row of the image with the filter by multiplying the spectrums
	mulSpectrums(this->filter, imageFourier, multipliedSpectrums, DFT_ROWS);
	// Perform the inverse transform
	dft(multipliedSpectrums, filterResult, DFT_INVERSE | DFT_ROWS, 0);

	/////
	/*Mat tmp;
	vector<Mat1d> parts2;
	split(this->filter, parts2);
	idft(this->filter, tmp, DFT_COMPLEX_OUTPUT | DFT_ROWS);
	split(tmp, parts2);

	struct { Mat operator()(Mat1d m) {
		Rect rIzq(0, 0, m.cols/2, m.rows);
		Rect rDer(m.cols/2, 0, m.cols/2, m.rows);
		Mat1d parteIzq = m(rIzq);
		Mat1d parteDer = m(rDer);

		Mat1d res(m.size());
		res.setTo(0);

		Mat1d resDer = res(rDer);
		Mat1d resIzq = res(rIzq);

		parteIzq.copyTo(resDer);
		parteDer.copyTo(resIzq);

		return res;
	} } acomodar;


	imshow("im1", tools::normalizeImage(acomodar(parts2[0])));
	imshow("im2", tools::normalizeImage(acomodar(parts2[1])));*/
	/////

	assert(filterResult.channels() == 2);

	// Split the result into real and imaginary parts
	vector<Mat1d> parts;
	split(filterResult, parts);
	assert(parts.size() == 2);
	Mat1d& real = parts[0];
	Mat1d& imag = parts[1];

	(this->type == FILTER_REAL ? real : imag).copyTo(dest);

	// Filter out elements with low response to the filter (low magnitude)
	GrayscaleImage responseMask;
	Mat1d mag;
	magnitude(real, imag, mag);
	compare(mag, 0.01, responseMask, CMP_GE);
	bitwise_and(mask, responseMask, destMask);
}

Mat_<Complexd> LogGabor1DFilter::createRowFilter(Size size, double f0, double sigmaOnF)
{
	Mat_<Complexd> filter(size);
	Mat_<Complexd> row(1, size.width);

	double q = 2.0*log(sigmaOnF)*log(sigmaOnF);
	const double x0 = 0.0, x1 = 0.5;

	for (int i = 0; i < size.width; i++) {
		double r = ((x1-x0)/double(size.width-1)) * double(i);
		double value = exp( -(log(r/f0)*log(r/f0)) / q);
		row(0, i) = Complexd(value, 0);
	}

	for (int y = 0; y < size.height; y++) {
		for (int x = 0; x < size.width; x++) {
			filter(y, x) = row(0, x);
		}
	}

	return filter;
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
