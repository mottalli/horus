#include "gaborencoder.h"

#include "tools.h"

GaborFilter::GaborFilter()
{
}

GaborFilter::GaborFilter(int width, int height, float u0, float v0, float alpha, float beta, FilterType type)
{
	this->width = width;
	this->height = height;
	this->u0 = u0;
	this->v0 = v0;
	this->alpha = alpha;
	this->beta = beta;
	this->type = type;

	float x0 = -1, y0 = -1;
	float x1 = 1, y1 = 1;

	this->filter.create(height, width);
	for (int i = 0; i < height; i++) {
		float y = y0 + ((y1-y0)/float(height-1)) * float(i);
		for (int j = 0; j < width; j++) {
			float x = x0 + ((x1-x0)/float(width-1)) * float(j);

			float env = exp(-M_PI* ((x*x)/(alpha*alpha)  + (y*y)/(beta*beta)));		// Gaussian envelope
			float f = 2.0*M_PI*(u0*x + v0*y);

			float carrier;
			if (type == FILTER_REAL) {
				carrier = cos(f);
			} else if (type == FILTER_IMAG) {
				carrier = -sin(f);
			}

			this->filter(i, j) = env*carrier;
		}
	}
}

GaborFilter::~GaborFilter()
{
}

void GaborFilter::applyFilter(const Mat_<float>& src, Mat_<float>& dest, const Mat_<uint8_t>& mask, Mat_<uint8_t>& destMask)
{
	assert(src.size() == dest.size());
	assert(mask.size() == destMask.size());

	filter2D(src, dest, dest.depth(), this->filter, Point(-1,-1), 0, BORDER_REPLICATE);

	mask.copyTo(destMask);
}

GaborEncoder::GaborEncoder()
{
	this->filterBank.push_back(GaborFilter(15, 15, 0.5, 0.5, 2, 2, GaborFilter::FILTER_IMAG));
	//this->filterBank.push_back(GaborFilter(15, 15, 0.5, -0.5, 2, 2, GaborFilter::FILTER_IMAG));
	//this->filterBank.push_back(GaborFilter(15, 15, 0.5, 1, 2, 2, GaborFilter::FILTER_IMAG));
	//this->filterBank.push_back(GaborFilter(15, 15, -0.5, 1, 2, 2, GaborFilter::FILTER_IMAG));
}

GaborEncoder::~GaborEncoder()
{
}

IrisTemplate GaborEncoder::encodeTexture(const Mat_<uint8_t>& texture, const Mat_<uint8_t>& mask)
{
	assert(texture.size() == mask.size());
	assert(texture.channels() == 1 && mask.channels() == 1);

	Size templateSize = GaborEncoder::getTemplateSize();
	size_t nFilters = this->filterBank.size();
	size_t nSlots = templateSize.width / nFilters;
	int slotSize = nFilters;

	this->resultTemplate.create(templateSize);
	this->resultMask.create(templateSize);

	this->floatTexture.create(texture.size());
	this->filteredTexture.create(texture.size());
	this->filteredMask.create(texture.size());

	texture.convertTo(this->floatTexture, this->floatTexture.type());		// Use decimal precision

	for (size_t f = 0; f < nFilters; f++) {
		GaborFilter& filter = this->filterBank[f];
		filter.applyFilter(this->floatTexture, this->filteredTexture, mask, this->filteredMask);

		for (size_t s = 0; s < nSlots; s++) {
			int xtemplate = s*slotSize + f;
			int xtexture = (texture.cols/nSlots) * s;
			for (int ytemplate = 0; ytemplate < templateSize.height; ytemplate++) {
				int ytexture = (texture.rows/templateSize.height) * ytemplate;
				assert(xtexture < this->filteredTexture.cols && ytexture < this->filteredTexture.rows);

				unsigned char templateBit =  (this->filteredTexture(ytexture, xtexture)  >  0.0 ? 1 : 0);
				this->resultTemplate(ytemplate, xtemplate) = templateBit;

				unsigned char maskBit1 = (this->filteredMask(ytexture, xtexture) ? 1 : 0);
				this->resultMask(ytemplate, xtemplate) = maskBit1;
			}
		}
	}

	IrisTemplate result(resultTemplate, resultMask);

	return result;
}
