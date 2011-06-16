/*
 * templatecomparator.h
 *
 *  Created on: Jun 14, 2009
 *      Author: marcelo
 */

#pragma once

#include "common.h"
#include "iristemplate.h"

namespace horus {

class TemplateComparator {
public:
	TemplateComparator(int nRots=20, int rotStep=2);
	TemplateComparator(const IrisTemplate& irisTemplate, int nRots=20, int rotStep=2);
	virtual ~TemplateComparator();

	void setSrcTemplate(const IrisTemplate& irisTemplate);
	double compare(const IrisTemplate& otherTemplate);
	
	// For "a contrario" matching
	std::vector<double> compareParts(const IrisTemplate& otherTemplate, int nParts = 4);
	std::vector<IrisTemplate> rotatedTemplates;

	const IrisTemplate& getBestRotatedTemplate();

	GrayscaleImage getComparationImage(const IrisTemplate& otherTemplate, bool showMask = true);

	static inline const GrayscaleImage getPart(const IrisTemplate& irisTemplate, int part, int nParts, bool fromMask)
	{
		const GrayscaleImage& packedMat = (fromMask ? irisTemplate.getPackedMask() : irisTemplate.getPackedTemplate());
		const int width = packedMat.cols;
		const int height = packedMat.rows;

		assert(width % nParts == 0);
		assert(packedMat.isContinuous());
		const int partWidth = width / nParts;

		// Faster version of the algorithm (using entire blocks) but less reliable
		Rect r(part*partWidth, 0, partWidth, height);
		return packedMat(r);

		// Slower version: interleave the columns in each part. More reliable.
		//TODO: Apply this in the CUDA version
		/*GrayscaleImage res(height, partWidth);
			for (int y = 0; y < height; y++) {
			const uint8_t* srcRow = packedMat.ptr(y);
			for (int xdest = 0; xdest < partWidth; xdest++) {
				res(y, xdest) = srcRow[xdest*nParts+part];
			}
		}
		return res;*/
	}

private:
	static void rotateMatrix(const Mat& src, Mat& dest, int step);
	IrisTemplate irisTemplate;

	GrayscaleImage maskIntersection;
	GrayscaleImage xorBuffer;

	double packedHammingDistance(const GrayscaleImage& template1, const GrayscaleImage& mask1, const GrayscaleImage& template2, const GrayscaleImage& mask2);

	int nRots, rotStep;
	int minHDIdx;
};

}
