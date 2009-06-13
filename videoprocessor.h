/*
 * videoprocessor.h
 *
 *  Created on: Jun 13, 2009
 *      Author: marcelo
 */

#ifndef VIDEOPROCESSOR_H_
#define VIDEOPROCESSOR_H_

#include "common.h"

class VideoProcessor {
public:
	VideoProcessor();
	virtual ~VideoProcessor();

	double imageQuality(const Image* image);
};

#endif /* VIDEOPROCESSOR_H_ */
