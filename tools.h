/*
 * tools.h
 *
 *  Created on: Jun 15, 2009
 *      Author: marcelo
 */

#ifndef TOOLS_H_
#define TOOLS_H_

#include "common.h"

namespace Tools
{
	void packBits(const CvMat* src, CvMat* dest);
	void unpackBits(const CvMat* src, CvMat* dest, int trueval = 1);
}

#endif /* TOOLS_H_ */
