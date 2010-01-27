/*
 * tools.h
 *
 *  Created on: Jun 15, 2009
 *      Author: marcelo
 */

#pragma once

#include "common.h"

//------------ Base64 code license BEGIN
/*
   base64.cpp and base64.h

   Copyright (C) 2004-2008 René Nyffenegger

   This source code is provided 'as-is', without any express or implied
   warranty. In no event will the author be held liable for any damages
   arising from the use of this software.

   Permission is granted to anyone to use this software for any purpose,
   including commercial applications, and to alter it and redistribute it
   freely, subject to the following restrictions:

   1. The origin of this source code must not be misrepresented; you must not
	  claim that you wrote the original source code. If you use this source code
	  in a product, an acknowledgment in the product documentation would be
	  appreciated but is not required.

   2. Altered source versions must be plainly marked as such, and must not be
	  misrepresented as being the original source code.

   3. This notice may not be removed or altered from any source distribution.

   René Nyffenegger rene.nyffenegger@adp-gmbh.ch

*/
//------------ Base64 code license END

namespace Tools
{
	void packBits(const CvMat* src, CvMat* dest);
	void unpackBits(const CvMat* src, CvMat* dest, int trueval = 1);

	// Useful debugging functions
	void drawHistogram(const IplImage* img);

	// Base64 methods
	std::string base64Encode(unsigned char const* buffer, unsigned int len);
	std::string base64Decode(std::string const& s);

	std::string base64EncodeMat(const CvMat* mat);
	CvMat* base64DecodeMat(const std::string &s);
}

