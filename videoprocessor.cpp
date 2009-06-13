/*
 * videoprocessor.cpp
 *
 *  Created on: Jun 13, 2009
 *      Author: marcelo
 */

#include "videoprocessor.h"
#include <iostream>
#include <cmath>

void cvShiftDFT(CvArr * src_arr, CvArr * dst_arr );

VideoProcessor::VideoProcessor() {

}

VideoProcessor::~VideoProcessor() {
}

double VideoProcessor::imageQuality(const Image* image)
{
	assert(image->nChannels == 1);

	Image* real = cvCreateImage(cvGetSize(image), IPL_DEPTH_32F, 1);
	Image* imaginary = cvCreateImage(cvGetSize(image), IPL_DEPTH_32F, 1);
	Image* resultFT = cvCreateImage(cvGetSize(image), IPL_DEPTH_32F, 2);
	Image* powerSpectrum = cvCreateImage(cvGetSize(image), IPL_DEPTH_32F, 1);

	cvConvert(image, real);

	cvZero(imaginary);
	cvMerge(real, imaginary, NULL, NULL, resultFT);
	cvDFT(resultFT, resultFT, CV_DXT_FORWARD);

	cvSplit(resultFT, real, imaginary, NULL, NULL);

	// Calculate the power spectrum: re^2 + imag^2
	cvMul(real, real, real);
	cvMul(imaginary, imaginary, imaginary);
	cvAdd(real, imaginary, powerSpectrum);

	// Put the DC coefficient in the middle of the image (makes calculations easier)
	double dc = cvGetReal2D(powerSpectrum, 0, 0);
	cvShiftDFT(powerSpectrum, powerSpectrum);

	// Now, calculate the PHF
	double totalPower = 0.0;
	double phf = 0.0;

	int x0 = powerSpectrum->width/2;
	int y0 = powerSpectrum->height/2;
	int radius2 = powerSpectrum->width/8;
	radius2 = radius2*radius2;

	for (int x = 0; x < powerSpectrum->width; x++) {
		for (int y = 0; y < powerSpectrum->height; y++) {
			double val = cvGetReal2D(powerSpectrum, y, x);

			totalPower += val;
			if ((x-0)*(x-x0)+(y-y0)*(y-y0) > radius2) {
				phf += val;
			}
		}
	}

	cvReleaseImage(&real);
	cvReleaseImage(&imaginary);
	cvReleaseImage(&resultFT);
	cvReleaseImage(&powerSpectrum);

	return 100.0*std::log10(phf)/std::log10(totalPower);
}

/******************************************************************************/
// Rearrange the quadrants of Fourier image so that the origin is at
// the image center
// src & dst arrays of equal size & type
void cvShiftDFT(CvArr * src_arr, CvArr * dst_arr )
{
    CvMat * tmp = NULL;
    CvMat q1stub, q2stub;
    CvMat q3stub, q4stub;
    CvMat d1stub, d2stub;
    CvMat d3stub, d4stub;
    CvMat * q1, * q2, * q3, * q4;
    CvMat * d1, * d2, * d3, * d4;

    CvSize size = cvGetSize(src_arr);
    CvSize dst_size = cvGetSize(dst_arr);
    int cx, cy;

    if(dst_size.width != size.width ||
       dst_size.height != size.height){
        cvError( CV_StsUnmatchedSizes,
		   "cvShiftDFT", "Source and Destination arrays must have equal sizes",
		   __FILE__, __LINE__ );
    }

    if(src_arr==dst_arr){
        tmp = cvCreateMat(size.height/2, size.width/2, cvGetElemType(src_arr));
    }

    cx = size.width/2;
    cy = size.height/2; // image center

    q1 = cvGetSubRect( src_arr, &q1stub, cvRect(0,0,cx, cy) );
    q2 = cvGetSubRect( src_arr, &q2stub, cvRect(cx,0,cx,cy) );
    q3 = cvGetSubRect( src_arr, &q3stub, cvRect(cx,cy,cx,cy) );
    q4 = cvGetSubRect( src_arr, &q4stub, cvRect(0,cy,cx,cy) );
    d1 = cvGetSubRect( src_arr, &d1stub, cvRect(0,0,cx,cy) );
    d2 = cvGetSubRect( src_arr, &d2stub, cvRect(cx,0,cx,cy) );
    d3 = cvGetSubRect( src_arr, &d3stub, cvRect(cx,cy,cx,cy) );
    d4 = cvGetSubRect( src_arr, &d4stub, cvRect(0,cy,cx,cy) );

    if(src_arr!=dst_arr){
        if( !CV_ARE_TYPES_EQ( q1, d1 )){
            cvError( CV_StsUnmatchedFormats,
			"cvShiftDFT", "Source and Destination arrays must have the same format",
			__FILE__, __LINE__ );
        }
        cvCopy(q3, d1, 0);
        cvCopy(q4, d2, 0);
        cvCopy(q1, d3, 0);
        cvCopy(q2, d4, 0);
    }
    else{
        cvCopy(q3, tmp, 0);
        cvCopy(q1, q3, 0);
        cvCopy(tmp, q1, 0);
        cvCopy(q4, tmp, 0);
        cvCopy(q2, q4, 0);
        cvCopy(tmp, q2, 0);

		cvReleaseMat(&tmp);
    }
}
/******************************************************************************/
