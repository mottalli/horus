/* 
 * File:   main.cpp
 * Author: marcelo
 *
 * Created on January 22, 2009, 2:01 PM
 */

#include <iostream>

#include "segmentator.h"

/*
 * 
 */
int main(int argc, char** argv) {
    IplImage* imagen = cvLoadImage("/home/marcelo/Mis_Documentos/Facu/Tesis/Bases de datos/UBA/marcelo_izq_1.bmp", 0);

    Segmentator segmentator;
    SegmentationResult res = segmentator.segmentImage(imagen);

    cvCircle(segmentator.buffers.workingImage, cvPoint(res.pupilCircle.xc, res.pupilCircle.yc), res.pupilCircle.radius, CV_RGB(255,255,255));

    cvNamedWindow("imagen");
    cvShowImage("imagen", segmentator.buffers.workingImage);


    cvWaitKey(0);
    cvReleaseImage(&imagen);
}

