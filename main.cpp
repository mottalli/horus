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
    IplImage* imagen = cvLoadImage("/home/marcelo/Mis_Documentos/Facu/Tesis/Bases de datos/UBA/marcelo_der_1.bmp", 0);

    Segmentator segmentator;
    segmentator.segmentImage(imagen);

    cvNamedWindow("imagen");
    cvShowImage("imagen", segmentator._pupilSegmentator.buffers.similarityImage);


    cvWaitKey(0);
    cvReleaseImage(&imagen);
}

