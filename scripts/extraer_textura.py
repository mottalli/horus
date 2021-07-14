#!/usr/bin/python

import sys
import horus
from opencv import cvCreateImage, IPL_DEPTH_8U, cvSize, cvGetSize
from opencv.highgui import *
from pylab import *
from math import pi

decorator = horus.Decorator()
segmentator = horus.Segmentator()
logGaborEncoder = horus.LogGaborEncoder()

if __name__ == '__main__':
	if len(sys.argv) != 2:
		print "Uso: ", sys.argv[0], "<imagen>"
		sys.exit(1)
	
	widthTextura, heightTextura = 512, 96
	theta0 = 0.0
	theta1 = 2.0*pi
	radius = 1.0
	
	imagen = cvLoadImage(sys.argv[1], 0)
	textura = cvCreateImage(cvSize(widthTextura, heightTextura), IPL_DEPTH_8U, 1)
	mascara = cvCreateImage(cvSize(widthTextura, heightTextura), IPL_DEPTH_8U, 1)
	
	sr = segmentator.segmentImage(imagen)
	logGaborEncoder.normalizeIris(imagen, textura, mascara, sr, theta0, theta1, radius)
	
	cvNamedWindow("textura")
	cvShowImage("textura", textura)
	
	cvWaitKey(0)
	
	print "Textura guardada en /tmp/textura.png"
	cvSaveImage("/tmp/textura.png", textura)
