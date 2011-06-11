#!/usr/bin/python
# -*- coding: UTF8 -*-

import horus
import sys
from opencv.highgui import *
from opencv import cvCloneImage, cvCreateImage, cvGetSize, IPL_DEPTH_8U, cvCvtColor, CV_GRAY2BGR, CV_RGB


if __name__ == '__main__':
	if len(sys.argv) != 3:
		print "Uso: %s <imagen entrada> <imagen salida>" % sys.argv[0]
	
	imagen = cvLoadImage(sys.argv[1], 0)
	decorator = horus.Decorator()
	encoder = horus.LogGaborEncoder()
	segmentator = horus.Segmentator()

	sr = segmentator.segmentImage(imagen)
	template = encoder.generateTemplate(imagen, sr)
	
	decorada = cvCreateImage(cvGetSize(imagen), IPL_DEPTH_8U, 3)
	cvCvtColor(imagen, decorada, CV_GRAY2BGR)

	decorator.drawSegmentationResult(imagen, sr)
	decorator.drawTemplateWRAP(imagen, template)
	
	cvNamedWindow("decorada")
	cvShowImage("decorada", imagen)
	
	while True:
		k = cvWaitKey(0)
		if k == 'q': break

	print "Guardando imagen en /tmp/decorada.jpg..."
	cvSaveImage("/tmp/decorada.jpg", imagen)
