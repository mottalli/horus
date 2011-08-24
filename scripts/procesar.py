#!/usr/bin/python
# -*- coding: UTF8 -*-

import pyhorus
import sys
from opencv.highgui import *
from opencv import cvCloneImage, cvCreateImage, cvGetSize, IPL_DEPTH_8U, cvCvtColor, CV_GRAY2BGR

paths = sys.argv[1:]
segmentator = pyhorus.Segmentator()
decorator = pyhorus.Decorator()
encoder = pyhorus.LogGaborEncoder()
#encoder = pyhorus.GaborEncoder()


imagenes = []
for path in paths:
	imagenes.append(cvLoadImage(path, 0))

while True:
	templates = []
	for i in range(len(imagenes)):
		rs = segmentator.segmentImage(imagenes[i])
		templates.append(encoder.generateTemplate(imagenes[i], rs))
	
		#decorada = cvCloneImage(imagenes[i])
		decorada = cvCreateImage(cvGetSize(imagenes[i]), IPL_DEPTH_8U, 3)
		cvCvtColor(imagenes[i], decorada, CV_GRAY2BGR)
		decorator.drawSegmentationResult(decorada, rs)
		#decorator.drawTemplate(decorada, templates[i])
		decorator.drawEncodingZone(decorada, rs)
		cvNamedWindow(paths[i])
		cvShowImage(paths[i], decorada)
	
	for i in range(len(imagenes)):
		comparator = pyhorus.TemplateComparator(templates[i])
		for j in range(len(imagenes)):
			if i == j: continue
			
			print 'Distancia entre %s y %s: %.4f' % (paths[i], paths[j], comparator.compare(templates[j]))
	
	while True:
		k = cvWaitKey(0)
		if k == 'q': sys.exit(0)

