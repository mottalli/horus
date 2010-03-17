#!/usr/bin/python
# -*- coding: UTF8 -*-

import horus
import sys
from opencv.highgui import *
from opencv import *

paths = sys.argv[1:]
segmentator = horus.Segmentator()
decorator = horus.Decorator()
encoder = horus.LogGaborEncoder()

#p = horus.Parameters.getParameters()
#p.normalizationWidth = p.templateWidth = 512
#p.normalizationHeight = p.templateHeight = 48
#encoder = horus.IrisDCTEncoder()

imagenes = []
for path in paths:
	imagenes.append(cvLoadImage(path, 1))

while True:
	templates = []
	for i in range(len(imagenes)):
		rs = segmentator.segmentImage(imagenes[i])
		templates.append(encoder.generateTemplate(imagenes[i], rs))
	
		decorada = cvCloneImage(imagenes[i])
		decorator.drawSegmentationResult(decorada, rs)
		decorator.drawTemplate(decorada, templates[i])
		decorator.drawEncodingZone(decorada, rs)
		cvNamedWindow(paths[i])
		cvShowImage(paths[i], decorada)
	
	for i in range(len(imagenes)):
		comparator = horus.TemplateComparator(templates[i])
		for j in range(len(imagenes)):
			if i == j: continue
			
			print 'Distancia entre %s y %s: %.4f' % (paths[i], paths[j], comparator.compare(templates[j]))
	
	while True:
		k = cvWaitKey(0)
		if k == 'q': sys.exit(0)
