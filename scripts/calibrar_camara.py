#!/usr/bin/python
# -*- coding: UTF8 -*-

from pyhorus import *
from opencv import cvFlip, CV_RGB, cvGetSize, cvCircle, cvPoint
from opencv.highgui import *

WIDTH_IRIS = 200

cap = cvCreateCameraCapture(0)
cvNamedWindow("video")
while True:
	frame = cvQueryFrame(cap)
	if not frame:
		raise IOError("Error capturando")
	
	cvFlip(frame, frame, 1)
	
	size = cvGetSize(frame)
	cvCircle(frame, cvPoint(size.width/2, size.height/2), WIDTH_IRIS/2, CV_RGB(255,255,255))
	
	cvShowImage("video", frame)
	
	if cvWaitKey(20) == 'q': break
