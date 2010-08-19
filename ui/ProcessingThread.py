# -*- coding: UTF8 -*-

import VideoThread
from PyQt4 import QtCore
from cv import cvGetSize, cvCvtColor, cvCreateImage, cvReleaseImage, IPL_DEPTH_8U, CV_BGR2GRAY
import horus

sigProcessedFrame = QtCore.SIGNAL("processedFrame(PyQt_PyObject, PyQt_PyObject)")

class ProcessingThread(QtCore.QThread):
	def __init__(self):
		super(QtCore.QThread,self).__init__()

		self.lastFrame = None
		QtCore.QObject.connect(VideoThread.videoThread, VideoThread.sigAvailableFrame, self.availableFrame, QtCore.Qt.BlockingQueuedConnection)
		
		self.videoProcessor = horus.VideoProcessor()
		
	def run(self):
		res = self.videoProcessor.processFrame(self.lastFrame)
		"""typedef enum {
			DEFOCUSED,
			INTERLACED,
			FOCUSED_NO_IRIS,
			IRIS_LOW_QUALITY,
			IRIS_TOO_CLOSE,
			IRIS_TOO_FAR,
			FOCUSED_IRIS,
			GOT_TEMPLATE
		} VideoStatus;"""
		self.emit(sigProcessedFrame, res, self.videoProcessor)
	
	def availableFrame(self, frame):
		if self.isRunning():
			return
		
		if not self.lastFrame or self.lastFrame.width != frame.width or self.lastFrame.height != frame.height:
			if self.lastFrame != None:
				cvReleaseImage(self.lastFrame)
			self.lastFrame = cvCreateImage(cvGetSize(frame), IPL_DEPTH_8U, 1)
		
		if frame.nChannels == 3:
			cvCvtColor(frame, self.lastFrame, CV_BGR2GRAY)
		else:
			cvCopy(frame, self.lastFrame)
		
		self.start()

processingThread = ProcessingThread()
