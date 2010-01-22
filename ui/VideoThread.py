# -*- coding: UTF8 -*-

from PyQt4.QtCore import QObject, QThread, SIGNAL
from opencv import cvGet2D, cvGetSubRect, cvRect
from opencv import highgui
import os

sigAvailableFrame = SIGNAL("availableFrame(PyQt_PyObject)")

class VideoThread(QThread):
	def __init__(self):
		QThread.__init__(self)
		
class VideoThreadCamera(VideoThread):
	def __init__(self, cameraNumber=0):
		super(VideoThreadCamera, self).__init__()
		self.cameraNumber = cameraNumber
		self.isWindows = (os.name == 'nt')
		self.finish = False
		
	def run(self):
		cap = highgui.cvCreateCameraCapture(self.cameraNumber)
		if not cap:
			raise IOError("Unable to initialize capture")
			
		while True:
			frame = highgui.cvQueryFrame(cap)
			if not frame:
				raise IOError("Error capturing frame")
			
			captured = frame
			# Fix for TVGo A03MCE
			if self.isWindows:
				value = cvGet2D(frame, 10, 10)
				if int(value[0]) == 5: continue
				cvFlip(frame, frame)
				if frame.width == 720 and frame.height == 480:
					captured = cvGetSubRect(frame, cvRect(30, 0, 640, 480))
			
			# Extract the borders (they usually come with black bands)
			delta = 10
			#subwindow = cvRect(delta, 0, frame.width-2*delta, frame.height)
			subwindow = cvRect(60, 40, frame.width-70, frame.height-40)
			portion = cvGetSubRect(captured, subwindow)
			self.emit(sigAvailableFrame, portion)
	
videoThread = VideoThreadCamera()
