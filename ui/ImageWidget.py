# -*- coding: UTF8 -*-

import cv as opencv
import highgui
from PyQt4 import QtCore, QtGui

class ImageWidget(QtGui.QWidget):
	def __init__(self, resize=False, parent=None):
		super(QtGui.QWidget, self).__init__(parent)
		
		self.resize = resize
		
		self.image = None
		self.buffer = None
		self.bufferR = None
		self.bufferG = None
		self.bufferB = None
	
	def paintEvent(self, event):
		if self.image is None:
			return
		
		painter = QtGui.QPainter(self)
		painter.drawImage(QtCore.QPoint(0,0), self.image)
	
	def showImage(self, image):
		self._convertImage(image)
		size = opencv.cvGetSize(image)
		self.setFixedSize(QtCore.QSize(size.width, size.height))
		self.repaint()
		
	def _convertImage(self, image):
		recreate = False
		size = opencv.cvGetSize(image)
		
		if self.buffer is None:
			recreate = True
		elif size.width != self.buffer.width or size.height != self.buffer.height:
			opencv.cvReleaseImage(self.buffer)
			opencv.cvReleaseImage(self.bufferR)
			opencv.cvReleaseImage(self.bufferG)
			opencv.cvReleaseImage(self.bufferB)
			recreate = True
			
		elemType = opencv.cvGetElemType(image)
		
		if recreate:
			self.buffer = opencv.cvCreateImage(size, opencv.IPL_DEPTH_8U, 4)
			self.bufferB = opencv.cvCreateImage(size, opencv.IPL_DEPTH_8U, 1)
			self.bufferG = opencv.cvCreateImage(size, opencv.IPL_DEPTH_8U, 1)
			self.bufferR = opencv.cvCreateImage(size, opencv.IPL_DEPTH_8U, 1)
		
		if elemType == opencv.CV_8UC1:
			opencv.cvMerge(image, None, None, None, self.buffer)
			opencv.cvMerge(None, image, None, None, self.buffer)
			opencv.cvMerge(None, None, image, None, self.buffer)
		elif elemType == opencv.CV_8UC3:
			opencv.cvSplit(image, self.bufferB, self.bufferG, self.bufferR, None)
			opencv.cvMerge(self.bufferB, None, None, None, self.buffer)
			opencv.cvMerge(None, self.bufferG, None, None, self.buffer)
			opencv.cvMerge(None, None, self.bufferR, None, self.buffer)
		else:
			raise Exception('Unsupported type')

		del self.image
		self.__data = self.buffer.imageData
		self.image = QtGui.QImage(self.__data, size.width, size.height, QtGui.QImage.Format_RGB32)
