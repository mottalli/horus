# -*- coding: UTF8 -*-

from opencv import *
from opencv import highgui
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
		size = cvGetSize(image)
		self.setFixedSize(QtCore.QSize(size.width, size.height))
		self.repaint()
		
	def _convertImage(self, image):
		recreate = False
		size = cvGetSize(image)
		
		if self.buffer is None:
			recreate = True
		elif size.width != self.buffer.width or size.height != self.buffer.height:
			cvReleaseImage(self.buffer)
			cvReleaseImage(self.bufferR)
			cvReleaseImage(self.bufferG)
			cvReleaseImage(self.bufferB)
			recreate = True
			
		elemType = cvGetElemType(image)
		
		if recreate:
			self.buffer = cvCreateImage(size, IPL_DEPTH_8U, 4)
			self.bufferB = cvCreateImage(size, IPL_DEPTH_8U, 1)
			self.bufferG = cvCreateImage(size, IPL_DEPTH_8U, 1)
			self.bufferR = cvCreateImage(size, IPL_DEPTH_8U, 1)
		
		if elemType == CV_8UC1:
			cvMerge(image, None, None, None, self.buffer)
			cvMerge(None, image, None, None, self.buffer)
			cvMerge(None, None, image, None, self.buffer)
		elif elemType == CV_8UC3:
			cvSplit(image, self.bufferB, self.bufferG, self.bufferR, None)
			cvMerge(self.bufferB, None, None, None, self.buffer)
			cvMerge(None, self.bufferG, None, None, self.buffer)
			cvMerge(None, None, self.bufferR, None, self.buffer)
		else:
			raise Exception('Unsupported type')

		del self.image
		self.image = QtGui.QImage(self.buffer.imageData, size.width, size.height, QtGui.QImage.Format_RGB32)
