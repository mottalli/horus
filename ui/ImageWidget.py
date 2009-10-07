# -*- coding: UTF8 -*-

from opencv import cvGetSize, cvCreateImage, cvReleaseImage, cvMerge, cvSplit, IPL_DEPTH_8U
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
		if self.buffer is None:
			recreate = True
		elif image.width != self.buffer.width or image.height != self.buffer.height:
			cvReleaseImage(self.buffer)
			cvReleaseImage(self.bufferR)
			cvReleaseImage(self.bufferG)
			cvReleaseImage(self.bufferB)
			recreate = True
		
		if recreate:
			self.buffer = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 4)
			self.bufferB = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1)
			self.bufferG = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1)
			self.bufferR = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1)
		
		if image.nChannels == 1:
			cvMerge(image, None, None, None, self.buffer)
			cvMerge(None, image, None, None, self.buffer)
			cvMerge(None, None, image, None, self.buffer)
		elif image.nChannels == 3:
			cvSplit(image, self.bufferB, self.bufferG, self.bufferR, None)
			cvMerge(self.bufferB, None, None, None, self.buffer)
			cvMerge(None, self.bufferG, None, None, self.buffer)
			cvMerge(None, None, self.bufferR, None, self.buffer)
		else:
			pass

		self.__data = self.buffer.imageData
		
		size = cvGetSize(image)
		del self.image
		self.image = QtGui.QImage(self.__data, size.width, size.height, QtGui.QImage.Format_RGB32)
