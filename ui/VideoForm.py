# -*- coding: UTF8 -*-

from PyQt4 import QtCore, QtGui
import VideoThread

class VideoForm(QtGui.QWidget):
	def __init__(self, parent=None):
		super(QtGui.QWidget, self).__init__(parent)

	def activate(self, parent):
		QtCore.QObject.connect(VideoThread.videoThread, VideoThread.sigAvailableFrame, self.videoWidget.showImage, QtCore.Qt.BlockingQueuedConnection)
	
	def deactivate(self, parent):
		QtCore.QObject.disconnect(VideoThread.videoThread, VideoThread.sigAvailableFrame, self.videoWidget.showImage)