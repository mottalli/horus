# -*- coding: UTF8 -*-
from PyQt4 import uic
from PyQt4 import QtGui, QtCore

import VideoThread

(Ui_MainForm, Base) = uic.loadUiType('forms/MainForm.ui')

class MainForm(QtGui.QMainWindow, Ui_MainForm):
	VIDEO_TAB = 0
	SETUP_TAB = 1

	def __init__(self):
		QtGui.QMainWindow.__init__(self)
		self.setupUi(self)
		
		QtCore.QObject.connect(self.mainTabs, QtCore.SIGNAL("currentChanged(int)"), self.changedTab)
		self.changedTab(self.mainTabs.currentIndex())

	@QtCore.pyqtSignature("int")
	def changedTab(self, tabIndex):
		if tabIndex == self.VIDEO_TAB:
			QtCore.QObject.disconnect(VideoThread.videoThread, VideoThread.sigAvailableFrame, self.setupVideoWidget.showImage)
			QtCore.QObject.connect(VideoThread.videoThread, VideoThread.sigAvailableFrame, self.videoWidget.showImage, QtCore.Qt.BlockingQueuedConnection)
		elif tabIndex == self.SETUP_TAB:
			QtCore.QObject.disconnect(VideoThread.videoThread, VideoThread.sigAvailableFrame, self.videoWidget.showImage)
			QtCore.QObject.connect(VideoThread.videoThread, VideoThread.sigAvailableFrame, self.setupVideoWidget.showImage, QtCore.Qt.BlockingQueuedConnection)
