# -*- coding: UTF8 -*-
from PyQt4 import uic
from PyQt4 import QtGui, QtCore
from opencv import *

import VideoThread, ProcessingThread
import horus


(Ui_MainForm, Base) = uic.loadUiType('forms/MainForm.ui')

class MainForm(QtGui.QMainWindow, Ui_MainForm):
	VIDEO_TAB = 0
	SETUP_TAB = 1
	

	def __init__(self):
		QtGui.QMainWindow.__init__(self)
		self.setupUi(self)
		
		QtCore.QObject.connect(self.mainTabs, QtCore.SIGNAL("currentChanged(int)"), self.changedTab)

		self.changedTab(self.mainTabs.currentIndex())
		self.lastSegmentationResult = None
		self.decorator = horus.Decorator()
		self.focusedIris = False

	@QtCore.pyqtSignature("int")
	def changedTab(self, tabIndex):
		# NO poner esto dentro de la clase (como un self.VIDEO_SIGNALS) porque falla a la salida
		VIDEO_SIGNALS = [
			(VideoThread.videoThread, VideoThread.sigAvailableFrame, self.availableFrame),
			(ProcessingThread.processingThread, ProcessingThread.sigProcessedFrame, self.processedFrame)
		]
		SETUP_SIGNALS = [
			(VideoThread.videoThread, VideoThread.sigAvailableFrame, self.setupVideoWidget.showImage)
		]

		if tabIndex == self.VIDEO_TAB:
			sIn = VIDEO_SIGNALS
			sOut = SETUP_SIGNALS
		elif tabIndex == self.SETUP_TAB:
			sIn = SETUP_SIGNALS
			sOut = VIDEO_SIGNALS
		
		for (obj, signal, callback) in sOut:
			QtCore.QObject.disconnect(obj, signal, callback)
		
		for (obj, signal, callback) in sIn:
			QtCore.QObject.connect(obj, signal, callback, QtCore.Qt.BlockingQueuedConnection)
		
		self.currentTab = tabIndex
	
	def inVideoTab(self):
		return self.currentTab == VIDEO_TAB
	
	def inSetupTab(self):
		return self.currentTab == SETUP_TAB
			
	def availableFrame(self, frame):
		if (self.lastSegmentationResult):
			self.decorator.drawSegmentationResult(frame, self.lastSegmentationResult)
		
		self.videoWidget.showImage(frame)
	
	def processedFrame(self, resultado, videoProcessor):
		if resultado >= horus.VideoProcessor.FOCUSED_NO_IRIS:
			self.lastSegmentationResult = videoProcessor.lastSegmentationResult
		else:
			self.lastSegmentationResult = None
		
		if resultado >= horus.VideoProcessor.IRIS_LOW_QUALITY:
			self.irisScore.setValue(videoProcessor.lastSegmentationScore)
		else:
			self.irisScore.setValue(0)
			
		if resultado >= horus.VideoProcessor.FOCUSED_IRIS:
			self.decorator.pupilColor = CV_RGB(0,255,0)
			self.decorator.irisColor = CV_RGB(255,0,0)
		else:
			self.decorator.pupilColor = CV_RGB(255,255,255)
			self.decorator.irisColor = CV_RGB(255,255,255)
		
		self.focusScore.setValue(videoProcessor.lastFocusScore)
	
		if resultado == horus.VideoProcessor.DEFOCUSED:
			print 'DEFOCUSED'
		elif resultado == horus.VideoProcessor.FOCUSED_NO_IRIS:
			print 'FOCUSED_NO_IRIS'
		elif resultado == horus.VideoProcessor.IRIS_LOW_QUALITY:
			print 'IRIS_LOW_QUALITY'
		elif resultado == horus.VideoProcessor.IRIS_TOO_CLOSE:
			print 'IRIS_TOO_CLOSE'
		elif resultado == horus.VideoProcessor.IRIS_TOO_FAR:
			print 'IRIS_TOO_FAR'
		elif resultado == horus.VideoProcessor.FOCUSED_IRIS:
			print 'FOCUSED_IRIS'
		elif resultado == horus.VideoProcessor.GOT_TEMPLATE:
			print 'GOT_TEMPLATE'

