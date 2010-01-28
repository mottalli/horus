# -*- coding: UTF8 -*-
from PyQt4 import uic
from PyQt4 import QtGui, QtCore
from opencv import *

import VideoThread, ProcessingThread
import horus


(Ui_MainForm, Base) = uic.loadUiType('forms/MainForm.ui')

class MainForm(QtGui.QMainWindow, Ui_MainForm):
	def __init__(self):
		QtGui.QMainWindow.__init__(self)
		self.setupUi(self)
		
		self.lastSegmentationResult = None
		self.decorator = horus.Decorator()
		self.focusedIris = False
		self.forzarIdentificacionProxima = False
		
		QtCore.QObject.connect(VideoThread.videoThread, VideoThread.sigAvailableFrame, self.availableFrame)
		QtCore.QObject.connect(ProcessingThread.processingThread, ProcessingThread.sigProcessedFrame, self.processedFrame)

	def availableFrame(self, frame):
		if (self.lastSegmentationResult):
			self.decorator.drawSegmentationResult(frame, self.lastSegmentationResult)
		
		self.videoWidget.showImage(frame)
		
		if self.forzarIdentificacionProxima:
			self.forzarIdentificacion(frame)
	
	def processedFrame(self, resultado, videoProcessor):
		if resultado >= horus.VideoProcessor.FOCUSED_NO_IRIS:
			self.lastSegmentationResult = videoProcessor.lastSegmentationResult
		else:
			self.lastSegmentationResult = None
		
		if resultado >= horus.VideoProcessor.IRIS_LOW_QUALITY:
			self.irisScore.setValue(videoProcessor.lastIrisQuality)
		else:
			self.irisScore.setValue(0)
			
		if resultado >= horus.VideoProcessor.FOCUSED_IRIS:
			self.decorator.pupilColor = CV_RGB(0,255,0)
			self.decorator.irisColor = CV_RGB(255,0,0)
		else:
			self.decorator.pupilColor = CV_RGB(255,255,255)
			self.decorator.irisColor = CV_RGB(255,255,255)
		
		if resultado == horus.VideoProcessor.GOT_TEMPLATE:
			self.decorator.pupilColor = CV_RGB(255,255,0)
			self.decorator.irisColor = CV_RGB(255,255,0)
			
		
		self.focusScore.setValue(videoProcessor.lastFocusScore)
	
		if resultado == horus.VideoProcessor.UNPROCESSED:
			print 'UNPROCESSED'
		elif resultado == horus.VideoProcessor.DEFOCUSED:
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
			self.gotTemplate(videoProcessor)
	
	def gotTemplate(self, videoProcessor):
		pass
	
	def on_btnForzarIdentificacion_clicked(self):
		self.forzarIdentificacionProxima = True

	def forzarIdentificacion(self, frame):
		self.forzarIdentificacionProxima = False
		print "Forzando"
