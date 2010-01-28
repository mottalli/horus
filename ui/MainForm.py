# -*- coding: UTF8 -*-
from PyQt4 import uic
from PyQt4 import QtGui, QtCore
from opencv import *
from opencv.highgui import *
import os.path

import VideoThread, ProcessingThread
import horus
import Database

(Ui_MainForm, Base) = uic.loadUiType('forms/MainForm.ui')

class MainForm(QtGui.QMainWindow, Ui_MainForm):
	def __init__(self):
		QtGui.QMainWindow.__init__(self)
		self.setupUi(self)
		
		self.lastSegmentationResult = None
		self.decorator = horus.Decorator()
		self.focusedIris = False
		self.forzarIdentificacionProxima = False
		self.capturarProxima = False
		self.frameDecorado = None
		self.frameCapturado = None
		self.frameCapturadoDecorado = None
		self.frameCapturadoDecoradoThumbnail = None
		self.lastTemplate = None
		
		self.database = Database.database

		self.segmentator = horus.Segmentator()
		self.encoder = horus.IrisEncoder()

		
		QtCore.QObject.connect(VideoThread.videoThread, VideoThread.sigAvailableFrame, self.availableFrame)
		QtCore.QObject.connect(ProcessingThread.processingThread, ProcessingThread.sigProcessedFrame, self.processedFrame)

	def availableFrame(self, frame):
		if (self.lastSegmentationResult):
			if not self.frameDecorado:
				self.frameDecorado = cvCreateImage(cvGetSize(frame), IPL_DEPTH_8U, 3)
			cvCopy(frame, self.frameDecorado)
			self.decorator.drawSegmentationResult(self.frameDecorado, self.lastSegmentationResult)
			self.videoWidget.showImage(self.frameDecorado)
		else:
			self.videoWidget.showImage(frame)
		
		if self.forzarIdentificacionProxima:
			self.forzarIdentificacion(frame)
		
		if self.capturarProxima:
			self.capturarProxima = False
			self.capturar(frame)
	
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
			self.statusBar.showMessage('Esperando...')
		elif resultado == horus.VideoProcessor.DEFOCUSED:
			self.statusBar.showMessage('Desenfocado')
		elif resultado == horus.VideoProcessor.FOCUSED_NO_IRIS:
			self.statusBar.showMessage('Imagen sin iris')
		elif resultado == horus.VideoProcessor.IRIS_LOW_QUALITY:
			self.statusBar.showMessage('Baja calidad de iris')
		elif resultado == horus.VideoProcessor.IRIS_TOO_CLOSE:
			self.statusBar.showMessage('Iris demasiado cerca')
		elif resultado == horus.VideoProcessor.IRIS_TOO_FAR:
			self.statusBar.showMessage('Iris demasiado lejos')
		elif resultado == horus.VideoProcessor.FOCUSED_IRIS:
			self.statusBar.showMessage('Iris enfocado')
		elif resultado == horus.VideoProcessor.GOT_TEMPLATE:
			self.statusBar.showMessage('Imagen capturada')
			self.gotTemplate(videoProcessor)
	
	def gotTemplate(self, videoProcessor):
		templateFrame = videoProcessor.getTemplateFrame()
		if not self.frameCapturado:
			size = cvGetSize(templateFrame)
			self.frameCapturado = cvCreateImage(cvGetSize(templateFrame), IPL_DEPTH_8U, 1)
			self.frameCapturadoDecorado = cvCreateImage(cvGetSize(templateFrame), IPL_DEPTH_8U, 3)
			#self.frameCapturadoDecoradoThumbnail = cvCreateImage(cvSize(size.width/2, size.height/2), IPL_DEPTH_8U, 3)
			self.frameCapturadoDecoradoThumbnail = cvCreateImage(cvSize(320,240), IPL_DEPTH_8U, 3)
		
		cvCopy(templateFrame, self.frameCapturado)
		cvCvtColor(self.frameCapturado, self.frameCapturadoDecorado, CV_GRAY2BGR)
		self.decorator.pupilColor = CV_RGB(0,255,0)
		self.decorator.irisColor = CV_RGB(255,0,0)
		self.decorator.drawSegmentationResult(self.frameCapturadoDecorado, videoProcessor.getTemplateSegmentation())
		cvResize(self.frameCapturadoDecorado, self.frameCapturadoDecoradoThumbnail, CV_INTER_CUBIC)
		
		self.capturedImage.showImage(self.frameCapturadoDecoradoThumbnail)
		self.lastTemplate = videoProcessor.getTemplate()
		
		if self.chkIdentificacionAutomatica.checkState() == QtCore.Qt.Checked:
			self.identificarTemplate(self.lastTemplate)
	
	@QtCore.pyqtSignature("")
	def on_btnForzarIdentificacion_clicked(self):
		self.forzarIdentificacionProxima = True
	
	@QtCore.pyqtSignature("")
	def on_btnIdentificar_clicked(self):
		if self.lastTemplate:
			self.identificarTemplate(self.lastTemplate)

	@QtCore.pyqtSignature("")
	def on_btnCapturar_clicked(self):
		self.capturarProxima = True


	def forzarIdentificacion(self, frame):
		self.forzarIdentificacionProxima = False
		print "Forzando"

	def identificarTemplate(self, template):
		self.database.doMatch(template)
		print (self.database.irisDatabase.getMinDistanceId(), self.database.irisDatabase.getMinDistance())
		
		self.database.irisDatabase.doAContrarioMatch(template)
		print (self.database.irisDatabase.getMinNFAId(), self.database.irisDatabase.getMinNFA())
	
	def capturar(self, frame):
		i = 0
		while True:
			nombreArchivo = "cap%i.jpg" % (i)
			if not os.path.exists(nombreArchivo):
				cvSaveImage(nombreArchivo, frame)
				print "Captura guardada en", nombreArchivo
				break
			else:
				i = i+1
