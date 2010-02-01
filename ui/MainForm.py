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
		self.thumbnailColorTmp = None
		self.thumbnail = None
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
			self.forzarIdentificacionProxima = False
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
		self.lastTemplate = videoProcessor.getTemplate()
		templateFrame = videoProcessor.getTemplateFrame()
		self.mostrarThumbnail(templateFrame, videoProcessor.getTemplateSegmentation(), self.lastTemplate())

		if self.chkIdentificacionAutomatica.checkState() == QtCore.Qt.Checked:
			self.identificarTemplate(self.lastTemplate)
	
	def mostrarThumbnail(self, imagen, segmentacion=None, template=None):
		size = cvGetSize(imagen)
		if not self.thumbnailColorTmp or self.thumbnailColorTmp.width != size.width or self.thumbnailColorTmp.height != size.height:
			self.thumbnailColorTmp = cvCreateImage(size, IPL_DEPTH_8U, 3)
			self.thumbnail = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 3)
		
		cvCvtColor(imagen, self.thumbnailColorTmp, CV_GRAY2BGR)
		if segmentacion:
			self.decorator.pupilColor = CV_RGB(0,255,0)
			self.decorator.irisColor = CV_RGB(255,0,0)
			self.decorator.drawSegmentationResult(self.thumbnailColorTmp, segmentacion)
		if template:
			self.decorator.drawTemplate(self.thumbnailColorTmp, template)
		cvResize(self.thumbnailColorTmp, self.thumbnail, CV_INTER_CUBIC)
		
		self.capturedImage.showImage(self.thumbnail)
	
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

	@QtCore.pyqtSignature("")
	def on_btnRegistrar_clicked(self):
		#TODO: Cambiar esto a la acción del menú "Procesar archivo"
		nombreArchivo = QtGui.QFileDialog.getOpenFileName(self, "Abrir archivo...")
		imagen = cvLoadImage(str(nombreArchivo), 0)
		self.forzarIdentificacion(imagen)

	def forzarIdentificacion(self, imagen):
		segmentacion = self.segmentator.segmentImage(imagen)
		template = self.encoder.generateTemplate(imagen, segmentacion)
		self.mostrarThumbnail(imagen, segmentacion, template)
		self.identificarTemplate(template)

	def identificarTemplate(self, template):
		self.database.doMatch(template)
		print (self.database.irisDatabase.getMinDistanceId(), self.database.irisDatabase.getMinDistance())
		
		self.database.irisDatabase.doAContrarioMatch(template)
		import pylab
		a = pylab.array(self.database.irisDatabase.resultNFAs)
		print a


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
