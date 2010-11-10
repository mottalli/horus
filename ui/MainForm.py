# -*- coding: UTF8 -*-
from PyQt4 import uic
from PyQt4 import QtGui, QtCore
import opencv
import opencv.highgui
import os.path

import VideoThread, ProcessingThread
import horus
import Database
import Utils

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
		self.lastCapture = None
		self.templateFrame = None
		self.templateFrameSegmentation = None
		self.ding = True
		
		self.database = Database.database

		self.segmentator = horus.Segmentator()
		self.encoder = horus.LogGaborEncoder()

		
		QtCore.QObject.connect(VideoThread.videoThread, VideoThread.sigAvailableFrame, self.availableFrame)
		QtCore.QObject.connect(ProcessingThread.processingThread, ProcessingThread.sigProcessedFrame, self.processedFrame)

	def availableFrame(self, frame):
		if (self.lastSegmentationResult):
			if not self.frameDecorado:
				self.frameDecorado = opencv.cvCreateImage(opencv.cvGetSize(frame), opencv.IPL_DEPTH_8U, 3)
			opencv.cvCopy(frame, self.frameDecorado)
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
			self.decorator.pupilColor = opencv.CV_RGB(0,255,0)
			self.decorator.irisColor = opencv.CV_RGB(255,0,0)
		else:
			self.decorator.pupilColor = opencv.CV_RGB(255,255,255)
			self.decorator.irisColor = opencv.CV_RGB(255,255,255)
		
		if resultado == horus.VideoProcessor.GOT_TEMPLATE:
			self.decorator.pupilColor = opencv.CV_RGB(255,255,0)
			self.decorator.irisColor = opencv.CV_RGB(255,255,0)
			
		
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
			if self.ding:
				QtGui.QSound.play('Ding.wav')
				self.ding = False
		elif resultado == horus.VideoProcessor.GOT_TEMPLATE:
			self.statusBar.showMessage('Imagen capturada')
			self.gotTemplate(videoProcessor)
	
	def gotTemplate(self, videoProcessor):
		self.lastTemplate = videoProcessor.getTemplate()
		self.templateFrame = horus.pyutilCloneFromHorus(videoProcessor.getTemplateFrame())
		self.templateFrameSegmentation = videoProcessor.getTemplateSegmentation()
		self.mostrarThumbnail(self.templateFrame, self.templateFrameSegmentation, self.lastTemplate)

		if self.radioIdentificar.isChecked():
			self.identificarTemplate(self.lastTemplate, self.templateFrame, self.templateFrameSegmentation)
		elif self.radioRegistrar.isChecked():
			pass
		elif self.radioCapturar.isChecked():
			pass
		
		self.lastCapture = opencv.cvClone(self.templateFrame)
		self.ding = True		# Pone el sonido en la próxima captura
	
	def mostrarThumbnail(self, imagen, segmentacion=None, template=None):
		size = opencv.cvGetSize(imagen)
		if not self.thumbnailColorTmp or self.thumbnailColorTmp.width != size.width or self.thumbnailColorTmp.height != size.height:
			self.thumbnailColorTmp = opencv.cvCreateImage(size, opencv.IPL_DEPTH_8U, 3)
			self.thumbnail = opencv.cvCreateImage(opencv.cvSize(320, 240), opencv.IPL_DEPTH_8U, 3)
		
		opencv.cvCvtColor(imagen, self.thumbnailColorTmp, opencv.CV_GRAY2BGR)
		if segmentacion:
			self.decorator.irisColor = opencv.CV_RGB(0,255,0)
			self.decorator.pupilColor = opencv.CV_RGB(255,0,0)
			self.decorator.drawSegmentationResult(self.thumbnailColorTmp, segmentacion)
		opencv.cvResize(self.thumbnailColorTmp, self.thumbnail, opencv.CV_INTER_CUBIC)
		
		if template:
			self.decorator.drawTemplate(self.thumbnail, template)

		self.capturedImage.showImage(self.thumbnail)
	
	@QtCore.pyqtSignature("")
	def on_btnForzarIdentificacion_clicked(self):
		self.forzarIdentificacionProxima = True
	
	@QtCore.pyqtSignature("")
	def on_btnIdentificar_clicked(self):
		if self.lastTemplate:
			self.identificarTemplate(self.lastTemplate, self.templateFrame, self.templateFrameSegmentation)

	@QtCore.pyqtSignature("")
	def on_btnCapturar_clicked(self):
		self.capturarProxima = True

	@QtCore.pyqtSignature("")
	def on_btnRegistrar_clicked(self):
		#TODO: Cambiar esto a la acción del menú "Procesar archivo"
		#nombreArchivo = QtGui.QFileDialog.getOpenFileName(self, "Abrir archivo...")
		#imagen = opencv.cvLoadImage(str(nombreArchivo), 0)
		#self.forzarIdentificacion(imagen)
		self.registrarTemplate(self.lastTemplate, self.templateFrame, self.templateFrameSegmentation)

	@QtCore.pyqtSignature("")
	def on_btnGuardarImagen_clicked(self):
		self.capturar(self.lastCapture)


	def forzarIdentificacion(self, imagen):
		imagen = Utils.aBlancoYNegro(imagen)
		segmentacion = self.segmentator.segmentImage(imagen)
		template = self.encoder.generateTemplate(imagen, segmentacion)
		self.mostrarThumbnail(imagen, segmentacion, template)
		self.identificarTemplate(template, imagen, segmentacion)

	def identificarTemplate(self, template, imagen=None, segmentacion=None):
		import Matching
		Matching.doMatch(self.database, template, imagen, segmentacion)

	def registrarTemplate(self, template, imagen, segmentacion):
		import Registracion
		Registracion.registrar(self.database, template, imagen, segmentacion)
	
	def capturar(self, frame):
		frame = opencv.cvCloneImage(frame)
		nombreArchivo = QtGui.QFileDialog.getSaveFileName(self, "Guardar archivo...")
		opencv.highgui.cvSaveImage(str(nombreArchivo), frame)
