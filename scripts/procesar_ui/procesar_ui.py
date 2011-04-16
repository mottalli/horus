#!/usr/bin/python
from PyQt4 import QtGui, QtCore, uic
import os
import sys
import horus
from opencv import cvCloneImage, cvNormalize, CV_MINMAX
from opencv.highgui import cvLoadImage

app = QtGui.QApplication(sys.argv)
(Ui_MainForm, Base) = uic.loadUiType('MainForm.ui')

def normalizarImagen(imagen):
	normalizada = cvCloneImage(imagen)
	cvNormalize(imagen, normalizada, 0, 255, CV_MINMAX)

class MainForm(QtGui.QMainWindow, Ui_MainForm):
	imagen = None
	imagenSegmentada = None
	pathImagen = None

	segmentator = horus.Segmentator()
	decorator = horus.Decorator()
	#qualityChecker = horus.QualityChecker()
	logGaborEncoder = horus.LogGaborEncoder()

	def __init__(self):
		QtGui.QMainWindow.__init__(self)
		self.setupUi(self)
	
	def procesarPath(self, path):
		imagen = cvLoadImage(path, 0)
		if not imagen: raise Exception("No se pudo abrir el archivo " + path)

		self.imagen = imagen
		self.pathImagen = path
		
		self.procesarImagen()
	
	def procesarImagen(self):
		sr = self.segmentator.segmentImage(self.imagen)
		self.imagenSegmentada = cvCloneImage(self.imagen)
		self.decorator.drawSegmentationResult(self.imagenSegmentada, sr)
		
		self.mostrarImagenSegmentada()
		self.mostrar
		
	def mostrarImagenSegmentada(self):
		self.imgImagen.showImage(self.imagenSegmentada)

	@QtCore.pyqtSignature("")
	def on_actionAbrir_triggered(self):
		#nombreArchivo = QtGui.QFileDialog.getOpenFileName(self, "Abrir archivo...")
		#if nombreArchivo:
		#	self.procesarPath(str(nombreArchivo))
		self.procesarPath("/home/marcelo/Desktop/iris_capturado.jpg")
		pass

mainForm = MainForm()
mainForm.show()
res = app.exec_()
sys.exit(res)
