# -*- coding: UTF8 -*-
from PyQt4 import uic, QtGui, QtCore
from horus import Decorator
import opencv
from opencv import highgui
import Utils

decorator = Decorator()
(Ui_Matching, Base) = uic.loadUiType('forms/Matching.ui')

class MatchingForm(QtGui.QDialog, Ui_Matching):
	database = None
	informacionUsuario = None
	imagen = None

	def __init__(self):
		QtGui.QMainWindow.__init__(self)
		self.setupUi(self)
		
	@QtCore.pyqtSignature("")
	def on_btnConfirmarIdentificacion_clicked(self):
		if self.database is None:
			return
			
		agregar = QtGui.QMessageBox.question(self, 'Agregar imagen', 'Agregar imagen a la base de datos?', QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)
		
		if agregar == QtGui.QMessageBox.Yes:	
			self.database.agregarImagen(self.informacionUsuario['id'], self.imagen)
		
		self.accept()
		


form = MatchingForm()

def doMatch(irisDatabase, template, imagen=None, segmentacion=None):
	if form.isVisible():
		return
	
	imagen = opencv.cvCloneImage(imagen)

	irisDatabase.doMatch(template, None)
	tiempoHamming = irisDatabase.getMatchingTime()
	irisDatabase.doAContrarioMatch(template, None)
	tiempoAContrario = irisDatabase.getMatchingTime()
	
	#informacionUsuario = irisDatabase.informacionUsuario(irisDatabase.getMinDistanceId())
	matchId = irisDatabase.getMinNFAId()
	informacionUsuario = irisDatabase.informacionUsuario(matchId)
	
	imagenUsuario = highgui.cvLoadImage(informacionUsuario['pathImagen'], 1)
	if imagenUsuario:
		decorator.drawSegmentationResult(imagenUsuario, informacionUsuario['segmentacion'])
		imagenUsuarioRes = opencv.cvCreateImage(opencv.cvSize(320,240), opencv.IPL_DEPTH_8U, 3)
		opencv.cvResize(imagenUsuario, imagenUsuarioRes)
		decorator.drawTemplate(imagenUsuarioRes, informacionUsuario['template'])
	else:
		imagenUsuarioRes = None
	
	if imagen:
		imagenDecorada = Utils.aColor(imagen)
		decorator.drawSegmentationResult(imagenDecorada, segmentacion)
		imagenRes = opencv.cvCreateImage(opencv.cvSize(320, 240), opencv.IPL_DEPTH_8U, 3)
		opencv.cvResize(imagenDecorada, imagenRes)
		decorator.drawTemplate(imagenRes, template)
		
	minNFA = irisDatabase.getNFAFor(matchId)
	minHD = irisDatabase.getDistanceFor(matchId)
	form.lblDistanciaHamming.setText(str(minHD))	
	form.lblNFA.setText(str(minNFA))
	form.lblUsuario.setText(informacionUsuario['usuario'])
	form.lblCantidadImagenes.setText('<font color="red">Sobre un total de %i personas - Tiempo de b&uacute;squeda: %.2f miliseg.</font>' % (irisDatabase.databaseSize(), tiempoHamming+tiempoAContrario))
	form.lblProbError.setText('%.5f%%' % (pow(10, minNFA)*100.0))
	
	if minNFA > -2 or minHD > 0.36:
		strIdentificacion = '<font color="red">Negativa</font>'
	else:
		strIdentificacion = '<font color="green">Positiva</font>'
	
	form.lblIdentificacion.setText(strIdentificacion)
	
	if imagenUsuarioRes:
		form.imagenBBDD.showImage(imagenUsuarioRes)
	if imagen:
		form.imagenCapturada.showImage(imagenRes)

	form.database = irisDatabase
	form.imagen = imagen
	form.informacionUsuario = informacionUsuario
	
	form.show()
	