from PyQt4 import uic
from horus import Decorator
import opencv
from opencv import highgui
import Utils

decorator = Decorator()
form = uic.loadUi('forms/Matching.ui')

def doMatch(irisDatabase, template, imagen=None, segmentacion=None):
	if form.isVisible():
		return

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
	form.lblCantidadImagenes.setText('<font color="red">Sobre un total de %i personas</font>' % (irisDatabase.databaseSize()))
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

	form.show()
	
