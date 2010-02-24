from PyQt4 import uic
from horus import Decorator
from opencv import *
from opencv.highgui import *
import Utils

decorator = Decorator()
def doMatch(irisDatabase, template, imagen=None, segmentacion=None):
	irisDatabase.doMatch(template, None)
	irisDatabase.doAContrarioMatch(template, None)
	
	informacionUsuario = irisDatabase.informacionUsuario(irisDatabase.getMinDistanceId())
	
	imagenUsuario = cvLoadImage(informacionUsuario['pathImagen'], 1)
	if imagenUsuario:
		decorator.drawSegmentationResult(imagenUsuario, informacionUsuario['segmentacion'])
		imagenUsuarioRes = cvCreateImage(cvSize(320,240), IPL_DEPTH_8U, 3)
		cvResize(imagenUsuario, imagenUsuarioRes)
		decorator.drawTemplate(imagenUsuarioRes, informacionUsuario['template'])
	else:
		imagenUsuarioRes = None
	
	if imagen:
		imagenDecorada = Utils.aColor(imagen)
		decorator.drawSegmentationResult(imagenDecorada, segmentacion)
		imagenRes = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 3)
		cvResize(imagenDecorada, imagenRes)
		decorator.drawTemplate(imagenRes, template)
		
	form = uic.loadUi('forms/Matching.ui')
	form.lblDistanciaHamming.setText(str(irisDatabase.getMinDistance()))	
	form.lblNFA.setText(str(irisDatabase.getMinNFA()))
	form.lblUsuario.setText(informacionUsuario['usuario'])
	if imagenUsuarioRes:
		form.imagenBBDD.showImage(imagenUsuarioRes)
	if imagen:
		form.imagenCapturada.showImage(imagenRes)

	form.show()
	
