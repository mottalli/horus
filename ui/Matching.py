from PyQt4 import uic
from horus import Decorator
from opencv import *
from opencv.highgui import *
import Utils

decorator = Decorator()
def doMatch(parent, irisDatabase, template, imagen=None, segmentacion=None):
	irisDatabase.doMatch(template, None)
	irisDatabase.doAContrarioMatch(template, None)
	
	form = uic.loadUi('forms/Matching.ui')
	
	informacionUsuario = irisDatabase.informacionUsuario(irisDatabase.getMinDistanceId())

	
	imagenUsuario = cvLoadImage(informacionUsuario['pathImagen'], 1)
	decorator.drawSegmentationResult(imagenUsuario, informacionUsuario['segmentacion'])
	imagenUsuarioRes = cvCreateImage(cvSize(320,240), IPL_DEPTH_8U, 3)
	cvResize(imagenUsuario, imagenUsuarioRes)
	
	if imagen:
		imagenDecorada = Utils.aColor(imagen)
		decorator.drawSegmentationResult(imagenDecorada, segmentacion)
		imagenRes = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 3)
		cvResize(imagenDecorada, imagenRes)
		

	form.lblDistanciaHamming.setText(str(irisDatabase.getMinDistance()))	
	form.lblNFA.setText(str(irisDatabase.getMinNFA()))
	form.lblUsuario.setText(informacionUsuario['usuario'])
	form.imagenBBDD.showImage(imagenUsuarioRes)
	if imagen:
		form.imagenCapturada.showImage(imagenRes)

	form.show()
	
