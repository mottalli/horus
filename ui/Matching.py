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

	form.lblDistanciaHamming.setText(str(irisDatabase.getMinDistance()))	
	form.lblUsuario.setText(informacionUsuario['usuario'])
	
	imagenUsuario = cvLoadImage(informacionUsuario['pathImagen'], 1)
	decorator.drawSegmentationResult(imagenUsuario, informacionUsuario['segmentacion'])
	imagenUsuarioRes = cvCreateImage(cvSize(320,240), IPL_DEPTH_8U, 3)
	cvResize(imagenUsuario, imagenUsuarioRes)
	form.imagenBBDD.showImage(imagenUsuarioRes)
	
	if imagen:
		imagenDecorada = Utils.aColor(imagen)
		decorator.drawSegmentationResult(imagenDecorada, segmentacion)
		imagenRes = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 3)
		cvResize(imagenDecorada, imagenRes)
		form.imagenCapturada.showImage(imagenRes)

	form.show()
	
