from PyQt4 import uic
import horus.Decorator

decorator = horus.Decorator()
def doMatch(parent, irisDatabase, template, imagen=None, segmentacion=None):
	irisDatabase.doMatch(template, None)
	
	form = uic.loadUi('forms/Matching.ui')
	form.imagenCapturada.showImage(imagen)
	
	
	form.show()
	
