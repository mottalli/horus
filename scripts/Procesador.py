from horus import Segmentator, IrisEncoder, Decorator, unserializeSegmentationResult
from opencv import *
from opencv.highgui import *

segmentator = Segmentator()
decorator = Decorator()
encoder = IrisEncoder()

def procesar(base, options):
	imagenes = options.imagen.split(',')
	
	id_imagen = int(imagenes[0])
	(id_imagen, path, segmentacion)  = base.conn.execute('SELECT id_imagen,imagen,segmentacion FROM base_iris WHERE id_imagen=%i' % (id_imagen)).fetchone()
	
	imagen = cvLoadImage(base.fullPath(path), 0)
	
	if options.soloVer:
		segmentationResult = unserializeSegmentationResult(str(segmentacion))
	else:
		segmentationResult = segmentator.segmentImage(imagen)
	
	template = encoder.generateTemplate(imagen, segmentationResult)

	decorator.drawSegmentationResult(imagen, segmentationResult)
	decorator.drawEncodingZone(imagen, segmentationResult)

	
	cvNamedWindow("imagen")
	cvShowImage("imagen", imagen)
	
	cvNamedWindow("textura")
	cvShowImage("textura", encoder.getNormalizedTexture())
	
	cvWaitKey(0)
