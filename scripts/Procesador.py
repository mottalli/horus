# -*- coding: UTF8 -*-
from horus import Segmentator, IrisEncoder, Decorator, unserializeSegmentationResult, serializeIrisTemplate, unserializeIrisTemplate
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

def codificar(base, options):
	filas = base.conn.execute('SELECT id_imagen,imagen,segmentacion FROM base_iris WHERE segmentacion_correcta=1').fetchall()
	
	for i in range(len(filas)):
		(id_imagen, path, segmentacion) = filas[i]
		
		print 'Codificando imagen %i...' % id_imagen
		
		imagen = cvLoadImage(base.fullPath(path), 0)
		segmentationResult = unserializeSegmentationResult(str(segmentacion))
		template = encoder.generateTemplate(imagen, segmentationResult)
		
		base.conn.execute('UPDATE base_iris SET codigo_gabor=? WHERE id_imagen=?', [serializeIrisTemplate(template), id_imagen])
	base.conn.commit()
