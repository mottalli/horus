# -*- coding: UTF8 -*-
from horus import Segmentator, IrisEncoder, Decorator, unserializeSegmentationResult, serializeIrisTemplate, unserializeIrisTemplate, TemplateComparator
from opencv import *
from opencv.highgui import *

segmentator = Segmentator()
decorator = Decorator()
encoder = IrisEncoder()

def procesar(base, options):
	imagenes = options.imagen.split(',')
	
	templates = {}
	
	for id_imagen in imagenes:
		id_imagen = int(id_imagen)
		row = base.conn.execute("SELECT imagen,segmentacion,codigo_gabor FROM base_iris WHERE id_imagen=?", [id_imagen]).fetchone()
		if not row: raise Exception('No existe la imagen %i' % id_imagen)
		(path, segmentacionSerializada, templateSerializado) = row
			
		
		if options.soloVer:
			imagen = cvLoadImage(base.fullPath(path))
			resultadoSegmentacion = unserializeSegmentationResult(str(segmentacionSerializada))
			template = unserializeIrisTemplate(str(templateSerializado))
		else:
			imagen = cvLoadImage(base.fullPath(path), 0)
			resultadoSegmentacion = segmentator.segmentImage(imagen)
			template = encoder.generateTemplate(imagen, resultadoSegmentacion)
			
		templates[id_imagen] = template
		
		cvNamedWindow(str(id_imagen))
		decorator.drawTemplate(imagen, template)
		decorator.drawSegmentationResult(imagen, resultadoSegmentacion)
		#decorator.drawEncodingZone(imagen, resultadoSegmentacion)
		
		cvShowImage(str(id_imagen), imagen)
	
	for (id_imagen1, template1) in templates.items():
		comparator = TemplateComparator(template1)
		for (id_imagen2, template2) in templates.items():
			if id_imagen1 == id_imagen2: continue
			print "Distancia de hamming entre %i y %i: %.4f" % (id_imagen1, id_imagen2, comparator.compare(template2))
	
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
