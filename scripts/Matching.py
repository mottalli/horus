#!/usr/bin/python
import horus
import sqlite3
import os
from opencv import *
from opencv.highgui import *

def testMatching(base):
	rows = base.conn.execute('SELECT * FROM base_iris WHERE segmentacion_correcta=1')
	
	templateComparator = horus.TemplateComparator()
	
	templates = {}
	clases = {}

	for row in rows:
		idImagen = int(row[0])
		idClase = int(row[1])
		imagePath = base.fullPath(row[2])
		serializedSegmentationResult = str(row[3])
		serializedTemplate = str(row[6])
		
	
		print "Cargando %i..." % idImagen

		if not len(serializedTemplate):
			raise Exception('No se codificaron todas las imagenes!  (correr iris.py con el parametro -c)')
		#image = cvLoadImage(imagePath, 0)
		#segmentationResult = horus.unserializeSegmentationResult(serializedSegmentationResult)
		#templates[idImagen] = irisEncoder.generateTemplate(image, segmentationResult)
		templates[idImagen] = horus.unserializeIrisTemplate(serializedTemplate)
		clases[idImagen] = idClase
		
	idsImagenes = templates.keys()
	comparaciones = []

	base.conn.execute("DELETE FROM comparaciones")
	base.conn.commit()

	for i in range(len(idsImagenes)):
		idImagen1 = idsImagenes[i]
		print "Comparando imagen %i..." % idImagen1
		templateComparator.setSrcTemplate(templates[idImagen1])
		
		for j in range(i+1, len(idsImagenes)):
			idImagen2 = idsImagenes[j]
			hd = templateComparator.compare(templates[idImagen2])
			intraClase = clases[idImagen1] == clases[idImagen2]
			base.conn.execute("INSERT INTO comparaciones(id_imagen1, id_imagen2, distancia, intra_clase) VALUES(%i,%i,%f,%i)" % (idImagen1, idImagen2, hd, 1 if intraClase else 0))
		
		if i % 10 == 0:
			base.conn.commit()
	base.conn.commit()
