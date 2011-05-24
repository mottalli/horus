#!/usr/bin/python
import pyhorus
import sqlite3
import os
from opencv import *
from opencv.highgui import *

def testMatching(base):
	rows = base.conn.execute('SELECT id_iris,id_usuario,imagen,segmentacion,image_template FROM base_iris WHERE entrada_valida=1')

	if pyhorus.HORUS_CUDA_SUPPORT:
		irisDatabase = pyhorus.IrisDatabaseCUDA()
	else:
		irisDatabase = pyhorus.IrisDatabase()
	
	templates = {}
	clases = {}

	for row in rows:
		idImagen = int(row[0])
		idClase = int(row[1])
		imagePath = base.fullPath(row[2])
		serializedSegmentationResult = str(row[3])
		serializedTemplate = str(row[4])
		
		print "Cargando %i..." % idImagen

		if not len(serializedTemplate):
			raise Exception('No se codificaron todas las imagenes! (correr iris.py con el parametro -c)')

		templates[idImagen] = pyhorus.unserializeIrisTemplate(serializedTemplate)
		clases[idImagen] = idClase
		
		irisDatabase.addTemplate(idImagen, templates[idImagen])
		
	idsImagenes = templates.keys()
	comparaciones = []

	base.conn.execute("DELETE FROM comparaciones")
	base.conn.commit()

	for i in range(len(idsImagenes)):
		idImagen1 = idsImagenes[i]
		print "Comparando imagen %i... " % idImagen1,
		
		irisDatabase.doMatch(templates[idImagen1])
		print "Tiempo: %.2f ms." % irisDatabase.getMatchingTime()
		
		distances = irisDatabase.getDistances()
		#for (j, distance) in enumerate(distances):			# Note: this triggers a bug with sequences in swig
		for j in range(len(distances)):
			distance = distances[j]
			idImagen2 = irisDatabase.ids[j]
			if idImagen1 >= idImagen2: continue
			intraClase = (clases[idImagen1] == clases[idImagen2])
			base.conn.execute("INSERT INTO comparaciones(id_iris1, id_iris2, distancia, intra_clase) VALUES(%i,%i,%f,%i)" % (idImagen1, idImagen2, distance, 1 if intraClase else 0))
		if i % 20 == 0:
			base.conn.commit()
		
	base.conn.commit()
