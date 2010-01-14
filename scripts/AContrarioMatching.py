# -*- coding: UTF8 -*-
import Database
import horus
import os.path
from opencv import *
from opencv.highgui import *
from scipy import *
from pylab import *
import herramientas

CANTIDAD_PARTES = 4
	
def correrMatchAContrario(base):
	irisDatabase = horus.IrisDatabase()

	rows = base.conn.execute('SELECT * FROM base_iris WHERE segmentacion_correcta=1')
	
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
			raise Exception('No se codificaron todas las imagenes! (correr iris.py con el parametro -c)')

		templates[idImagen] = horus.unserializeIrisTemplate(serializedTemplate)
		clases[idImagen] = idClase
		
		irisDatabase.addTemplate(idImagen, templates[idImagen])

	print "Borrando datos viejos..."
	base.conn.execute("DELETE FROM comparaciones_a_contrario")
	base.conn.execute("DELETE FROM nfa_a_contrario")
	base.conn.commit()

	idsImagenes = templates.keys()
	for i in range(len(idsImagenes)):
		idImagen1 = idsImagenes[i]
		print "Haciendo matching a contrario de imagen %i... " % idImagen1,

		irisDatabase.doAContrarioMatch(templates[idImagen1], CANTIDAD_PARTES)
		print "Tiempo: %.2f ms." % irisDatabase.getMatchingTime()
		
		# Escribe los datos
		for (j, nfa) in enumerate(irisDatabase.resultNFAs):
			idImagen2 = irisDatabase.ids[j]
			if idImagen1 == idImagen2: continue
			intraClase = 1 if clases[idImagen1] == clases[idImagen2] else 0
			base.conn.execute('INSERT INTO nfa_a_contrario VALUES(?,?,?,?)', [idImagen1, idImagen2, nfa, intraClase])
		
		for (parte, distancias) in enumerate(irisDatabase.resultPartsDistances):
			for (j, distancia) in enumerate(distancias):
				idImagen2 = irisDatabase.ids[j]
				if idImagen1 >= idImagen2: continue
				intraClase = 1 if clases[idImagen1] == clases[idImagen2] else 0
				base.conn.execute('INSERT INTO comparaciones_a_contrario VALUES(?,?,?,?,?)', [idImagen1, idImagen2, distancia, parte, intraClase])
		
		if i % 10 == 0: base.conn.commit()
	
	base.conn.commit()
