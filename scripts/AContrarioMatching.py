# -*- coding: UTF8 -*-
import Database
import pyhorus
import os.path
from opencv import *
from opencv.highgui import *
from scipy import *
from pylab import *
import herramientas

CANTIDAD_PARTES = 4
	
def correrMatchAContrario(base):
	if pyhorus.HORUS_CUDA_SUPPORT:
		print "NOTA: usando aceleración CUDA"
		irisDatabase = pyhorus.IrisDatabaseCUDA()
	else:
		irisDatabase = pyhorus.IrisDatabase()

	#rows = base.conn.execute('SELECT id_iris,id_usuario,imagen,segmentacion,image_template FROM base_iris WHERE entrada_valida=1')
	rows = base.conn.execute('SELECT id_iris,id_usuario,imagen,segmentacion,template FROM vw_base_iris WHERE entrada_valida=1')
	
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
		for j in range(len(irisDatabase.resultNFAs)):
			nfa = irisDatabase.resultNFAs[j]
			idImagen2 = irisDatabase.ids[j]
			if idImagen1 == idImagen2: continue
			intraClase = 1 if clases[idImagen1] == clases[idImagen2] else 0
			base.conn.execute('INSERT INTO nfa_a_contrario VALUES(?,?,?,?)', [idImagen1, idImagen2, nfa, intraClase])
		
		for parte in range(len(irisDatabase.resultPartsDistances)):
			distancias = irisDatabase.resultPartsDistances[parte]
			for j in range(len(distancias)):
				distancia = distancias[j]
				idImagen2 = irisDatabase.ids[j]
				if idImagen1 >= idImagen2: continue
				intraClase = 1 if clases[idImagen1] == clases[idImagen2] else 0
				base.conn.execute('INSERT INTO comparaciones_a_contrario VALUES(?,?,?,?,?)', [idImagen1, idImagen2, distancia, parte, intraClase])
		
		if i % 10 == 0: base.conn.commit()
	
	base.conn.commit()
