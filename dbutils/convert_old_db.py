#!/usr/bin/python
import pyhorus
import sqlite3
import sys
import os, os.path

if len(sys.argv) != 3:
	print "Uso: %s <base vieja> <base nueva>" % (sys.argv[0])
	sys.exit(1)

vieja = sqlite3.connect(sys.argv[1])
nueva = sqlite3.connect(sys.argv[2])

usuarios = []

for row in vieja.execute('SELECT id_imagen,id_clase,imagen,segmentacion,segmentacion_correcta,codigo_gabor FROM base_iris'):
	(idImagen, idClase, pathImagen, segmentacionSerializada, segmentacionCorrecta, codigoSerializado) = row
	if segmentacionCorrecta:
		segmentacion = pyhorus.unserializeSegmentationResultOLD(str(segmentacionSerializada))
		segmentacionSerializada = pyhorus.serializeSegmentationResult(segmentacion)
	else:
		segmentacionSerializada = ""
		
	if idClase not in usuarios:
		usuarios.append(idClase)
		nombre = str(os.path.dirname(pathImagen))
		nueva.execute('INSERT INTO usuarios(id_usuario,nombre) VALUES(?,?)', (idClase, nombre))

	print "Importando %i..." % (idImagen)
	nueva.execute('''INSERT INTO base_iris(id_iris,id_usuario,imagen,segmentacion,entrada_valida,image_template)
					VALUES(?,?,?,?,?,?)''', (idImagen,idClase,pathImagen,segmentacionSerializada,segmentacionCorrecta,codigoSerializado))
nueva.commit()
