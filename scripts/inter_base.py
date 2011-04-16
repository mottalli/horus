#!/usr/bin/python
# -*- coding: UTF8 -*-
import horus
import Database
from opencv import *
from opencv.highgui import *
from scipy import *
from pylab import *

base = Database.getDatabase('bath')
irisDatabase = horus.IrisDatabase()

rows = base.conn.execute('SELECT id_imagen,codigo_gabor FROM base_iris WHERE segmentacion_correcta=1')
for row in rows:
	idImagen = int(row[0])
	serializedTemplate = str(row[1])

	print "Cargando %i..." % idImagen
	if not len(serializedTemplate):
		raise Exception('No se codificaron todas las imagenes! (correr iris.py con el parametro -c)')

	irisDatabase.addTemplate(idImagen, horus.unserializeIrisTemplate(serializedTemplate))

decorator = horus.Decorator()
segmentator = horus.Segmentator()
encoder = horus.LogGaborEncoder()

imagen = cvLoadImage('/home/marcelo/Mis_Documentos/Facu/Tesis/Bases de datos/UBA/marcelo_der_2.bmp', 0)
imagenColor = cvCreateImage(cvGetSize(imagen), IPL_DEPTH_8U, 3)
cvCvtColor(imagen, imagenColor, CV_GRAY2BGR)

segmentation = segmentator.segmentImage(imagen)
template = encoder.generateTemplate(imagen, segmentation)

irisDatabase.doMatch(template)
irisDatabase.doAContrarioMatch(template)

dashes = ['--', '-', '-.', ':']
for parte in range(4):
	a = array(irisDatabase.resultPartsDistances[parte])
	(h, b) = histogram(a, bins=50)
	plot(b[:-1], h, linestyle=dashes[parte], c='black', label='Parte '+str(parte))
legend(loc='upper left')

print (irisDatabase.getMinDistanceId(), irisDatabase.getMinDistance())
print (irisDatabase.getMinNFAId(), irisDatabase.getMinNFA())

show()



