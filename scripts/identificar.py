#!/usr/bin/python
# -*- coding: UTF8 -*-

import horus
import sys
from opencv.highgui import *
from opencv import *
from optparse import OptionParser
import Database

segmentator = horus.Segmentator()
decorator = horus.Decorator()
encoder = horus.LogGaborEncoder()

parser = OptionParser()
parser.add_option('-b', dest='base', help="Base")

(options, args) = parser.parse_args()
base = Database.getDatabase(options.base)

pathImagen = args[0]

print "Cargando base..."
irisDatabase = horus.IrisDatabase()
rows = base.conn.execute('SELECT id_imagen,codigo_gabor FROM base_iris WHERE segmentacion_correcta=1')
for row in rows:
	idImagen = int(row[0])
	template = horus.unserializeIrisTemplate(str(row[1]))
	irisDatabase.addTemplate(idImagen, template)

imagen = cvLoadImage(pathImagen, 1)
rs = segmentator.segmentImage(imagen)
template = encoder.generateTemplate(imagen, rs)
imagenDecorada = cvCloneImage(imagen)
decorator.drawSegmentationResult(imagenDecorada, rs)
decorator.drawTemplate(imagenDecorada, template)

irisDatabase.doMatch(template)
minHD = irisDatabase.getMinDistance()
minHDId = irisDatabase.getMinDistanceId()

irisDatabase.doAContrarioMatch(template)
minNFA = irisDatabase.getMinNFA()
minNFAId =irisDatabase.getMinNFAId()

match = base.conn.execute('SELECT imagen,segmentacion,codigo_gabor FROM base_iris WHERE id_imagen=?', [minHDId]).fetchone()
imagenMatch = cvLoadImage(base.fullPath(match[0]))
rs = horus.unserializeSegmentationResult(str(match[1]))
template = horus.unserializeIrisTemplate(str(match[2]))
decorator.drawSegmentationResult(imagenMatch, rs)
decorator.drawTemplate(imagenMatch, template)

matchAC = base.conn.execute('SELECT imagen,segmentacion,codigo_gabor FROM base_iris WHERE id_imagen=?', [minNFAId]).fetchone()
imagenMatchAC = cvLoadImage(base.fullPath(matchAC[0]))
rs = horus.unserializeSegmentationResult(str(matchAC[1]))
template = horus.unserializeIrisTemplate(str(matchAC[2]))
decorator.drawSegmentationResult(imagenMatchAC, rs)
decorator.drawTemplate(imagenMatchAC, template)


cvNamedWindow("imagen")
cvShowImage("imagen", imagenDecorada)
cvNamedWindow("matchHD")
cvShowImage("matchHD", imagenMatch)
cvNamedWindow("matchAC")
cvShowImage("matchAC", imagenMatchAC)

print 'Match (DH): %i (%s), distancia: %.4f' % (minHDId, base.fullPath(match[0]), minHD)
print 'Match (AC): %i (%s), NFA: %.4f' % (minNFAId, base.fullPath(match[0]), minNFA)

while True:
	k = cvWaitKey(0)
	if k == 'q':
		break
