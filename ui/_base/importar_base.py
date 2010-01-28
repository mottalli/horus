#!/usr/bin/python
# -*- coding: UTF8 -*-

import sqlite3
import sys
import os.path

if len(sys.argv) != 2:
	print 'Uso: %s /path/de/base' % (sys.argv[0])
	sys.exit(1)	

pathBase = os.path.join(sys.argv[1], 'base.db')
if not os.path.exists(pathBase):
	print 'No existe', pathBase
	sys.exit(1)

conn = sqlite3.connect('./base.db')
if not conn:
	print 'No se pudo abrir ./base.db'
	sys.exit(1)

otraConn = sqlite3.connect(pathBase)
if not otraConn:
	print 'No se pudo abrir', pathBase
	sys.exit(1)

filas = otraConn.execute('SELECT imagen,segmentacion,codigo_gabor FROM base_iris WHERE segmentacion_correcta=1')
for fila in filas:
	nombre = str(fila[0])
	imagen = os.path.join(pathBase, str(fila[0]))
	segmentacion = str(fila[1])
	codigo_gabor = str(fila[2])
	
	print "Importando", imagen
	conn.execute('INSERT INTO base_iris(nombre, imagen, segmentacion, codigo_gabor) VALUES(?, ?, ?, ?)', (nombre, imagen, segmentacion, codigo_gabor))
conn.commit()
