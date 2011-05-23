#!/usr/bin/python
# -*- coding: UTF8 -*-
import sqlite3
import sys
import os, os.path

if len(sys.argv) != 2:
	print "Uso: %s /path/a/base.db" % sys.argv[0]

base = sqlite3.connect("./base.db")
otrabase = sqlite3.connect(sys.argv[1])

basePath = os.path.dirname(sys.argv[1])

(deltaIdUsuario,) = otrabase.execute("SELECT MAX(id_usuario) FROM usuarios").fetchone()

for row in otrabase.execute('SELECT * FROM base_iris'):
	(idIris,idUsuario,pathImagen,segmentacion,entradaValida,imageTemplate,averageTemplate) = row
	
	pathImagen = str(os.path.join(basePath, pathImagen))
	print "Importando imagen %s..." % (pathImagen)
	
	base.execute('''INSERT INTO base_iris(id_usuario,imagen,segmentacion,entrada_valida,image_template,average_template)
					VALUES(?,?,?,?,?,?)''', (idUsuario+deltaIdUsuario,pathImagen,segmentacion,entradaValida,imageTemplate,averageTemplate))

for row in otrabase.execute('SELECT * FROM usuarios'):
	(idUsuario,nombre) = row
	base.execute('INSERT INTO usuarios(id_usuario,nombre) VALUES(?,?)', (idUsuario+deltaIdUsuario,nombre))

base.commit()
