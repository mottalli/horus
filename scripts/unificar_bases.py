#!/usr/bin/python

import os.path
import sqlite3

paths_bases = ['Bath/', 'CASIA3-Preprocesada', 'MMU/']

unificada = sqlite3.connect('base.db')
if not unificada:
	raise Exception('No existe la unificada')
	
unificada.execute('DELETE FROM base_iris')
unificada.commit()

id_imagen_unificada = 1
max_clase = 0
for path_base in paths_bases:
	base = str(os.path.join(path_base, 'base.db'))
	if not os.path.exists(base):
		raise Exception('No existe la base ' + base)
	base = sqlite3.connect(base)
	
	rs = base.execute('SELECT * FROM base_iris')

	while True:
		row = rs.fetchone()
		if not row: break
		(id_imagen,id_clase,imagen,segmentacion,segmentacion_correcta,codigo_dct,codigo_gabor,mascara_codigo) = row
		
		id_imagen = id_imagen_unificada
		imagen = str(os.path.join(path_base, str(imagen)))
		
		id_clase = id_clase+max_clase
		print id_imagen, imagen, id_clase
		
		unificada.execute('INSERT INTO base_iris VALUES(?,?,?,?,?,?,?,?)', [id_imagen,id_clase,imagen,segmentacion,segmentacion_correcta,codigo_dct,codigo_gabor,mascara_codigo])
		
		id_imagen_unificada = id_imagen_unificada+1
	
	unificada.commit()
	r = unificada.execute('SELECT MAX(id_clase) FROM base_iris').fetchone()
	max_clase = r[0]

# Engania pichanga: estas dos son la misma persona!!
print "Actualizando BATH 0002 y CASIA3 224..."
unificada.execute("UPDATE base_iris SET id_clase=(SELECT id_clase FROM base_iris WHERE imagen='Bath/0002/R/0001.jpg') WHERE id_imagen IN (SELECT id_imagen FROM base_iris WHERE imagen LIKE 'CASIA3-Preprocesada/224/R/%')")
unificada.execute("UPDATE base_iris SET id_clase=(SELECT id_clase FROM base_iris WHERE imagen='Bath/0002/L/0001.jpg') WHERE id_imagen IN (SELECT id_imagen FROM base_iris WHERE imagen LIKE 'CASIA3-Preprocesada/224/L/%')")
