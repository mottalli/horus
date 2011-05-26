#!/usr/bin/python
import os, os.path
import sqlite3
import re

base = sqlite3.connect('./base.db')

aBorrar = []
idsValidos = []

for row in base.execute('SELECT id_usuario,nombre FROM usuarios'):
	(id_usuario,nombre) = row
	
	if re.match(r'[0-9]{4}/[LR]', nombre):		# De la base Bath
		aBorrar.append(id_usuario)
	#elif (...) 			# Otra base
	else:
		idsValidos.append(id_usuario)

# Borro las entradas correspondientes a las otras bases
for id_usuario in aBorrar:
	print "Borrando ID", id_usuario
	base.execute('DELETE FROM usuarios WHERE id_usuario=?', (id_usuario,))
base.commit()

# Ahora acomodo los IDs
n = len(idsValidos)
for nuevoId in range(1,n+1):
	if nuevoId in idsValidos:
		continue
	
	# Esto significa que hay un "agujero" en los IDs. Lo relleno con el del maximo ID.
	viejoId = max(idsValidos)
	idsValidos.remove(viejoId)
	idsValidos.append(nuevoId)
	print "Cambiado ID %i a %i..." % (viejoId, nuevoId)
	base.execute('UPDATE base_iris SET id_usuario=? WHERE id_usuario=?', (nuevoId, viejoId))
	base.execute('UPDATE usuarios SET id_usuario=? WHERE id_usuario=?', (nuevoId, viejoId))
	
	# Ahora mueve los archivos
	archivosViejos = filter(lambda fname: re.match(r'^%i_[0-9]+\.jpg' % (viejoId,), fname), os.listdir('.'))
	archivosNuevos = map(lambda fname: re.sub('^%i_' % (viejoId,), '%i_' % (nuevoId,), fname), archivosViejos) 
	for i in range(len(archivosViejos)):
		print "    * Moviendo %s a %s..." % (archivosViejos[i], archivosNuevos[i])
		os.rename(archivosViejos[i], archivosNuevos[i])
		base.execute('UPDATE base_iris SET imagen=? WHERE imagen=?', (archivosNuevos[i], archivosViejos[i]))

	base.commit()
