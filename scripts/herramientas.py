#!/usr/bin/python
# -*- coding: UTF8 -*-
from numpy import where, array
import os

def obtenerImagen(imagen, base=None):
	try:
		idImagen = int(imagen)
		if not base:
			raise Exception('obtenerImagen: se definió ID pero no se definió la base!')
		BASE = BaseIris.cargarBase(base)
		fila = BASE.conn.cursor().execute('SELECT * FROM base_iris WHERE id_imagen=?', (idImagen,)).fetchone()
		(idImagen, idClase, pathImagen, segmentacion, segmentacionCorrecta, codigoDCT, codigoGabor, mascaraCodigo) = fila
		fullPathImagen = str(os.path.join(BASE.pathBase, pathImagen))
	except ValueError:
		fullPathImagen = imagen

	return fullPathImagen
