# -*- coding: UTF8 -*-
from PyIrisLib import BaseIris
import horus
import os
from opencv import *
from opencv.highgui import *
import herramientas
import sys

ACCION_SIGUIENTE = 'S'
ACCION_ANTERIOR = 'Q'
ACCION_SALIR = 'q'
ACCION_GUARDAR = 'w'
ACCION_SEGMENTACION_CORRECTA = 's'
ACCION_SEGMENTACION_ERRONEA = 'n'
ACCION_GUARDAR_IMAGEN = 'i'

segmentator = horus.Segmentator()
decorator = horus.Decorator()

def segmentarBase(options):
	BASE = BaseIris.cargarBase(options.base)
	q = "SELECT * FROM base_iris WHERE 1=1"
	o = []
	
	if options.soloMalas:
		q += ' AND segmentacion_correcta=0'
	elif options.soloBuenas:
		q += ' AND segmentacion_correcta=1'
	
	if not options.forzar:
		q += ' AND (segmentacion=? OR segmentacion IS NULL)'
		o.append('')
		
	if options.imagen:
		q += ' AND id_imagen >= ?'
		o.append(options.imagen)
	
	filas = BASE.conn.execute(q,o).fetchall()
	i = 0
	buffer = []
	
	if not filas:
		print "No hay imagenes a procesar"
		sys.exit(0)
	while 1:
		if i >= len(filas): 
			print 'Fin!'
			break

		(idImagen, idClase, pathImagen, segmentacion, segmentacionCorrecta, codigoDCT, codigoGabor, mascaraCodigo) = filas[i]
		
		fullPathImagen = str(os.path.join(BASE.pathBase, pathImagen))
		
		if options.soloVer and segmentacion:
			imagen = cvLoadImage(fullPathImagen, 0)
			if not imagen:
				raise Exception('No se pudo abrir la imagen ' + imagen)

			resultadoSegmentacion = horus.unserializeSegmentationResult(str(segmentacion))
			decorada = mostrarSegmentada(imagen, resultadoSegmentacion)
		else:
			print 'Segmentando %i (%s)' % (idImagen, fullPathImagen)
			(resultadoSegmentacion, decorada) = segmentarYMostrar(fullPathImagen)
		
		while 1:
			accion = obtenerAccion()
			if accion == ACCION_ANTERIOR:
				i = i-1
				break
			elif accion == ACCION_SIGUIENTE:
				i = i+1
				break
			elif accion == ACCION_SEGMENTACION_CORRECTA or accion == ACCION_SEGMENTACION_ERRONEA:
				buffer = filter(lambda x: x['id_imagen'] != idImagen, buffer)		# La quito si ya está
				buffer.append({
						'id_imagen': idImagen,
						'segmentacion_correcta': (accion == ACCION_SEGMENTACION_CORRECTA),
						'segmentacion': horus.serializeSegmentationResult(resultadoSegmentacion)
					})
				i = i+1
				break
			elif accion == ACCION_GUARDAR:
				buffer = flushBuffer(BASE, buffer)
				continue
			elif accion == ACCION_SALIR:
				buffer = flushBuffer(BASE, buffer)
				sys.exit(0);
			
		if len(buffer) and (len(buffer) % 10) == 0:
				buffer = flushBuffer(BASE, buffer)
		
	buffer = flushBuffer(BASE, buffer)

def segmentarUna(options):
	pathImagen = herramientas.obtenerImagen(options.imagen, options.base)
	
	(resultadoSegmentacion, imagenDecorada) = segmentarYMostrar(pathImagen)

	while True:
		accion = cvWaitKey(0)
		if accion == ACCION_GUARDAR_IMAGEN:
			cvSaveImage('decorada.png', imagenDecorada)
			print 'Imagen guardada en decorada.png'
		else:
			break

def segmentarYMostrar(imagen):
	
	if imagen.__class__ == str:
		imagen = cvLoadImage(imagen, 0)
		if not imagen:
			raise Exception('No se pudo abrir la imagen ' + imagen)
	
	rs = segmentator.segmentImage(imagen)
	#segmentator.segmentEyelids(imagen, rs)
	imagenDecorada = mostrarSegmentada(imagen, rs)
	return (rs, imagenDecorada)


def mostrarSegmentada(imagen, rs):	
	imagenDecorada = cvCreateImage(cvGetSize(imagen), IPL_DEPTH_8U, 3)
	cvCvtColor(imagen, imagenDecorada, CV_GRAY2BGR)
	decorator.drawSegmentationResult(imagenDecorada, rs)
	decorator.drawEncodingZone(imagenDecorada, rs)
	
	cvNamedWindow("segmentada")
	cvShowImage("segmentada", imagenDecorada)
	return imagenDecorada

def obtenerAccion():
	return cvWaitKey(0)

def flushBuffer(base, buffer):
	print "Haciendo flush del buffer..."
	for fila in buffer:
		#print "Actualizando %i..." % (fila['id_imagen'],)
		base.conn.cursor().execute('UPDATE base_iris SET segmentacion_correcta=?, segmentacion=? WHERE id_imagen=?', 
			[fila['segmentacion_correcta'],fila['segmentacion'],fila['id_imagen']])
		base.conn.commit()
	return []

