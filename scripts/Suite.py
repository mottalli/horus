# -*- coding: UTF8 -*-
import sys
sys.path.append('/home/marcelo/Mis_Documentos/Facu/Tesis/implementacion/scripts_nuevos')

import horus
from PyIrisLib import BaseIris

import os
from pylab import *
from opencv import *
from opencv.highgui import *


WIDTH_FILTRO = 240
HEIGHT_FILTRO = 20
CANT_ROTACIONES = 20
STEP_ROTACIONES = 2

BANCO_FILTROS = [
		FiltroLogGabor1DPrueba.FiltroLogGabor1DPrueba(WIDTH_FILTRO, HEIGHT_FILTRO, 1/32.0, 0.5)
	]


def testSuite(options):
	BASE = BaseIris.cargarBase(options.base)

	suite = []

	suite.append(TestSuite.TestBase(BASE, BANCO_FILTROS, CANT_ROTACIONES, STEP_ROTACIONES))

	for prueba in suite:
		prueba.correr()

def estadisticas(options):
	## Cambiar esto
	BASE = BaseIris.cargarBase(options.base)

	print 'Cargando datos...'
	datosIntra = array(BASE.conn.execute('SELECT * FROM comparaciones WHERE intra_clase=1').fetchall())
	datosInter = array(BASE.conn.execute('SELECT * FROM comparaciones WHERE intra_clase=0').fetchall())
	distanciasIntra = datosIntra[:, 2]
	distanciasInter = datosInter[:, 2]

	muIntra = distanciasIntra.mean()
	muInter = distanciasInter.mean()
	stdIntra = distanciasIntra.std()
	stdInter = distanciasInter.std()

	separabilidad = abs(muIntra-muInter)/sqrt( (stdIntra**2 + stdInter**2)/2.0 )
	print "Separabilidad: ", separabilidad

	(FARs, FRRs, EER, thresholdOptimo) = herramientas.calcularROC_EER(distanciasIntra, distanciasInter, linspace(0.3, 0.5, 200))
	# Genera la ROC
	figure()
	plot(FARs, FRRs, linewidth=2, color='black')
	u = EER*2.0
	plot([0, u],[0, u], color='black', linestyle=':')
	plot([EER], [EER], 'ko')
	axis([0, u, 0, u])
	xlabel('FAR')
	ylabel('FRR')

	print "Threshold optimo:", thresholdOptimo
	print "EER:", EER

	distanciasIntra = array(datosIntra[:, 2])
	distanciasInter = array(datosInter[:, 2])

	figure()
	(h1, b1) = histogram(distanciasInter, bins=30, new=True)
	(h2, b2) = histogram(distanciasIntra, bins=30, new=True)
	plot(b1[:-1], h1/float(len(distanciasInter)), color='black', label='Inter-class distribution', linestyle='-')
	plot(b2[:-1], h2/float(len(distanciasIntra)), color='black', label='Intra-class distribution', linestyle='--')
	legend(loc='upper left')
	#yticks([])
	xlabel('Hamming distance')

	show()

# Compara dos imágenes
def comparar(options):
	(im1, im2) = options.comparar.split(',')

	segmentador = PyIrisLib.Segmentador()
	segmentadorParpados = PyIrisLib.SegmentadorParpados()
	normalizador = PyIrisLib.Normalizador()
	codificador = PyIrisLib.CodificadorPrueba.CodificadorPrueba()
	codificador.banco = BANCO_FILTROS

	fullPathImagen1 = herramientas.obtenerImagen(im1, options.base)
	fullPathImagen2 = herramientas.obtenerImagen(im2, options.base)

	print "Comparando", fullPathImagen1, "y", fullPathImagen2

	imagen1 = cvLoadImage(fullPathImagen1, 0)
	rs1 = segmentador.segmentar(imagen1)
	segmentadorParpados.segmentarParpados(imagen1, rs1)
	imagen1Decorada = cvCreateImage(cvGetSize(imagen1), IPL_DEPTH_8U, 3)
	cvCvtColor(imagen1, imagen1Decorada, CV_GRAY2BGR)
	rs1.dibujarSegmentacion(imagen1Decorada)
	normalizador.dibujarNormalizacion(imagen1Decorada, rs1, CV_RGB(255,255,0))
	cvNamedWindow("imagen1")

	imagen2 = cvLoadImage(fullPathImagen2, 0)
	rs2 = segmentador.segmentar(imagen2)
	segmentadorParpados.segmentarParpados(imagen2, rs2)
	imagen2Decorada = cvCreateImage(cvGetSize(imagen2), IPL_DEPTH_8U, 3)
	cvCvtColor(imagen2, imagen2Decorada, CV_GRAY2BGR)
	rs2.dibujarSegmentacion(imagen2Decorada)
	normalizador.dibujarNormalizacion(imagen2Decorada, rs2, CV_RGB(255,255,0))
	cvNamedWindow("imagen2")

	codificacion1 = codificador.normalizarYCodificar(imagen1, rs1)
	comparador1 = PyIrisLib.ComparadorCodigoPrueba.ComparadorCodigoPrueba(codificacion1.codigo, codificacion1.mascaraCodigo, CANT_ROTACIONES, STEP_ROTACIONES)
	codificacion2 = codificador.normalizarYCodificar(imagen2, rs2)
	comparador2 = PyIrisLib.ComparadorCodigoPrueba.ComparadorCodigoPrueba(codificacion2.codigo, codificacion2.mascaraCodigo, CANT_ROTACIONES, STEP_ROTACIONES)

	dh = comparador1.menorDistanciaHamming(comparador2)
	print "Distancia de Hamming:", dh

	codificacion1.codigo.dibujar(imagen1Decorada, cvPoint(10,10))
	codificacion2.codigo.dibujar(imagen2Decorada, cvPoint(10,10))

	cvShowImage("imagen1", imagen1Decorada)
	cvShowImage("imagen2", imagen2Decorada)

	cvWaitKey(0)

def codificarUna(options):
	pathImagen = herramientas.obtenerImagen(options.imagen, options.base)
	segmentador = PyIrisLib.Segmentador()
	segmentadorParpados = PyIrisLib.SegmentadorParpados()
	normalizador = PyIrisLib.Normalizador()
	codificador = PyIrisLib.CodificadorPrueba.CodificadorPrueba()
	codificador.banco = BANCO_FILTROS

	imagen = cvLoadImage(pathImagen, 0)
	rs = segmentador.segmentar(imagen)
	segmentadorParpados.segmentarParpados(imagen, rs)
	codificacion = codificador.normalizarYCodificar(imagen, rs)

	imagenDecorada = cvCreateImage(cvGetSize(imagen), IPL_DEPTH_8U, 3)
	cvCvtColor(imagen, imagenDecorada, CV_GRAY2BGR)
	rs.dibujarSegmentacion(imagenDecorada)
	normalizador.dibujarNormalizacion(imagenDecorada, rs)

	cvNamedWindow('imagen')
	cvShowImage('imagen', imagenDecorada)

	cvNamedWindow('codigo')
	cvShowImage('codigo', codificacion.codigo.matriz)

	k = cvWaitKey(0)
	if k == 'i':
		cvSaveImage('codigo.png', codificacion.codigo.matriz)
		print 'Guardado código en codigo.png'
