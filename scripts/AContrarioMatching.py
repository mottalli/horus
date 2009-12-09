# -*- coding: UTF8 -*-
import Database
import horus
import os.path
from opencv import *
from opencv.highgui import *
from scipy import *
from pylab import *
import herramientas

CANTIDAD_PARTES = 4
	
def hacerComparaciones(base):
	encoder = horus.IrisEncoder()
	
	templates = {}
	clases = {}
	
	for (idImagen,idClase,pathImagen,resultadoSegmentacionSerializado) in base.conn.execute("SELECT id_imagen,id_clase,imagen,segmentacion FROM base_iris WHERE segmentacion_correcta=1 AND segmentacion IS NOT NULL"):
		resultadoSegmentacion = horus.unserializeSegmentationResult(str(resultadoSegmentacionSerializado))
		fullPathImagen = base.fullPath(pathImagen)
		
		print "Codificando imagen %i (%s...)" % (idImagen ,fullPathImagen)
		
		imagen = cvLoadImage(fullPathImagen, 0)
		if not imagen: raise Exception('No se pudo abrir %s' % (fullPathImagen))
		templates[idImagen] = encoder.generateTemplate(imagen, resultadoSegmentacion)
		clases[idImagen] = idClase

	# Ahora comparo todas contra todas
	print 'Comparando...'
	base.conn.execute('DELETE FROM comparaciones_a_contrario')
	base.conn.commit()
	for idImagen1 in templates.keys():
		print "Comparando partes de imagen %i..." % (idImagen1)
		comparator = horus.TemplateComparator(templates[idImagen1], 20, 2)
		for idImagen2 in templates.keys():
			ds = comparator.compareParts(templates[idImagen2], CANTIDAD_PARTES)
			for numParte in range(CANTIDAD_PARTES):
				base.conn.execute('INSERT INTO comparaciones_a_contrario(id_imagen1,id_imagen2,distancia,parte, intra_clase) VALUES(%i,%i,%f,%i,%i)'
					% (idImagen1, idImagen2, ds[numParte], numParte, 1 if clases[idImagen1] == clases[idImagen2] else 0))
		
		if idImagen1 % 10 == 0: 
			base.conn.commit()
	
	base.conn.commit()

######################################################################

def correrMatchAContrario(base):
	clases = {}
	for (idImagen, idClase) in base.conn.execute('SELECT id_imagen,id_clase FROM base_iris WHERE segmentacion_correcta=1 AND segmentacion IS NOT NULL').fetchall():
		clases[idImagen] = idClase

	# Proceso cada imagen por separado
	base.conn.execute('DELETE FROM nfa_a_contrario')
	for idImagen in clases.keys():
		print "Haciendo matching de imagen %i..." % (idImagen)
		correrTestImagen(base, idImagen, clases)
		
def correrTestImagen(base, idImagen, clases):
	# Calculo los histogramas de distancias para cada parte
	histogramasPartes = []
	histAcumPartes = []
	binsPartes = []
	for parte in range(CANTIDAD_PARTES):
		distancias = array(base.conn.execute('SELECT distancia FROM comparaciones_a_contrario WHERE id_imagen1=%i AND parte=%i' % (idImagen, parte)).fetchall())
		(histograma, bins) = histogram(distancias, bins=50)
		histograma = histograma / float(len(distancias))
		histogramasPartes.append(histograma)
		histAcumPartes.append(cumsum(histograma))
		binsPartes.append(bins[:-1])

	# Contra qué imágenes comparé
	imagenesContra = []
	for id_imagen2 in base.conn.execute('SELECT DISTINCT id_imagen2 FROM comparaciones_a_contrario WHERE id_imagen1=%i ORDER BY id_imagen1, id_imagen2, parte' % (idImagen)):
		imagenesContra.append(id_imagen2)
		
	matchesSignificativos = []
	c = 0
	for (idOtraImagen,) in imagenesContra:
		c += 1
		comparaciones = array(base.conn.execute('SELECT * FROM comparaciones_a_contrario WHERE id_imagen1=%i AND id_imagen2=%i'%(idImagen, idOtraImagen)).fetchall())
		assert len(comparaciones) == CANTIDAD_PARTES

		# NOTA: uso el logaritmo!
		lNFA = log10(len(imagenesContra)) + sum([log10(probaAcum(comparaciones[parte,2], histAcumPartes[parte], binsPartes[parte])) for parte in range(CANTIDAD_PARTES)])
		base.conn.execute('INSERT INTO nfa_a_contrario VALUES(%i,%i,%f,%i)'%(idImagen, idOtraImagen, lNFA, 1 if clases[idImagen] == clases[idOtraImagen] else 0))

		if c % 10 == 0:
			base.conn.commit()

	base.conn.commit()

def probaAcum(d, acum, bins):
	assert len(acum) == len(bins)
	i = -1
	for i in range(len(bins)):
		if bins[i] >= d:
			break

	assert i >= 0
	return acum[i]
