#!/usr/bin/python
# -*- coding: UTF8 -*-

import horus
import Database
from optparse import OptionParser

parser = OptionParser()
parser.add_option('-b', dest='base', help="Base")
parser.add_option('-a', dest='acontrario', action='store_true', help='Simular a contrario')
parser.add_option('-d', dest='hamming', action='store_true', help='Simular Distancia de Hamming')


(options, args) = parser.parse_args()
base = Database.getDatabase(options.base)

clases = {}
imagenesClase = {}

print 'Cargando datos...'
for (id_imagen, id_clase) in base.conn.execute('SELECT id_imagen, id_clase FROM base_iris WHERE segmentacion_correcta=1'):
	clases[id_imagen] = id_clase
	if not id_clase in imagenesClase: imagenesClase[id_clase] = []
	imagenesClase[id_clase].append(id_imagen)

cantidadClases = len(clases)

def simularHamming():
	distancias = {}

	print 'Cargando distancias...'
	for (id_imagen1, id_imagen2, distancia) in base.conn.execute('SELECT id_imagen1, id_imagen2, distancia FROM comparaciones'):
		if id_imagen1 not in distancias: distancias[id_imagen1] = {}
		if id_imagen2 not in distancias: distancias[id_imagen2] = {}
		
		distancias[id_imagen1][id_imagen2] = distancia
		distancias[id_imagen2][id_imagen1] = distancia
	print 'Fin de carga'

	cantidadImagenes = len(distancias.keys())
	matchesMalos = 0

	for (id_imagen, distancias_imagen) in distancias.iteritems():
		minHD = min([d for (id_imagen2, d) in distancias_imagen.iteritems()])
		id_match = (filter(lambda x: x[1] == minHD, distancias_imagen.iteritems()))[0][0]
		
		id_clase = clases[id_imagen]

		if clases[id_imagen] == clases[id_match]:
			#print 'Imagen %i identificacion OK, match: %i, distancia: %.4f' % (id_imagen, id_match, minHD)
			pass
		elif len(imagenesClase[id_clase]) == 1:
			# Significa que esta era la única imagen de su clase, por lo tanto el minHD seguro que se 
			# corresponde con otra clase
			continue
		else:
			# Imagen mal identificada
			print 'Imagen %i identificacion MAL, match: %i, distancia: %.4f' % (id_imagen, id_match, minHD)
			# Muestro las imágenes de la misma clase y sus distancias
			print 'Distancias intra-clase:', filter(lambda (i,d): i in imagenesClase[id_clase], distancias_imagen.iteritems())
			matchesMalos = matchesMalos + 1

	print "Total identificaciones incorrectas: %i (%.2f%%)" % (matchesMalos, 100.0*matchesMalos/cantidadImagenes)

def simularAContrario():
	############################## Método a contrario
	print 'Cargando NFAs...'
	nfas = {}
	# Recordar que nfa(id_imagen1, id_imagen2) != nfa(id_imagen2, id_imagen1)
	for (id_imagen1, id_imagen2, nfa) in base.conn.execute('SELECT id_imagen1, id_imagen2, nfa FROM nfa_a_contrario'):
		if id_imagen1 not in nfas: nfas[id_imagen1] = {}
		nfas[id_imagen1][id_imagen2] = nfa

	print 'Fin de carga'

	cantidadImagenes = len(nfas.keys())
	matchesMalos = 0

	for (id_imagen, nfas_imagen) in nfas.iteritems():
		minNFA = min([nfa for (id_imagen2, nfa) in nfas_imagen.iteritems()])
		id_match = (filter(lambda x: x[1] == minNFA, nfas_imagen.iteritems()))[0][0]
		
		id_clase = clases[id_imagen]

		if clases[id_imagen] == clases[id_match]:
			#print 'Imagen %i identificacion OK, match: %i, NFA: %.4f' % (id_imagen, id_match, minNFA)
			pass
		elif len(imagenesClase[id_clase]) == 1:
			# Significa que esta era la única imagen de su clase, por lo tanto el minHD seguro que se 
			# corresponde con otra clase
			continue
		else:
			# Imagen mal identificada
			print 'Imagen %i identificacion MAL, match: %i, NFA: %.4f' % (id_imagen, id_match, minNFA)
			# Muestro las imágenes de la misma clase y sus distancias
			print 'NFAs intra-clase:', filter(lambda (i,d): i in imagenesClase[id_clase], nfas_imagen.iteritems())
			matchesMalos = matchesMalos + 1

	print "Total identificaciones incorrectas: %i (%.2f%%)" % (matchesMalos, 100.0*matchesMalos/cantidadImagenes)


if __name__ == '__main__':
	if options.hamming: simularHamming()
	if options.acontrario: simularAContrario()
