#!/usr/bin/python
# -*- coding: UTF8 -*-

import sys

import horus
import Database
from optparse import OptionParser
import os
import Segmentar
import Matching
import Estadisticas
import Procesador

parser = OptionParser()
parser.add_option('-b', dest='base', help="Base")
parser.add_option('--id', '-i', dest='imagen', default=0, help="ID o path de la imagen correspondiente")
parser.add_option('-s', dest='segmentarUna', action='store_true', default=False, help="Segmentar unicamente esta imagen")
parser.add_option('-S', dest='segmentarBase', action='store_true', default=False, help="Segmentar todas las imagenes a partir de la imagen especificada (opcional)")
parser.add_option('-f', dest='forzar', action="store_true", default=False, help="Fuerza a procesar todas las imagenes")
parser.add_option('--malas', dest='soloMalas', action="store_true", default=False, help="Procesar solo las imagenes mal segmentadas")
parser.add_option('--buenas', dest='soloBuenas', action="store_true", default=False, help="Procesar solo las imagenes bien segmentadas")
parser.add_option('-e', '--estadisticas', dest='estadisticas', action="store_true", default=False)
parser.add_option('-v', dest='soloVer', action="store_true", default=False, help="Solo ver, no segmentar (usar solo el resultado guardado)")
parser.add_option('-m', '--matching', dest='matching', action="store_true", default=False, help="Hacer las pruebas de matching")
parser.add_option('-p', dest='procesar', action='store_true', help='Procesar las imagenes especificadas')
parser.add_option('--full', dest='full', action='store_true', default=False, help='Equivalente a -cme')
parser.add_option('-c', dest='codificar', action='store_true', default=False, help='Codificar todas las imagenes de la base')

(options, args) = parser.parse_args()
base = Database.getDatabase(options.base)

# Base por default
#if not options.base: options.base = 'casia3p'

if __name__ == '__main__':
	if options.segmentarBase:
		Segmentar.segmentarBase(options)
	if options.segmentarUna:
		Segmentar.segmentarUna(options)
	if options.codificar or options.full:
		Procesador.codificar(base, options)
	if options.matching or options.full:
		Matching.testMatching(base)
	if options.estadisticas or options.full:
		Estadisticas.estadisticas(base)
	if options.procesar:
		Procesador.procesar(base, options)
