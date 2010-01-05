#!/usr/bin/python
# -*- coding: UTF8 -*-
import AContrarioMatching
import Database
from optparse import OptionParser
import Estadisticas

parser = OptionParser()
parser.add_option('-b', dest='base', help="Base")
parser.add_option('-m', '--matching', dest='matching', action='store_true', default=False, help="Correr el match a contrario (necesita las comparaciones)")
parser.add_option('-f', '--full', dest='full', action='store_true', default=False, help="Realiza todas las acciones")
parser.add_option('-e', '--estadisticas', dest='estadisticas', action='store_true', default=False, help="Estadisticas")

(options, args) = parser.parse_args()

nombreBase = options.base or 'bath'
base = Database.getDatabase(nombreBase)

if __name__ == '__main__':
	if options.matching or options.full:
		AContrarioMatching.correrMatchAContrario(base)
	if options.estadisticas or options.full:
		Estadisticas.estadisticasFull(base)
