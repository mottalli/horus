#!/usr/bin/python
# -*- coding: UTF8 -*-

import horus, Database
from scipy import *
from pylab import *

#bases = ['casia3p', 'mmu', 'bath', 'unificada']
#titulosBases = ['CASIA', 'MMU', 'Bath', 'Unified']
bases = ['casia3p', 'bath', 'unificada']
titulosBases = ['CASIA', 'Bath', 'Unified']

colores = map(str, linspace(0, 0.75, len(bases)))

subplot(121)

for (i, sbase) in enumerate(bases):
	print sbase, "..."
	base = Database.getDatabase(sbase)

	nIntra = (base.conn.execute('SELECT COUNT(*) FROM nfa_a_contrario WHERE intra_clase=1').fetchone())[0]
	nInter = (base.conn.execute('SELECT COUNT(*) FROM nfa_a_contrario WHERE intra_clase=0').fetchone())[0]

	print 'Cargando datos... (%i comparaciones)' % (nIntra+nInter)
	if nIntra == 0 or nInter == 0:
		raise Exception('No hay datos en la tabla nfa_a_contrario!')
	
	# De este modo es más rápido y consume menos memoria
	nfaIntra = zeros(nIntra)
	nfaInter = zeros(nInter)
	rs = base.conn.execute('SELECT nfa FROM nfa_a_contrario WHERE intra_clase=1')
	for (j,d) in enumerate(rs):
		nfaIntra[j] = d[0]

	rs = base.conn.execute('SELECT nfa FROM nfa_a_contrario WHERE intra_clase=0')
	for (j,d) in enumerate(rs):
		nfaInter[j] = d[0]
	
	print "Datos cargados. Calculando..."	

	(hinter, binter) = histogram(nfaInter, bins=30)
	(hintra, bintra) = histogram(nfaIntra, bins=40)
	
	hinter = hinter / float(len(nfaInter))
	hintra = hintra / float(len(nfaIntra))
	binter = binter[:-1]
	bintra = bintra[:-1]

	plot(binter, hinter, linestyle='-', color=colores[i], label=titulosBases[i])
	plot(bintra, hintra, linestyle='--', color=colores[i])

legend(loc='upper left')
xlabel('log(NFA)')

subplot(122)

for (i, sbase) in enumerate(bases):
	print sbase, "..."
	base = Database.getDatabase(sbase)

	nIntra = (base.conn.execute('SELECT COUNT(*) FROM comparaciones WHERE intra_clase=1').fetchone())[0]
	nInter = (base.conn.execute('SELECT COUNT(*) FROM comparaciones WHERE intra_clase=0').fetchone())[0]

	print 'Cargando datos... (%i comparaciones)' % (nIntra+nInter)
	if nIntra == 0 or nInter == 0:
		raise Exception('No hay datos en la tabla nfa_a_contrario!')
	
	# De este modo es más rápido y consume menos memoria
	distanciasIntra = zeros(nIntra)
	distanciasInter = zeros(nInter)
	rs = base.conn.execute('SELECT distancia FROM comparaciones WHERE intra_clase=1')
	for (j,d) in enumerate(rs):
		distanciasIntra[j] = d[0]

	rs = base.conn.execute('SELECT distancia FROM comparaciones WHERE intra_clase=0')
	for (j,d) in enumerate(rs):
		distanciasInter[j] = d[0]
	
	print "Datos cargados. Calculando..."	

	(hinter, binter) = histogram(distanciasInter, bins=30)
	(hintra, bintra) = histogram(distanciasIntra, bins=40)
	
	hinter = hinter / float(len(distanciasInter))
	hintra = hintra / float(len(distanciasIntra))
	binter = binter[:-1]
	bintra = bintra[:-1]
	
	plot(binter, hinter, linestyle='-', color=colores[i], label=titulosBases[i])
	plot(bintra, hintra, linestyle='--', color=colores[i])

legend(loc='upper left')
xlabel('Hamming distance')

show()
