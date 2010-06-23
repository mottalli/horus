# -*- coding: UTF8 -*-

from scipy import *
from pylab import *
from AContrarioMatching import CANTIDAD_PARTES

def FAR(valoresInter, threshold):
	try:
		return (len(where(valoresInter <= threshold)[0]) / float(len(valoresInter))) * 100.0
	except IndexError:
		return 0		# No hay falsas aceptaciones (genera una excepción "tuple index out of range")

def FRR(valoresIntra, threshold):
	try:
		return (len(where(valoresIntra >= threshold)[0]) / float(len(valoresIntra))) * 100.0
	except IndexError:
		return 0		# No hay falsos rechazos (genera una excepción "tuple index out of range")

def calcularTablaFAR(distanciasInter, distanciasIntra, thresholds):
	ret = []
	for threshold in thresholds:
		far = (len(where(distanciasInter <= threshold)[0]) / float(len(distanciasInter))) * 100.0
		frr = (len(where(distanciasIntra >= threshold)[0]) / float(len(distanciasIntra))) * 100.0
		ret.append( (threshold, far, frr) )
	
	return ret

def calcularROC_EER(valoresIntra, valoresInter, thresholds):
	FARs = []
	FRRs = []
	for threshold in thresholds:
		FARs.append(FAR(valoresInter, threshold))
		FRRs.append(FRR(valoresIntra, threshold))

	# EER
	D = array(FARs)-array(FRRs)
	a = max(where(D <= 0)[0])
	b = min(where(D > 0)[0])
	assert a+1 == b

	EER = (FARs[a]+FARs[b]+FRRs[a]+FRRs[b])/4.0

	thresholdOptimo = (thresholds[a]+thresholds[b])/2.0

	return (FARs, FRRs, EER, thresholdOptimo)

def estadisticas(base, doShow=True):
	nIntra = (base.conn.execute('SELECT COUNT(*) FROM comparaciones WHERE intra_clase=1').fetchone())[0]
	nInter = (base.conn.execute('SELECT COUNT(*) FROM comparaciones WHERE intra_clase=0').fetchone())[0]

	print 'Cargando datos... (%i comparaciones)' % (nIntra+nInter)
	if nIntra == 0 or nInter == 0:
		raise Exception('No hay datos en la tabla comparaciones!')
	
	# De este modo es más rápido y consume menos memoria
	distanciasIntra = zeros(nIntra)
	distanciasInter = zeros(nInter)
	rs = base.conn.execute('SELECT distancia FROM comparaciones WHERE intra_clase=1')
	i = 0
	for (d,) in rs:
		distanciasIntra[i] = d
		i = i+1

	rs = base.conn.execute('SELECT distancia FROM comparaciones WHERE intra_clase=0')
	i = 0
	for (d,) in rs:
		distanciasInter[i] = d
		i = i+1
	
	print "Datos cargados. Calculando..."
	
	figure()

	muIntra = distanciasIntra.mean()
	muInter = distanciasInter.mean()
	stdIntra = distanciasIntra.std()
	stdInter = distanciasInter.std()

	separabilidad = abs(muIntra-muInter)/sqrt( (stdIntra**2 + stdInter**2)/2.0 )

	(FARs, FRRs, EER, thresholdOptimo) = calcularROC_EER(distanciasIntra, distanciasInter, linspace(0.3, 0.5, 200))
	# Genera la ROC
	# subplot(121)
	# plot(FARs, FRRs, linewidth=2, color='black')
	# u = EER*2.0
	# plot([0, u],[0, u], color='black', linestyle=':')
	# plot([EER], [EER], 'ko')
	# axis([0, u, 0, u])
	# xlabel('FAR')
	# ylabel('FRR')

	#distanciasIntra = array(datosIntra[:, 2])
	#distanciasInter = array(datosInter[:, 2])
	
	# Calcula FRR para FAR = 0
	minDistanciasInter = min(distanciasInter)
	FRRFAR0 = (len(where(distanciasIntra > minDistanciasInter)[0]) / float(len(distanciasIntra))) * 100.0
	
	# subplot(122)
	(h1, b1) = histogram(distanciasInter, bins=30)
	(h2, b2) = histogram(distanciasIntra, bins=30)
	plot(b1[:-1], h1/float(len(distanciasInter)), color='black', label='Inter-class distribution', linestyle='-')
	plot(b2[:-1], h2/float(len(distanciasIntra)), color='black', label='Intra-class distribution', linestyle='--')
	legend(loc='upper left')
	#yticks([])
	xlabel('Hamming distance')

	if doShow: 
		print "Sobre un total de %i comparaciones intra-clase y %i comparaciones inter-clase:" % (len(distanciasIntra), len(distanciasInter))
		print "Separabilidad: ", separabilidad
		print "Threshold optimo:", thresholdOptimo
		print "EER (%):", EER

		show()
		
	tablaFAR = calcularTablaFAR(distanciasInter, distanciasIntra, linspace(0.32, 0.45, 15))
	
	return (separabilidad, thresholdOptimo, EER, tablaFAR, FRRFAR0, FARs, FRRs)

def estadisticasAContrario(base, doShow=True):

	nIntra = (base.conn.execute('SELECT COUNT(*) FROM nfa_a_contrario WHERE intra_clase=1').fetchone())[0]
	nInter = (base.conn.execute('SELECT COUNT(*) FROM nfa_a_contrario WHERE intra_clase=0').fetchone())[0]

	print 'Cargando datos... (%i comparaciones)' % (nIntra+nInter)
	if nIntra == 0 or nInter == 0:
		raise Exception('No hay datos en la tabla nfa_a_contrario!')
	
	# De este modo es más rápido y consume menos memoria
	nfaIntra = zeros(nIntra)
	nfaInter = zeros(nInter)
	rs = base.conn.execute('SELECT nfa FROM nfa_a_contrario WHERE intra_clase=1')
	for (i,d) in enumerate(rs):
		nfaIntra[i] = d[0]

	rs = base.conn.execute('SELECT nfa FROM nfa_a_contrario WHERE intra_clase=0')
	for (i,d) in enumerate(rs):
		nfaInter[i] = d[0]
	
	print "Datos cargados. Calculando..."	
	
	figure()

	muIntra = nfaIntra.mean()
	muInter = nfaInter.mean()
	stdIntra = nfaIntra.std()
	stdInter = nfaInter.std()
	separabilidad = abs(muIntra-muInter)/sqrt( (stdIntra**2 + stdInter**2)/2.0 )

	# subplot(221)

	(h1, b1) = histogram(nfaInter, bins=30)
	(h2, b2) = histogram(nfaIntra, bins=40)
	plot(b1[:-1], h1/float(len(nfaInter)), color='black', label='Inter-class distribution', linestyle='-')
	plot(b2[:-1], h2/float(len(nfaIntra)), color='black', label='Intra-class distribution', linestyle='--')
	legend(loc='upper left')
	xlabel('log(NFA)')

	if True:
		print 'Calculando ROC...'

		(FARs, FRRs, EER, thresholdOptimo) = calcularROC_EER(nfaIntra, nfaInter, linspace(-5,3,50))
		# Genera la ROC
		# subplot(222)
		# plot(FARs, FRRs, linewidth=2, color='black')
		# u = EER*10.0
		# plot([0, u],[0, u], color='black', linestyle=':')
		# plot([EER], [EER], 'ko')
		# axis([0, u, 0, u])
		# xlabel('FAR')
		# ylabel('FRR')

	if True:
		# Histograma de distancias entre partes
		print 'Calculando histograma de distancias...'

		romanas = ['I', 'II', 'III', 'IV']
		dashes = ['--', '-', '-.', ':']
		# subplot(223)
		figure()
		for parte in range(CANTIDAD_PARTES):
			distancias = array([x[0] for x in base.conn.execute('SELECT distancia FROM comparaciones_a_contrario WHERE parte=%i AND intra_clase=0' % (parte,))])
			(h, b) = histogram(distancias, bins=30)
			plot(b[:-1], h/float(len(distancias)), label='Part %s' % (romanas[parte],), linestyle=dashes[parte], c='black')
		legend(loc='upper left')
		xlabel('Hamming Distance')
		#yticks([])

	if doShow: 
		print "Separabilidad: ", separabilidad
		print "Threshold optimo:", thresholdOptimo
		print "EER:", EER

		show()

	tablaFAR = calcularTablaFAR(nfaInter, nfaIntra, range(-5, 0))
	
	# Calcula FRR para FAR = 0
	minNFAInter = min(nfaInter)
	FRRFAR0 = (len(where(nfaIntra > minNFAInter)[0]) / float(len(nfaIntra))) * 100.0
	
	return (separabilidad, thresholdOptimo, EER, tablaFAR, FRRFAR0, FARs, FRRs)

def estadisticasFull(base):
	print 'Calculando estadisticas método clásico...'
	(separabilidad, thresholdOptimo, EER, tablaFAR, FRRFAR0, FARs, FRRs) = estadisticas(base, False)
	print 'Calculando estadisticas a contrario...'
	(separabilidadAC, thresholdOptimoAC, EERAC, tablaFARAC, FRRFAR0AC, FARsAC, FRRsAC) = estadisticasAContrario(base, False)
	
	N = (base.conn.execute('SELECT COUNT(*) FROM base_iris WHERE segmentacion_correcta=1').fetchone())[0]
	
	print 'Metodo clásico'
	print '-------------'
	print 'Separabilidad:', separabilidad
	print 'Threshold optimo (DH):', thresholdOptimo
	print 'EER (%):', EER
	print 'FRR para FAR=0 (%):', FRRFAR0
	print
	print 'Tabla FAR'
	print 'HD        |   FAR (%)    |  FRR (%)'
	print '-----------------------------------'
	for (hd, far, frr) in tablaFAR:
		print '%f     | %f     | %f' % (hd, far, frr)

	print

	print 'A Contrario'
	print '-------------'
	print 'Separabilidad:', separabilidadAC
	print 'Threshold optimo (epsilon-significatividad):', thresholdOptimoAC
	print 'EER (%):', EERAC
	print 'FRR para FAR=0 (%):', FRRFAR0AC
	print
	print 'Tabla FAR'
	print 'epsilon   |   FAR (%)    |  FRR (%)     |  N*FAR'
	print '-------------------------------------------------'
	for (epsilon, far, frr) in tablaFARAC:
		print '10^%i     | %f     | %f         | %f' % (epsilon, far, frr, N*far/100.0)


	figure()
	plot(FARs, FRRs, linewidth=2, color='black', label='ROC using HD')
	plot(FARsAC, FRRsAC, linewidth=2, color='black', linestyle='--', label='ROC using NFA')
	xlabel('FAR')
	ylabel('FRR')
	legend(loc='upper right')
	q = max(EER, EERAC) * 2
	plot([0, q], [0, q], color='black', linestyle=':', linewidth=1)
	axis([0, q, 0, q])

	show()

