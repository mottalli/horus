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

def calcularTablaFAR(nfaInter, epsilons):
	ret = []
	for epsilon in epsilons:
		a = (len(where(nfaInter <= epsilon)[0]) / float(len(nfaInter))) * 100.0
		ret.append( (epsilon, a) )
	
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
	n = base.conn.execute('SELECT COUNT(*) FROM comparaciones').fetchall()
	n = n[0][0]
	if n == 0:
		raise Exception('No hay datos en la tabla comparaciones!')
	
	print 'Cargando datos...'
	datosIntra = array(base.conn.execute('SELECT * FROM comparaciones WHERE intra_clase=1').fetchall())
	datosInter = array(base.conn.execute('SELECT * FROM comparaciones WHERE intra_clase=0').fetchall())
	distanciasIntra = datosIntra[:, 2]
	distanciasInter = datosInter[:, 2]
	
	figure()

	muIntra = distanciasIntra.mean()
	muInter = distanciasInter.mean()
	stdIntra = distanciasIntra.std()
	stdInter = distanciasInter.std()

	separabilidad = abs(muIntra-muInter)/sqrt( (stdIntra**2 + stdInter**2)/2.0 )

	(FARs, FRRs, EER, thresholdOptimo) = calcularROC_EER(distanciasIntra, distanciasInter, linspace(0.3, 0.5, 200))
	# Genera la ROC
	subplot(121)
	plot(FARs, FRRs, linewidth=2, color='black')
	u = EER*2.0
	plot([0, u],[0, u], color='black', linestyle=':')
	plot([EER], [EER], 'ko')
	axis([0, u, 0, u])
	xlabel('FAR')
	ylabel('FRR')

	distanciasIntra = array(datosIntra[:, 2])
	distanciasInter = array(datosInter[:, 2])

	subplot(122)
	(h1, b1) = histogram(distanciasInter, bins=30)
	(h2, b2) = histogram(distanciasIntra, bins=30)
	plot(b1[:-1], h1/float(len(distanciasInter)), color='black', label='Inter-class distribution', linestyle='-')
	plot(b2[:-1], h2/float(len(distanciasIntra)), color='black', label='Intra-class distribution', linestyle='--')
	legend(loc='upper left')
	#yticks([])
	xlabel('Hamming distance')

	if doShow: 
		print "Separabilidad: ", separabilidad
		print "Threshold optimo:", thresholdOptimo
		print "EER:", EER

		show()
	
	return (separabilidad, thresholdOptimo, EER)

def estadisticasAContrario(base, doShow=True):
	print 'Cargando datos...'
	
	n = base.conn.execute('SELECT COUNT(*) FROM nfa_a_contrario').fetchall()
	n = n[0][0]
	if n == 0:
		raise Exception('No hay datos en la tabla nfa_a_contrario!')
	
	datosIntra = array(base.conn.execute('SELECT * FROM nfa_a_contrario WHERE intra_clase=1').fetchall())
	datosInter = array(base.conn.execute('SELECT * FROM nfa_a_contrario WHERE intra_clase=0').fetchall())
	nfaIntra = datosIntra[:, 2]
	nfaInter = datosInter[:, 2]
	
	figure()

	muIntra = nfaIntra.mean()
	muInter = nfaInter.mean()
	stdIntra = nfaIntra.std()
	stdInter = nfaInter.std()
	separabilidad = abs(muIntra-muInter)/sqrt( (stdIntra**2 + stdInter**2)/2.0 )

	subplot(221)

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
		subplot(222)
		plot(FARs, FRRs, linewidth=2, color='black')
		u = EER*10.0
		plot([0, u],[0, u], color='black', linestyle=':')
		plot([EER], [EER], 'ko')
		axis([0, u, 0, u])
		xlabel('FAR')
		ylabel('FRR')

	if True:
		# Histograma de distancias entre partes
		print 'Calculando histograma de distancias...'

		romanas = ['I', 'II', 'III', 'IV']
		dashes = ['--', '-', '-.', ':']
		subplot(223)
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

	tablaFAR = calcularTablaFAR(nfaInter, range(-5, 0))
	return (separabilidad, thresholdOptimo, EER, tablaFAR)

def estadisticasFull(base):
	print 'Calculando estadisticas método clásico...'
	(separabilidad, thresholdOptimo, EER) = estadisticas(base, False)
	print 'Calculando estadisticas a contrario...'
	(separabilidadAC, thresholdOptimoAC, EERAC, tablaFAR) = estadisticasAContrario(base, False)
	
	print 'Metodo clásico'
	print '-------------'
	print 'Separabilidad:', separabilidad
	print 'Threshold optimo (DH):', thresholdOptimo
	print 'EER (%):', EER

	print 'A Contrario'
	print '-------------'
	print 'Separabilidad:', separabilidadAC
	print 'Threshold optimo (epsilon-significatividad):', thresholdOptimoAC
	print 'EER (%):', EERAC
	
	print 'Tabla FAR'
	print 'epsilon   |   FAR (%)'
	print '-----------------'
	for (epsilon, far) in tablaFAR:
		print '10^%i     | %f' % (epsilon, far)

	show()
