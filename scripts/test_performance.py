#!/usr/bin/python
# -*- coding: UTF8 -*-
import sys

import horus
import Database
from pylab import *

def matchesPorSegundo(v):
	return (1000.0*v[-1, 0])/v[-1, 1];

base = Database.getDatabase('bath')
#irisDatabase = horus.IrisDatabase()
irisDatabaseCUDA = horus.IrisDatabaseCUDA()
irisDatabase = horus.IrisDatabase()

step = 1000
cant_total = 20000
#cant_total = 5000

codigo_base = base.conn.execute('SELECT codigo_gabor FROM base_iris WHERE segmentacion_correcta=1').fetchone()
codigo_base = horus.unserializeIrisTemplate(str(codigo_base[0]))

count = 0
print "Tamaño | Tiempo match | Tiempo a contrario | Tiempo match CUDA | Tiempo a contrario CUDA | Dif. prom. HD | Dif. prom. NFA"
res = []
while count < cant_total:
	rows = base.conn.execute('SELECT id_imagen,codigo_gabor FROM base_iris WHERE segmentacion_correcta=1')
	for row in rows:
		idTemplate = int(row[0])
		codigo = horus.unserializeIrisTemplate(str(row[1]))
		irisDatabase.addTemplate(count, codigo)
		irisDatabaseCUDA.addTemplate(count, codigo)
		count += 1
		
		if count > 1 and count % step == 0:
			irisDatabase.doMatch(codigo_base)
			matchTime = irisDatabase.getMatchingTime()
			irisDatabase.doAContrarioMatch(codigo_base)
			aContrarioMatchTime = irisDatabase.getMatchingTime()

			irisDatabaseCUDA.doMatch(codigo_base)
			matchTimeCUDA = irisDatabaseCUDA.getMatchingTime()
			irisDatabaseCUDA.doAContrarioMatch(codigo_base)
			aContrarioMatchTimeCUDA = irisDatabaseCUDA.getMatchingTime()
			
			rd = array(irisDatabase.resultDistances)
			rdcuda = array(irisDatabaseCUDA.resultDistances)
			
			rnfa = array(irisDatabase.resultNFAs)
			rnfacuda = array(irisDatabaseCUDA.resultNFAs)
			
			error = abs(mean(rd-rdcuda))
			errorcuda = abs(mean(rnfa-rnfacuda))
			
			res.append([count, matchTime, aContrarioMatchTime, matchTimeCUDA, aContrarioMatchTimeCUDA, error, errorcuda])

			print '%i, %.4f, %.4f, %.4f, %.4f %.8f, %.8f' % (count, matchTime, aContrarioMatchTime, matchTimeCUDA, aContrarioMatchTimeCUDA, error, errorcuda)

res = array(res)

print 'Matches por segundo (HD): ', matchesPorSegundo(res[:, [0,1]])
print 'Matches por segundo (A Contrario): ', matchesPorSegundo(res[:, [0,2]])
print 'Matches por segundo (HD c/CUDA): ', matchesPorSegundo(res[:, [0,3]])
print 'Matches por segundo (A Contrario c/CUDA): ', matchesPorSegundo(res[:, [0,4]])

plot(res[:, 0], res[:, 1], label='HD')
plot(res[:, 0], res[:, 2], label='A Contrario')
plot(res[:, 0], res[:, 3], label='HD w/CUDA')
plot(res[:, 0], res[:, 4], label='A Contrario w/CUDA')
xlabel('Database size')
ylabel('Matching time (ms.)')
legend()
show()
