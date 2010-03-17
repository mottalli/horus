#!/usr/bin/python
import sys

import horus
import Database
from pylab import *

base = Database.getDatabase('bath')
#irisDatabase = horus.IrisDatabase()
irisDatabaseCUDA = horus.IrisDatabaseCUDA()
irisDatabase = horus.IrisDatabase()

step = 1000
cant_total = 20000

codigo_base = base.conn.execute('SELECT codigo_gabor FROM base_iris WHERE segmentacion_correcta=1').fetchone()
codigo_base = horus.unserializeIrisTemplate(str(codigo_base[0]))

count = 0
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
			
			error = abs(mean(rd-rdcuda))

			print '%i, %.4f, %.4f, %.4f, %.4f %.4f' % (count, matchTime, aContrarioMatchTime, matchTimeCUDA, aContrarioMatchTimeCUDA, error)
