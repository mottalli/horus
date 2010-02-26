#!/usr/bin/python

import horus
import Database

base = Database.getDatabase('bath')
irisDatabase = horus.IrisDatabase()

step = 1000
cant_total = 50000

codigo_base = base.conn.execute('SELECT codigo_gabor FROM base_iris WHERE segmentacion_correcta=1').fetchone()
codigo_base = horus.unserializeIrisTemplate(str(codigo_base[0]))

count = 0
while count < cant_total:
	rows = base.conn.execute('SELECT id_imagen,codigo_gabor FROM base_iris WHERE segmentacion_correcta=1')
	for row in rows:
		idTemplate = int(row[0])
		codigo = horus.unserializeIrisTemplate(str(row[1]))
		irisDatabase.addTemplate(count, codigo)
		count += 1
		
		if count > 1 and count % step == 0:
			irisDatabase.doMatch(codigo_base)
			matchTime = irisDatabase.getMatchingTime()
			irisDatabase.doAContrarioMatch(codigo_base)
			aContrarioMatchTime = irisDatabase.getMatchingTime()
			print '%i, %.4f, %.4f' % (count, matchTime, aContrarioMatchTime)
