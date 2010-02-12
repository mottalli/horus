# -*- coding: UTF8 -*-
import sqlite3
import os.path
import horus

class Database:
	def __init__(self, basePath='_base'):
		self.basePath = basePath
		self.baseFile = os.path.join(basePath, 'base.db')
		self.irisDatabase = horus.IrisDatabase()
		
		if not os.path.exists(self.baseFile):
			raise Exception('No existe el archivo ' + self.baseFile)

		self.conn = sqlite3.connect(self.baseFile)
		
		rows = self.conn.execute('SELECT id_imagen,codigo_gabor FROM base_iris')
		for row in rows:
			idImagen = int(row[0])
			templateSerializado = str(row[1])
			
			if not len(templateSerializado):
				raise Exception('La imagen %i no está codificada!' % (idImagen))
			
			template = horus.unserializeIrisTemplate(templateSerializado)
			self.irisDatabase.addTemplate(idImagen, template)
	
	#def callback(self):
	#	print '.'
	
	#def addTemplate(self, nombre, imagen, template, segmentacion):
	#	pass
	
	def doMatch(self, template, statusCallback=None):
		self.irisDatabase.doMatch(template)
	
	#def doAContrarioMatch(self, template, statusCallback=None):
	#	pass

database = Database()
