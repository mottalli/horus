# -*- coding: UTF8 -*-
import sqlite3
import os.path
import horus
from opencv import highgui

class Database:
	def __init__(self, basePath='_base'):
		self.basePath = basePath
		self.baseFile = os.path.join(basePath, 'base.db')
		if horus.HORUS_CUDA_SUPPORT:
			print "Nota: Activando aceleración CUDA"
			self.irisDatabase = horus.IrisDatabaseCUDA()
		else:
			self.irisDatabase = horus.IrisDatabase()
		
		if not os.path.exists(self.baseFile):
			raise Exception('No existe el archivo ' + self.baseFile)

		self.conn = sqlite3.connect(self.baseFile)
		
		rows = self.conn.execute('SELECT id_usuario,codigo_gabor FROM usuarios')
		for row in rows:
			idUsuario = int(row[0])
			templateSerializado = str(row[1])
			
			if not len(templateSerializado):
				raise Exception('La imagen %i no está codificada!' % (idUsuario))
			
			template = horus.unserializeIrisTemplate(templateSerializado)
			self.irisDatabase.addTemplate(idUsuario, template)
	
	#def callback(self):
	#	print '.'
	
	#def addTemplate(self, nombre, imagen, template, segmentacion):
	#	pass
	
	def doMatch(self, template, statusCallback=None):
		self.irisDatabase.doMatch(template)
	
	def getMinDistanceId(self):
		return self.irisDatabase.getMinDistanceId()

	def getMinDistance(self):
		return self.irisDatabase.getMinDistance()
	

	def doAContrarioMatch(self, template, statusCallback=None):
		self.irisDatabase.doAContrarioMatch(template)

	def getMinNFAId(self):
		return self.irisDatabase.getMinNFAId()

	def getMinNFA(self):
		return self.irisDatabase.getMinNFA()
	
	def informacionUsuario(self, id_usuario):
		row = self.conn.execute('SELECT id_usuario, nombre, imagen, segmentacion, codigo_gabor FROM usuarios WHERE id_usuario=?', [id_usuario]).fetchone()
		if not row:
			return None
		
		idUsuario = int(str(row[0]))
		usuario = str(row[1])
		pathImagen = os.path.join(self.basePath, str(row[2]))
		segmentacion = horus.unserializeSegmentationResult(str(row[3]))
		template = horus.unserializeIrisTemplate(str(row[4]))
		
		return { 'id': idUsuario, 'usuario': usuario, 'pathImagen': pathImagen, 'segmentacion': segmentacion, 'template': template }
	
	def agregarTemplate(self, nombre, imagen, template, segmentacion):
		if not nombre:
			raise Exception('El nombre no puede estar vacio')
			
		existe = self.conn.execute('SELECT id_usuario FROM usuarios WHERE LOWER(nombre)=LOWER(?)', [str(nombre)]).fetchone()
		if existe:
			raise Exception('Ya existe el nombre de usuario ' + nombre)
		
		if not imagen or not template or not segmentacion:
			raise Exception('Imagen, template o segmentacion no especificado')
		
		templateSerializado = horus.serializeIrisTemplate(template)
		segmentacionSerializada = horus.serializeSegmentationResult(segmentacion)
		
		fullPathImagen = '-xxxxx-'		# Valor temporario
		
		self.conn.execute('INSERT INTO usuarios(nombre, imagen, segmentacion, codigo_gabor) VALUES (?,?,?,?)', [str(nombre), fullPathImagen, segmentacionSerializada, templateSerializado])
		id = self.conn.execute('SELECT LAST_INSERT_ROWID() AS ri').fetchone()
		id = id[0]
		
		# Guarda la imagen
		nombreImagen = '%i.jpg' % (id)
		pathRelativoImagen = '%i/' % (id)
		pathFullImagen = str(os.path.join(self.basePath, pathRelativoImagen))
		
		nombreRelativoImagen = str(os.path.join(pathRelativoImagen, nombreImagen))
		nombreFullImagen = str(os.path.join(self.basePath, nombreRelativoImagen))
		
		os.mkdir(pathFullImagen)
		highgui.cvSaveImage(nombreFullImagen, imagen)
		self.conn.execute('UPDATE usuarios SET imagen=? WHERE id_usuario=?', [nombreRelativoImagen, id])
		
		self.conn.commit()
		
		self.irisDatabase.addTemplate(id, template)
	
	def agregarImagen(self, id, imagen, template=None, segmentacion=None):
		pathImagen = '%i' % (id)
		
		i = 1
		while True:
			nombreImagen = '%i_%i.jpg' % (id, i)
			pathRelativoImagen = str(os.path.join(pathImagen, nombreImagen))
			fullPathImagen = os.path.join(self.basePath, pathRelativoImagen)
			
			if not os.path.exists(fullPathImagen): break
			
			i = i+1
		
		highgui.cvSaveImage(fullPathImagen, imagen)
	
	def databaseSize(self):
		return self.irisDatabase.databaseSize()
	
	def getNFAFor(self, templateId):
		return self.irisDatabase.getNFAFor(templateId)
	
	def getDistanceFor(self, templateId):
		return self.irisDatabase.getDistanceFor(templateId)
	
	def getMatchingTime(self):
		return self.irisDatabase.getMatchingTime()

database = Database()
