import sqlite3
import os.path
import horus

PATH_BASE = '/home/marcelo/iris/BBDD'

PATHS_BASES = {
		#'unificada': PATH_BASE,
		'unificada': '/home/marcelo/Mis_Documentos/Facu/Tesis/Bases de datos',
		'casia1': str(os.path.join(PATH_BASE, 'CASIA1')),
		#'casia3': '/home/marcelo/Mis_Documentos/Facu/Tesis/Bases de datos/CASIA-IrisV3-Interval',
		'casia3p': str(os.path.join(PATH_BASE, 'CASIA3-Preprocesada')),
		'mmu': str(os.path.join(PATH_BASE, 'MMU')),
		'bath': str(os.path.join(PATH_BASE, 'Bath')),
		'uba': '/home/marcelo/iris/horus/ui/_base'
	}

class IrisDatabase:
	def __init__(self, path):
		self.path = path
		baseFile = os.path.join(path, 'base.db')
		
		if not os.path.exists(baseFile):
			raise Exception('No such file: ' + baseFile)
		
		self.path = path
		self.conn = sqlite3.connect(baseFile)
	
	def fullPath(self, path):
		return str(os.path.join(self.path, path))
		
def getDatabase(name):
	if not name in PATHS_BASES.keys():
		raise Exception('Base ' + name + ' doesn\'t exist')
	
	database = IrisDatabase(PATHS_BASES[name])
	loadParameters(name)
	return database

def loadParameters(name):
	parameters = horus.Parameters.getParameters()
	
	if name == 'mmu':
		#parameters.normalizationWidth = parameters.templateWidth
		#parameters.normalizationHeight = parameters.templateHeight
		pass
	elif name == 'casia1':
		#parameters.normalizationWidth = parameters.templateWidth
		#parameters.normalizationHeight = parameters.templateHeight
		pass
	elif name == 'casia3p':
		#parameters.normalizationWidth = parameters.templateWidth
		#parameters.normalizationHeight = parameters.templateHeight
		pass
	elif name == 'bath':
		#parameters.normalizationWidth = parameters.templateWidth
		#parameters.normalizationHeight = parameters.templateHeight
		parameters.muPupil = 0
		parameters.sigmaPupil = 10
		pass
