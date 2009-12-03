import sqlite3
import os.path

INSTALLED_BASES = ['casia1', 'casia3', 'casia3p', 'mmu', 'bath']
PATHS_BASES = {
		'casia1': '/home/marcelo/Mis_Documentos/Facu/Tesis/Bases de datos/CASIA1',
		'casia3': '/home/marcelo/Mis_Documentos/Facu/Tesis/Bases de datos/CASIA-IrisV3-Interval',
		'casia3p': '/home/marcelo/Mis_Documentos/Facu/Tesis/Bases de datos/CASIA3-Preprocesada',
		'mmu': '/home/marcelo/Mis_Documentos/Facu/Tesis/Bases de datos/MMU Iris Database',
		'bath': '/home/marcelo/Mis_Documentos/Facu/Tesis/Bases de datos/Bath',
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
	if not name in INSTALLED_BASES:
		raise Exception('Base ' + name + 'doesn\'t exist')
	
	return IrisDatabase(PATHS_BASES[name])
