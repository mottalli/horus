# -*- coding: UTF8 -*-

from PyQt4 import QtCore, QtGui

class SetupForm(QtGui.QWidget):
	def __init__(self, parent=None):
		super(QtGui.QWidget, self).__init__(parent)
	
	def activate(self):
		print "SETUP"
	
	def deactivate(self):
		print "NO SETUP"