#!/usr/bin/python
# -*- coding: UTF8 -*-

from PyQt4 import QtGui
import sys
import MainForm
import VideoThread
import ProcessingThread
import horus
from opencv import highgui

app = QtGui.QApplication(sys.argv)

parameters = horus.Parameters.getParameters()

# PARAMETROS
parameters.segmentEyelids = False
parameters.focusThreshold = 35

cap = highgui.cvCreateCameraCapture(0)
if not cap:
	raise IOError("Unable to initialize capture")

VideoThread.videoThread.cap = cap
VideoThread.videoThread.start()

mainForm = MainForm.MainForm()
mainForm.show()

horus.Parameters.getParameters().focusThreshold = 30


res = app.exec_()
sys.exit(res)
