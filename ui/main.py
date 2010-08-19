#!/usr/bin/python
# -*- coding: UTF8 -*-

from PyQt4 import QtGui
import sys
import MainForm
import VideoThread
import ProcessingThread
import horus
import highgui

app = QtGui.QApplication(sys.argv)

parameters = horus.Parameters.getParameters()
parameters.segmentEyelids = False

cap = highgui.cvCreateCameraCapture(0)
if not cap:
	raise IOError("Unable to initialize capture")

VideoThread.videoThread.cap = cap
VideoThread.videoThread.start()

mainForm = MainForm.MainForm()
mainForm.show()


res = app.exec_()
sys.exit(res)
