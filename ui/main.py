#!/usr/bin/python
# -*- coding: UTF8 -*-

from PyQt4 import QtGui
import sys
import MainForm
import VideoThread
import ProcessingThread
import horus

app = QtGui.QApplication(sys.argv)
mainForm = MainForm.MainForm()
mainForm.show()

parameters = horus.Parameters.getParameters()
parameters.segmentEyelids = False

VideoThread.videoThread.start()

res = app.exec_()
sys.exit(res)
