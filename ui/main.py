#!/usr/bin/python
# -*- coding: UTF8 -*-

from PyQt4 import QtGui
import sys
import MainForm
import VideoThread
import ProcessingThread

app = QtGui.QApplication(sys.argv)
mainForm = MainForm.MainForm()
mainForm.show()

VideoThread.videoThread.start()

res = app.exec_()
sys.exit(res)
