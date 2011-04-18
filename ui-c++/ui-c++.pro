#-------------------------------------------------
#
# Project created by QtCreator 2011-04-16T14:36:22
#
#-------------------------------------------------

QT       += core gui

TARGET = ui-c++
TEMPLATE = app

LIBS += -lhorus -lml -lcvaux -lhighgui -lcv -lcxcore


SOURCES += main.cpp\
        mainwindow.cpp \
    videothread.cpp \
    imagewidget.cpp \
    processingthread.cpp \
    registerdialog.cpp \
    matchingdialog.cpp

HEADERS  += mainwindow.h \
    videothread.h \
    imagewidget.h \
    common.h \
    processingthread.h \
    registerdialog.h \
    matchingdialog.h

FORMS    += mainwindow.ui \
    registerdialog.ui \
    matchingdialog.ui
