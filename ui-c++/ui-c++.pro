#-------------------------------------------------
#
# Project created by QtCreator 2011-04-16T14:36:22
#
#-------------------------------------------------

QT       += core gui

TARGET = ui-c++
TEMPLATE = app

LIBS += -lml -lcvaux -lhighgui -lcv -lcxcore

INCLUDEPATH += ../src ./external/boost/

SOURCES += main.cpp\
        mainwindow.cpp \
    videothread.cpp \
    imagewidget.cpp \
    processingthread.cpp \
    registerdialog.cpp \
	matchingdialog.cpp \
	external/sqlite3/sqlite3.c \
	sqlite3irisdatabase.cpp

SOURCES += ../src/clock.cpp              ../src/loggaborencoder.cpp     ../src/segmentator.cpp \
	../src/decorator.cpp          ../src/serializer.cpp \
	../src/eyelidsegmentator.cpp  ../src/irisencoder.cpp       ../src/pupilsegmentator.cpp    ../src/templatecomparator.cpp \
	../src/gaborencoder.cpp       ../src/irissegmentator.cpp   ../src/qualitychecker.cpp      ../src/tools.cpp \
	../src/irisdatabase.cpp       ../src/iristemplate.cpp      ../src/segmentationresult.cpp  ../src/videoprocessor.cpp

SOURCES += external/boost/libs/system/src/error_code.cpp \
	external/boost/libs/system/src/local_free_on_destruction.hpp \
	external/boost/libs/filesystem/v3/src/windows_file_codecvt.hpp \
	external/boost/libs/filesystem/v3/src/windows_file_codecvt.cpp \
	external/boost/libs/filesystem/v3/src/codecvt_error_category.cpp \
	external/boost/libs/filesystem/v3/src/path_traits.cpp \
	external/boost/libs/filesystem/v3/src/unique_path.cpp \
	external/boost/libs/filesystem/v3/src/utf8_codecvt_facet.cpp \
	external/boost/libs/filesystem/v3/src/portability.cpp \
	external/boost/libs/filesystem/v3/src/operations.cpp \
	external/boost/libs/filesystem/v3/src/path.cpp \

HEADERS  += mainwindow.h \
    videothread.h \
    imagewidget.h \
    common.h \
    processingthread.h \
    registerdialog.h \
    matchingdialog.h \
	external/sqlite3/sqlite3.h \
	external/sqlite3/sqlite3ext.h \
	sqlite3irisdatabase.h \
	../src/clock.h              ../src/gaborencoder.h      ../src/irissegmentator.h   ../src/segmentationresult.h  ../src/types.h \
	../src/common.h             ../src/iristemplate.h      ../src/segmentator.h         ../src/videoprocessor.h \
	../src/irisdatabase.h      ../src/loggaborencoder.h   ../src/serializer.h \
	../src/decorator.h          ../src/pupilsegmentator.h  ../src/templatecomparator.h \
	../src/eyelidsegmentator.h  ../src/irisencoder.h       ../src/qualitychecker.h    ../src/tools.h

FORMS    += mainwindow.ui \
    registerdialog.ui \
    matchingdialog.ui
