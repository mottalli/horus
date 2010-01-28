QT -= core \
    gui
TARGET = horus
CONFIG += console
CONFIG -= app_bundle
TEMPLATE = app
LIBS = -lcv \
    -lhighgui \
    -lml
HEADERS += videoprocessor.h \
    types.h \
    tools.h \
    templatecomparator.h \
    serializer.h \
    segmentator.h \
    segmentationresult.h \
    qualitychecker.h \
    pupilsegmentator.h \
    parameters.h \
    loggabor1dfilter.h \
    iristemplate.h \
    irissegmentator.h \
    irisencoder.h \
    irisdctencoder.h \
    irisdatabase.h \
    helperfunctions.h \
    eyelidsegmentator.h \
    decorator.h \
    common.h \
    clock.h
SOURCES += videoprocessor.cpp \
    tools.cpp \
    templatecomparator.cpp \
    serializer.cpp \
    segmentator.cpp \
    segmentationresult.cpp \
    qualitychecker.cpp \
    pupilsegmentator.cpp \
    parameters.cpp \
    main.cpp \
    loggabor1dfilter.cpp \
    iristemplate.cpp \
    irissegmentator.cpp \
    irisencoder.cpp \
    irisdctencoder.cpp \
    irisdatabase.cpp \
    helperfunctions.cpp \
    eyelidsegmentator.cpp \
    decorator.cpp \
    clock.cpp
