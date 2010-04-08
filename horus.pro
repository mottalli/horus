LIBS = -lcxcore \
    -lcv \
    -lhighgui \
    -lcvaux \
    -lml

# OTHER_FILES += src/irisdatabase_kernel.cu
HEADERS += src/clock.h \
    src/helperfunctions.h \
	src/segmentationresult.h \
#    src/irisdatabasecuda.h \
    src/common.h \
    src/qualitychecker.h \
    src/irissegmentator.h \
    src/segmentator.h \
    src/types.h \
    src/serializer.h \
    src/cudacommon.h \
    src/decorator.h \
    src/parameters.h \
	src/templatecomparator.h \
    src/eyelidsegmentator.h \
    src/loggaborencoder.h \
    src/irisencoder.h \
#    src/irisdatabase.h \
    src/iristemplate.h \
	src/videoprocessor.h \
    src/pupilsegmentator.h \
	src/qualitychecker.h \
#    src/irisdctencoder.h \
    src/tools.h \
    src/gaborencoder.h
SOURCES += src/segmentationresult.cpp \
    src/clock.cpp \
    src/parameters.cpp \
    src/helperfunctions.cpp \
	src/main.cpp \
#    src/irisdatabasecuda.cpp \
    src/segmentator.cpp \
    src/iristemplate.cpp \
    src/serializer.cpp \
	src/videoprocessor.cpp \
    src/decorator.cpp \
    src/eyelidsegmentator.cpp \
#    src/irisdatabase.cpp \
    src/qualitychecker.cpp \
    src/irisencoder.cpp \
    src/irissegmentator.cpp \
#    src/irisdctencoder.cpp \
    src/loggaborencoder.cpp \
	src/templatecomparator.cpp \
    src/tools.cpp \
    src/pupilsegmentator.cpp \
	src/gaborencoder.cpp
OTHER_FILES += src/irisdatabase_kernel.cu
