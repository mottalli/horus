EXECUTABLE	:= horus
CUFILES		:= irisdatabase.cu
CCFILES		:= clock.cpp \
			irisdatabase.cpp \
			irisdatabasecuda.cpp \
			iristemplate.cpp \
			pupilsegmentator.cpp \
			serializer.cpp \
			decorator.cpp \
			irisdctencoder.cpp \
			loggaborencoder.cpp \
			qualitychecker.cpp \
			templatecomparator.cpp \
			eyelidsegmentator.cpp \
			irisencoder.cpp \
			main.cpp \
			segmentationresult.cpp \
			tools.cpp \
			helperfunctions.cpp \
			irissegmentator.cpp \
			parameters.cpp \
			segmentator.cpp \
			videoprocessor.cpp
EXTRA_LIBS	:= `pkg-config --libs opencv sqlite3`

BINDIR = ./
ROOTDIR := /home/marcelo/Mis_Documentos/Programacion/NVIDIA_GPU_Computing_SDK/C/common
include $(ROOTDIR)/../common/common.mk
