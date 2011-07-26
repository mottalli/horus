CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(Prueba-SDK)

# Definir estas cosas!

SET(BOOST_ROOT "C:/boost_1_47_0")
SET(SDK_DIR "C:/Users/Marcelo/Desktop/Samples SDK ISCAN")
SET(HORUS_DIR "c:/Users/Marcelo/Desktop/horus")


FIND_PACKAGE(OpenCV REQUIRED)

SET(Boost_USE_STATIC_LIBS ON)
FIND_PACKAGE(Boost REQUIRED COMPONENTS system filesystem date_time program_options thread)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})

SET(HORUS_SOURCES 
	${HORUS_DIR}/src/decorator.cpp 
	${HORUS_DIR}/src/eyelidsegmentator.cpp
	${HORUS_DIR}/src/irisdatabase.cpp 
	${HORUS_DIR}/src/irisencoder.cpp
	${HORUS_DIR}/src/irissegmentator.cpp 
	${HORUS_DIR}/src/iristemplate.cpp 
	${HORUS_DIR}/src/loggaborencoder.cpp
	${HORUS_DIR}/src/pupilsegmentator.cpp 
	${HORUS_DIR}/src/qualitychecker.cpp 
	${HORUS_DIR}/src/segmentator.cpp 
	${HORUS_DIR}/src/serializer.cpp 
	${HORUS_DIR}/src/templatecomparator.cpp 
	${HORUS_DIR}/src/tools.cpp
	${HORUS_DIR}/src/videoprocessor.cpp
	${HORUS_DIR}/src/gaborencoder.cpp 
	${HORUS_DIR}/src/eyedetect.cpp
	${HORUS_DIR}/src/external/base64.cpp 
)

SET(HORUS_HEADERS 
	${HORUS_DIR}/src/clock.h
	${HORUS_DIR}/src/common.h
	${HORUS_DIR}/src/decorator.h
	${HORUS_DIR}/src/eyelidsegmentator.h
	${HORUS_DIR}/src/irisdatabase.h
	${HORUS_DIR}/src/irisencoder.h
	${HORUS_DIR}/src/irissegmentator.h
	${HORUS_DIR}/src/iristemplate.h
	${HORUS_DIR}/src/loggaborencoder.h
	${HORUS_DIR}/src/pupilsegmentator.h
	${HORUS_DIR}/src/qualitychecker.h
	${HORUS_DIR}/src/segmentator.h
	${HORUS_DIR}/src/serializer.h
	${HORUS_DIR}/src/templatecomparator.h
	${HORUS_DIR}/src/tools.h
	${HORUS_DIR}/src/videoprocessor.h
	${HORUS_DIR}/src/gaborencoder.h
	${HORUS_DIR}/src/types.h
	${HORUS_DIR}/src/external/base64.h
	${HORUS_DIR}/src/eyedetect.h
	${HORUS_DIR}/src/horus.h
)

INCLUDE_DIRECTORIES("${SDK_DIR}/Inc" ${HORUS_DIR}/src/)

FIND_PACKAGE(Qt4 REQUIRED)
INCLUDE(${QT_USE_FILE})
ADD_DEFINITIONS(${QT_DEFINITIONS})

SET(Prueba_HEADERS common.h imagewidget.h irisvideowidget.h matchingdialog.h registerdialog.h sqlite3wrapper.h
		debugdialog.h irisvideocapture.h mainwindow.h processingthread.h sqlite3irisdatabase.h videothread.h
		external/sqlite3/sqlite3.h
		ISCAN2CaptureDriver.h "${SDK_DIR}/Inc/biob.h" "${SDK_DIR}/Inc/biob_defs.h" "${SDK_DIR}/Inc/biob_defs_common.h" 
		"${SDK_DIR}/Inc/biob_defs_ise.h" "${SDK_DIR}/Inc/biob_errorcodes.h"
)

SET(Prueba_SOURCES debugdialog.cpp irisvideocapture.cpp main.cpp matchingdialog.cpp registerdialog.cpp sqlite3wrapper.cpp
		imagewidget.cpp irisvideowidget.cpp mainwindow.cpp processingthread.cpp sqlite3irisdatabase.cpp videothread.cpp
		external/sqlite3/sqlite3.c
		main.cpp ISCAN2CaptureDriver.cpp "${SDK_DIR}/Inc/biob_defs_common.c"
)

SET(Prueba_FORMS debugdialog.ui mainwindow.ui matchingdialog.ui registerdialog.ui)

QT4_WRAP_CPP(Prueba_HEADERS_MOC ${Prueba_HEADERS})
QT4_WRAP_UI(Prueba_FORMS_HEADERS ${Prueba_FORMS})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})		# Without this, it cannot include the .h files (!)

SET(BIOB_LIBS "${SDK_DIR}/Lib/bba_iscan.lib" "${SDK_DIR}/Lib/BioB_XML.lib")

ADD_EXECUTABLE(prueba ${Prueba_SOURCES} ${Prueba_HEADERS_MOC} ${Prueba_FORMS_HEADERS} ${Prueba_HEADERS} ${HORUS_SOURCES} ${HORUS_HEADERS})
TARGET_LINK_LIBRARIES(prueba ${OpenCV_LIBS} ${BIOB_LIBS} ${Boost_LIBRARIES} ${QT_LIBRARIES})
