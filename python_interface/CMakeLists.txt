FIND_PACKAGE(SWIG REQUIRED)
INCLUDE(${SWIG_USE_FILE})

FIND_PACKAGE(PythonLibs REQUIRED)
INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})

INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/src)

# Find the site-packages dir
EXECUTE_PROCESS(COMMAND python -c "from distutils.sysconfig import get_python_lib; print get_python_lib()" 
	OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)
MESSAGE("Installing python extensions to ${PYTHON_SITE_PACKAGES}")

SET_SOURCE_FILES_PROPERTIES(horus.i PROPERTIES CPLUSPLUS ON)

SET(SOURCES ../src/clock.cpp ../src/decorator.cpp ../src/eyelidsegmentator.cpp ../src/irisdatabase.cpp ../src/irisencoder.cpp
	../src/irissegmentator.cpp ../src/iristemplate.cpp ../src/loggaborencoder.cpp ../src/pupilsegmentator.cpp ../src/qualitychecker.cpp ../src/segmentationresult.cpp
	../src/segmentator.cpp ../src/serializer.cpp ../src/templatecomparator.cpp ../src/tools.cpp ../src/videoprocessor.cpp ../src/gaborencoder.cpp
	../src/external/base64.cpp ../src/eyedetect.cpp)

SET(HEADERS ../src/clock.h ../src/decorator.h ../src/eyelidsegmentator.h ../src/irisdatabase.h ../src/irisencoder.h
	../src/irissegmentator.h ../src/iristemplate.h ../src/loggaborencoder.h ../src/pupilsegmentator.h ../src/qualitychecker.h ../src/segmentationresult.h
	../src/segmentator.h ../src/serializer.h ../src/templatecomparator.h ../src/tools.h ../src/videoprocessor.h ../src/gaborencoder.h 
	../src/external/base64.h ../src/eyedetect.h)

IF (USE_CUDA)
	SET(SOURCES ${SOURCES} ../src/irisdatabasecuda.cpp)
	SET(HEADERS ${HEADERS} ../src/cudacommon.h)
	SET(CMAKE_SWIG_FLAGS "-DUSE_CUDA")
ENDIF ()
	
SWIG_ADD_MODULE(horus python horus.i ${SOURCES} ${HEADERS})
SWIG_LINK_LIBRARIES(horus ${PYTHON_LIBRARIES} ${OpenCV_LIBS})

IF (USE_CUDA)
	LINK_DIRECTORIES(${CMAKE_BINARY_DIR}/src)
	SWIG_LINK_LIBRARIES(horus horus_cuda)
ENDIF ()

IF (MSVC)
	INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/horus.py ${CMAKE_CURRENT_BINARY_DIR}/Release/_horus.pyd
		DESTINATION ${PYTHON_SITE_PACKAGES})
ELSE ()
	INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/horus.py ${CMAKE_CURRENT_BINARY_DIR}/_horus.so
		DESTINATION ${PYTHON_SITE_PACKAGES})
ENDIF ()
