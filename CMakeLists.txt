CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(Horus)

OPTION(USE_CUDA "Use CUDA acceleration" OFF)

SET(HORUS_LIB_SOURCES 
		${PROJECT_SOURCE_DIR}/src/clock.cpp
		${PROJECT_SOURCE_DIR}/src/decorator.cpp 
		${PROJECT_SOURCE_DIR}/src/eyelidsegmentator.cpp 
		${PROJECT_SOURCE_DIR}/src/irisdatabase.cpp 
		${PROJECT_SOURCE_DIR}/src/irisencoder.cpp
		${PROJECT_SOURCE_DIR}/src/irissegmentator.cpp 
		${PROJECT_SOURCE_DIR}/src/iristemplate.cpp 
		${PROJECT_SOURCE_DIR}/src/loggaborencoder.cpp 
		${PROJECT_SOURCE_DIR}/src/pupilsegmentator.cpp 
		${PROJECT_SOURCE_DIR}/src/qualitychecker.cpp 
		${PROJECT_SOURCE_DIR}/src/segmentator.cpp 
		${PROJECT_SOURCE_DIR}/src/serializer.cpp 
		${PROJECT_SOURCE_DIR}/src/templatecomparator.cpp 
		${PROJECT_SOURCE_DIR}/src/tools.cpp
		${PROJECT_SOURCE_DIR}/src/videoprocessor.cpp
		${PROJECT_SOURCE_DIR}/src/gaborencoder.cpp 
		${PROJECT_SOURCE_DIR}/src/eyedetect.cpp
		${PROJECT_SOURCE_DIR}/src/external/base64.cpp 
)

SET(HORUS_LIB_HEADERS 
		${PROJECT_SOURCE_DIR}/src/clock.h
		${PROJECT_SOURCE_DIR}/src/common.h
		${PROJECT_SOURCE_DIR}/src/decorator.h
		${PROJECT_SOURCE_DIR}/src/eyelidsegmentator.h
		${PROJECT_SOURCE_DIR}/src/irisdatabase.h
		${PROJECT_SOURCE_DIR}/src/irisencoder.h
		${PROJECT_SOURCE_DIR}/src/irissegmentator.h
		${PROJECT_SOURCE_DIR}/src/iristemplate.h
		${PROJECT_SOURCE_DIR}/src/loggaborencoder.h
		${PROJECT_SOURCE_DIR}/src/pupilsegmentator.h
		${PROJECT_SOURCE_DIR}/src/qualitychecker.h
		${PROJECT_SOURCE_DIR}/src/segmentator.h
		${PROJECT_SOURCE_DIR}/src/serializer.h
		${PROJECT_SOURCE_DIR}/src/templatecomparator.h
		${PROJECT_SOURCE_DIR}/src/tools.h
		${PROJECT_SOURCE_DIR}/src/videoprocessor.h
		${PROJECT_SOURCE_DIR}/src/gaborencoder.h
		${PROJECT_SOURCE_DIR}/src/types.h
		${PROJECT_SOURCE_DIR}/src/external/base64.h
		${PROJECT_SOURCE_DIR}/src/eyedetect.h
		${PROJECT_SOURCE_DIR}/src/horus.h
)

INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/src)

IF (WIN32)
	FIND_PACKAGE(OpenCV)
ELSE (WIN32)
	FIND_PACKAGE(PkgConfig)
	PKG_CHECK_MODULES(OPENCV opencv)
	SET(OpenCV_LIBS ${OPENCV_LIBRARIES})
ENDIF (WIN32)
	
IF (USE_CUDA)
	FIND_PACKAGE(CUDA)
	IF (CUDA_FOUND)
		ADD_DEFINITIONS("-DUSE_CUDA")
		SET(HORUS_LIB_SOURCES ${HORUS_LIB_SOURCES} ${PROJECT_SOURCE_DIR}/src/irisdatabasecuda.cpp)
		SET(HORUS_LIB_HEADERS ${HORUS_LIB_HEADERS} ${PROJECT_SOURCE_DIR}/src/cudacommon.h)
		SET(CUDA_SOURCES ${PROJECT_SOURCE_DIR}/src/irisdatabase_kernel.cu ${PROJECT_SOURCE_DIR}/src/gpuwrapper.cu)
		#SET(CUDA_LIBRARIES ${CUDA_LIBRARIES} ${LIBRARIES})
		# Enable this to compile CUDA support in a dynamic library
		SET(CUDA_NVCC_FLAGS --ptxas-options=-v --compiler-options '-fPIC')
		CUDA_ADD_LIBRARY(horus_cuda ${CUDA_SOURCES})
	ELSE (CUDA_FOUND)
		SET(USE_CUDA OFF)
	ENDIF (CUDA_FOUND)
ENDIF (USE_CUDA)

ADD_LIBRARY(horus SHARED ${HORUS_LIB_SOURCES})

IF (USE_CUDA)
	TARGET_LINK_LIBRARIES(horus horus_cuda)
ENDIF (USE_CUDA)

INSTALL(TARGETS horus LIBRARY DESTINATION lib)
INSTALL(FILES ${HORUS_LIB_HEADERS} DESTINATION include/horus)

ADD_SUBDIRECTORY(python_interface)
ADD_SUBDIRECTORY(test)

CONFIGURE_FILE(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
  IMMEDIATE @ONLY)

ADD_CUSTOM_TARGET(uninstall
  "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake")
