OPTION(USE_CUDA "Use CUDA acceleration" ON)

IF (NOT CMAKE_BUILD_TYPE )
   SET(CMAKE_BUILD_TYPE "Release")
ENDIF ()


FIND_PACKAGE(PkgConfig)
PKG_CHECK_MODULES(OPENCV opencv)
PKG_CHECK_MODULES(SQLITE3 sqlite3)

SET(SOURCES clock.cpp decorator.cpp eyelidsegmentator.cpp helperfunctions.cpp irisdatabase.cpp irisdctencoder.cpp irisencoder.cpp
	irissegmentator.cpp iristemplate.cpp loggaborencoder.cpp parameters.cpp pupilsegmentator.cpp qualitychecker.cpp segmentationresult.cpp
	segmentator.cpp serializer.cpp templatecomparator.cpp tools.cpp videoprocessor.cpp)
	
IF (USE_CUDA)
	FIND_PACKAGE(CUDA)

	SET(CUDASDK /home/marcelo/Mis_Documentos/Programacion/NVIDIA_GPU_Computing_SDK/C)
	SET(SOURCES ${SOURCES} irisdatabasecuda.cpp irisdatabase_kernel.cu)
	
	CUDA_INCLUDE_DIRECTORIES(${CUDASDK}/common/inc)
	LINK_DIRECTORIES(${CUDASDK}/lib)
ENDIF (USE_CUDA)

IF (USE_CUDA)
	CUDA_ADD_EXECUTABLE(horus ${SOURCES} main.cpp)
	TARGET_LINK_LIBRARIES(horus ${OPENCV_LIBRARIES} ${SQLITE3_LIBRARIES} cutil_x86_64)
ELSE ()
	ADD_LIBRARY(horus SHARED ${SOURCES})
ENDIF ()

