FIND_PACKAGE(PkgConfig)
PKG_CHECK_MODULES(OPENCV opencv)


SET(SOURCES clock.cpp decorator.cpp eyelidsegmentator.cpp helperfunctions.cpp irisdatabase.cpp irisdctencoder.cpp irisencoder.cpp
	irissegmentator.cpp iristemplate.cpp loggaborencoder.cpp parameters.cpp pupilsegmentator.cpp qualitychecker.cpp segmentationresult.cpp
	segmentator.cpp serializer.cpp templatecomparator.cpp tools.cpp videoprocessor.cpp
)

SET(HEADERS clock.h common.h decorator.h eyelidsegmentator.h helperfunctions.h irisdatabase.h irisdctencoder.h irisencoder.h
	irissegmentator.h iristemplate.h loggaborencoder.h parameters.h pupilsegmentator.h qualitychecker.h segmentationresult.h
	segmentator.h serializer.h templatecomparator.h tools.h videoprocessor.h
)

IF (CUDA_FOUND)
	SET(SOURCES ${SOURCES} irisdatabasecuda.cpp)
	SET(HEADERS ${HEADERS} cudacommon.h)
	
	SET(CUDA_SOURCES irisdatabase_kernel.cu)
	SET(CUDA_LIBRARIES ${CUDA_LIBRARIES} ${LIBRARIES})
	SET(CUDA_NVCC_FLAGS --ptxas-options=-v --compiler-options '-fPIC')
	CUDA_ADD_LIBRARY(horus_cuda ${CUDA_SOURCES})
ENDIF (CUDA_FOUND)

ADD_LIBRARY(horus SHARED ${SOURCES})
INSTALL(TARGETS horus LIBRARY DESTINATION lib)

TARGET_LINK_LIBRARIES(horus ${OPENCV_LIBRARIES})
ADD_DEFINITIONS("-Wall -O3")

IF (CUDA_FOUND)
	TARGET_LINK_LIBRARIES(horus horus_cuda)
ENDIF (CUDA_FOUND)
