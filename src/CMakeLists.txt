SET(SOURCES clock.cpp decorator.cpp eyelidsegmentator.cpp irisdatabase.cpp irisencoder.cpp
	irissegmentator.cpp iristemplate.cpp loggaborencoder.cpp parameters.cpp pupilsegmentator.cpp qualitychecker.cpp segmentationresult.cpp
	segmentator.cpp serializer.cpp templatecomparator.cpp tools.cpp videoprocessor.cpp gaborencoder.cpp main.cpp
)

SET(HEADERS clock.h common.h decorator.h eyelidsegmentator.h irisdatabase.h irisencoder.h
	irissegmentator.h iristemplate.h loggaborencoder.h parameters.h pupilsegmentator.h qualitychecker.h segmentationresult.h
	segmentator.h serializer.h templatecomparator.h tools.h videoprocessor.h gaborencoder.h
)

IF (CUDA_FOUND)
	SET(SOURCES ${SOURCES} irisdatabasecuda.cpp)
	SET(HEADERS ${HEADERS} cudacommon.h)
	
	SET(CUDA_SOURCES irisdatabase_kernel.cu gpuwrapper.cu)
	SET(CUDA_LIBRARIES ${CUDA_LIBRARIES} ${LIBRARIES})
	SET(CUDA_NVCC_FLAGS --ptxas-options=-v --compiler-options '-fPIC')
	CUDA_ADD_LIBRARY(horus_cuda ${CUDA_SOURCES})
ENDIF (CUDA_FOUND)

ADD_EXECUTABLE(horus_bin ${SOURCES} ${HEADERS})
TARGET_LINK_LIBRARIES(horus_bin ${OpenCV_LIBS})

IF (CUDA_FOUND)
	TARGET_LINK_LIBRARIES(horus_bin horus_cuda)
ENDIF (CUDA_FOUND)
