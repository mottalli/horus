SET(SOURCES clock.cpp decorator.cpp eyelidsegmentator.cpp irisdatabase.cpp irisencoder.cpp
	irissegmentator.cpp iristemplate.cpp loggaborencoder.cpp pupilsegmentator.cpp qualitychecker.cpp segmentationresult.cpp
	segmentator.cpp serializer.cpp templatecomparator.cpp tools.cpp videoprocessor.cpp gaborencoder.cpp external/base64.cpp
	eyedetect.cpp
)

SET(HEADERS clock.h common.h decorator.h eyelidsegmentator.h irisdatabase.h irisencoder.h
	irissegmentator.h iristemplate.h loggaborencoder.h pupilsegmentator.h qualitychecker.h segmentationresult.h
	segmentator.h serializer.h templatecomparator.h tools.h videoprocessor.h gaborencoder.h types.h external/base64.h
	eyedetect.h horus.h
)

IF (USE_CUDA)
	SET(SOURCES ${SOURCES} irisdatabasecuda.cpp)
	SET(HEADERS ${HEADERS} cudacommon.h)
	
	SET(CUDA_SOURCES irisdatabase_kernel.cu gpuwrapper.cu)
	SET(CUDA_LIBRARIES ${CUDA_LIBRARIES} ${LIBRARIES})
	SET(CUDA_NVCC_FLAGS --ptxas-options=-v --compiler-options '-fPIC')			# Enable this to compile CUDA support in a dynamic library
	CUDA_ADD_LIBRARY(horus_cuda ${CUDA_SOURCES})
ENDIF (USE_CUDA)

#ADD_EXECUTABLE(horus_bin ${SOURCES} ${HEADERS})
ADD_LIBRARY(horus SHARED ${SOURCES})
TARGET_LINK_LIBRARIES(horus ${OpenCV_LIBS})

IF (USE_CUDA)
	TARGET_LINK_LIBRARIES(horus horus_cuda)
ENDIF (USE_CUDA)

INSTALL(TARGETS horus LIBRARY DESTINATION lib)
INSTALL(FILES ${HEADERS} DESTINATION include/horus)
