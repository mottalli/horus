SET(SOURCES ../src/clock.cpp ../src/decorator.cpp ../src/eyelidsegmentator.cpp ../src/irisdatabase.cpp ../src/irisencoder.cpp
	../src/irissegmentator.cpp ../src/iristemplate.cpp ../src/loggaborencoder.cpp ../src/pupilsegmentator.cpp ../src/qualitychecker.cpp ../src/segmentationresult.cpp
	../src/segmentator.cpp ../src/serializer.cpp ../src/templatecomparator.cpp ../src/tools.cpp ../src/videoprocessor.cpp ../src/gaborencoder.cpp
	main.cpp)

SET(HEADERS ../src/clock.h ../src/decorator.h ../src/eyelidsegmentator.h ../src/irisdatabase.h ../src/irisencoder.h
	../src/irissegmentator.h ../src/iristemplate.h ../src/loggaborencoder.h ../src/pupilsegmentator.h ../src/qualitychecker.h ../src/segmentationresult.h
	../src/segmentator.h ../src/serializer.h ../src/templatecomparator.h ../src/tools.h ../src/videoprocessor.h ../src/gaborencoder.h)

INCLUDE_DIRECTORIES(../src)

IF (USE_CUDA)
	SET(SOURCES ${SOURCES} ../src/irisdatabasecuda.cpp)
	SET(HEADERS ${HEADERS} ../src/cudacommon.h)
	
	SET(CUDA_SOURCES ../src/irisdatabase_kernel.cu ../src/gpuwrapper.cu)
	SET(CUDA_LIBRARIES ${CUDA_LIBRARIES} ${LIBRARIES})
	SET(CUDA_NVCC_FLAGS --ptxas-options=-v --compiler-options '-fPIC')			# Enable this to compile CUDA support in a dynamic library
	CUDA_ADD_LIBRARY(test_cuda ${CUDA_SOURCES})
ENDIF (USE_CUDA)

ADD_EXECUTABLE(test ${SOURCES} ${HEADERS})
TARGET_LINK_LIBRARIES(test ${OpenCV_LIBS})


IF (USE_CUDA)
	TARGET_LINK_LIBRARIES(test test_cuda)
ENDIF (USE_CUDA)

#INSTALL(TARGETS horus LIBRARY DESTINATION lib)
#INSTALL(FILES ${HEADERS} DESTINATION include/horus)
