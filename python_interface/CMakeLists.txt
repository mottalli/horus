FIND_PACKAGE(PkgConfig)
PKG_CHECK_MODULES(OPENCV opencv)

FIND_PACKAGE(SWIG REQUIRED)
INCLUDE(${SWIG_USE_FILE})

FIND_PACKAGE(PythonLibs)
INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})

SET(CMAKE_SWIG_FLAGS "")

SET(SOURCES ../src/clock.cpp ../src/decorator.cpp ../src/eyelidsegmentator.cpp ../src/helperfunctions.cpp ../src/irisdatabase.cpp ../src/irisdctencoder.cpp ../src/irisencoder.cpp
	../src/irissegmentator.cpp ../src/iristemplate.cpp ../src/loggaborencoder.cpp ../src/parameters.cpp ../src/pupilsegmentator.cpp ../src/qualitychecker.cpp ../src/segmentationresult.cpp
	../src/segmentator.cpp ../src/serializer.cpp ../src/templatecomparator.cpp ../src/tools.cpp ../src/videoprocessor.cpp)

SET(LIBRARIES ${PYTHON_LIBRARIES} ${OPENCV_LIBRARIES})
	
IF (USE_CUDA)
	FIND_PACKAGE(CUDA)

	SET(CUDASDK /home/marcelo/Mis_Documentos/Programacion/NVIDIA_GPU_Computing_SDK/C)
	SET(SOURCES ${SOURCES} ../src/irisdatabasecuda.cpp)
	
	CUDA_INCLUDE_DIRECTORIES(${CUDASDK}/common/inc)
	LINK_DIRECTORIES(${CUDASDK}/lib)
	CUDA_ADD_LIBRARY(horus_cuda SHARED ../src/irisdatabase_kernel.cu)
	SET(LIBRARIES ${LIBRARIES} horus_cuda cutil_x86_64)
ENDIF (USE_CUDA)

SET_SOURCE_FILES_PROPERTIES(horus.i PROPERTIES CPLUSPLUS ON)
SWIG_ADD_MODULE(horus python ${SOURCES} horus.i)
SWIG_LINK_LIBRARIES(horus ${LIBRARIES})
