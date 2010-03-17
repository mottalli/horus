FIND_PACKAGE(PkgConfig)
PKG_CHECK_MODULES(OPENCV opencv)

FIND_PACKAGE(SWIG REQUIRED)
INCLUDE(${SWIG_USE_FILE})

FIND_PACKAGE(PythonLibs REQUIRED)
INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})

SET(CMAKE_SWIG_FLAGS "")

SET(SOURCES ../src/clock.cpp ../src/decorator.cpp ../src/eyelidsegmentator.cpp ../src/helperfunctions.cpp ../src/irisdatabase.cpp ../src/irisdctencoder.cpp ../src/irisencoder.cpp
	../src/irissegmentator.cpp ../src/iristemplate.cpp ../src/loggaborencoder.cpp ../src/parameters.cpp ../src/pupilsegmentator.cpp ../src/qualitychecker.cpp ../src/segmentationresult.cpp
	../src/segmentator.cpp ../src/serializer.cpp ../src/templatecomparator.cpp ../src/tools.cpp ../src/videoprocessor.cpp)

SET(LIBRARIES ${PYTHON_LIBRARIES} ${OPENCV_LIBRARIES})
	
IF (USE_CUDA)
	MESSAGE("Configuring python interface with CUDA...")
	FIND_PACKAGE(CUDA)

	SET(CUDA_FILES ../src/irisdatabase_kernel.cu)

	#SET(CUDASDK /home/marcelo/Mis_Documentos/Programacion/NVIDIA_GPU_Computing_SDK/C)
	SET(SOURCES ${SOURCES} ../src/irisdatabasecuda.cpp)
	#SET(CUDASDK_INC ${CUDASDK}/common/inc)
	#SET(CUDASDK_LIBS ${CUDASDK}/lib	)
	
	#CUDA_INCLUDE_DIRECTORIES(${CUDASDK_INC})
	#LINK_DIRECTORIES(${CUDASDK_LIBS})
	SET(LIBRARIES ${LIBRARIES} horus_cuda)
	SET(CUDA_NVCC_FLAGS --ptxas-options=-v --compiler-options '-fPIC')
	SET(CUDA_LIBRARIES ${CUDA_LIBRARIES} ${LIBRARIES})
	CUDA_ADD_LIBRARY(horus_cuda ${CUDA_FILES})
ENDIF (USE_CUDA)

CONFIGURE_FILE(setup.py.in setup.py)

ADD_LIBRARY(horus SHARED ${SOURCES})
INSTALL(TARGETS horus LIBRARY DESTINATION lib)

IF (USE_CUDA)
	TARGET_LINK_LIBRARIES(horus horus_cuda)
ENDIF (USE_CUDA)
