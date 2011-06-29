CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(Horus)

# ----------------------------------- Program options --------------------------------------
OPTION(USE_CUDA "Use CUDA acceleration" OFF)
OPTION(PYTHON_INTERFACE "Generate Python interface" ON)

# ----------------------------------- SOURCE FILES --------------------------------------
SET(HORUS_LIB_SOURCES 
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

# ----------------------------------- Compiler settings --------------------------------------
IF (CMAKE_COMPILER_IS_GNUCXX)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")		# Enable c++0x support (requires GCC >= 4.5)
ENDIF()

# ----------------------------------- OpenCV 2.2 --------------------------------------
#FIND_PACKAGE(OpenCV 2.2 REQUIRED HINTS ".")			 # This doesn't work (see CMake manual)
FIND_PACKAGE(OpenCV REQUIRED HINTS ".")
IF (${OpenCV_VERSION_MAJOR} EQUAL 2 AND ${OpenCV_VERSION_MINOR} LESS 2)
	MESSAGE(FATAL_ERROR "Needs at least OpenCV 2.2 (If present on system, copy the file OpenCVConfig.cmake to the current dir)")
ELSE()
	MESSAGE(STATUS "Found OpenCV ${OpenCV_VERSION}")
ENDIF()

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

# ----------------------------------- Boost --------------------------------------
IF (WIN32)
	SET(Boost_USE_STATIC_LIBS ON)
ENDIF()

FIND_PACKAGE(Boost REQUIRED COMPONENTS system filesystem date_time program_options thread serialization)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})

# ----------------------------------- CUDA --------------------------------------
IF (USE_CUDA)
	FIND_PACKAGE(CUDA REQUIRED)
	ADD_DEFINITIONS("-DUSE_CUDA")
	SET(CUDA_SOURCES
		${PROJECT_SOURCE_DIR}/src/irisdatabase_kernel.cu
	)
	
	SET(HORUS_LIB_SOURCES ${HORUS_LIB_SOURCES} ${PROJECT_SOURCE_DIR}/src/irisdatabasecuda.cpp)
	SET(HORUS_LIB_HEADERS ${HORUS_LIB_HEADERS} ${PROJECT_SOURCE_DIR}/src/irisdatabasecuda.h)
	
	# Enable this to compile CUDA support in a dynamic library
	SET(CUDA_NVCC_FLAGS --ptxas-options=-v --compiler-options '-fPIC' --compiler-bindir=/opt/gcc-4.4)
	SET(CUDA_PROPAGATE_HOST_FLAGS OFF)			# To disable a problem with the -std=c++0x flag
	CUDA_ADD_LIBRARY(horus_cuda ${CUDA_SOURCES})
ELSE ()
	SET(USE_CUDA OFF)
ENDIF ()


# ----------------------------------- SHARED LIBRARY --------------------------------------
ADD_LIBRARY(horus SHARED ${HORUS_LIB_SOURCES})
TARGET_LINK_LIBRARIES(horus ${OpenCV_LIBS} ${Boost_LIBRARIES})

IF (USE_CUDA)
	TARGET_LINK_LIBRARIES(horus horus_cuda)
ENDIF (USE_CUDA)

# ----------------------------------- INSTALLATION --------------------------------------

INSTALL(TARGETS horus DESTINATION lib)
INSTALL(FILES ${HORUS_LIB_HEADERS} DESTINATION include/horus)

# ----------------------------------- SUBDIRS --------------------------------------
IF (PYTHON_INTERFACE)
	ADD_SUBDIRECTORY(python_interface)
ENDIF()
ADD_SUBDIRECTORY(test)
ADD_SUBDIRECTORY(ui-c++)

# ----------------------------------- UNINSTALL --------------------------------------

CONFIGURE_FILE(
	"${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
	"${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
	IMMEDIATE @ONLY
)

ADD_CUSTOM_TARGET(uninstall
  "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake")
