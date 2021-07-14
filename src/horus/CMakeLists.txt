# --- PROGRAM OPTIONS ---
OPTION(ENABLE_UEYE_DRIVER "Enable uEye driver" OFF)
OPTION(HORUS_SHARED_LIB "Build as a shared library" ON)

# --- DIRECTORIES ---
SET(HORUS_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/src/horus)
SET(HORUS_SRC_DIR ${PROJECT_SOURCE_DIR}/src/horus)

INCLUDE_DIRECTORIES(${HORUS_INCLUDE_DIR})

find_package(Boost REQUIRED COMPONENTS system filesystem)

# --- HORUS LIBRARY ---
SET(HORUS_LIB_SOURCES 
    ${HORUS_SRC_DIR}/decorator.cpp 
    ${HORUS_SRC_DIR}/eyelidsegmentator.cpp
    ${HORUS_SRC_DIR}/irisdatabase.cpp 
    ${HORUS_SRC_DIR}/irisencoder.cpp
    ${HORUS_SRC_DIR}/irissegmentator.cpp 
    ${HORUS_SRC_DIR}/iristemplate.cpp 
    ${HORUS_SRC_DIR}/loggaborencoder.cpp
    ${HORUS_SRC_DIR}/pupilsegmentator.cpp 
    ${HORUS_SRC_DIR}/videopupilsegmentator.cpp 
    ${HORUS_SRC_DIR}/qualitychecker.cpp 
    ${HORUS_SRC_DIR}/segmentator.cpp 
    ${HORUS_SRC_DIR}/videosegmentator.cpp 
    ${HORUS_SRC_DIR}/serializer.cpp 
    ${HORUS_SRC_DIR}/templatecomparator.cpp 
    ${HORUS_SRC_DIR}/tools.cpp
    ${HORUS_SRC_DIR}/videoprocessor.cpp
    ${HORUS_SRC_DIR}/gaborencoder.cpp 
    ${HORUS_SRC_DIR}/eyedetect.cpp
    ${HORUS_SRC_DIR}/external/base64.cpp 
    ${HORUS_SRC_DIR}/drivers/basedriver.cpp
    ${HORUS_SRC_DIR}/drivers/v4ldriver.cpp
    ${HORUS_SRC_DIR}/drivers/imagesequencedriver.cpp
)

SET(HORUS_LIB_HEADERS 
    ${HORUS_INCLUDE_DIR}/clock.h
    ${HORUS_INCLUDE_DIR}/common.h
    ${HORUS_INCLUDE_DIR}/decorator.h
    ${HORUS_INCLUDE_DIR}/eyelidsegmentator.h
    ${HORUS_INCLUDE_DIR}/irisdatabase.h
    ${HORUS_INCLUDE_DIR}/irisencoder.h
    ${HORUS_INCLUDE_DIR}/irissegmentator.h
    ${HORUS_INCLUDE_DIR}/iristemplate.h
    ${HORUS_INCLUDE_DIR}/loggaborencoder.h
    ${HORUS_INCLUDE_DIR}/pupilsegmentator.h
    ${HORUS_INCLUDE_DIR}/videopupilsegmentator.h
    ${HORUS_INCLUDE_DIR}/qualitychecker.h
    ${HORUS_INCLUDE_DIR}/segmentator.h
    ${HORUS_INCLUDE_DIR}/videosegmentator.h
    ${HORUS_INCLUDE_DIR}/serializer.h
    ${HORUS_INCLUDE_DIR}/templatecomparator.h
    ${HORUS_INCLUDE_DIR}/tools.h
    ${HORUS_INCLUDE_DIR}/videoprocessor.h
    ${HORUS_INCLUDE_DIR}/gaborencoder.h
    ${HORUS_INCLUDE_DIR}/types.h
    ${HORUS_INCLUDE_DIR}/external/base64.h
    ${HORUS_INCLUDE_DIR}/eyedetect.h
    ${HORUS_INCLUDE_DIR}/horus.h
    ${HORUS_INCLUDE_DIR}/drivers/basedriver.hpp
    ${HORUS_INCLUDE_DIR}/drivers/v4ldriver.hpp
    ${HORUS_INCLUDE_DIR}/drivers/imagesequencedriver.hpp
)

IF (ENABLE_UEYE_DRIVER)
    SET(HORUS_LIB_HEADERS ${HORUS_LIB_HEADERS} ${HORUS_INCLUDE_DIR}/drivers/ueyedriver.hpp)
    SET(HORUS_LIB_SOURCES ${HORUS_LIB_SOURCES} ${HORUS_SRC_DIR}/drivers/ueyedriver.cpp)
ENDIF (ENABLE_UEYE_DRIVER)

IF (HORUS_SHARED_LIB)
    ADD_LIBRARY(horus SHARED ${HORUS_LIB_SOURCES})
    TARGET_LINK_LIBRARIES(horus ${OpenCV_LIBS} ${Boost_LIBRARIES})

    IF (ENABLE_UEYE_DRIVER)
        TARGET_LINK_LIBRARIES(horus ueye_api)
    ENDIF (ENABLE_UEYE_DRIVER)

    # --- INSTALLATION ---
    INSTALL(TARGETS horus DESTINATION lib)
    INSTALL(FILES ${HORUS_LIB_HEADERS} DESTINATION include/horus)
ENDIF (HORUS_SHARED_LIB)
