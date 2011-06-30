PROJECT(test)
SET(test_SOURCES main.cpp)

ADD_EXECUTABLE(test ${test_SOURCES})
TARGET_LINK_LIBRARIES(test horus ${OpenCV_LIBS})
