PROJECT(tests)

SET(test_SOURCES main.cpp)
ADD_EXECUTABLE(test ${test_SOURCES})
TARGET_LINK_LIBRARIES(test horus ${OpenCV_LIBS})

SET(dual_cap_SOURCES dual_cap.cpp)
ADD_EXECUTABLE(dual_cap ${dual_cap_SOURCES})
TARGET_LINK_LIBRARIES(dual_cap horus ${OpenCV_LIBS})
