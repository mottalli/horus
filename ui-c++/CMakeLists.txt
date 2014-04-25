PROJECT(ui)

FIND_PACKAGE(Qt4 REQUIRED)
INCLUDE(${QT_USE_FILE})
ADD_DEFINITIONS(${QT_DEFINITIONS})

SET(ui_HEADERS common.h imagewidget.h irisvideowidget.h matchingdialog.h registerdialog.h sqlite3wrapper.h
		debugdialog.h irisvideocapture.h mainwindow.h processingthread.h sqlite3irisdatabase.h videothread.h
		external/sqlite3/sqlite3.h
)

SET(ui_SOURCES debugdialog.cpp irisvideocapture.cpp main.cpp matchingdialog.cpp registerdialog.cpp sqlite3wrapper.cpp
		imagewidget.cpp irisvideowidget.cpp mainwindow.cpp processingthread.cpp sqlite3irisdatabase.cpp videothread.cpp
		external/sqlite3/sqlite3.c
)

SET(ui_FORMS debugdialog.ui mainwindow.ui matchingdialog.ui registerdialog.ui)

QT4_WRAP_CPP(ui_HEADERS_MOC ${ui_HEADERS})
QT4_WRAP_UI(ui_FORMS_HEADERS ${ui_FORMS})

FIND_PACKAGE(Boost REQUIRED COMPONENTS system filesystem date_time program_options thread)

INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})		# Without this, it cannot include the .h files (!)

IF (HORUS_SHARED_LIB)
	ADD_EXECUTABLE(ui ${ui_SOURCES}
		${ui_HEADERS_MOC}
		${ui_HEADERS}
		${ui_FORMS_HEADERS}
	)

	TARGET_LINK_LIBRARIES(ui horus dl ${QT_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})
ELSE()
	ADD_EXECUTABLE(ui ${ui_SOURCES}
		${ui_HEADERS_MOC}
		${ui_HEADERS}
		${ui_FORMS_HEADERS}
		${HORUS_LIB_SOURCES} 
		${HORUS_LIB_HEADERS}
	)

	TARGET_LINK_LIBRARIES(ui ${QT_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})
ENDIF()
