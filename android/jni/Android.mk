LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)
OPENCV_CAMERA_MODULES:=off

OPENCV_MK_PATH:=/home/marcelo/Programas/OpenCV-trunk/android/build_armeabi/OpenCV.mk
ifeq ("$(wildcard $(OPENCV_MK_PATH))","")
	# Try to load OpenCV.mk from default install location
	include $(TOOLCHAIN_PREBUILT_ROOT)/user/share/OpenCV/OpenCV.mk
else
	include $(OPENCV_MK_PATH)
endif

LOCAL_MODULE    := horus_wrapper
LOCAL_SRC_FILES := horus_wrapper.cpp ../../src/segmentator.cpp ../../src/irissegmentator.cpp ../../src/pupilsegmentator.cpp ../../src/eyelidsegmentator.cpp ../../src/tools.cpp  
LOCAL_LDLIBS +=  -llog -ldl

include $(BUILD_SHARED_LIBRARY)