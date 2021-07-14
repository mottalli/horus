export ANDROID_NDK=/home/marcelo/Documents/Programacion/android-ndk-r6
export ANDROID_SDK=/home/marcelo/Documents/Programacion/android-sdk-linux_x86
export OPENCV=/home/marcelo/Programas/OpenCV-trunk

# For android-cmake
export ANDROID_NDK_TOOLCHAIN_ROOT=$ANDROID_NDK/toolchains
export ANDTOOLCHAIN=$OPENCV/android/android.toolchain.cmake

export PATH=$ANDTOOLCHAIN:$ANDROID_NDK:$ANDROID_SDK/platform-tools:$ANDROID_SDK/tools:$ANDROID_NDK_TOOLCHAIN_ROOT:$PATH

alias cmake-android='cmake -DCMAKE_TOOLCHAIN_FILE=$ANDTOOLCHAIN -DARM_TARGET=armeabi'
