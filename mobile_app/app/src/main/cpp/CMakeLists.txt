cmake_minimum_required(VERSION 3.22.1)
project(swarved_voice_guard)

set(ONNXRUNTIME_INCLUDE_DIR
    "${CMAKE_CURRENT_LIST_DIR}/third_party/onnxruntime/include"
    CACHE PATH "Directory containing onnxruntime_cxx_api.h")
set(ONNXRUNTIME_LIBRARY
    "${CMAKE_CURRENT_LIST_DIR}/../jniLibs/${ANDROID_ABI}/libonnxruntime.so"
    CACHE FILEPATH "ABI-specific ONNX Runtime shared library")

if(NOT EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_cxx_api.h")
    message(FATAL_ERROR "ONNX Runtime C++ headers are missing. Add them under app/src/main/cpp/third_party/onnxruntime/include.")
endif()
if(NOT EXISTS "${ONNXRUNTIME_LIBRARY}")
    message(FATAL_ERROR "ONNX Runtime library is missing for ${ANDROID_ABI}. Add libonnxruntime.so under app/src/main/jniLibs/${ANDROID_ABI}.")
endif()

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}")
add_library(swarved_voice_guard SHARED native-lib.cpp)
target_compile_features(swarved_voice_guard PRIVATE cxx_std_17)
target_include_directories(swarved_voice_guard PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}")
find_library(android-log-lib log)
target_link_libraries(swarved_voice_guard PRIVATE onnxruntime ${android-log-lib})
