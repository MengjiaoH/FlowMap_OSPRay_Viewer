cmake_minimum_required(VERSION 3.16)
project(CPP_INFERENCE)

set(rkcommom_DIR "/home/mengjiao/Desktop/externals/rkcommon/install/lib/cmake/rkcommon-1.10.0/")
find_package(rkcommon REQUIRED)

# list(APPEND PLUGINS "nvinfer")
# list(APPEND PLUGINS "nvonnxparser")
# list(APPEND PLUGINS "nvparsers")

# foreach(libName ${PLUGINS})
#     find_library(${libName}_lib NAMES ${libName} "/usr" PATH_SUFFIXES lib)
#     list(APPEND PLUGIN_LIBS "${${libName}_lib}")
# endforeach()

# find_library(NVINFER NAMES libnvinfer.so)
# find_library(NVPARSERS NAMES nvparsers)
# find_library(NVONNXPARSERS NAMES nvonnxparser)
# if(NVINFER)
#    message("TensorRT is available!")
#    message("NVINFER: ${NVINFER}")
#    message("NVPARSERS: ${NVPARSERS}")
#    message("NVONNXPARSERS: ${NVONNXPARSERS}")
#    set(TRT_AVAIL ON)
# else()
#   message("TensorRT is NOT Available")
#   set(TRT_AVAIL OFF)
# endif()

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(TBB REQUIRED)

add_subdirectory(utils)
include_directories(utils)


add_executable(inference_2D main_2D.cpp)
target_link_libraries(inference_2D PRIVATE rkcommon::rkcommon utils "${TORCH_LIBRARIES}" TBB::tbb)
set_property(TARGET inference_2D PROPERTY CXX_STANDARD 17)

add_executable(inference_3D main_3D.cpp)
target_link_libraries(inference_3D PRIVATE rkcommon::rkcommon utils "${TORCH_LIBRARIES}" TBB::tbb)
set_property(TARGET inference_3D PROPERTY CXX_STANDARD 17)
