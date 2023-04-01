cmake_minimum_required(VERSION 3.13)

# Project
project(onnxruntime_samples C CXX)
if (WIN32)
 string(APPEND CMAKE_CXX_FLAGS " /W4")
else()
 string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra")
 string(APPEND CMAKE_C_FLAGS " -Wall -Wextra")
endif()

#onnxruntime providers
option(onnxruntime_USE_CUDA "Build with CUDA support" ON)
option(onnxruntime_USE_OPENVINO "Build with OpenVINO support" OFF)
option(onnxruntime_USE_NNAPI_BUILTIN "Build with builtin NNAPI lib for Android NNAPI support" OFF)
option(onnxruntime_USE_DNNL "Build with DNNL support" OFF)
option(onnxruntime_USE_NUPHAR "Build with Nuphar" OFF)
option(onnxruntime_USE_TENSORRT "Build with TensorRT support" OFF)
option(LIBPNG_ROOTDIR "libpng root dir")
# option(ONNXRUNTIME_ROOTDIR "/usr/local/include/onnxruntime")
include(FetchContent)


set(CMAKE_CXX_STANDARD 17)
# set(ONNXRUNTIME_ROOTDIR "/usr/")

set(rkcommom_DIR "/home/mengjiao/Desktop/externals/rkcommon/install/lib/cmake/rkcommon-1.10.0/")
find_package(rkcommon REQUIRED)

find_package(TBB REQUIRED)

add_subdirectory(utils)
include_directories(utils)

set (ONNXRUNTIME_ROOTDIR "/usr/local")

include_directories("/usr/local/include/onnxruntime")


message(STATUS "ONNXRUNTIME ROOT: ${ONNXRUNTIME_ROOTDIR}")

#TODO: we should only need one of them.
include_directories("${ONNXRUNTIME_ROOTDIR}/include" "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session")
# link_directories("${ONNXRUNTIME_ROOTDIR}/lib")


if(onnxruntime_USE_CUDA)
  add_definitions(-DUSE_CUDA)
endif()
if(onnxruntime_USE_OPENVINO)
  add_definitions(-DUSE_OPENVINO)
endif()
if(onnxruntime_USE_NNAPI_BUILTIN)
  add_definitions(-DUSE_NNAPI)
endif()
if(onnxruntime_USE_DNNL)
  add_definitions(-DUSE_DNNL)
endif()
if(onnxruntime_USE_NUPHAR)
  add_definitions(-DUSE_NUPHAR)
endif()
if(onnxruntime_USE_TENSORRT)
  add_definitions(-DUSE_TENSORRT)
endif()
if(onnxruntime_USE_DML)
  message("Enabling DML")
  add_definitions(-DUSE_DML)
endif()

add_executable(main "main.cpp")

# target_include_directories(main PRIVATE ${PROJECT_SOURCE_DIR}/include ${PNG_INCLUDE_DIRS})
target_link_libraries(main PRIVATE onnxruntime rkcommon::rkcommon utils TBB::tbb)

add_executable(main_2d "main_2D.cpp")

# target_include_directories(main PRIVATE ${PROJECT_SOURCE_DIR}/include ${PNG_INCLUDE_DIRS})
target_link_libraries(main_2d PRIVATE onnxruntime rkcommon::rkcommon utils TBB::tbb)
