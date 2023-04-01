cmake_minimum_required(VERSION 3.5)
project(miniscivis)

if (NOT WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
endif()

include(ExternalProject)

find_package(SDL2 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(ospray 2.0 REQUIRED)
find_package(rkcommon REQUIRED)
find_package(TBB REQUIRED)
find_package(OpenVisus)
include(cmake/glm.cmake)
find_package(VTK REQUIRED)

option(USE_EXPLICIT_ISOSURFACE "Use explicit isosurface extraction with VTK" OFF)
if (USE_EXPLICIT_ISOSURFACE)
    find_package(VTK)
endif()


find_package(CGAL)
include(CGAL_TBB_support) 

add_subdirectory(imgui)
add_subdirectory(util)

add_executable(mini_scivis
    main.cpp
    loader.cpp
    load_off.cpp
    load_vtk.cpp
    imgui_impl_opengl3.cpp
    imgui_impl_sdl.cpp)

set_target_properties(mini_scivis PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON)

target_include_directories(mini_scivis PUBLIC
    $<BUILD_INTERFACE:${SDL2_INCLUDE_DIRS}>
	$<BUILD_INTERFACE:${OPENGL_INCLUDE_DIR}>)

target_link_libraries(mini_scivis PUBLIC
    ospray::ospray
    rkcommon::rkcommon
    TBB::tbb
    util
    imgui
    CGAL::CGAL 
    CGAL::TBB_support
    ${SDL2_LIBRARIES}
    ${OPENGL_LIBRARY})

target_compile_definitions(mini_scivis PUBLIC
    -DNOMINMAX
    -DSDL_MAIN_HANDLED
    -DOSPRAY_CPP_RKCOMMON_TYPES)

if ("${VTK_FOUND}")
    target_compile_definitions(mini_scivis PUBLIC
        -DUSE_EXPLICIT_ISOSURFACE=1)
    target_include_directories(mini_scivis PUBLIC
        ${VTK_INCLUDE_DIRS})

    target_link_libraries(mini_scivis PUBLIC
        ${VTK_LIBRARIES})
else()
    message(WARNING "VTK not found, but is required for testing explicit isosurfaces. "
        "Implicit isosurfaces will be used instead.")
endif()

if (${OpenVisus_FOUND})
    target_compile_definitions(mini_scivis PUBLIC
        -DOPENVISUS_FOUND=1)
    target_link_libraries(mini_scivis PUBLIC
        OpenVisus::Idx)
else()
    message(WARNING "OpenVisus not found, IDX support will be disabled")
endif()

