cmake_minimum_required(VERSION 2.8)

project(MKCF)

find_package(OpenCV REQUIRED)

set(CMAKE_BUILD_TYPE Release)

include_directories(${OpenCV_INCLUDE_DIRS})

add_definitions(-DWITH_OPENMP=true)
FIND_PACKAGE(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")


set(SOURCE_FILES src/main.cpp src/tracker.cpp)

add_executable(MKCF ${SOURCE_FILES})

target_link_libraries(MKCF ${OpenCV_LIBS})
