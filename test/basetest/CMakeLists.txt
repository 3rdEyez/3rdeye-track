cmake_minimum_required(VERSION 3.10)
project(basetest)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
find_package(ncnn REQUIRED)

add_executable(basetest
${CMAKE_SOURCE_DIR}/utils/utils.cpp
${CMAKE_SOURCE_DIR}/utils/cv_utils.cpp
../ncnn_detector.cpp
main.cpp
)

include_directories(
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/3rdparty
    ${CMAKE_SOURCE_DIR}/3rdparty/hungarian-algorithm-cpp
    ${CMAKE_SOURCE_DIR}/SORTTracker/inc
    ${CMAKE_SOURCE_DIR}/SORTTracker/share
    ..
)

target_link_libraries(
    basetest PRIVATE 
    ${OpenCV_LIBS}
    ncnn
    SORTTracker
)

# some definitions
add_definitions(-DNCNN_USE_GPU=1
                -DTEST_IMAGE_FILE="${TEST_IMAGE_FILE}")

