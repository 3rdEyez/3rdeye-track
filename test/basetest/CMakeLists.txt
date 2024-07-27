cmake_minimum_required(VERSION 3.10)
project(basetest)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
find_package(ncnn REQUIRED)

add_executable(basetest
${CMAKE_SOURCE_DIR}/3rdeye-track/src/detect_utils.cpp
${CMAKE_SOURCE_DIR}/3rdeye-track/src/sort_tracker.cpp
${CMAKE_SOURCE_DIR}/3rdeye-track/src/kalman_filter.cpp
${CMAKE_SOURCE_DIR}/3rdparty/hungarian-algorithm-cpp/Hungarian.cpp
../ncnn_detector.cpp
main.cpp
)

include_directories(
    ${CMAKE_SOURCE_DIR}/3rdparty
    ${CMAKE_SOURCE_DIR}/3rdparty/hungarian-algorithm-cpp
    ${CMAKE_SOURCE_DIR}/3rdeye-track/inc
    ..
)

target_link_libraries(
    basetest PRIVATE 
    ${OpenCV_LIBS}
    ncnn
)

# some definitions
add_definitions(-DNCNN_USE_GPU=0
                -DTEST_IMAGE_FILE="${TEST_IMAGE_FILE}")

