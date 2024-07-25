cmake_minimum_required(VERSION 3.10)

project(kalman_filter_test)

set(CMAKE_CXX_STANDARD 11)

include_directories(
    ${CMAKE_SOURCE_DIR}/3rdparty
    ${CMAKE_SOURCE_DIR}/3rdeye-track/inc
)

add_executable(kalman_filter_test 
    ${CMAKE_SOURCE_DIR}/3rdeye-track/src/kalman_filter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)