cmake_minimum_required(VERSION 3.10)

project(kalman_filter_test)

set(CMAKE_CXX_STANDARD 11)

include_directories(${CMAKE_SOURCE_DIR}/3rd_party)

add_executable(kalman_filter_test 
    ${CMAKE_CURRENT_SOURCE_DIR}/kalman_filter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)