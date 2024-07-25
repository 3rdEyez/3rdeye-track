cmake_minimum_required(VERSION 3.10)

project(eigen_test)

include_directories(${CMAKE_SOURCE_DIR}/3rdparty)

add_executable(eigen_test main.cpp)
